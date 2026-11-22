"""Export the trained model to ONNX for the browser.

Two stateful graphs, so the web worker can run them separately:

classifier.onnx  (fed once per new stroke, state carried between calls)
    inputs   x      float32 (B, T, 5)    stroke-5 rows (start with SOS)
             h0, c0 float32 (L, B, H)    LSTM state (zeros at the start)
    outputs  probs  float32 (B, T, C)    calibrated softmax(logits / T)
             h1, c1 float32 (L, B, H)    state after the last step

generator.onnx   (rerun over the prefix for the current top class, then
                  stepped one sampled point at a time)
    inputs   x      float32 (B, T, 5)
             cls    int64   (B,)         class whose ghost to generate
             h0, c0 float32 (L, B, H)
    outputs  mdn    float32 (B, T, 6M+3) raw MDN params (layout in model.json)
             h1, c1 float32 (L, B, H)

T is dynamic. The temperature from calibrate.py is baked into `probs`.
Everything the browser preprocessing must match (canvas size, origin, offset
scale, SOS, class list) goes into model.json next to the .onnx files.

Usage (from train/):
    python export_onnx.py --ckpt checkpoints/v3/last.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from data.strokes import CANVAS, OFFSET_SCALE, ORIGIN, SOS
from sample import load_model

OPSET = 17
WEB_PUBLIC = Path(__file__).resolve().parent.parent / "web" / "public"
STATE_AXES = {1: "batch"}
SEQ_AXES = {0: "batch", 1: "steps"}


class ClassifierStep(nn.Module):
    def __init__(self, model, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("inv_t", torch.tensor(1.0 / temperature))

    def forward(self, x, h0, c0):
        logits, (h1, c1) = self.model.classify(x, (h0, c0))
        return torch.softmax(logits * self.inv_t, dim=-1), h1, c1


class GeneratorStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cls, h0, c0):
        mdn, (h1, c1) = self.model.generate(x, cls, (h0, c0))
        return mdn, h1, c1


def zero_state(model, batch: int = 1):
    shape = (model.lstm.num_layers, batch, model.lstm.hidden_size)
    return torch.zeros(shape), torch.zeros(shape)


def export(model, temperature: float, out_dir: Path):
    """Writes classifier.onnx and generator.onnx to `out_dir`."""
    model = model.cpu().eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.as_tensor(SOS)[None, None].repeat(1, 3, 1)
    cls = torch.zeros(1, dtype=torch.long)
    h0, c0 = zero_state(model)
    state_axes = {"h0": STATE_AXES, "c0": STATE_AXES, "h1": STATE_AXES, "c1": STATE_AXES}
    torch.onnx.export(
        ClassifierStep(model, temperature).eval(), (x, h0, c0), str(out_dir / "classifier.onnx"),
        input_names=["x", "h0", "c0"], output_names=["probs", "h1", "c1"],
        dynamic_axes={"x": SEQ_AXES, "probs": SEQ_AXES, **state_axes},
        opset_version=OPSET, dynamo=False,
    )
    torch.onnx.export(
        GeneratorStep(model).eval(), (x, cls, h0, c0), str(out_dir / "generator.onnx"),
        input_names=["x", "cls", "h0", "c0"], output_names=["mdn", "h1", "c1"],
        dynamic_axes={"x": SEQ_AXES, "cls": {0: "batch"}, "mdn": SEQ_AXES, **state_axes},
        opset_version=OPSET, dynamo=False,
    )


def metadata(model, classes, temperature: float) -> dict:
    m = model.n_mix
    return {
        "classes": list(classes),
        "temperature": temperature,
        "canvas": CANVAS,
        "origin": [float(v) for v in ORIGIN],
        "offset_scale": OFFSET_SCALE,
        "sos": [float(v) for v in SOS],
        "max_len": config.MAX_LEN,
        "hidden": model.lstm.hidden_size,
        "layers": model.lstm.num_layers,
        "n_mix": m,
        "mdn_layout": {
            "pi_logits": [0, m],
            "mu_x": [m, 2 * m],
            "mu_y": [2 * m, 3 * m],
            "log_sigma_x": [3 * m, 4 * m],
            "log_sigma_y": [4 * m, 5 * m],
            "rho_raw (tanh, clamp +-0.999)": [5 * m, 6 * m],
            "pen_logits (down, lift, end)": [6 * m, 6 * m + 3],
        },
        "files": {"classifier": "classifier.onnx", "generator": "generator.onnx"},
        "opset": OPSET,
    }


def _compare(sess, feed_extra, step, x, h, c, n_seq_out):
    """Max |onnx - torch| for one full call and for T single-step calls."""
    with torch.no_grad():
        ref = [t.numpy() for t in step(x, *feed_extra.values(), h, c)]
    extra = {k: v.numpy() for k, v in feed_extra.items()}

    def run(xs, hs, cs):
        return sess.run(None, {"x": xs, **extra, "h0": hs, "c0": cs})

    full = run(x.numpy(), h.numpy(), c.numpy())
    worst = max(np.abs(a - b).max() for a, b in zip(full, ref))
    hs, cs, seq = h.numpy(), c.numpy(), [[] for _ in range(n_seq_out)]
    for t in range(x.shape[1]):
        *outs, hs, cs = run(x[:, t:t + 1].numpy(), hs, cs)
        for i, o in enumerate(outs):
            seq[i].append(o)
    stepped = [np.concatenate(s, 1) for s in seq] + [hs, cs]
    return max(worst, max(np.abs(a - b).max() for a, b in zip(stepped, ref)))


def verify(out_dir: Path, model, temperature: float, prefix: np.ndarray, cls: int = 0, atol: float = 1e-4) -> float:
    """Compare onnxruntime with PyTorch on `prefix` (T, 5, SOS first) for both
    graphs, in one call and step by step. Raises if a difference exceeds atol."""
    import onnxruntime as ort

    model = model.cpu().eval()
    x = torch.as_tensor(prefix, dtype=torch.float32)[None]
    h, c = zero_state(model)
    sess = lambda name: ort.InferenceSession(str(out_dir / name), providers=["CPUExecutionProvider"])  # noqa: E731
    worst = max(
        _compare(sess("classifier.onnx"), {}, ClassifierStep(model, temperature).eval(), x, h, c, 1),
        _compare(sess("generator.onnx"), {"cls": torch.tensor([cls])}, GeneratorStep(model).eval(), x, h, c, 1),
    )
    if worst > atol:
        raise AssertionError(f"ONNX differs from PyTorch by {worst:.2e} (atol {atol})")
    return float(worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=None,
                    help="calibration.json (default: next to the checkpoint)")
    ap.add_argument("--out-dir", type=Path, default=WEB_PUBLIC)
    ap.add_argument("--raw-dir", type=Path, default=Path(__file__).parent / "raw")
    args = ap.parse_args()

    model, classes = load_model(args.ckpt, "cpu")
    cal = args.calibration or args.ckpt.parent / "calibration.json"
    if cal.exists():
        temperature = json.loads(cal.read_text())["temperature"]
    else:
        print(f"warning: {cal} not found, exporting without calibration (T = 1)")
        temperature = 1.0

    export(model, temperature, args.out_dir)
    (args.out_dir / "model.json").write_text(json.dumps(metadata(model, classes, temperature), indent=2) + "\n")
    old = args.out_dir / "model.onnx"  # single-graph export from v1/v2
    if old.exists():
        old.unlink()

    # Check on a real validation drawing, not only random input.
    from data.dataset import QuickDrawDataset

    ds = QuickDrawDataset(args.raw_dir, classes, split="val", max_len=config.MAX_LEN, per_class=50)
    x, _, n, label = ds[0]
    worst = verify(args.out_dir, model, temperature, x[: int(n) + 1].numpy(), cls=int(label))
    size = sum((args.out_dir / f).stat().st_size for f in ("classifier.onnx", "generator.onnx")) / 1e6
    print(f"saved classifier.onnx + generator.onnx ({size:.1f} MB) and model.json to {args.out_dir}, "
          f"T = {temperature:.3f}, max |onnx - torch| = {worst:.1e}")


if __name__ == "__main__":
    main()
