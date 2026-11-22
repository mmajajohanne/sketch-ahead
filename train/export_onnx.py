"""Export the trained model to ONNX for the browser (milestone 5).

The exported graph is stateful, so the web worker can run one step per new
point and carry the LSTM state between calls:

    inputs   x      float32 (B, T, 5)   stroke-5 rows (start with SOS)
             cls    int64   (B,)        class whose ghost to generate
             h0     float32 (L, B, H)   LSTM state (zeros at the start)
             c0     float32 (L, B, H)
    outputs  probs  float32 (B, T, C)   calibrated softmax(logits / T)
             mdn    float32 (B, T, 6M+3) raw MDN params (layout in model.json)
             h1, c1 float32 (L, B, H)   state after the last step

T is dynamic, so the same model also replays a whole prefix in one call (on
undo, or to restart a ghost from a saved state). The temperature from
calibrate.py is baked into `probs`. Everything the browser preprocessing must
match (canvas size, origin, offset scale, SOS, class list) goes into
model.json next to the .onnx file.

Usage (from train/):
    python export_onnx.py --ckpt checkpoints/gen-long/last.pt
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


class StepModel(nn.Module):
    def __init__(self, model, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("inv_t", torch.tensor(1.0 / temperature))

    def forward(self, x, cls, h0, c0):
        logits, mdn, (h1, c1) = self.model(x, cls, (h0, c0))
        probs = torch.softmax(logits * self.inv_t, dim=-1)
        return probs, mdn, h1, c1


def zero_state(model, batch: int = 1):
    shape = (model.lstm.num_layers, batch, model.lstm.hidden_size)
    return torch.zeros(shape), torch.zeros(shape)


def export(model, temperature: float, path: Path):
    model = model.cpu().eval()
    step = StepModel(model, temperature).eval()
    x = torch.as_tensor(SOS)[None, None].repeat(1, 3, 1)
    cls = torch.zeros(1, dtype=torch.long)
    h0, c0 = zero_state(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        step, (x, cls, h0, c0), str(path),
        input_names=["x", "cls", "h0", "c0"],
        output_names=["probs", "mdn", "h1", "c1"],
        dynamic_axes={
            "x": {0: "batch", 1: "steps"},
            "cls": {0: "batch"},
            "h0": {1: "batch"},
            "c0": {1: "batch"},
            "probs": {0: "batch", 1: "steps"},
            "mdn": {0: "batch", 1: "steps"},
            "h1": {1: "batch"},
            "c1": {1: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
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
        "opset": OPSET,
    }


def verify(path: Path, model, temperature: float, prefix: np.ndarray, cls: int = 0, atol: float = 1e-4) -> float:
    """Compare onnxruntime with PyTorch on `prefix` (T, 5, SOS first), both
    in one call and one step at a time with the state carried over. Returns
    the largest absolute difference; raises if it exceeds `atol`."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    step = StepModel(model.cpu().eval(), temperature).eval()
    x = torch.as_tensor(prefix, dtype=torch.float32)[None]
    c = torch.tensor([cls])
    h, cc = zero_state(model)
    with torch.no_grad():
        ref = [t.numpy() for t in step(x, c, h, cc)]

    feed = lambda xs, h, c_: {"x": xs, "cls": c.numpy(), "h0": h, "c0": c_}  # noqa: E731
    full = sess.run(None, feed(x.numpy(), h.numpy(), cc.numpy()))
    worst = max(np.abs(a - b).max() for a, b in zip(full, ref))

    hs, cs = h.numpy(), cc.numpy()
    probs, mdn = [], []
    for t in range(x.shape[1]):
        p, m, hs, cs = sess.run(None, feed(x[:, t:t + 1].numpy(), hs, cs))
        probs.append(p)
        mdn.append(m)
    stepped = [np.concatenate(probs, 1), np.concatenate(mdn, 1), hs, cs]
    worst = max(worst, max(np.abs(a - b).max() for a, b in zip(stepped, ref)))
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

    onnx_path = args.out_dir / "model.onnx"
    export(model, temperature, onnx_path)
    (args.out_dir / "model.json").write_text(json.dumps(metadata(model, classes, temperature), indent=2) + "\n")

    # Check on a real validation drawing, not only random input.
    from data.dataset import QuickDrawDataset

    ds = QuickDrawDataset(args.raw_dir, classes, split="val", max_len=config.MAX_LEN, per_class=50)
    x, _, n, label = ds[0]
    worst = verify(onnx_path, model, temperature, x[: int(n) + 1].numpy(), cls=int(label))
    size = onnx_path.stat().st_size / 1e6
    print(f"saved {onnx_path} ({size:.1f} MB) and model.json, T = {temperature:.3f}, "
          f"max |onnx - torch| = {worst:.1e}")


if __name__ == "__main__":
    main()
