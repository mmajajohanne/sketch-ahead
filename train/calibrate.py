"""Temperature scaling for the class head (milestone 4).

The belief chart shows probabilities, not just the top guess, so they should
mean what they say: of all the times the model says 70 %, it should be right
about 70 % of the time. Temperature scaling fits a single scalar T on the
validation split (softmax(logits / T)) and leaves the argmax, and so the
accuracy, unchanged. It is reported on the test split, which was not used
for training or for fitting T.

Logits are taken at the same points as the accuracy curve (10 %, 20 %, ...,
100 % of each drawing), so short and long drawings count equally and partial
drawings, which is what the app mostly shows, make up most of the data.

Usage (from train/):
    python calibrate.py --ckpt checkpoints/gen-long/last.pt
Writes <ckpt dir>/calibration.json and outputs/calibration.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data.dataset import QuickDrawDataset
from evaluate import FRACTIONS
from sample import load_model


@torch.no_grad()
def collect_logits(model, loader, device, fractions=FRACTIONS):
    """Returns logits (N, C), labels (N,), fraction (N,) with one row per
    drawing and fraction (same positions as `accuracy_by_fraction`)."""
    model.eval()
    out_logits, out_labels, out_frac = [], [], []
    for x, _, n, label in loader:
        x, n, label = x.to(device), n.to(device), label.to(device)
        logits, _, _ = model(x)
        for f in fractions:
            t = torch.clamp(torch.round(f * n.float()).long(), min=1)
            out_logits.append(logits[torch.arange(len(t)), t].cpu())
            out_labels.append(label.cpu())
            out_frac.append(torch.full((len(t),), f))
    return torch.cat(out_logits), torch.cat(out_labels), torch.cat(out_frac)


def fit_temperature(logits, labels, iters: int = 200) -> float:
    """Scalar T > 0 minimizing NLL of softmax(logits / T)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def ece(probs, labels, bins: int = 15) -> float:
    """Expected calibration error on the top-1 confidence."""
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return total


def reliability(probs, labels, bins: int = 15):
    """(bin centre, mean confidence, accuracy, count) for non-empty bins."""
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((conf[m].mean().item(), correct[m].mean().item(), int(m.sum())))
    return rows


def report(logits, labels, frac, temperature):
    """Dict with NLL / ECE before and after scaling, overall and per fraction."""
    def stats(lg, lb):
        p0, p1 = F.softmax(lg, -1), F.softmax(lg / temperature, -1)
        return {
            "nll": F.cross_entropy(lg, lb).item(),
            "nll_scaled": F.cross_entropy(lg / temperature, lb).item(),
            "ece": ece(p0, lb),
            "ece_scaled": ece(p1, lb),
            "accuracy": (lg.argmax(-1) == lb).float().mean().item(),
        }

    out = {"all": stats(logits, labels)}
    for f in FRACTIONS:
        m = torch.isclose(frac, torch.tensor(f))
        out[f"{int(round(f * 100))}%"] = stats(logits[m], labels[m])
    return out


def plot(logits, labels, temperature, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls="--", label="perfect")
    for lg, name, color in [(logits, "raw", "#d62728"), (logits / temperature, f"T = {temperature:.2f}", "#1f77b4")]:
        rows = reliability(F.softmax(lg, -1), labels)
        conf, acc, _ = zip(*rows)
        ax.plot(conf, acc, marker="o", ms=3, color=color, label=name)
    ax.set_xlabel("confidence (top-1 probability)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Reliability on test split (all fractions)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, default=Path(__file__).parent / "raw")
    ap.add_argument("--per-class", type=int, default=config.PER_CLASS)
    ap.add_argument("--plot", type=Path, default=Path(__file__).parent / "outputs" / "calibration.png")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, classes = load_model(args.ckpt, device)

    def logits_for(split):
        ds = QuickDrawDataset(args.raw_dir, classes, split=split, max_len=config.MAX_LEN, per_class=args.per_class)
        return collect_logits(model, DataLoader(ds, batch_size=256), device)

    val = logits_for("val")
    temperature = fit_temperature(val[0], val[1])
    test = logits_for("test")
    rep = report(*test, temperature)

    print(f"temperature T = {temperature:.3f} (fit on val, {len(val[1])} rows)")
    print(f"test ({len(test[1])} rows)       acc    NLL raw -> scaled    ECE raw -> scaled")
    for k, s in rep.items():
        print(f"  {k:>5}  {s['accuracy']:.3f}    {s['nll']:.3f} -> {s['nll_scaled']:.3f}"
              f"       {s['ece']:.3f} -> {s['ece_scaled']:.3f}")

    out = args.ckpt.parent / "calibration.json"
    out.write_text(json.dumps({"temperature": temperature, "test": rep}, indent=2))
    plot(test[0], test[1], temperature, args.plot)
    print(f"saved {out} and {args.plot}")


if __name__ == "__main__":
    main()
