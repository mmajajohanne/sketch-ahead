"""Plot ghost completions for partial validation drawings (milestone 3).

For each example drawing and each cut fraction, the drawing is cut at the
last stroke that ends before that fraction (the app samples on pointer-up),
the model guesses the class, and a ghost completion is sampled for each of
the top-k guesses.

Usage (from train/):
    python sample.py --ckpt checkpoints/gen/last.pt
    python sample.py --ckpt checkpoints/gen/last.pt --fractions 0.2 0.4 0.6 --temperature 0.3
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from data.dataset import QuickDrawDataset
from data.strokes import SOS, stroke5_to_points
from model import SketchAhead, sample_continuation

GHOST_COLORS = ["#d62728", "#1f77b4", "#2ca02c"]


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = SketchAhead(
        num_classes=len(ckpt["classes"]), hidden=config.HIDDEN, layers=config.LAYERS,
        class_emb=config.CLASS_EMB, n_mix=config.N_MIX, dropout=config.DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["classes"]


def cut_at_stroke(seq, n: int, fraction: float) -> np.ndarray:
    """First k rows of `seq`, where k is the end of the last stroke that ends at
    or before round(fraction * n). Always keeps at least the first stroke."""
    seq = np.asarray(seq)
    ends = np.flatnonzero(seq[:n, 3] > 0.5) + 1  # row counts at stroke ends
    target = round(fraction * n)
    ok = ends[ends <= target]
    k = int(ok[-1]) if len(ok) else int(ends[0])
    return seq[:k]


@torch.no_grad()
def class_probs(model, prefix) -> np.ndarray:
    x = torch.cat([torch.as_tensor(SOS)[None], torch.as_tensor(prefix)], 0)[None]
    logits, _, _ = model(x.float())
    return torch.softmax(logits[0, -1], -1).numpy()


def strokes_of(points: np.ndarray):
    """Split (N, 3) absolute points into a list of (k, 2) strokes."""
    out, cur = [], []
    for x, y, lift in points:
        cur.append((x, y))
        if lift > 0.5:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


def plot_panel(ax, prefix, ghosts, title):
    pts = stroke5_to_points(prefix)
    for s in strokes_of(pts):
        ax.plot(s[:, 0], s[:, 1], color="black", lw=2)
    for (ghost, color) in ghosts:
        if len(ghost) == 0:
            continue
        full = stroke5_to_points(np.vstack([prefix, ghost]))[len(prefix):]
        for s in strokes_of(full):
            ax.plot(s[:, 0], s[:, 1], color=color, lw=1.5, alpha=0.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # canvas y points down
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, default=Path(__file__).parent / "raw")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs" / "samples.png")
    ap.add_argument("--n", type=int, default=6, help="number of example drawings")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, classes = load_model(args.ckpt)
    ds = QuickDrawDataset(args.raw_dir, classes, split="val", max_len=config.MAX_LEN, per_class=400)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    idx = rng.choice(len(ds), size=args.n, replace=False)

    fig, axes = plt.subplots(args.n, len(args.fractions),
                             figsize=(2.4 * len(args.fractions), 2.6 * args.n), squeeze=False)
    for r, i in enumerate(idx):
        _, y, n, label = ds[int(i)]
        seq = y.numpy()
        for c, f in enumerate(args.fractions):
            prefix = cut_at_stroke(seq, int(n), f)
            p = class_probs(model, prefix)
            top = np.argsort(p)[::-1][: args.top_k]
            ghosts = [
                (sample_continuation(model, prefix, int(k), temperature=args.temperature, generator=gen).numpy(),
                 GHOST_COLORS[j % len(GHOST_COLORS)])
                for j, k in enumerate(top)
            ]
            guesses = "  ".join(f"{classes[k]} {p[k]:.2f}" for k in top)
            title = f"{classes[int(label)]} · {len(prefix)}/{int(n)} pts\n{guesses}"
            plot_panel(axes[r, c], prefix, ghosts, title)

    fig.suptitle("black = drawn so far · red = ghost for top guess · blue = 2nd guess", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
