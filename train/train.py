"""Train the sketch-ahead model.

Usage (from train/):
    python -m data.download                 # once
    python train.py --epochs 10
    python train.py --no-generator          # milestone 2: class head only
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data.dataset import QuickDrawDataset
from evaluate import accuracy_by_fraction, format_curve
from losses import compute_losses
from model import SketchAhead


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=Path(__file__).parent / "raw")
    ap.add_argument("--out", default=Path(__file__).parent / "checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--per-class", type=int, default=config.PER_CLASS)
    ap.add_argument("--no-generator", action="store_true",
                    help="train only the class head (milestone 2)")
    ap.add_argument("--init", type=Path, default=None,
                    help="start from this checkpoint (e.g. a class-head-only run)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    common = dict(classes=config.CLASSES, max_len=config.MAX_LEN, per_class=args.per_class)
    train_ds = QuickDrawDataset(args.raw_dir, split="train", augment=True, **common)
    val_ds = QuickDrawDataset(args.raw_dir, split="val", augment=False, **common)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    model = SketchAhead(
        num_classes=len(config.CLASSES), hidden=config.HIDDEN, layers=config.LAYERS,
        class_emb=config.CLASS_EMB, n_mix=config.N_MIX, dropout=config.DROPOUT,
    ).to(device)
    if args.init is not None:
        ckpt = torch.load(args.init, map_location=device)
        assert ckpt["classes"] == config.CLASSES, "checkpoint was trained on other classes"
        model.load_state_dict(ckpt["model"])
        print(f"initialized from {args.init} (epoch {ckpt['epoch']})")
    opt =torch.optim.Adam(model.parameters(), lr=args.lr)
    w_gen = 0.0 if args.no_generator else 1.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total": 0.0, "class": 0.0, "xy": 0.0, "pen": 0.0}
        for step, (x, y, n, label) in enumerate(train_dl, 1):
            x, y, n, label = x.to(device), y.to(device), n.to(device), label.to(device)
            losses = compute_losses(model, x, y, n, label, w_xy=w_gen, w_pen=w_gen)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in sums:
                sums[k] += losses[k].detach().item()
            if step % 100 == 0:
                avg = {k: v / step for k, v in sums.items()}
                print(f"epoch {epoch} step {step}  " + "  ".join(f"{k}={v:.3f}" for k, v in avg.items()))

        curve = accuracy_by_fraction(model, val_dl, device)
        print(f"epoch {epoch} val accuracy by fraction drawn:  {format_curve(curve)}")
        torch.save(
            {"model": model.state_dict(), "classes": config.CLASSES, "epoch": epoch, "val_curve": curve},
            out / "last.pt",
        )


if __name__ == "__main__":
    main()
