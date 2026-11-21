"""Evaluation: accuracy as a function of how much of the drawing is done.

This is the project's main result: how early is the model right?
"""

import torch

FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@torch.no_grad()
def accuracy_by_fraction(model, loader, device, fractions=FRACTIONS):
    """Returns {fraction: accuracy}. Fraction f means the guess after
    max(1, round(f * n)) of the drawing's n points."""
    model.eval()
    correct = torch.zeros(len(fractions))
    total = 0
    for x, _, n, label in loader:
        x, n, label = x.to(device), n.to(device), label.to(device)
        logits, _, _ = model(x)
        pred = logits.argmax(-1)  # (B, T)
        for i, f in enumerate(fractions):
            t = torch.clamp(torch.round(f * n.float()).long(), min=1)
            p = pred.gather(1, t[:, None]).squeeze(1)
            correct[i] += (p == label).sum().item()
        total += len(label)
    return {f: (correct[i] / max(total, 1)).item() for i, f in enumerate(fractions)}


def format_curve(curve) -> str:
    return "  ".join(f"{int(f * 100):>3}%:{acc:.2f}" for f, acc in curve.items())
