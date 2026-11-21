"""PyTorch dataset over Quick, Draw! simplified ndjson files.

Each item is (x, y, n, label):
    x     (max_len, 5)  model input:  [SOS, p_1, ..., p_{max_len-1}]
    y     (max_len, 5)  target:       [p_1, ..., p_n, END, END, ...]
    n     int           number of real points
    label int           class index

At timestep t the model has seen x[0..t] = SOS, p_1..p_t and predicts y[t] =
p_{t+1}. So the class logits at step t are "the guess after t points".
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .strokes import SOS, augment, drawing_to_points, normalize, to_stroke5

# Deterministic split by line index: 1/20 test, 1/20 val, rest train.
_SPLIT_OF = {0: "test", 1: "val"}


def _split(i: int) -> str:
    return _SPLIT_OF.get(i % 20, "train")


class QuickDrawDataset(Dataset):
    def __init__(
        self,
        raw_dir,
        classes,
        split: str = "train",
        max_len: int = 200,
        per_class: int | None = 20_000,
        augment: bool = False,
    ):
        assert split in ("train", "val", "test")
        self.classes = list(classes)
        self.max_len = max_len
        self.augment = augment
        self.items: list[tuple[np.ndarray, int]] = []

        for label, name in enumerate(self.classes):
            path = Path(raw_dir) / f"{name}.ndjson"
            with open(path) as f:
                for i, line in enumerate(f):
                    if per_class is not None and i >= per_class:
                        break
                    if _split(i) != split:
                        continue
                    rec = json.loads(line)
                    if not rec.get("recognized", True):
                        continue
                    pts = drawing_to_points(rec["drawing"])
                    if 0 < len(pts) < max_len:
                        self.items.append((normalize(pts), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pts, label = self.items[idx]
        if self.augment:
            pts = augment(pts, np.random.default_rng())
        seq, n = to_stroke5(pts, self.max_len)
        x = np.vstack([SOS, seq[:-1]])
        return (
            torch.from_numpy(x),
            torch.from_numpy(seq),
            torch.tensor(n),
            torch.tensor(label),
        )
