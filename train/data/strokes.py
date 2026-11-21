"""Stroke representations and preprocessing.

IMPORTANT: web/src/preprocess.ts must reproduce `drawing_to_points`,
`normalize` and `to_stroke5` exactly. Any change here must be mirrored there.

Representations
---------------
points   (N, 3) float32: absolute (x, y, lift), lift = 1 on the last point
         of each stroke. Coordinates are in canvas units.
stroke-5 (max_len, 5) float32: (dx, dy, p_down, p_lift, p_end) per row.
         - dx, dy: offset from the previous point, divided by OFFSET_SCALE.
           The first point is offset from the canvas centre (0.5, 0.5), so
           start position is kept.
         - p_down: pen stays down after this point (next point is same stroke)
         - p_lift: pen is lifted after this point (end of a stroke)
         - p_end:  drawing is finished; all padding rows are end tokens.
"""

import numpy as np

# Quick, Draw! simplified coordinates lie in [0, 255]. The live canvas is
# mapped to the same range, so normalization is by a constant (canvas size),
# NOT by each drawing's bounding box: that box is unknown mid-drawing.
CANVAS = 256.0
ORIGIN = np.array([0.5, 0.5], dtype=np.float32)

# Std of (dx, dy) in normalized coordinates, measured on the training set
# (≈ 0.12 / 0.10 for x / y, 0.11 joint). Offsets in stroke-5 are divided by
# this so they have roughly unit scale, which the MDN head needs to learn
# useful mixture components (Sketch-RNN does the same).
OFFSET_SCALE = 0.11

# Start-of-sequence token fed to the model before the first point.
SOS = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def drawing_to_points(drawing) -> np.ndarray:
    """Simplified ndjson `drawing` ([[xs, ys], ...] per stroke) -> (N, 3)."""
    rows = []
    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        n = len(xs)
        for i in range(n):
            rows.append((xs[i], ys[i], 1.0 if i == n - 1 else 0.0))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def normalize(points: np.ndarray) -> np.ndarray:
    """Canvas units -> [0, 1]."""
    out = points.copy()
    out[:, :2] /= CANVAS
    return out


def augment(
    points: np.ndarray,
    rng: np.random.Generator,
    scale_range=(0.5, 1.1),
    stretch: float = 0.1,
    shift: float = 0.15,
) -> np.ndarray:
    """Random scale, aspect stretch and shift around the canvas centre.

    Operates on normalized points. Makes the model robust to how big and
    where on the canvas people draw, which matters because we normalize by
    canvas size rather than by bounding box.
    """
    out = points.copy()
    s = rng.uniform(*scale_range)
    sx = s * (1.0 + rng.uniform(-stretch, stretch))
    sy = s * (1.0 + rng.uniform(-stretch, stretch))
    out[:, 0] = (out[:, 0] - 0.5) * sx + 0.5 + rng.uniform(-shift, shift)
    out[:, 1] = (out[:, 1] - 0.5) * sy + 0.5 + rng.uniform(-shift, shift)
    return out


def to_stroke5(points: np.ndarray, max_len: int):
    """Normalized points (N, 3) -> (stroke5 (max_len, 5), n).

    Rows n..max_len-1 are end tokens, so N must be < max_len (at least one end
    token, which the generator learns to emit).
    """
    n = len(points)
    if n == 0 or n >= max_len:
        raise ValueError(f"need 0 < n < max_len, got n={n}, max_len={max_len}")
    seq = np.zeros((max_len, 5), dtype=np.float32)
    xy = np.vstack([ORIGIN, points[:, :2]])
    seq[:n, 0:2] = np.diff(xy, axis=0) / OFFSET_SCALE
    lift = points[:, 2]
    seq[:n, 2] = 1.0 - lift
    seq[:n, 3] = lift
    seq[n:, 4] = 1.0
    return seq, n


def stroke5_to_points(seq: np.ndarray) -> np.ndarray:
    """Inverse of `to_stroke5` (stops at the first end token)."""
    seq = np.asarray(seq, dtype=np.float32)
    end = np.flatnonzero(seq[:, 4] > 0.5)
    n = int(end[0]) if len(end) else len(seq)
    xy = np.cumsum(seq[:n, 0:2] * OFFSET_SCALE, axis=0) + ORIGIN
    return np.column_stack([xy, seq[:n, 3]]).astype(np.float32)
