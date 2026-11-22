"""Shared configuration for training.

The class list deliberately mixes classes that look alike early in a drawing
(cat / dog / pig / lion / owl / bear / rabbit / mouse: they all start with a
round head) with a couple of clearly different ones, so the belief chart has
something interesting to show.
"""

CLASSES = [
    "cat",
    "dog",
    "pig",
    "lion",
    "owl",
    "bear",
    "rabbit",
    "mouse",
    "house",
    "bicycle",
]

# Max number of points per drawing (after RDP simplification in the dataset).
# Drawings with more points are skipped. One slot is reserved for the end token.
MAX_LEN = 200

# Drawings read per class before the train/val/test split.
PER_CLASS = 20_000

# Model
HIDDEN = 256
LAYERS = 2
CLASS_EMB = 32
N_MIX = 20
# Must stay 0.0 while training on Apple MPS: with inter-layer dropout > 0,
# PyTorch 2.8's MPS LSTM reuses its training graph for later eval calls, so
# the model learns a function that a standard LSTM (CPU, ONNX, browser) does
# not reproduce. See tests/test_mps.py.
DROPOUT = 0.0
