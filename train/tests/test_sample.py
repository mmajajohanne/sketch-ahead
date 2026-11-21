import numpy as np
import torch

from data.strokes import drawing_to_points, normalize, to_stroke5
from model import SketchAhead
from sample import class_probs, cut_at_stroke, plot_panel, strokes_of

# Three strokes with 3, 2 and 4 points -> stroke ends after rows 3, 5, 9.
DRAWING = [
    [[10, 50, 90], [20, 20, 20]],
    [[50, 50], [30, 100]],
    [[10, 20, 30, 40], [200, 210, 220, 230]],
]


def seq_and_n():
    seq, n = to_stroke5(normalize(drawing_to_points(DRAWING)), max_len=12)
    return seq, n


def test_cut_at_stroke_respects_boundaries():
    seq, n = seq_and_n()
    assert n == 9
    assert len(cut_at_stroke(seq, n, 0.1)) == 3   # never less than the first stroke
    assert len(cut_at_stroke(seq, n, 0.6)) == 5   # round(5.4) = 5 -> end of stroke 2
    assert len(cut_at_stroke(seq, n, 1.0)) == 9
    assert cut_at_stroke(seq, n, 0.6)[-1, 3] == 1  # ends on a pen lift


def test_strokes_of_splits_on_lift():
    pts = normalize(drawing_to_points(DRAWING))
    assert [len(s) for s in strokes_of(pts)] == [3, 2, 4]


def test_class_probs_and_plot_run():
    torch.manual_seed(0)
    model = SketchAhead(num_classes=3, hidden=16, layers=1, class_emb=4, n_mix=3).eval()
    seq, n = seq_and_n()
    prefix = cut_at_stroke(seq, n, 0.5)
    p = class_probs(model, prefix)
    assert p.shape == (3,) and np.isclose(p.sum(), 1.0, atol=1e-5)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_panel(ax, prefix, [(seq[5:9], "red"), (np.zeros((0, 5)), "blue")], "t")
    plt.close(fig)
