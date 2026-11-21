import numpy as np
import pytest

from data.strokes import (
    CANVAS,
    augment,
    drawing_to_points,
    normalize,
    stroke5_to_points,
    to_stroke5,
)

# Two strokes: a 3-point line and a 2-point line.
DRAWING = [
    [[10, 50, 90], [20, 20, 20]],
    [[50, 50], [30, 100]],
]


def test_drawing_to_points_marks_stroke_ends():
    pts = drawing_to_points(DRAWING)
    assert pts.shape == (5, 3)
    np.testing.assert_array_equal(pts[:, 2], [0, 0, 1, 0, 1])
    np.testing.assert_array_equal(pts[0, :2], [10, 20])


def test_normalize_divides_by_canvas_not_bbox():
    pts = normalize(drawing_to_points(DRAWING))
    np.testing.assert_allclose(pts[2, :2], [90 / CANVAS, 20 / CANVAS])


def test_stroke5_roundtrip():
    pts = normalize(drawing_to_points(DRAWING))
    seq, n = to_stroke5(pts, max_len=10)
    assert n == 5
    np.testing.assert_allclose(stroke5_to_points(seq), pts, atol=1e-6)


def test_stroke5_pen_states_and_end_padding():
    pts = normalize(drawing_to_points(DRAWING))
    seq, n = to_stroke5(pts, max_len=10)
    # Exactly one pen state per row.
    np.testing.assert_array_equal(seq[:, 2:].sum(1), np.ones(10))
    np.testing.assert_array_equal(seq[:n, 3], [0, 0, 1, 0, 1])
    assert (seq[n:, 4] == 1).all() and (seq[n:, :2] == 0).all()


def test_stroke5_requires_room_for_end_token():
    pts = normalize(drawing_to_points(DRAWING))
    with pytest.raises(ValueError):
        to_stroke5(pts, max_len=5)


def test_augment_keeps_shape_and_pen():
    pts = normalize(drawing_to_points(DRAWING))
    out = augment(pts, np.random.default_rng(0))
    assert out.shape == pts.shape
    np.testing.assert_array_equal(out[:, 2], pts[:, 2])
    assert not np.allclose(out[:, :2], pts[:, :2])
