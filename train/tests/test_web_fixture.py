"""Python side of the preprocessing parity check with the web app.

web/test/fixtures/preprocess.json holds a few drawings and the stroke-5 rows
Python produces for them. This test checks the fixture still matches Python;
web/test/preprocess.test.ts checks that preprocess.ts matches the fixture.
If you change data/strokes.py on purpose, regenerate the fixture:

    python -m tests.test_web_fixture   (from train/)
"""

import json
from pathlib import Path

import numpy as np

from data.strokes import drawing_to_points, normalize, to_stroke5

FIXTURE = Path(__file__).resolve().parents[2] / "web" / "test" / "fixtures" / "preprocess.json"

# Small hand-made drawings covering: single-point stroke, several strokes,
# points at the canvas edges.
DRAWINGS = [
    [[[10, 60, 120], [20, 25, 200]]],
    [[[128], [128]], [[0, 255, 255, 0], [0, 0, 255, 255]], [[40, 41], [90, 91]]],
    [[[200, 180, 150, 120, 100], [30, 60, 70, 65, 40]], [[150, 150], [70, 140]], [[90, 210], [140, 140]]],
]


def rows_for(drawing):
    points = normalize(drawing_to_points(drawing))
    seq, n = to_stroke5(points, max_len=len(points) + 1)
    return seq[:n]


def test_fixture_matches_python():
    data = json.loads(FIXTURE.read_text())
    assert len(data["cases"]) == len(DRAWINGS)
    for case in data["cases"]:
        np.testing.assert_allclose(rows_for(case["drawing"]), np.array(case["rows"]), atol=1e-6)


if __name__ == "__main__":
    cases = [{"drawing": d, "rows": rows_for(d).astype(float).round(7).tolist()} for d in DRAWINGS]
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    print(f"wrote {FIXTURE}")
