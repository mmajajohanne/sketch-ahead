import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from calibrate import ece, fit_temperature
from data.strokes import SOS
from export_onnx import export, metadata, verify
from model import SketchAhead

pytest.importorskip("onnxruntime")


def _tiny():
    torch.manual_seed(0)
    return SketchAhead(num_classes=4, hidden=16, layers=2, class_emb=4, n_mix=3, dropout=0.0).eval()


def _prefix(T=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.zeros((T, 5), dtype=np.float32)
    rows[:, :2] = rng.normal(size=(T, 2))
    lift = rng.random(T) < 0.2
    rows[:, 2], rows[:, 3] = ~lift, lift
    return np.vstack([SOS, rows])


def test_onnx_matches_torch_full_and_stepwise(tmp_path):
    model = _tiny()
    path = tmp_path / "m.onnx"
    export(model, temperature=1.7, path=path)
    assert verify(path, model, 1.7, _prefix(), cls=2) < 1e-4


def test_onnx_dynamic_batch(tmp_path):
    import onnxruntime as ort

    model = _tiny()
    path = tmp_path / "m.onnx"
    export(model, 1.0, path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.stack([_prefix(seed=0), _prefix(seed=1)])
    h = np.zeros((2, 2, 16), np.float32)
    probs, mdn, h1, _ = sess.run(None, {"x": x, "cls": np.array([0, 3]), "h0": h, "c0": h})
    assert probs.shape == (2, 13, 4) and mdn.shape == (2, 13, 6 * 3 + 3) and h1.shape == (2, 2, 16)
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-5)


def test_metadata_is_json_and_layout_covers_mdn():
    model = _tiny()
    meta = json.loads(json.dumps(metadata(model, ["a", "b", "c", "d"], 1.3)))
    spans = sorted(meta["mdn_layout"].values())
    assert spans[0][0] == 0 and spans[-1][1] == 6 * 3 + 3
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_fit_temperature_recovers_known_scale():
    torch.manual_seed(0)
    true = torch.randn(4000, 5) * 2
    labels = torch.distributions.Categorical(logits=true).sample()
    # Overconfident model: logits 3x too large -> T should come out near 3.
    t = fit_temperature(true * 3, labels)
    assert 2.5 < t < 3.5
    assert ece(F.softmax(true * 3 / t, -1), labels) < ece(F.softmax(true * 3, -1), labels)
