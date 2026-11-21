import json

import torch
from torch.utils.data import DataLoader

from data.dataset import QuickDrawDataset
from evaluate import accuracy_by_fraction
from losses import compute_losses
from model import SketchAhead, sample_continuation

C, T = 3, 30


def small_model():
    torch.manual_seed(0)
    return SketchAhead(num_classes=C, hidden=32, layers=2, class_emb=8, n_mix=4)


def fake_raw_dir(tmp_path, classes, per_class=40):
    """Write tiny ndjson files: each class is a random walk with a class-specific drift."""
    g = torch.Generator().manual_seed(0)
    for ci, name in enumerate(classes):
        with open(tmp_path / f"{name}.ndjson", "w") as f:
            for _ in range(per_class):
                strokes = []
                for _ in range(2):
                    k = int(torch.randint(3, 8, (1,), generator=g))
                    xs = (torch.rand(k, generator=g) * 100 + ci * 50).int().tolist()
                    ys = (torch.rand(k, generator=g) * 200).int().tolist()
                    strokes.append([xs, ys])
                f.write(json.dumps({"word": name, "recognized": True, "drawing": strokes}) + "\n")
    return tmp_path


def test_forward_shapes():
    m = small_model()
    x = torch.randn(4, T, 5)
    logits, mdn, state = m(x, torch.tensor([0, 1, 2, 0]))
    assert logits.shape == (4, T, C)
    assert mdn.shape == (4, T, 6 * 4 + 3)
    _, mdn_none, _ = m(x)
    assert mdn_none is None


def test_model_is_causal():
    """Changing future points must not change earlier guesses."""
    m = small_model().eval()
    x = torch.randn(1, T, 5)
    x2 = x.clone()
    x2[:, 15:] = torch.randn(1, T - 15, 5)
    a, _, _ = m(x)
    b, _, _ = m(x2)
    torch.testing.assert_close(a[:, :15], b[:, :15])
    assert not torch.allclose(a[:, 15:], b[:, 15:])


def test_stateful_steps_match_full_sequence():
    """Feeding points one by one (as the browser will) equals one full pass."""
    m = small_model().eval()
    x = torch.randn(1, T, 5)
    full, _, _ = m(x)
    state, steps = None, []
    for t in range(T):
        logits, _, state = m(x[:, t:t + 1], state=state)
        steps.append(logits)
    torch.testing.assert_close(torch.cat(steps, 1), full)


def test_dataset_and_losses(tmp_path):
    classes = ["a", "b", "c"]
    raw = fake_raw_dir(tmp_path, classes)
    ds = QuickDrawDataset(raw, classes, split="train", max_len=T, per_class=None, augment=True)
    assert len(ds) == 3 * 36  # 2 of every 20 lines go to val/test
    x, y, n, label = ds[0]
    assert x.shape == y.shape == (T, 5)
    torch.testing.assert_close(x[1:], y[:-1])  # x is y shifted by one, with SOS first
    assert y[n, 4] == 1 and y[n - 1, 4] == 0

    m = small_model()
    xb, yb, nb, lb = next(iter(DataLoader(ds, batch_size=8)))
    losses = compute_losses(m, xb, yb, nb, lb)
    assert all(torch.isfinite(v) for v in losses.values())
    losses["total"].backward()


def test_overfits_tiny_batch(tmp_path):
    """Sanity check that the whole loss is trainable."""
    classes = ["a", "b", "c"]
    ds = QuickDrawDataset(fake_raw_dir(tmp_path, classes), classes, max_len=T, per_class=None)
    xb, yb, nb, lb = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))
    m = small_model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = compute_losses(m, xb, yb, nb, lb)["total"].item()
    for _ in range(60):
        loss = compute_losses(m, xb, yb, nb, lb)["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first - 1.0


def test_accuracy_by_fraction_runs(tmp_path):
    classes = ["a", "b", "c"]
    ds = QuickDrawDataset(fake_raw_dir(tmp_path, classes), classes, split="val", max_len=T, per_class=None)
    curve = accuracy_by_fraction(small_model(), DataLoader(ds, batch_size=4), "cpu")
    assert len(curve) == 10 and all(0.0 <= v <= 1.0 for v in curve.values())


def test_sample_continuation_rows_are_valid():
    m = small_model()
    prefix = torch.zeros(5, 5)
    prefix[:, 2] = 1.0
    g = torch.Generator().manual_seed(0)
    out = sample_continuation(m, prefix, cls=1, max_steps=20, generator=g)
    assert out.ndim == 2 and out.shape[1] == 5 and len(out) <= 20
    if len(out):
        torch.testing.assert_close(out[:, 2:].sum(1), torch.ones(len(out)))
        assert (out[:, 4] == 0).all()
