"""Guard against a PyTorch MPS LSTM bug.

With inter-layer dropout > 0, the first train-mode call on MPS caches a graph
that is reused for later eval calls, so MPS and CPU disagree in eval. A model
trained that way reports good validation accuracy during training but loses
most of it once reloaded (or exported to ONNX). Found 2026-11-21 with
PyTorch 2.8.0: val accuracy 0.88 during training, 0.23 after reload.
"""

import pytest
import torch

import config
from model import SketchAhead

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")


def _eval_diff_after_train_call(dropout: float) -> float:
    torch.manual_seed(0)
    cpu = SketchAhead(num_classes=3, hidden=32, layers=2, class_emb=4, n_mix=3, dropout=dropout)
    gpu = SketchAhead(num_classes=3, hidden=32, layers=2, class_emb=4, n_mix=3, dropout=dropout).to("mps")
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(4, 20, 5)
    gpu.train()
    gpu(x.to("mps"))[0].sum().backward()  # first MPS call: train mode + backward
    gpu.eval()
    cpu.eval()
    with torch.no_grad():
        return (cpu(x)[0] - gpu(x.to("mps"))[0].cpu()).abs().max().item()


@mps
def test_configured_dropout_gives_same_eval_on_mps_and_cpu():
    assert _eval_diff_after_train_call(config.DROPOUT) < 1e-4


@mps
def test_bug_still_present_with_dropout():
    """If this starts failing, PyTorch has fixed the bug and dropout can come back."""
    assert _eval_diff_after_train_call(0.1) > 1e-2
