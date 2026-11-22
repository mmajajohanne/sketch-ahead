"""Two causal LSTMs over the same stroke-5 input: a classifier and a
class-conditioned generator.

    x_t ──────────────► LSTM ──► h_t ──► class_head ──► logits_t        (belief)
    [x_t ; emb(c)] ───► gen LSTM ──► g_t ──► mdn_head ──► params_t      (ghost)

v1 had one shared LSTM with the class embedding only at the MDN head. The
generator then barely used the class (telling it the wrong class cost only
0.13 nats per point) and drifted after a few strokes, because the recurrent
state that has to carry "what is left to draw" never knew what was being
drawn. Here the generator gets the class at every step, Sketch-RNN style,
and has its own state, so it does not share capacity with the classifier.

The generator state depends on the class, so when the top guess changes the
generator must be rerun over the prefix (one batched call; cheap).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.strokes import SOS


class SketchAhead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden: int = 256,
        layers: int = 2,
        class_emb: int = 32,
        n_mix: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_mix = n_mix
        drop = dropout if layers > 1 else 0.0
        self.lstm = nn.LSTM(5, hidden, layers, batch_first=True, dropout=drop)
        self.class_head = nn.Linear(hidden, num_classes)
        self.class_emb = nn.Embedding(num_classes, class_emb)
        self.gen_lstm = nn.LSTM(5 + class_emb, hidden, layers, batch_first=True, dropout=drop)
        self.mdn_head = nn.Linear(hidden + class_emb, 6 * n_mix + 3)

    def classify(self, x, state=None):
        """x: (B, T, 5) -> class logits (B, T, C), classifier LSTM state."""
        h, state = self.lstm(x, state)
        return self.class_head(h), state

    def generate(self, x, cls, state=None):
        """x: (B, T, 5), cls: (B,) -> MDN params (B, T, 6M+3), generator state."""
        e = self.class_emb(cls)[:, None, :].expand(-1, x.size(1), -1)
        g, state = self.gen_lstm(torch.cat([x, e], dim=-1), state)
        return self.mdn_head(torch.cat([g, e], dim=-1)), state

    def forward(self, x, cls=None, state=None):
        """x: (B, T, 5). cls: (B,) long or None (skip the generator).

        `state` is (classifier_state, generator_state); either may be None.
        Returns class logits (B, T, C), MDN params (B, T, 6M+3) or None, and
        the new state pair (for stateful step-by-step inference).
        """
        cs, gs = state if state is not None else (None, None)
        logits, cs = self.classify(x, cs)
        mdn = None
        if cls is not None:
            mdn, gs = self.generate(x, cls, gs)
        return logits, mdn, (cs, gs)


def mdn_split(params, n_mix: int):
    """Raw MDN output -> (log_pi, mu_x, mu_y, log_sx, log_sy, rho, pen_logits)."""
    pi, mux, muy, log_sx, log_sy, rho = params[..., : 6 * n_mix].split(n_mix, dim=-1)
    pen_logits = params[..., 6 * n_mix:]
    log_pi = F.log_softmax(pi, dim=-1)
    rho = torch.tanh(rho).clamp(-0.999, 0.999)
    return log_pi, mux, muy, log_sx, log_sy, rho, pen_logits


def bivariate_log_prob(dx, dy, mux, muy, log_sx, log_sy, rho):
    """log N((dx, dy) | mu, sigma, rho), broadcast over mixture components."""
    sx, sy = log_sx.exp(), log_sy.exp()
    zx = (dx - mux) / sx
    zy = (dy - muy) / sy
    one_m_r2 = 1.0 - rho**2
    z = zx**2 + zy**2 - 2.0 * rho * zx * zy
    return (
        -z / (2.0 * one_m_r2)
        - math.log(2.0 * math.pi)
        - log_sx
        - log_sy
        - 0.5 * torch.log(one_m_r2)
    )


@torch.no_grad()
def sample_continuation(
    model: SketchAhead,
    prefix,
    cls: int,
    max_steps: int = 150,
    temperature: float = 0.4,
    generator: torch.Generator | None = None,
):
    """Sample how a drawing continues ("ghost completion").

    prefix: (T, 5) stroke-5 rows of the drawing so far (no SOS, no END rows).
    Returns (K, 5) sampled stroke-5 rows, ending before the end token.
    """
    model.eval()
    device = next(model.parameters()).device
    prefix = torch.as_tensor(prefix, dtype=torch.float32, device=device)
    sos = torch.as_tensor(SOS, device=device)[None]
    x = torch.cat([sos, prefix], dim=0)[None]
    c = torch.tensor([cls], device=device)

    mdn, state = model.generate(x, c)
    params = mdn[0, -1]
    out = []
    for _ in range(max_steps):
        log_pi, mux, muy, log_sx, log_sy, rho, pen_logits = mdn_split(params, model.n_mix)
        k = torch.multinomial(F.softmax(log_pi / temperature, -1), 1, generator=generator).item()
        sx = log_sx[k].exp() * math.sqrt(temperature)
        sy = log_sy[k].exp() * math.sqrt(temperature)
        z1, z2 = torch.randn(2, device=device, generator=generator)
        dx = mux[k] + sx * z1
        dy = muy[k] + sy * (rho[k] * z1 + torch.sqrt(1 - rho[k] ** 2) * z2)
        pen = torch.multinomial(F.softmax(pen_logits / temperature, -1), 1, generator=generator).item()
        if pen == 2:
            break
        row = torch.zeros(5, device=device)
        row[0], row[1], row[2 + pen] = dx, dy, 1.0
        out.append(row)
        mdn, state = model.generate(row[None, None], c, state)
        params = mdn[0, -1]
    if not out:
        return torch.zeros(0, 5)
    return torch.stack(out).cpu()
