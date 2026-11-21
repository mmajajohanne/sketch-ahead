"""Causal LSTM with a class head and a class-conditioned MDN head.

    x_t (stroke-5) ──► LSTM ──► h_t ──┬─► class_head ──► logits_t        (belief)
                                      └─► [h_t ; emb(c)] ─► mdn_head ─► params_t (ghost)

Both heads read the same hidden state, so one forward step per new point
gives both the belief update and the next-point distribution.
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
        self.lstm = nn.LSTM(
            5, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.class_head = nn.Linear(hidden, num_classes)
        self.class_emb = nn.Embedding(num_classes, class_emb)
        self.mdn_head = nn.Linear(hidden + class_emb, 6 * n_mix + 3)

    def forward(self, x, cls=None, state=None):
        """x: (B, T, 5). cls: (B,) long or None (skip the generator head).

        Returns class logits (B, T, C), MDN params (B, T, 6M+3) or None, and
        the LSTM state (for stateful step-by-step inference).
        """
        h, state = self.lstm(x, state)
        logits = self.class_head(h)
        mdn = None
        if cls is not None:
            e = self.class_emb(cls)[:, None, :].expand(-1, h.size(1), -1)
            mdn = self.mdn_head(torch.cat([h, e], dim=-1))
        return logits, mdn, state


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

    _, mdn, state = model(x, c)
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
        _, mdn, state = model(row[None, None], c, state)
        params = mdn[0, -1]
    if not out:
        return torch.zeros(0, 5)
    return torch.stack(out).cpu()
