"""Training losses.

total = class CE (every timestep after >= 1 real point)
      + offset NLL (MDN, only where the target is a real point)
      + pen-state CE (every step up to and including the first end token)

The per-timestep class loss is what trains the model on every prefix of every
drawing, i.e. "guess before it's finished".
"""

import torch
import torch.nn.functional as F

from model import bivariate_log_prob, mdn_split


def step_masks(n, T: int):
    """n: (B,) real point counts. Returns (cls_mask, gen_mask, xy_mask) builders."""
    steps = torch.arange(T, device=n.device)[None, :]
    gen_mask = steps <= n[:, None]           # targets p_1..p_n and the first END
    cls_mask = (steps >= 1) & gen_mask       # model has seen >= 1 real point
    xy_mask = steps < n[:, None]             # targets that are real points
    return cls_mask, gen_mask, xy_mask


def compute_losses(model, x, y, n, label, w_class=1.0, w_xy=1.0, w_pen=1.0):
    logits, mdn, _ = model(x, label)
    B, T, C = logits.shape
    cls_mask, gen_mask, xy_mask = step_masks(n, T)

    # Class: same label at every valid step.
    lab = label[:, None].expand(B, T)
    loss_class = F.cross_entropy(logits[cls_mask], lab[cls_mask])

    # Offsets: mixture NLL.
    log_pi, mux, muy, log_sx, log_sy, rho, pen_logits = mdn_split(mdn, model.n_mix)
    dx, dy = y[..., 0:1], y[..., 1:2]
    lp = torch.logsumexp(log_pi + bivariate_log_prob(dx, dy, mux, muy, log_sx, log_sy, rho), dim=-1)
    loss_xy = -lp[xy_mask].mean()

    # Pen state.
    pen_target = y[..., 2:5].argmax(-1)
    loss_pen = F.cross_entropy(pen_logits[gen_mask], pen_target[gen_mask])

    total = w_class * loss_class + w_xy * loss_xy + w_pen * loss_pen
    return {
        "total": total,
        "class": loss_class.detach(),
        "xy": loss_xy.detach(),
        "pen": loss_pen.detach(),
    }
