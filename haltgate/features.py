"""Cheap per-loop features read from the looped state, plus the offline Nmax oracle trace."""
import torch
from .data import SLOT

SCALARS = ["loop", "ent_mean", "ent_max", "conf_min", "delta_norm", "flip"]


def loop_features(prev, h, logits, x, i, nmax, prev_pred):
    """[B, d + len(SCALARS)]: mean-pool of h, slot entropy stats, min confidence, loop index, delta-norm, flip rate."""
    tok, slot = (x != 0).unsqueeze(-1).float(), x == SLOT  # answer slots are visible in the input; no labels used
    pool = (h * tok).sum(1) / tok.sum(1)
    lp = logits.log_softmax(-1)
    ent = -(lp.exp() * lp).sum(-1)
    n = slot.sum(1)
    ent_mean = (ent * slot).sum(1) / n
    ent_max = ent.masked_fill(~slot, 0).amax(1)
    conf_min = lp.amax(-1).exp().masked_fill(~slot, 1).amin(1)
    dn = ((h - prev).norm(dim=-1) / (prev.norm(dim=-1) + 1e-6) * tok[..., 0]).sum(1) / tok[..., 0].sum(1)
    pred = logits.argmax(-1)
    flip = ((pred != prev_pred) & slot).sum(1) / n if prev_pred is not None else torch.ones_like(n, dtype=h.dtype)
    loop = torch.full_like(n, i, dtype=h.dtype) / nmax
    return torch.cat([pool, torch.stack([loop, ent_mean, ent_max, conf_min, dn.clamp(max=10), flip], 1)], 1), pred


@torch.no_grad()
def trace(model, x, y, bs=1024):
    """Run every item to Nmax offline. Returns feats [M, Nmax, F] and exact-match correct [M, Nmax] (bool)."""
    model.eval()
    F, C = [], []
    for s in range(0, len(x), bs):
        xb, yb = x[s : s + bs], y[s : s + bs]
        fs, cs, pp = [], [], None
        for i, (prev, h, logits) in enumerate(model(xb), 1):
            f, pp = loop_features(prev, h, logits, xb, i, model.nmax, pp)
            fs.append(f), cs.append(((pp == yb) | (yb == -100)).all(1))
        F.append(torch.stack(fs, 1)), C.append(torch.stack(cs, 1))
    return torch.cat(F), torch.cat(C)
