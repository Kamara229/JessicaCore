"""Training signals only: hidden trajectories plus hard token ranking.

No KL/NLL/PPL gate. These objectives are not answer-quality measurements.
"""
import torch
from .protocol import HIDDEN_WEIGHT, MARGIN_WEIGHT, MARGIN


def hard_margin(logits, target, mask):
    flat = logits[mask]
    ids = target[mask]
    if not len(ids):
        raise ValueError('No supervised token positions')
    good = flat.gather(1, ids[:, None]).squeeze(1)
    # Top two suffice to exclude the target without allocating a full mask.
    top, topids = flat.topk(2, dim=-1)
    bad = torch.where(topids[:, 0] == ids, top[:, 1], top[:, 0])
    return (MARGIN + bad - good).relu().mean()


def objective(result, reference):
    if len(result.hidden_states) != len(reference['hidden']):
        raise ValueError('Missing original layer targets')
    errors = [(s-t).square().mean() / t.square().mean().clamp_min(1e-8)
              for s, t in zip(result.hidden_states, reference['hidden'])]
    hidden = torch.stack(errors).mean()
    margin = hard_margin(result.logits, reference['targets'], reference['mask'])
    return HIDDEN_WEIGHT * hidden + MARGIN_WEIGHT * margin, hidden, margin

