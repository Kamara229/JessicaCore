"""CALIBRI S92: research-only first-order protected-gradient projection.

This is a NUMERICAL HELPER, NOT a strict no-forgetting certificate.
The real full-output CPU guard must validate every physical checkpoint.
No QA/TEST/LIVE access.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


def _dot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum((x.float() * y.float()).sum() for x, y in zip(a, b))


@torch.no_grad()
def project_primary_gradient(
    primary: Sequence[torch.Tensor],
    protected: Sequence[Sequence[torch.Tensor]],
    *,
    sweeps: int = 64,
    relative_tolerance: float = 1e-6,
) -> tuple[list[torch.Tensor], dict]:
    """Minimize interference with protected losses to first order.

    Find a direction g close to the proposed primary-loss gradient, satisfying
    <g, grad L_protected> >= 0 for each supplied protected example. Taking
    SGD step -eta*g therefore does not INCREASE the *first-order* loss of
    sampled protected examples. The finite-step nonlinear model can still
    forget, so an exact full/EOS CPU inference guard is indispensable.

    Uses Dykstra cyclic projections onto homogeneous half-spaces to compute
    the Euclidean projection. If it does not converge, returns the all-zero
    step (feasible, but no claimed learning). Does not edit caller tensors.
    """
    if not primary or not protected:
        raise ValueError("Nonempty primary and per-sample protected gradients required")
    if sweeps < 1 or not (0 < relative_tolerance < 1e-2):
        raise ValueError("Invalid projection budget or tolerance")
    shapes = [p.shape for p in primary]
    devices = [p.device for p in primary]
    for batch in protected:
        if (len(batch) != len(primary) or
            any(x.shape != sh or x.device != dv
                for x, sh, dv in zip(batch, shapes, devices))):
            raise ValueError("Each protection vector must match all primary tensors")
    if any(not torch.isfinite(x).all().item() for x in primary):
        raise ValueError("Nonfinite proposed gradient")
    if any(not torch.isfinite(x).all().item() for batch in protected for x in batch):
        raise ValueError("Nonfinite protection gradient")
    p = [[z.detach().float() for z in row] for row in protected]
    g = [z.detach().float().clone() for z in primary]
    corrections = [[torch.zeros_like(z) for z in g] for _ in p]
    lengths = [max(float(_dot(row, row).item()), 1e-30) for row in p]
    required = relative_tolerance
    max_violation = 0.0
    completed = 0
    for cycle in range(sweeps):
        for k, row in enumerate(p):
            old = [a + b for a, b in zip(g, corrections[k])]
            signed = float(_dot(old, row).item())
            if signed < 0:
                t = -signed / lengths[k]
                nxt = [a + t * b for a, b in zip(old, row)]
            else:
                nxt = old
            corrections[k] = [a - b for a, b in zip(old, nxt)]
            g = nxt
        completed = cycle + 1
        max_violation = max(
            max(0.0, -float(_dot(g, row).item())) / (lengths[k] ** 0.5)
            for k, row in enumerate(p)
        )
        if max_violation <= required:
            break
    feasible = max_violation <= required
    if not feasible:
        g = [torch.zeros_like(a) for a in g]
    norm = float(_dot(g, g).sqrt().item())
    return g, {
        "projection_feasible": feasible,
        "projection_sweeps": completed,
        "max_normalized_protected_violation": max_violation,
        "projected_direction_norm": norm,
        "can_apply_gradient": feasible and norm > 0,
    }
