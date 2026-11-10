"""CALIBRI S105: project a desired gain update to protect old TRAIN losses.

This is a diagnostic/direction-selection utility, not a quality guarantee.
Actual inference safety still requires complete physical 456/456 S101
full-answer + EOS/first-token checks, inclusive bpp and CPU speed guards.

Goal: compare gradients of NEW wrong TRAIN answers against the gradients
of OLD S101-correct answers, at each of the ten all-layer shared GT gains.
Compute a projected candidate gain direction that is locally (first-order)
non-increasing for the chosen protected examples.

Do not train on previously exposed QA, DEV or TEST; do not edit LIVE.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ProtectedGainDirection:
    direction: torch.Tensor
    original_direction: torch.Tensor
    constraints: int
    original_violations: int
    remaining_violations: int
    desired_improvement: float
    projected_improvement: float
    norm_ratio: float
    iterations: int


def project_protected_gain_direction(
    target_gradient: torch.Tensor,
    protected_gradients: torch.Tensor,
    *,
    max_iterations: int = 500,
    atol: float = 1e-7,
) -> ProtectedGainDirection:
    """Dykstra projection of -target_gradient onto halfspaces g_old · d <= 0.

    Assumes target-gradient = d(loss_wrong)/d(gains). No update is applied.
    All work occurs on CPU float64 to make small 10D constraints stable.
    Iterative convex projection is only a FIRST ORDER approximation.
    """
    goal = target_gradient.detach().to("cpu", dtype=torch.float64).flatten()
    guards = protected_gradients.detach().to("cpu", dtype=torch.float64)
    if guards.ndim != 2 or guards.shape[1] != goal.numel() or not guards.shape[0]:
        raise ValueError("Need one or more matched protected gain gradients")
    if not torch.isfinite(goal).all() or not torch.isfinite(guards).all():
        raise ValueError("Nonfinite training gradient")
    if not max_iterations > 0 or atol <= 0:
        raise ValueError("Invalid convex-projection tolerance")
    raw = -goal
    if goal.norm().item() == 0:
        raise ValueError("Target update has no gradient signal")
    # Normalize each halfspace's normal; positive scaling changes no constraint.
    n = guards / guards.norm(dim=1, keepdim=True).clamp_min(1e-18)
    valid = guards.norm(dim=1) > 1e-15
    n = n[valid]
    v = raw.clone()
    correction = torch.zeros_like(n)
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        before = v.clone()
        for index in range(n.shape[0]):
            shifted = v + correction[index]
            violation = torch.dot(shifted, n[index])
            if violation > 0:
                projection = shifted - violation * n[index]
            else:
                projection = shifted
            correction[index] = shifted - projection
            v = projection
        violations = n @ v if n.shape[0] else torch.zeros(0, dtype=v.dtype)
        if ((violations.numel() == 0 or violations.max().item() <= atol)
                and (v - before).norm().item() <= atol):
            break
    # Reject directions numerically violating constraints. Zero movement is safe.
    post = n @ v if n.shape[0] else torch.zeros(0, dtype=v.dtype)
    if bool((post > atol).any()):
        v.zero_()
        post = n @ v if n.shape[0] else torch.zeros(0, dtype=v.dtype)
    if torch.dot(goal, v).item() > atol:
        v.zero_()
        post = n @ v if n.shape[0] else torch.zeros(0, dtype=v.dtype)
    prior = n @ raw if n.shape[0] else torch.zeros(0, dtype=v.dtype)
    return ProtectedGainDirection(
        direction=v,
        original_direction=raw,
        constraints=int(n.shape[0]),
        original_violations=int((prior > atol).sum().item()),
        remaining_violations=int((post > atol).sum().item()),
        desired_improvement=float(-torch.dot(goal, raw)),
        projected_improvement=float(-torch.dot(goal, v)),
        norm_ratio=float(v.norm() / raw.norm()),
        iterations=iterations,
    )
