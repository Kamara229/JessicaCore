"""S139 optimizer-aware J7: protect the *actual* Adam displacement.

S131 projects raw gradients; Adam's preconditioning/momentum can rotate
its subsequent step back into old-loss increasing directions. Project the
displacement on the 24 trainable stage2 channels as a separate last-mile
first-order guard. The real persisted 456+128 FP16 CPU J7 remains decisive.

Research TRAIN only; no heldout QA, production, or automatic acceptance.
"""
from __future__ import annotations

import math
import torch
from calibri2.s131.protected_gradients import project_against_old_loss_increase


def correct_adam_stage2_displacement(
    before: torch.Tensor,
    after: torch.Tensor,
    old_gradients: list[torch.Tensor],
    *,
    passes: int = 32,
    cosine_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Return a first-order protected post-Adam weight proposal.

    Gradient of old loss g and displacement d=w_after-w_before must have
    g·d <= 0 to avoid first-order old-loss increase. We project -d into
    the positive gradient halfspaces; only the caller's new-only channel
    slice is supplied. If cyclic projection fails tolerance, return the
    original weights rather than accepting an unverified movement.

    This is NOT a guarantee of full/EOS output preservation. Physical J7
    serialization, exact old/new identity checks and optimizer rollback
    remain mandatory.
    """
    if (
        before.shape != after.shape or before.device != after.device
        or before.dtype != after.dtype or not before.is_floating_point()
        or not bool(torch.isfinite(before).all())
        or not bool(torch.isfinite(after).all())
        or not old_gradients or not 1 <= passes <= 32
        or not math.isfinite(cosine_tolerance)
        or not 0 < cosine_tolerance < 1e-3
    ):
        raise ValueError("S139 invalid optimizer displacement or settings")

    constraints = []
    for grad in old_gradients:
        if (
            grad.shape != before.shape or grad.device != before.device
            or not grad.is_floating_point()
            or not bool(torch.isfinite(grad).all())
        ):
            raise ValueError("S139 invalid old TRAIN constraint")
        grad32 = grad.detach().float()
        norm = float(torch.linalg.vector_norm(grad32))
        if norm > 1e-12:
            constraints.append((grad32, norm))
    if not constraints:
        raise AssertionError("S139 no nonzero protected TRAIN gradients")

    direction = before.detach().float() - after.detach().float()
    original_size = float(torch.linalg.vector_norm(direction))

    def minimum_cosine(candidate: torch.Tensor) -> float:
        length = float(torch.linalg.vector_norm(candidate))
        if length <= 1e-12:
            return 0.0
        return min(
            float(torch.sum(candidate * grad)) / (length * norm)
            for grad, norm in constraints
        )

    initial = minimum_cosine(direction)
    audit = {
        "old_gradient_constraints": len(constraints),
        "post_adam_min_cosine_before": initial,
        "post_adam_min_cosine_after": initial,
        "post_adam_raw_displacement_norm": original_size,
        "post_adam_projection_required": initial < -cosine_tolerance,
        "post_adam_zero_fallback": False,
        "post_adam_first_order_only": True,
    }
    if initial >= -cosine_tolerance:
        return after.detach().clone(), audit

    projected, projection_info = project_against_old_loss_increase(
        direction, [g for g, _ in constraints], passes=passes,
    )
    aligned = minimum_cosine(projected.float())
    audit["post_adam_min_cosine_after"] = aligned
    audit["post_adam_projection_corrections"] = (
        projection_info["cyclic_projection_corrections"]
    )

    if aligned < -cosine_tolerance:
        audit["post_adam_zero_fallback"] = True
        audit["post_adam_min_cosine_after"] = 0.0
        return before.detach().clone(), audit

    safe = (before.detach().float() - projected.float()).to(before.dtype)
    if not bool(torch.isfinite(safe).all()):
        raise AssertionError("S139 projected candidate nonfinite")
    # Check the actual rounded candidate, not only its float32 projection.
    actual = minimum_cosine(before.detach().float() - safe.float())
    if actual < -cosine_tolerance:
        audit["post_adam_zero_fallback"] = True
        audit["post_adam_min_cosine_after"] = 0.0
        return before.detach().clone(), audit
    audit["post_adam_min_cosine_after"] = actual
    return safe.detach(), audit
