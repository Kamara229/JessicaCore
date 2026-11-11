"""S142 bounded dual-cone correction of actual Adam displacement, research TRAIN only.

Only 24 new additive-core channels are modified. This first-order cone
projection never replaces persisted FP16 456+128 CPU J7 and exact rollback.
No QA, production Jessica, Render, Supabase or APK modifications.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.optimize import minimize


def correct_adam_stage2_displacement(
    before: torch.Tensor,
    after: torch.Tensor,
    old_gradients: list[torch.Tensor],
    *,
    passes: int = 32,
    cosine_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Project negative actual update d=before-after into TRAIN halfspaces.

    Each protected old gradient g requires g·d>=0 for a non-increasing
    first-order loss. L-BFGS-B solves the nonnegative dual of the cone QP,
    which may be more accurate than S139's 32 cyclic corrections. If the
    *actual rounded* proposal violates the tolerance, return the baseline.
    No first-order method guarantees preservation of complete CPU answers.
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
        raise ValueError("S142 invalid optimizer displacement or settings")
    rows = []
    for grad in old_gradients:
        if (
            grad.shape != before.shape or grad.device != before.device
            or not grad.is_floating_point()
            or not bool(torch.isfinite(grad).all())
        ):
            raise ValueError("S142 invalid protected TRAIN constraint")
        flat = grad.detach().to(device="cpu", dtype=torch.float64).reshape(-1).numpy()
        norm = float(np.linalg.norm(flat))
        if norm > 1e-12:
            rows.append(flat / norm)
    if not rows:
        raise AssertionError("S142 no nonzero protected TRAIN gradients")
    unit = np.stack(rows, axis=0)
    start = before.detach().to(device="cpu", dtype=torch.float64).numpy()
    raw = start.reshape(-1) - after.detach().to(
        device="cpu", dtype=torch.float64).reshape(-1).numpy()
    raw_norm = float(np.linalg.norm(raw))

    def minimum_cosine(direction: np.ndarray) -> float:
        length = float(np.linalg.norm(direction))
        return float(np.min(unit @ direction / length)) if length > 1e-12 else 0.0

    initial = minimum_cosine(raw)
    audit = {
        "old_gradient_constraints": len(rows),
        "post_adam_min_cosine_before": initial,
        "post_adam_min_cosine_after": initial,
        "post_adam_raw_displacement_norm": raw_norm,
        "post_adam_projection_required": initial < -cosine_tolerance,
        "post_adam_zero_fallback": False,
        "post_adam_first_order_only": True,
        "post_adam_solver": "S142_cpu_fp64_dual_LBFGSB",
        "post_adam_dual_iterations": 0,
    }
    if initial >= -cosine_tolerance:
        return after.detach().clone(), audit

    # Dual min_{lambda>=0} 0.5 lambda^T (G G^T) lambda +
    # (G d0 - margin)^T lambda; primal correction d=d0+G^T lambda.
    gram = unit @ unit.T
    margin = 5e-7 * raw_norm
    linear = unit @ raw - margin

    def objective(lam):
        product = gram @ lam
        return float(0.5 * np.dot(lam, product) +
                     np.dot(linear, lam)), product + linear

    solution = minimize(
        objective, np.zeros(len(rows), dtype=np.float64),
        jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(rows),
        options={"maxiter": 3000, "maxls": 50, "ftol": 1e-15, "gtol": 1e-12},
    )
    audit["post_adam_dual_iterations"] = int(solution.nit)
    audit["post_adam_dual_converged"] = bool(solution.success)
    direction = raw + unit.T @ solution.x
    projected = start - direction.reshape(start.shape)
    result = torch.as_tensor(projected, dtype=before.dtype, device=before.device)
    if not bool(torch.isfinite(result).all()):
        raise AssertionError("S142 nonfinite projected post-Adam coefficients")
    # Verify actual FP32/MPS rounded proposal (not just FP64 dual solution).
    realized = start.reshape(-1) - result.detach().to(
        "cpu", dtype=torch.float64).reshape(-1).numpy()
    actual = minimum_cosine(realized)
    audit["post_adam_min_cosine_after"] = actual
    audit["post_adam_dual_feasible"] = bool(actual >= -cosine_tolerance)
    if actual < -cosine_tolerance:
        audit["post_adam_zero_fallback"] = True
        audit["post_adam_min_cosine_after"] = 0.0
        return before.detach().clone(), audit
    return result.detach(), audit
