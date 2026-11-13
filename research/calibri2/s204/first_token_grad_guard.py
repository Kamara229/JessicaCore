"""S204 TRAIN-only projection of real NRA Adam displacement against old-token loss.

Protected directions must be genuine gradients of precommitted original TRAIN
first-token teacher-forced cross entropy. First-order constraints are NOT
a proof of preserved full answers: 456+128 native CPU J7 remains mandatory.
No QA/TEST, no inference-time label routing, no LIVE.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
import torch

NRA_PREFIX = "neuro_representation_area."


def project_protected_nra_step(
    before: Mapping[str, torch.Tensor],
    after_adam: Mapping[str, torch.Tensor],
    protected_first_token_grads: list[Mapping[str, torch.Tensor]],
    *,
    tolerance: float = 1e-6,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Project REAL post-Adam update, fail closed after physical FP16 rounding.

    The caller is responsible for obtaining authentic per-sample protected
    first-token gradients from original permitted TRAIN *before* the optimizer
    step. A rejected update restores the original FP16-equivalent NRA state.
    """
    keys = tuple(sorted(before))
    if (not keys or keys != tuple(sorted(after_adam))
        or not protected_first_token_grads
        or not (math.isfinite(tolerance) and 0 < tolerance <= 1e-5)):
        raise ValueError("S204 requires matched NRA tensors and protected TRAIN gradients")
    shapes = {}
    devices = set()
    for key in keys:
        b, a = before[key], after_adam[key]
        if (not key.startswith(NRA_PREFIX) or not b.is_floating_point()
            or b.shape != a.shape or not a.is_floating_point()
            or not torch.isfinite(b).all() or not torch.isfinite(a).all()
            or b.device != a.device):
            raise ValueError("S204 requires finite NRA-only pre/post-Adam state")
        devices.add(b.device)
        shapes[key] = b.shape
    if len(devices) != 1:
        raise ValueError("S204 single device required")
    old = torch.cat([before[k].detach().float().reshape(-1) for k in keys])
    proposed = torch.cat([after_adam[k].detach().float().reshape(-1) for k in keys])
    gradients = []
    for row in protected_first_token_grads:
        if set(row) != set(keys):
            raise ValueError("S204 missing protected first-token TRAIN gradient")
        parts = []
        for key in keys:
            g = row[key]
            if (g.shape != shapes[key] or g.device != before[key].device
                or not g.is_floating_point() or not torch.isfinite(g).all()):
                raise ValueError("S204 malformed old TRAIN gradient")
            parts.append(g.detach().float().reshape(-1))
        direction = torch.cat(parts)
        if float(torch.linalg.vector_norm(direction)) <= 1e-12:
            raise AssertionError("S204 zero old TRAIN gradient cannot certify")
        gradients.append(direction)
    # Perform a cyclic projection of ACTUAL Adam displacement onto the
    # permitted first-order halfspaces, not of pre-optimizer gradients.
    displacement_proposed = old - proposed
    direction = displacement_proposed.clone()
    units = [g / torch.linalg.vector_norm(g) for g in gradients]
    corrections = 0
    for _pass in range(32):
        changed = 0
        for u in units:
            dot = torch.dot(direction, u)
            if float(dot) < 0:
                direction -= dot * u
                changed += 1
        corrections += changed
        if changed == 0:
            break
    projected = old - direction
    # The physically persisted NRA tensors are FP16. Verify *rounded* update.
    actual = projected.to(dtype=torch.float16).float()
    prior = old.to(dtype=torch.float16).float()
    displacement = prior - actual
    norm = float(torch.linalg.vector_norm(displacement))
    cosine = [
        float(torch.dot(displacement, g)) /
        (norm * float(torch.linalg.vector_norm(g)))
        if norm > 1e-12 else 0.0
        for g in gradients
    ]
    feasible = bool(min(cosine) >= -tolerance)
    fallback = not feasible or not bool(torch.any(prior != actual))
    if fallback:
        actual = prior
        cosine = [0.0] * len(gradients)
    offset = 0
    result = {}
    for key in keys:
        count = before[key].numel()
        result[key] = actual[offset:offset+count].reshape(shapes[key]).to(
            dtype=before[key].dtype, device=before[key].device
        ).detach().clone()
        offset += count
    return result, {
        "s204_train_only": True,
        "protected_first_token_train_examples": len(gradients),
        "post_adam_projection_corrections": corrections,
        "fp16_min_protected_gradient_cosine": min(cosine),
        "fp16_candidate_feasible_first_order_only": feasible,
        "fp16_zero_update_fallback": fallback,
        "must_run_independent_full_456_128_32_cpu_j7": True,
        "model_accepted": False,
    }
