"""S204 connect first-token TRAIN gradient constraints to an actual Adam step.

The caller must precompute real four S73 original TRAIN first-token
gradients under the frozen original source profile and prefill scope, then
compute the NEW TRAIN loss/backward. Only the NRA optimizer is permitted.
This is research TRAIN code, never an inference-time answer selector.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
import torch

from calibri2.s204.first_token_grad_guard import project_protected_nra_step


def guarded_nra_adam_step(
    *,
    area: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    protected_first_token_grads: list[Mapping[str, torch.Tensor]],
) -> dict:
    """Execute one real Adam step, project its rounded displacement, fail closed.

    A zero or infeasible FP16 update is rejected and optimizer moment buffers
    are rolled back along with original NRA parameters. A successful step is
    *not* a valid model until independently checked by the complete CPU J7.
    """
    parameters = dict(area.named_parameters())
    expected_ids = {id(p) for p in parameters.values()}
    actual_ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if (not parameters or set(actual_ids) != expected_ids
        or len(actual_ids) != len(expected_ids)
        or any(not p.requires_grad or p.grad is None
               or not bool(torch.isfinite(p.grad).all())
               for p in parameters.values())):
        raise AssertionError("S204 requires one gradient-bearing NRA-only Adam optimizer")
    names = {"neuro_representation_area." + k: p for k, p in parameters.items()}
    before = {k: p.detach().clone() for k, p in names.items()}
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    try:
        optimizer.step()
        after = {k: p.detach().clone() for k, p in names.items()}
        projected, audit = project_protected_nra_step(
            before, after, protected_first_token_grads
        )
        if audit["fp16_zero_update_fallback"]:
            raise AssertionError("S204 protected first-token Adam displacement infeasible")
        with torch.no_grad():
            for key, p in names.items():
                p.copy_(projected[key])
        return {
            **audit,
            "actual_nra_adam_step": True,
            "optimizer_adam_state_retained_for_projected_step": True,
            "independent_original_456_new128_novel32_native_cpu_j7_required": True,
            "model_accepted": False,
        }
    except BaseException:
        with torch.no_grad():
            for key, p in names.items():
                p.copy_(before[key])
        optimizer.load_state_dict(optimizer_before)
        raise
