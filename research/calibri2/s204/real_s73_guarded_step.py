"""S204 TRAIN-only wiring: authentic four S73 old-token gradients -> Adam guard.

This is a low-level adapter, not an executable CPU model experiment.
The host research driver must bind source_scope to S197's exact frozen
original route/bridge/Green Tunnel for EACH S73 TRAIN item, and must use
the same actual model for the new32 TRAIN loss. Never infer a source route
from the expected answer. Recheck native 456+128+32 CPU J7 after FP16 reload.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
import torch

from calibri2.s204.protected_s73_train import (
    precommitted_s73_original_train,
    real_s73_nra_gradient,
)
from calibri2.s204.guarded_nra_optimizer import guarded_nra_adam_step

VALID_SOURCE_ROUTES = frozenset(("S87", "S91", "S93"))


def real_four_s73_guarded_step(
    *,
    original_456: Mapping,
    source_routes: Mapping[str, str],
    source_scope: Callable[[str, str], AbstractContextManager],
    model: torch.nn.Module,
    cpu: tuple,
    tok,
    area: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    backward_new32_train_loss: Callable[[], torch.Tensor],
) -> dict:
    """One original-source-bound *research* NRA step; fail closed.

    source_scope(id, route) must restore the original frozen route and
    all prefill-only tunnel/profile states, not use any gold-token content.
    This function trains only on 32 precommitted TRAIN sources, and obtains
    separate protected first-token gradients from original S73 TRAIN.
    """
    records = precommitted_s73_original_train(original_456)
    if (not callable(source_scope) or not callable(backward_new32_train_loss)
        or len(cpu) < 6 or not getattr(area, "_active", False)
        or not bool(getattr(area, "_handles", []))
        or not all(isinstance(source_routes.get(k), str)
                   and source_routes[k] in VALID_SOURCE_ROUTES
                   for k, _ in records)):
        raise AssertionError("S204 real installed NRA and frozen original S73 source routes required")
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("S204 cannot train frozen real core/Green Tunnel")
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise AssertionError("S204 real original TRAIN gradient requires torch autograd")
    protected = []
    try:
        for identity, row in records:
            # S73 gold is used only in TRAIN-time cross-entropy, never in
            # source routing; the route comes from a frozen source map.
            route = source_routes[identity]
            optimizer.zero_grad(set_to_none=True)
            with source_scope(identity, route):
                one = real_s73_nra_gradient(
                    model=model, cpu=cpu, tok=tok, area=area,
                    question=row["question"],
                    first_token_id=row["expected"][0],
                )
            protected.append(one)
        optimizer.zero_grad(set_to_none=True)
        loss = backward_new32_train_loss()
        if (not isinstance(loss, torch.Tensor) or loss.ndim != 0
            or not bool(torch.isfinite(loss)) or not loss.requires_grad):
            raise AssertionError("S204 needs real scalar NEW32 TRAIN loss")
        loss.backward()
        result = guarded_nra_adam_step(
            area=area,
            optimizer=optimizer,
            protected_first_token_grads=protected,
        )
        if (result["protected_first_token_train_examples"] != 4
            or result["model_accepted"] is not False):
            raise AssertionError("S204 cannot bypass four old TRAIN constraints")
        return {
            **result,
            "original_S73_TRAIN_ids": [k for k, _ in records],
            "S204_NEW32_TRAIN_loss_before_step": float(loss.detach()),
            "original_456_new128_novel32_real_CPU_J7_pending": True,
            "model_accepted": False,
        }
    finally:
        # Clear any residual protected gradients from the adapter.
        optimizer.zero_grad(set_to_none=True)
