"""S204 original frozen S73 TRAIN source-route prefill scope, not serving code.

Mirrors S160's prefill-only S93/S87/S91 profile, bridge, GT12 and grown
core activations with autograd enabled solely for protected TRAIN gradients.
S115.frozen_cpu_phase_scope is intentionally NOT used: it requires
torch.inference_mode(), which would silently block true NRA training.
All four S73 identities and their question digests are TRAIN-only.
No QA/TEST, LIVE, Render, Supabase or model acceptance.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import torch

from calibri2.s204.protected_s73_train import ORIGINAL_S73_SHA

VALID_ROUTES = frozenset(("S87", "S91", "S93"))


def ensure_original_s73_source(*, identity: str, route: str, row: dict,
                               source_routes: dict):
    """Validate provenance; the route is frozen input, NEVER gold-derived."""
    if (identity not in ORIGINAL_S73_SHA
        or route not in VALID_ROUTES
        or source_routes.get(identity) != route
        or not isinstance(row, dict)
        or not isinstance(row.get("question"), str)
        or hashlib.sha256(row["question"].encode()).hexdigest()
            != ORIGINAL_S73_SHA[identity]):
        raise AssertionError("S204 S73 TRAIN source/route does not match frozen commitment")


@contextmanager
def original_s73_train_profile_scope(
    *, identity: str, route: str, old_456: dict, source_routes: dict,
    cpu: tuple, tok, bank: dict, bridge, full, expansion,
):
    """One real S73 gradient prefill with original profile and causal hooks.

    Must be entered only from S204 research TRAIN code with original S197
    native CPU fully loaded and original frozen S95 routes. No gold or
    desired token enters route/profile decisions.
    """
    if identity not in old_456:
        raise AssertionError("S204 missing old TRAIN identity")
    ensure_original_s73_source(identity=identity, route=route,
                               row=old_456[identity],
                               source_routes=source_routes)
    if (len(cpu)<9 or torch.is_inference_mode_enabled()
        or not torch.is_grad_enabled()
        or not getattr(bridge,"_handles",None)
        or not getattr(full,"_handles",None)
        or not getattr(expansion,"_handles",None)):
        raise AssertionError("S204 needs installed frozen real CPU source with autograd")
    from calibri2.s160 import real_parallel_slots_j7 as s160
    from calibri2.s127.q2_prefill_gate_control import prefill_only_tunnel
    from v176.operators import clear

    model,sources,*_ = cpu
    features = (
        torch.as_tensor(
            s160.prompt_only_vector(tok,old_456[identity]["question"]),
            dtype=torch.float32)
        if route == "S93" else None)
    # Uses EXACT S99 frozen profile bank, no answer-aware route override.
    s160.apply_profile(sources,cpu[6],route,bank,cpu[7],cpu[8])
    try:
        with expansion.prefill_only(model), prefill_only_tunnel(model,full):
            with bridge.prompt_scope(route,features):
                clear(sources)
                yield
    finally:
        # S204 autonomous TRAIN call must not leak a prior S73 profile into
        # subsequent new32 optimization or future protected probes.
        s160.apply_profile(sources,cpu[6],"S93",bank,cpu[7],cpu[8])
        clear(sources)


def make_original_s73_training_scope(
    *, old_456: dict, source_routes: dict, cpu: tuple,
    tok, bank: dict, bridge, full, expansion,
):
    """Provide source_scope(id,route) expected by real_four_s73_guarded_step."""
    if len(old_456) != 456 or any(
        k not in source_routes or k not in old_456
        for k in ORIGINAL_S73_SHA):
        raise AssertionError("S204 full 456 TRAIN and frozen S73 source routes required")
    def source_scope(identity,route):
        return original_s73_train_profile_scope(
            identity=identity,route=route,old_456=old_456,
            source_routes=source_routes,cpu=cpu,tok=tok,bank=bank,
            bridge=bridge,full=full,expansion=expansion)
    return source_scope
