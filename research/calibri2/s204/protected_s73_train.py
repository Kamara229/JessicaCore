"""S204 authenticated original S73 TRAIN supervision for independent NRA gradient guard.

This module is TRAIN-ONLY. It must NEVER be imported by a serving decoder,
or used to route answers at inference. The four SHA commitments are from the
original permitted S73 TRAIN in S202's verified native CPU diagnostic.
Actual candidate admission still requires independent 456+128+32 J7, bpp,
CPU speed, and zero full+EOS forgetting. No QA/TEST, no LIVE.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch
from torch.nn import functional as F

ORIGINAL_S73_SHA = {
    "S73/73012/left": "f83736e83814ae27a25ba558e8b937d92b0460c67e947ab5f372c9254eca9a7f",
    "S73/73012/right": "f45f58b356add1302eb1502ae5f9e88100854293503dcb33ca986968125a7fa8",
    "S73/73015/left": "f21dfbced10f22a3da43267210beeaa627fa6caecf52fff845aa1586db318aa0",
    "S73/73015/right": "4597a12d3d9ca11061369fcd9c9147872e7fa076f9ee7b3875285d0d59a4cda1",
}


def precommitted_s73_original_train(old: Mapping) -> tuple[tuple[str, dict], ...]:
    """Validate four real, frozen original S73 TRAIN rows, never synthetic QA."""
    if len(old) != 456:
        raise AssertionError("S204 needs all 456 original authorized TRAIN rows")
    records = []
    for identity, digest in ORIGINAL_S73_SHA.items():
        if identity not in old:
            raise AssertionError("S204 original protected S73 TRAIN record absent")
        row = old[identity]
        question, expected = row.get("question"), row.get("expected")
        if (not isinstance(question, str) or not question.startswith("TRAIN S73 ")
            or hashlib.sha256(question.encode("utf-8")).hexdigest() != digest
            or not isinstance(expected, (list, tuple)) or len(expected) != 3
            or not all(isinstance(t, int) and t >= 0 for t in expected)
            or expected[-1] != 2):
            raise AssertionError("S204 immutable S73 TRAIN token source mismatch")
        records.append((identity, row))
    return tuple(records)


def real_s73_first_token_loss(*, model, cpu, tok, question: str,
                              first_token_id: int) -> torch.Tensor:
    """Teacher-forced first-token CE through the real frozen core and NRA.

    Must be called with the *original S73 source profile / prefill scopes*
    already installed by the research training driver. It never samples
    a candidate response, selects an inference route, or uses heldout data.
    """
    if not torch.is_grad_enabled() or not 0 <= first_token_id < int(tok.vocab_size):
        raise AssertionError("S204 requires grad-enabled original TRAIN first token")
    from calibri2.s160.real_parallel_slots_j7 import prompt_tokens
    prompt = prompt_tokens(tok, question)
    if not prompt or not question.startswith("TRAIN S73 "):
        raise AssertionError("S204 first-token CE only on precommitted original TRAIN")
    ids = torch.tensor([prompt], dtype=torch.long)
    result = model(input_ids=ids, past_key_values=None, use_cache=True,
                   output_hidden_states=True)
    hidden = result.hidden_states[-1][0, -1]
    logits = result.logits[0, -1] + F.linear(cpu[2](hidden), cpu[5])
    target = torch.tensor([first_token_id], dtype=torch.long)
    return F.cross_entropy(logits.unsqueeze(0), target)


def real_s73_nra_gradient(*, model, cpu, tok, area, question: str,
                          first_token_id: int) -> dict[str, torch.Tensor]:
    """Independent real CPU per-example old-token gradients, not mock data."""
    if not all(p.requires_grad for p in area.parameters()):
        raise AssertionError("S204 protected NRA must be trainable")
    parameters = tuple(sorted(area.named_parameters()))
    if not parameters:
        raise AssertionError("S204 original area has no trainable parameters")
    loss = real_s73_first_token_loss(
        model=model, cpu=cpu, tok=tok, question=question,
        first_token_id=first_token_id)
    if not bool(torch.isfinite(loss)):
        raise AssertionError("S204 protected old first-token CE nonfinite")
    gradients = torch.autograd.grad(
        loss, [p for _name, p in parameters],
        allow_unused=False, retain_graph=False)
    result = {}
    for (key, _param), grad in zip(parameters, gradients):
        if not bool(torch.isfinite(grad).all()):
            raise AssertionError("S204 old TRAIN gradient nonfinite")
        result["neuro_representation_area." + key] = grad.detach().clone()
    if not any(bool(torch.any(g)) for g in result.values()):
        raise AssertionError("S204 original old-token training gradient is zero")
    return result
