"""S87-J2: train the *actual deployed autoregressive scoring path* jointly.

Research-only: frozen original S82 controllers / source-pointer + trainable
internal hidden-state Green Tunnel and existing paid CB59 centroid tensors.
Unlike S87's raw-LM-only CE, this differentiates the same first/later
controller logits and second-token copy bonus used by S82 at inference.
No QA/TEST import, prompt-ID routing, secret, new uncounted inference weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from calibri2.s82.real_mps_generic_token_context_copy import candidate_positions
from v176.operators import clear


@dataclass(frozen=True)
class AlignedStep:
    aligned_full_eos_loss: float
    core_gradient_norm: float
    tunnel_gradient_norm: float
    nonzero_core_gradient_tensors: int
    nonzero_tunnel_gradient_tensors: int
    zero_tunnel_gradient_tensors: tuple[str, ...]
    selected_codebook_count: int
    trained_authorized_examples: int


def actual_path_teacher_forced_loss(model, tok, question, answer, first,
                                    later, pointer, head):
    """Gold-prefix CE: same controller and conditional pointer as S82 CPU.

    This is teacher-forced TRAIN loss, not a claim of correct greedy generation.
    Every token, including EOS, receives a gradient through original compressed
    model layers, the internal residuals and selected paid codebooks.
    """
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    gold = tok.encode(answer, add_special_tokens=False) + [tok.eos_token_id]
    if not prompt_ids or len(gold) < 2 or len(prompt_ids) + len(gold) > 320:
        raise AssertionError("Unexpected authorized TRAIN input length")
    device = head.device
    inp = torch.tensor([prompt_ids + gold[:-1]], dtype=torch.long, device=device)
    gold_ids = torch.tensor(gold, dtype=torch.long, device=device)
    out = model(input_ids=inp, use_cache=False, output_hidden_states=True)
    last_hidden = out.hidden_states[-1][0]
    positions = candidate_positions(tok, prompt_ids) if pointer is not None else []
    selected = torch.tensor(positions, dtype=torch.long, device=device)
    keys = last_hidden.index_select(0, selected) if positions else None
    candidate_token_ids = torch.tensor(
        [prompt_ids[p] for p in positions], dtype=torch.long, device=device)
    losses = []
    for step in range(len(gold)):
        location = len(prompt_ids) - 1 + step
        h = last_hidden[location]
        scores = out.logits[0, location] + F.linear(
            first(h) if step == 0 else later(h), head)
        if step == 1 and positions and int(scores.detach().argmax()) != tok.eos_token_id:
            bonuses = pointer.full_vocab_bonus(h, keys, candidate_token_ids)
            scores = scores + torch.zeros_like(scores).index_add(
                0, candidate_token_ids, bonuses)
        losses.append(F.cross_entropy(scores[None, :], gold_ids[step:step + 1]))
    return torch.stack(losses).mean()


def aligned_joint_step(model, sources, tunnel, selected_codebooks, tok, examples,
                       optimizer, first, later, pointer, *, step):
    """One bounded MPS training update with explicit gradient diagnostic."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("REAL MPS required")
    if not examples or any(p.device.type != "mps" for p in tunnel.parameters()):
        raise AssertionError("Missing MPS internal trainable residual")
    if any(s.centroids.device.type != "mps" for s in selected_codebooks.values()):
        raise AssertionError("Paid CB59 centroids not on MPS")
    if any(p.device.type != "mps" for m in (first, later, pointer)
           for p in m.parameters()):
        raise AssertionError("Frozen S82 controllers must execute on MPS")
    clear(sources)
    optimizer.zero_grad(set_to_none=True)
    try:
        # Frozen LM-head source: it is already paid, no separate output model.
        with torch.no_grad():
            head = model.lm_head.source().detach()
        if head.device.type != "mps":
            raise AssertionError("LM head is not on MPS")
        losses = []
        for question, answer in examples:
            clear(sources)
            losses.append(actual_path_teacher_forced_loss(
                model, tok, question, answer, first, later, pointer, head))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite full+EOS controller-aligned loss")
        loss.backward()
        core = [s.centroids for s in selected_codebooks.values()]
        inner = list(tunnel.parameters())
        if any(p.grad is None for p in core + inner):
            raise AssertionError("Missing differentiable core/tunnel gradient")
        nz_core = sum(int(bool(torch.count_nonzero(p.grad).item())) for p in core)
        nz_inner = sum(int(bool(torch.count_nonzero(p.grad).item())) for p in inner)
        zero_names = tuple(name for name, p in tunnel.named_parameters()
                           if not bool(torch.count_nonzero(p.grad).item()))
        if nz_core != len(core) or any(n.endswith(".up.weight") for n in zero_names):
            raise AssertionError("Core or final internal tunnel gradient vanished")
        # Zero-initialized up matrix blocks down/gate gradient only on step 0.
        if step > 0 and nz_inner != len(inner):
            raise AssertionError("Full internal tunnel gradient was not unlocked")
        core_norm = float(torch.nn.utils.clip_grad_norm_(core, 1.0))
        inner_norm = float(torch.nn.utils.clip_grad_norm_(inner, 1.0))
        optimizer.step()
        with torch.no_grad():
            for p in core:
                p.clamp_(-16.0, 16.0)
        return AlignedStep(
            float(loss.detach()), core_norm, inner_norm, nz_core, nz_inner,
            zero_names, len(core), len(examples))
    finally:
        clear(sources)
