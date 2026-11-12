"""S159: source-bound, parallel multi-slot decoder (research-only).

A single FROZEN model prefill supplies contextual hidden states. The lightweight
slot projections compute all answer positions in ONE vectorized operation. A
shared span score binds attributes to the SAME source record; independent
argmax over attributes is intentionally forbidden (would mix owners).
This is NOT a rule-based answer lookup and never uses question/answer hashes.

By default the fast path is DISABLED: parent's decoding is bitwise untouched.
Only an external real S142 physical 456+128 J7 audit may enable the new route.
No QA/TEST, LIVE, Render, Supabase or APK integration here.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class ParallelSlotBindingHead(nn.Module):
    """Predict N contiguous source-token slots in parallel from one model prefill.

    Inputs:
      states: [batch, source_length, hidden] frozen causal model hidden states
      source_ids: matching token IDs from the SAME already-performed prefill
      source_mask: optional [batch, source_length] causal source-valid positions
    Output logits score whole N-token source spans; no independent color/noun
    cross-product can violate the common owner. All token identities are copied
    from context; neither a fixed vocabulary nor exact prompt-ID lookup exists.

    The frozen model's last contextual hidden state acts as an answer query.
    Learnable slot-specific projections are NOT independent inference passes.
    """

    def __init__(self, hidden_size: int = 320, slots: int = 2, rank: int = 16):
        super().__init__()
        if hidden_size < 1 or not 1 <= slots <= 4 or rank < 1:
            raise ValueError("hidden_size, slots (1..4) and rank must be positive")
        self.hidden_size = hidden_size
        self.slots = slots
        self.rank = rank
        self.query = nn.Linear(hidden_size, slots * rank, bias=False)
        self.keys = nn.Linear(hidden_size, slots * rank, bias=False)
        self.binding_query = nn.Linear(hidden_size, rank, bias=False)
        self.binding_keys = nn.Linear(hidden_size, rank, bias=False)
        # A common query must select the owner, then all slots use its span.
        self.binding_strength = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        states: torch.Tensor,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (states.ndim != 3 or states.size(-1) != self.hidden_size or
            source_ids.ndim != 2 or source_ids.shape != states.shape[:2] or
            source_ids.dtype not in (torch.int64, torch.int32) or
            states.size(1) < self.slots):
            raise ValueError("expected [B,L,H] states with aligned [B,L] token IDs")
        batch, length, hidden = states.shape
        if source_mask is None:
            source_mask = torch.ones_like(source_ids, dtype=torch.bool)
        if (source_mask.dtype != torch.bool or source_mask.shape != source_ids.shape):
            raise ValueError("source_mask must be boolean [B,L]")
        span_count = length - self.slots + 1
        # Each contiguous candidate is evaluated once. No repeated 43M pass.
        windows = torch.stack(
            [states[:, j:j + span_count] for j in range(self.slots)], dim=2
        )  # [B, candidates, slots, H]
        word_windows = torch.stack(
            [source_ids[:, j:j + span_count] for j in range(self.slots)], dim=2
        )
        valid = torch.stack(
            [source_mask[:, j:j + span_count] for j in range(self.slots)], dim=2
        ).all(dim=-1)
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("each question must contain a fully valid source span")
        # Query is shared across slots, with each slot's own low-rank projection.
        # Use the LAST valid token, rather than leaking a pad position as query.
        last_idx = source_mask.long().sum(dim=1) - 1
        if bool((last_idx < 0).any()):
            raise ValueError("missing question query token")
        context = states[torch.arange(batch, device=states.device), last_idx]
        queries = self.query(context).view(batch, self.slots, self.rank)
        # Key projections differ by slot, but both slot logits are produced by
        # the SAME batched operation (no sequential autoregressive decoding).
        slot_keys = self.keys(windows).view(
            batch, span_count, self.slots, self.slots, self.rank
        )
        diagonal = slot_keys.diagonal(dim1=2, dim2=3).permute(0, 1, 3, 2)
        slot_logits = (diagonal * queries[:, None]).sum(dim=-1) / math.sqrt(self.rank)
        joint_key = self.binding_keys(windows.mean(dim=2))
        joint_query = self.binding_query(context)
        binding_logits = (joint_key * joint_query[:, None]).sum(dim=-1)
        span_logits = slot_logits.mean(dim=-1) + (
            self.binding_strength.sigmoid() * binding_logits / math.sqrt(self.rank)
        )
        span_logits = span_logits.masked_fill(~valid, -torch.inf)
        selected_span = span_logits.argmax(dim=-1)
        chosen = word_windows[
            torch.arange(batch, device=states.device), selected_span
        ].long()
        probs = span_logits.softmax(dim=-1)
        confidence = probs.gather(1, selected_span[:, None]).squeeze(1)
        if span_count >= 2:
            top = span_logits.topk(2, dim=-1).values
            margin = top[:, 0] - top[:, 1]
        else:
            margin = torch.zeros_like(confidence)
        return {
            "slot_ids": chosen, "span_logits": span_logits,
            "slot_logits": slot_logits, "selected_span": selected_span,
            "confidence": confidence, "margin": margin,
            "candidate_count": valid.sum(dim=-1),
        }


def source_span_nll(
    outputs: dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    expected_slot_ids: torch.Tensor,
    source_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multiple matching spans use marginal likelihood; TRAIN examples only.

    Targets may come only from original authorized S87/S122 TRAIN, not QA.
    Raises if the exact target source span does not exist: no hallucinated copy.
    """
    logits = outputs["span_logits"]
    if (source_ids.ndim != 2 or expected_slot_ids.ndim != 2 or
        source_ids.size(0) != expected_slot_ids.size(0)):
        raise ValueError("source IDs and labels must have aligned batch dimensions")
    slots = expected_slot_ids.size(1)
    if logits.shape != (source_ids.size(0), source_ids.size(1) - slots + 1):
        raise ValueError("candidate logits don't match number of slots")
    if source_mask is None:
        source_mask = torch.ones_like(source_ids, dtype=torch.bool)
    matches = torch.ones_like(logits, dtype=torch.bool)
    for offset in range(slots):
        matches &= source_ids[:, offset:offset + logits.size(1)] == expected_slot_ids[:, offset:offset + 1]
        matches &= source_mask[:, offset:offset + logits.size(1)]
    if not bool(matches.any(dim=-1).all()):
        raise ValueError("expected answer cannot be copied as one contiguous span")
    logp = F.log_softmax(logits, dim=-1)
    return -(torch.logsumexp(logp.masked_fill(~matches, -torch.inf), dim=-1)).mean()


class GuardedParallelDecode:
    """Research-only optional early exit; exact original callback is the default.

    The full S142 decoder must be preserved on fallback. Do NOT arm until real
    frozen physical J7 across 456+128, bpp and paired CPU has explicitly
    accepted a saved FP16 candidate. This class is an interface, not that audit.
    """

    def __init__(
        self,
        head: ParallelSlotBindingHead,
        *,
        eos_id: int,
        min_confidence: float = 0.99,
        min_margin: float = 3.0,
    ):
        if not 0.5 <= min_confidence <= 1.0 or min_margin <= 0 or eos_id < 0:
            raise ValueError("invalid early-exit safety threshold or EOS")
        self.head = head
        self.eos_id = eos_id
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.enabled = False

    def decode(
        self,
        states: torch.Tensor,
        source_ids: torch.Tensor,
        original_decode: Callable[[], torch.Tensor],
        source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool]:
        if not self.enabled:
            return original_decode(), False
        if states.size(0) != 1:
            # Multi-query batches require per-row fallback audit first.
            return original_decode(), False
        with torch.inference_mode():
            ans = self.head(states, source_ids, source_mask)
            admissible = (
                ans["candidate_count"].item() >= 2 and
                ans["confidence"].item() >= self.min_confidence and
                ans["margin"].item() >= self.min_margin
            )
            if admissible:
                eos = torch.full((1, 1), self.eos_id, dtype=torch.long,
                                 device=source_ids.device)
                return torch.cat((ans["slot_ids"], eos), dim=1), True
        return original_decode(), False
