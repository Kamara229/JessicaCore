"""S177: research-only adaptive 5/10 candidate consensus over ONE frozen prefill.

This does not create 5/10 neural models, modify weights, or validate factual
correctness via popularity. Candidates are contiguous 2-token *source spans*.
A distinct, prompt-only owner verifier is mandatory before any proposal can
replace the legacy choice. Independent QA/TEST and production remain untouched.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class ConsensusDecision:
    proposed: Optional[tuple[int, int]]
    legacy: Optional[tuple[int, int]]
    branch_count: int
    votes: int
    agreement: float
    reason: str
    unique_span_count: int

    @property
    def changes_legacy(self) -> bool:
        return self.proposed is not None and self.proposed != self.legacy


def _legacy_pair(choice: Optional[Sequence[int]]) -> Optional[tuple[int, int]]:
    if choice is None:
        return None
    if len(choice) != 2:
        raise ValueError("legacy answer must have exactly two slots")
    return (int(choice[0]), int(choice[1]))


def _unique_spans(
    span_logits: torch.Tensor,
    canonical_spans: torch.Tensor,
    eos_id: int,
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """Collapse identical spans by MAX logit, preventing repeated-source bias."""
    if span_logits.ndim != 1 or canonical_spans.ndim != 2:
        raise ValueError("expected [L] logits and [L,2] source spans")
    if canonical_spans.shape != (len(span_logits), 2):
        raise ValueError("unaligned contiguous source span logits")
    if canonical_spans.dtype not in (torch.int32, torch.int64):
        raise ValueError("source spans must be integer tokens")
    if span_logits.device.type != "cpu" or canonical_spans.device.type != "cpu":
        raise ValueError("native CPU research only; no hidden GPU transfer")
    scores: dict[tuple[int, int], float] = {}
    for score, ids in zip(span_logits.tolist(), canonical_spans.tolist()):
        pair = (int(ids[0]), int(ids[1]))
        if not math.isfinite(float(score)) or min(pair) < 0 or eos_id in pair:
            continue
        if pair not in scores or score > scores[pair]:
            scores[pair] = float(score)
    ordered = sorted(scores)  # deterministic tie order, independent of dict
    return ordered, torch.tensor([scores[p] for p in ordered], dtype=torch.float64)


def adaptive_consensus(
    span_logits: torch.Tensor,
    canonical_spans: torch.Tensor,
    *,
    question: str,
    verified_owner_ids: Optional[Sequence[int]],
    legacy_choice: Optional[Sequence[int]],
    eos_id: int,
    salt: str = "calibri-s177-v1",
) -> ConsensusDecision:
    """Return a candidate ONLY on agreement + independent prompt-only proof.

    Verifier input must be derived from the literal prompt/source alone,
    NEVER expected answer, output correctness, QA labels, or protected ids.
    Selection is precommitted and uses a *local* RNG (no global seed change).
    A caller must preserve the original S176 route until real full J7 passes.
    """
    legacy = _legacy_pair(legacy_choice)
    if not isinstance(question, str) or not question or eos_id < 0:
        raise ValueError("literal prompt and EOS are required")
    pool, scores = _unique_spans(span_logits, canonical_spans, eos_id)
    no = lambda why, n=0, v=0: ConsensusDecision(
        None, legacy, n, v, v / n if n else 0.0, why, len(pool))
    if verified_owner_ids is None:
        return no("no_independent_owner_proof")
    if len(verified_owner_ids) != 2:
        return no("owner_not_two_tokens")
    owner = tuple(int(x) for x in verified_owner_ids)
    if min(owner) < 0 or eos_id in owner or owner not in pool:
        return no("owner_not_exact_source_span")
    if len(pool) < 2:
        return no("insufficient_distinct_candidates")
    if not torch.isfinite(scores).all():
        raise AssertionError("nonfinite pooled logits")
    generator = torch.Generator(device="cpu")
    digest = hashlib.sha256((salt + "\0" + question).encode("utf-8")).digest()
    generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
    # Independently sampled candidates from three fixed temperatures. No labels.
    temperatures = (0.75, 1.0, 1.35, 0.9, 1.15,
                    0.75, 1.0, 1.35, 0.9, 1.15)
    samples: list[tuple[int, int]] = []
    for temp in temperatures:
        distribution = torch.softmax((scores - scores.max()) / temp, dim=0)
        index = int(torch.multinomial(distribution, 1, generator=generator).item())
        samples.append(pool[index])
    five = Counter(samples[:5])
    top_five, votes_five = sorted(five.items(), key=lambda z: (-z[1], z[0]))[0]
    if votes_five >= 4 and top_five == owner:
        return ConsensusDecision(top_five, legacy, 5, votes_five,
                                 votes_five / 5.0, "verified_consensus_5", len(pool))
    ten = Counter(samples)
    top_ten, votes_ten = sorted(ten.items(), key=lambda z: (-z[1], z[0]))[0]
    if votes_ten >= 7 and top_ten == owner:
        return ConsensusDecision(top_ten, legacy, 10, votes_ten,
                                 votes_ten / 10.0, "verified_consensus_10", len(pool))
    return no("no_verified_consensus", 10, votes_ten)
