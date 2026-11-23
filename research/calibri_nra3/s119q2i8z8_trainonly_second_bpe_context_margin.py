"""S119q2i8z8: TRAIN-only second-native-BPE contextual competitor objective contract.

RESEARCH-ONLY. This is a proposed loss component, NOT a trained Calibri model.
Prior real original S142 43M/12GT/Core/NRA3/NPA1/rank4 native49152
S119q2i8z7 found the contextual rank4 term increases WRONG-vs-human-gold
winner gap at SECOND BPE on 8/14 already-spent internal synthetic Z2 DEV.
At THIRD it often helps gold. Prior simple strongest-nonEOS wrong penalty
on another X source did not improve true OWN generation. Therefore z8
defines a DISTINCT smooth top-k native-competitor objective; it is a
hypothesis to test with FRESH PRECOMMITTED TRAIN/DEV, NOT on spent Z2/X.

A later controlled research run may compare identical parent, data and
optimizer with baseline native CE versus baseline+this TRAIN-only BPE2
objective. This file must never evaluate DEV, J7/EVAL128 or GOLD, change
pretrained 43M/GT donor, alter generation logits, mask native tokens at
inference, add areas/parameters, select a model or promote to LIVE.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

STAGE = "CALIBRI_S119Q2I8Z8_TRAINONLY_SECOND_NATIVE_BPE_TOPK_CONTEXT_MARGIN_CONTRACT"
NATIVE_VOCAB = 49152
EOS_NATIVE = 2
SECOND_NATIVE_INDEX = 1
COMPETITOR_TOPK = 8
SMOOTH_TEMPERATURE = 0.5
GOLD_LOGIT_MARGIN = 1.0
MARGIN_LOSS_WEIGHT = 0.20


def trainonly_second_native_margin(
    logits: torch.Tensor,
    target_native_id: int,
    *,
    bpe_index: int,
    source_split: str,
    human_previous_gold: bool,
) -> torch.Tensor:
    """Extra differentiable TRAIN-only objective, separate from all-position CE.

    Apply exactly once at BPE index 1 of each human-gold TRAIN answer.
    Treat ALL wrong native IDs, including EOS=2, as possible competitors.
    Exclude only the correct gold ID *within this additional loss*; do not
    mask logits in normal CE or in any decoder. The top8 smooth aggregate
    is a contextual alternative to repeating the prior top1 hard-negative.

    The loss is a PREDECLARED mechanistic hypothesis, not proof of quality.
    BPE index 2 and human terminal EOS retain their original CE objective.
    """
    if source_split != "TRAIN":
        raise ValueError("Fail closed: no DEV, EVAL, J7 or GOLD in objective")
    if not human_previous_gold:
        raise ValueError("Fail closed: no own/generated-prefix pseudo-targets")
    if not isinstance(logits, torch.Tensor) or logits.shape != (1, NATIVE_VOCAB):
        raise ValueError("Requires one real full 49152 native-vocabulary logit vector")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite or nonfloating full-native logits forbidden")
    if type(target_native_id) is not int or not (0 <= target_native_id < NATIVE_VOCAB):
        raise ValueError("Invalid human native BPE ID")
    if type(bpe_index) is not int or bpe_index < 0:
        raise ValueError("Invalid zero-based native position")
    if bpe_index != SECOND_NATIVE_INDEX:
        # No additional loss at FIRST, THIRD, or EOS; all-position CE remains.
        return logits.float().sum() * 0.0
    if target_native_id == EOS_NATIVE:
        raise ValueError("Second gold native BPE must be human continuation, not EOS")

    native = logits.float()[0]
    wrong_only = native.clone()
    wrong_only[target_native_id] = float("-inf")
    top_wrong, wrong_ids = torch.topk(wrong_only, k=COMPETITOR_TOPK)
    if bool((wrong_ids == target_native_id).any()) or not bool(torch.isfinite(top_wrong).all()):
        raise ValueError("Wrong-only topK must contain finite non-gold native IDs")
    # Soft maximum of eight contextual wrong logits, centered to avoid
    # a free +log(8) disadvantage relative to the gold logit.
    smooth_wrong = SMOOTH_TEMPERATURE * (
        torch.logsumexp(top_wrong / SMOOTH_TEMPERATURE, dim=0)
        - math.log(COMPETITOR_TOPK)
    )
    gap = smooth_wrong - native[target_native_id]
    penalty = MARGIN_LOSS_WEIGHT * F.softplus(GOLD_LOGIT_MARGIN + gap)
    if penalty.ndim != 0 or not bool(torch.isfinite(penalty)):
        raise ValueError("Finite scalar second-BPE contextual margin required")
    return penalty


def training_objective(
    logits: torch.Tensor,
    target_native_id: int,
    *,
    bpe_index: int,
    source_split: str,
    human_previous_gold: bool,
    native_ce_position_weight: float,
) -> torch.Tensor:
    """Full-native CE + predeclared add-on; unchanged CE at all other BPEs."""
    if native_ce_position_weight not in (1.0, 3.0):
        raise ValueError("Existing baseline human CE weights only")
    if bpe_index == SECOND_NATIVE_INDEX and native_ce_position_weight != 3.0:
        raise ValueError("Matched existing 3x second native CE required")
    target = torch.tensor([target_native_id], dtype=torch.long, device=logits.device)
    base = native_ce_position_weight * F.cross_entropy(logits.float(), target)
    return base + trainonly_second_native_margin(
        logits, target_native_id, bpe_index=bpe_index,
        source_split=source_split, human_previous_gold=human_previous_gold,
    )
