"""S143: classify persisted J7 TRAIN gains; never promote a model from this report.

A commit may improve any previously incorrect old-456 full/EOS answer or
a new S122 full/EOS answer/pair, but must lose zero previously correct
first/full identities. Final physically-reloaded CPU proof is still required.
"""
from __future__ import annotations

import math


def classify_j7_development_gain(
    *, committed: bool, before_old_full: int, after_old_full: int,
    before_new_full: int, after_new_full: int,
    before_new_pairs: int, after_new_pairs: int,
    lost_old_full: int, lost_old_first: int,
    lost_new_full: int, lost_new_first: int,
    fp16_reloaded: bool, immutable_source: bool,
    physical_guard: bool, j7_guard: bool,
    physical_bpp: float, cpu_ratio: float,
) -> dict:
    """Describe TRAIN-only improvements; never equate them to held-out quality."""
    counters = (before_old_full, after_old_full, before_new_full, after_new_full,
                before_new_pairs, after_new_pairs, lost_old_full, lost_old_first,
                lost_new_full, lost_new_first)
    if (any(type(c) is not int or c < 0 for c in counters)
        or not 0 <= before_old_full <= 456 or not 0 <= after_old_full <= 456
        or not 0 <= before_new_full <= 128 or not 0 <= after_new_full <= 128
        or not 0 <= before_new_pairs <= 64 or not 0 <= after_new_pairs <= 64
        or not all(type(x) is bool for x in (committed, fp16_reloaded,
                                           immutable_source, physical_guard, j7_guard))
        or not math.isfinite(physical_bpp) or not math.isfinite(cpu_ratio)
        or physical_bpp <= 0 or cpu_ratio <= 0):
        raise ValueError("S143 invalid J7 full/EOS development evidence")
    old_gain = after_old_full - before_old_full
    new_gain = after_new_full - before_new_full
    pair_gain = after_new_pairs - before_new_pairs
    safe = (
        committed
        and fp16_reloaded and immutable_source and physical_guard and j7_guard
        and physical_bpp <= .7 and cpu_ratio >= .9
        and lost_old_full == 0 and lost_old_first == 0
        and lost_new_full == 0 and lost_new_first == 0
        and old_gain >= 0 and new_gain >= 0 and pair_gain >= 0
        and (old_gain > 0 or new_gain > 0 or pair_gain > 0)
    )
    return {
        "historical_old456_full_delta": old_gain,
        "S122_new128_full_delta": new_gain,
        "S122_both_sides_pairs_delta": pair_gain,
        "all_previously_correct_full_first_identities_preserved": (
            lost_old_full == 0 and lost_old_first == 0
            and lost_new_full == 0 and lost_new_first == 0
        ),
        "safe_development_TRAIN_improvement": safe,
        "research_only_not_model_acceptance": True,
        "independent_unseen_QA_not_run": True,
    }
