"""CALIBRI v204 precommitted TRAIN-only global per-step acceptance.

Protect all 24 contradictory TRAIN pairs at first token and all 8 frozen
TRAIN six-token opposing-answer rollouts at EVERY proposed step.
No QA, VALID, independently held-out TEST, or Jessica calls are permitted.
"""
from v198.guard import anti_collapse_gate
from v200.rollout import sequence_no_regression_gate


def evaluate_candidate(
    immutable_full_first, candidate_full_first, immutable_full_sequence,
    candidate_full_sequence, frozen_sampled_ids, full_diversity_floor,
    objective_before, objective_after, max_relative_increase=0.01,
    sampled_diversity_floor=0.25,
):
    if len(immutable_full_first) != 24 or len(candidate_full_first) != 24:
        raise AssertionError("Full frozen 24-pair TRAIN first-token gate required")
    if len(immutable_full_sequence) != 8 or len(candidate_full_sequence) != 8:
        raise AssertionError("All eight immutable TRAIN complete-sequence pairs required")
    keys=tuple(frozen_sampled_ids)
    if len(set(keys))!=8 or set(keys)!=set(immutable_full_sequence):
        raise AssertionError("TRAIN sampled pairs must exactly match frozen full sequences")
    if not set(keys)<=set(immutable_full_first):
        raise AssertionError("Subset of original 24 TRAIN pairs expected")
    if not 0<=full_diversity_floor<=1:
        raise ValueError("Invalid frozen baseline-relative diversity floor")
    if not 0<max_relative_increase<=0.1:
        raise ValueError("Invalid precommitted loss tolerance")
    small_before={key:immutable_full_first[key] for key in keys}
    small_after={key:candidate_full_first[key] for key in keys}
    sampled=anti_collapse_gate(small_before,small_after,
        minimum_distinct_fraction=sampled_diversity_floor)
    full=anti_collapse_gate(immutable_full_first,candidate_full_first,
        minimum_distinct_fraction=full_diversity_floor)
    sequence=sequence_no_regression_gate(
        immutable_full_sequence,candidate_full_sequence)
    loss_ok=(float(objective_after)<=float(objective_before)*(1+max_relative_increase))
    accepted=all((
        sampled["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"],
        full["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"],
        sequence["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"],
        loss_ok
    ))
    return {
        "accept":accepted,
        "sampled_eight_first_token_guard":sampled,
        "global_twentyfour_first_token_guard":full,
        "global_eight_full_sequence_guard":sequence,
        "TRAIN_loss_no_more_than_1percent_increase":loss_ok,
        "per_step_full_all_pairs_checked":True,
        "used_only_frozen_TRAIN":True,
        "not_informed_by_public_DEV_or_VALID":True
    }
