"""TRAIN-only anti-collapse guard for authentic 135M Green Tunnel recovery.

This is NOT a QA grader and cannot certify FP32 parity. Every guard input
is frozen TRAIN; untouched VALID and independently sealed DEV stay out of
optimizer, rollback choices, and source-index acceptance.
"""
from collections import Counter
import torch
from v176.operators import clear


@torch.no_grad()
def probe_first_tokens(student, sources, pairs):
    if not pairs:
        raise ValueError("Frozen TRAIN guard needs opposite-response pairs")
    result = {}
    try:
        for pair in pairs:
            if pair["split"] != "TRAIN":
                raise AssertionError("Do not tune on held-out validation or QA")
            sides = {}
            for side in ("left", "right"):
                sample = pair[side]
                clear(sources)
                logits = student(input_ids=sample["prompt_ids"][None],
                                 use_cache=False).logits[0, -1].detach()
                expected = int(sample["first_expected_id"])
                actual = int(logits.argmax())
                if not torch.isfinite(logits).all():
                    raise AssertionError("Nonfinite native-source TRAIN logits")
                sides[side] = {"first_token_id": actual, "expected": expected,
                               "first_token_correct": actual == expected}
            result[pair["pair_id"]] = {"domain": pair["domain"], "sides": sides}
    finally:
        clear(sources)
    return result


def summarize_guard(probes):
    if not probes:
        raise ValueError("Empty TRAIN probe")
    per_domain = Counter()
    winners = []
    correct_ids = set()
    both_correct = set()
    for pair_id, row in sorted(probes.items()):
        sides = row["sides"]
        if set(sides) != {"left", "right"}:
            raise ValueError("Both contradictory TRAIN prompts required")
        if sides["left"]["expected"] == sides["right"]["expected"]:
            raise AssertionError("This is not an opposite-answer TRAIN pair")
        correct = 0
        for side in ("left", "right"):
            x = sides[side]
            winners.append(x["first_token_id"])
            correct += int(x["first_token_correct"])
            if x["first_token_correct"]:
                correct_ids.add((pair_id, side))
        per_domain[row["domain"]] += correct
        if correct == 2:
            both_correct.add(pair_id)
    return {"per_domain_correct": dict(per_domain),
            "correct_prompt_ids": sorted(correct_ids),
            "both_correct_pair_ids": sorted(both_correct),
            "unique_first_token_ids": sorted(set(winners)),
            "first_token_diversity": len(set(winners)) / len(winners),
            "prompt_count": len(winners)}


def anti_collapse_gate(immutable_v186, candidate, *,
                       minimum_distinct_fraction=0.25):
    """Compare a candidate with the signed immutable v186 TRAIN behavior.

    No previously correct original TRAIN first token or fully-correct
    opposing TRAIN pair may be destroyed. Domain correct count and
    diversity must not regress. 25% diversity floor is a TRAIN safeguard,
    NOT a test of actual generation correctness or final model equality.
    """
    if set(immutable_v186) != set(candidate):
        raise ValueError("Training probe pair IDs changed")
    for key in immutable_v186:
        if (immutable_v186[key]["domain"] != candidate[key]["domain"] or
            {s: immutable_v186[key]["sides"][s]["expected"]
             for s in ("left", "right")} !=
            {s: candidate[key]["sides"][s]["expected"]
             for s in ("left", "right")}):
            raise ValueError("Frozen TRAIN questions or answers modified")
    old, new = summarize_guard(immutable_v186), summarize_guard(candidate)
    old_correct = set(map(tuple, old["correct_prompt_ids"]))
    new_correct = set(map(tuple, new["correct_prompt_ids"]))
    old_pairs = set(old["both_correct_pair_ids"])
    new_pairs = set(new["both_correct_pair_ids"])
    lost = sorted(old_correct - new_correct)
    lost_pairs = sorted(old_pairs - new_pairs)
    domains = sorted(set(old["per_domain_correct"]) | set(new["per_domain_correct"]))
    domain_regressions = {
        d: {"v186": old["per_domain_correct"].get(d, 0),
            "candidate": new["per_domain_correct"].get(d, 0)}
        for d in domains
        if new["per_domain_correct"].get(d, 0) <
           old["per_domain_correct"].get(d, 0)}
    diversity_regressed = (len(new["unique_first_token_ids"]) <
                           len(old["unique_first_token_ids"]))
    diversity_below_floor = new["first_token_diversity"] < minimum_distinct_fraction
    accepted = not (lost or lost_pairs or domain_regressions or
                    diversity_regressed or diversity_below_floor)
    return {"ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE": accepted,
            "lost_previous_correct_TRAIN_first_tokens": lost,
            "lost_original_both_correct_opposite_TRAIN_pairs": lost_pairs,
            "correct_by_domain_regressions": domain_regressions,
            "distinct_first_tokens_before": len(old["unique_first_token_ids"]),
            "distinct_first_tokens_after": len(new["unique_first_token_ids"]),
            "TRAIN_first_token_diversity_before": old["first_token_diversity"],
            "TRAIN_first_token_diversity_after": new["first_token_diversity"],
            "diversity_regressed": diversity_regressed,
            "diversity_below_precommitted_floor": diversity_below_floor,
            "TRAIN_only_not_full_answer_quality": True}
