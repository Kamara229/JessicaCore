"""S180: post-inference second-token margins for the 18 first-right failures.

This module reads immutable S179 REAL CPU observation plus frozen S87 TRAIN
targets *after* all 456 model outputs exist. It never executes a model,
derives an inference-time policy, or writes to deployment branches.
The ranks and logits are observations from the second-token step, NOT
post-second-token EOS logits. Values may predate copy-net override.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median

from calibri2.s180.offline_failure_attribution import analyze


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    if any(not math.isfinite(value) for value in values):
        raise AssertionError("S180 invalid nonfinite score margin")
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def margins(trace_archive: dict, gold_archive: dict) -> dict:
    """Read-only, gold used for OFFLINE explanation, never for selection."""
    score = analyze(trace_archive, gold_archive)
    assert score["full_eos_correct"] == 428
    error_hashes = {
        r["prompt_sha256"]
        for r in score["incorrect_sha_only"]
        if r["second_wrong_after_first_right"]
    }
    if len(error_hashes) != 18:
        raise AssertionError("S180 expected 18 frozen first-right token2 errors")
    labels = {
        r["question_sha256"]: r["target_ids"]
        for r in gold_archive["all_456_per_identity_results"]
    }
    traces = {r["prompt_sha256"]: r for r in trace_archive["cases"]}
    if set(labels) != set(traces):
        raise AssertionError("S180 trace/label cohort mismatch")

    events = Counter()
    routes = Counter()
    missing = Counter()
    gold_top5 = 0
    raw_top1_equals_generated = Counter()
    top1_minus_gold = []
    top1_minus_top2 = []
    rows = []

    for sha in sorted(error_hashes):
        trace = traces[sha]
        target = labels[sha]
        generated = trace["full_generated_ids"]
        if not (generated and generated[0] == target[0]
                and (len(generated) < 2 or generated[1] != target[1])):
            raise AssertionError("S180 only first-right token2 errors admissible")
        routes[trace["route"]] += 1
        for event in trace["guard_events"]:
            events[event] += 1

        observed = trace.get("conditional_step2")
        detail = {
            "prompt_sha256": sha,
            "route": trace["route"],
            "guard_events": trace["guard_events"],
            "step2_logits_present": observed is not None,
            "gold_second_rank_within_observed_top5": None,
            "top1_minus_gold_step2_logit": None,
            "top1_minus_top2_step2_logit": None,
            "observed_raw_top1_equals_frozen_generated_second": None,
            "eos_rank_at_step2_not_after_step2": None,
        }
        if observed is None:
            missing["no_conditional_step2"] += 1
            rows.append(detail)
            continue
        if observed.get("score_type") != "actual_ar_decoder_conditional_step2_logits":
            raise AssertionError("S180 refuses surrogate second-token scores")
        top = observed.get("top5_token_ids_and_logits")
        if (not isinstance(top, list) or len(top) > 5 or len(top) < 1
                or any(not isinstance(p, list) or len(p) != 2
                       or not isinstance(p[0], int)
                       or not isinstance(p[1], (float, int))
                       or not math.isfinite(p[1]) for p in top)
                or any(top[i][1] < top[i + 1][1] for i in range(len(top) - 1))
                or len({p[0] for p in top}) != len(top)):
            raise AssertionError("S180 malformed observed top5")
        detail["eos_rank_at_step2_not_after_step2"] = observed.get("eos_rank")
        if len(top) > 1:
            gap = float(top[0][1] - top[1][1])
            detail["top1_minus_top2_step2_logit"] = gap
            top1_minus_top2.append(gap)
        generated_top1 = len(generated) > 1 and top[0][0] == generated[1]
        raw_top1_equals_generated[str(bool(generated_top1)).lower()] += 1
        detail["observed_raw_top1_equals_frozen_generated_second"] = generated_top1

        for rank, (token, value) in enumerate(top, 1):
            if token == target[1]:
                detail["gold_second_rank_within_observed_top5"] = rank
                gap = float(top[0][1] - value)
                if gap < -1e-8:
                    raise AssertionError("S180 expected logit cannot exceed raw top1")
                detail["top1_minus_gold_step2_logit"] = gap
                top1_minus_gold.append(gap)
                gold_top5 += 1
                break
        else:
            missing["gold_second_not_in_top5"] += 1
        rows.append(detail)

    if len(rows) != 18:
        raise AssertionError("S180 incomplete 18 token2 error analysis")
    return {
        "stage": "S180_SECOND_TOKEN_OFFLINE_MARGIN_DIAGNOSTIC_ONLY",
        "source_s179_run_id": 43323981995,
        "source_s180_posthoc_run_id": 43326099868,
        "original_train_correct": 428,
        "token2_errors_after_correct_first": 18,
        "conditional_step2_available": sum(row["step2_logits_present"] for row in rows),
        "gold_second_in_observed_step2_top5": gold_top5,
        "missing_or_out_of_top5": dict(sorted(missing.items())),
        "observed_raw_top1_matches_generated_second": dict(
            sorted(raw_top1_equals_generated.items())),
        "routes": dict(sorted(routes.items())),
        "guard_events": dict(sorted(events.items())),
        "top1_minus_gold_logit_distribution": summarize(top1_minus_gold),
        "top1_minus_top2_logit_distribution": summarize(top1_minus_top2),
        "sha_only_cases": rows,
        "gold_was_offline_only": True,
        "no_gold_in_inference_or_selector": True,
        "no_candidate_policy_or_model_training": True,
        "model_accepted": False,
        "note": "Step2 EOS rank is the probability of EOS as second token, NOT EOS after second; raw logits may precede copy-net override.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_s179_trace", type=Path)
    parser.add_argument("frozen_s87_gold", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("S180 margin artifact must not overwrite")
    data = margins(json.loads(args.frozen_s179_trace.read_text()),
                   json.loads(args.frozen_s87_gold.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "S180_OFFLINE_SECOND_TOKEN_MARGINS",
        "second_token_errors": 18,
        "conditional_observed": data["conditional_step2_available"],
        "gold_second_top5": data["gold_second_in_observed_step2_top5"],
        "raw_top1_matches_generated": data[
            "observed_raw_top1_matches_generated_second"],
        "median_top1_minus_gold": data[
            "top1_minus_gold_logit_distribution"]["median"],
        "model_accepted": False,
    }), flush=True)


if __name__ == "__main__":
    main()
