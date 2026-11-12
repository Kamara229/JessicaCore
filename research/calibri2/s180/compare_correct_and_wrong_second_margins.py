"""S180: compare observed token-2 logit margins in 428 frozen correct
and 18 first-token-correct / second-token-incorrect TRAIN cases.

OFFLINE ONLY: same immutable S179 real CPU J7 traces + S87 TRAIN labels.
No model inference, training, selector, threshold tuning, gold-driven routing,
held-out evaluation, or production/Jessica/LIVE changes.
Only compare logits when the frozen AR decoder really exposed token-2 logits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median

from calibri2.s180.offline_failure_attribution import analyze


def distribution(values):
    if any(not math.isfinite(v) for v in values):
        raise AssertionError("S180 nonfinite margin")
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(values), "min": min(values),
            "median": median(values), "max": max(values)}


def compare(trace_archive: dict, s87_archive: dict) -> dict:
    baseline = analyze(trace_archive, s87_archive)
    if baseline["full_eos_correct"] != 428 or baseline["incorrect_count"] != 28:
        raise AssertionError("S180 frozen correctness baseline drift")
    targets = {r["question_sha256"]: r["target_ids"]
               for r in s87_archive["all_456_per_identity_results"]}
    traces = {r["prompt_sha256"]: r for r in trace_archive["cases"]}
    if set(targets) != set(traces) or len(targets) != 456:
        raise AssertionError("S180 SHA cohort mismatch")

    buckets = {
        "full_correct": {"observed": 0, "missing": 0,
                         "routes": Counter(), "raw_matches": Counter(),
                         "margins": [], "gold_top1_gaps": []},
        "token2_wrong_first_correct": {"observed": 0, "missing": 0,
                         "routes": Counter(), "raw_matches": Counter(),
                         "margins": [], "gold_top1_gaps": []},
        "other_wrong": {"observed": 0, "missing": 0,
                         "routes": Counter(), "raw_matches": Counter(),
                         "margins": [], "gold_top1_gaps": []},
    }
    per_route = defaultdict(lambda: defaultdict(list))
    for sha in sorted(traces):
        row = traces[sha]
        gen, gold = row["full_generated_ids"], targets[sha]
        if gen == gold:
            group = "full_correct"
        elif gen[0] == gold[0] and (len(gen) < 2 or gen[1] != gold[1]):
            group = "token2_wrong_first_correct"
        else:
            group = "other_wrong"
        bucket = buckets[group]
        bucket["routes"][row["route"]] += 1
        token2 = row.get("conditional_step2")
        if token2 is None:
            bucket["missing"] += 1
            continue
        if token2.get("score_type") != "actual_ar_decoder_conditional_step2_logits":
            raise AssertionError("S180 forbids surrogate span head logits")
        raw = token2.get("top5_token_ids_and_logits")
        if (not isinstance(raw, list) or not raw or len(raw) > 5
                or any(not isinstance(pair, list) or len(pair) != 2
                       or not isinstance(pair[0], int)
                       or not isinstance(pair[1], (int, float))
                       or not math.isfinite(pair[1]) for pair in raw)
                or len(set(pair[0] for pair in raw)) != len(raw)
                or any(raw[i][1] < raw[i + 1][1] for i in range(len(raw) - 1))):
            raise AssertionError("S180 malformed actual decoder top5")
        bucket["observed"] += 1
        raw_matches = len(gen) > 1 and raw[0][0] == gen[1]
        bucket["raw_matches"][str(raw_matches).lower()] += 1
        if len(raw) > 1:
            gap = float(raw[0][1] - raw[1][1])
            if gap < 0:
                raise AssertionError("S180 inverted first/second scores")
            bucket["margins"].append(gap)
            per_route[row["route"]][group].append(gap)
        for token, logit in raw:
            if token == gold[1]:
                bucket["gold_top1_gaps"].append(float(raw[0][1] - logit))
                break

    if sum(buckets[k]["observed"] + buckets[k]["missing"] for k in buckets) != 456:
        raise AssertionError("S180 full cohort coverage drift")
    if (buckets["token2_wrong_first_correct"]["observed"] +
        buckets["token2_wrong_first_correct"]["missing"]) != 18:
        raise AssertionError("S180 exact 18 second token wrong cohort drift")
    if (buckets["full_correct"]["observed"] +
        buckets["full_correct"]["missing"]) != 428:
        raise AssertionError("S180 exact 428 full-correct cohort drift")

    summarized = {}
    for name, group in buckets.items():
        summarized[name] = {
            "total": group["observed"] + group["missing"],
            "actual_token2_logits_observed": group["observed"],
            "no_actual_token2_logits": group["missing"],
            "routes": dict(sorted(group["routes"].items())),
            "raw_top1_matches_generated": dict(sorted(group["raw_matches"].items())),
            "top1_minus_top2_logits": distribution(group["margins"]),
            "top1_minus_gold_token2_logits_when_gold_in_top5":
                distribution(group["gold_top1_gaps"]),
        }
    route_comp = {}
    for route in sorted(per_route):
        good = per_route[route].get("full_correct", [])
        misses = per_route[route].get("token2_wrong_first_correct", [])
        route_comp[route] = {
            "correct_observed_top2_margin": distribution(good),
            "second_wrong_observed_top2_margin": distribution(misses),
            "comparable_both_groups": bool(good and misses),
        }
    return {
        "stage": "S180_OFFLINE_TOP2_MARGIN_428_VS_18_NO_POLICY",
        "source_s179_run_id": 43323981995,
        "original_train_456": 456,
        "frozen_correct_428": summarized["full_correct"],
        "first_correct_second_wrong_18":
            summarized["token2_wrong_first_correct"],
        "other_wrong_10": summarized["other_wrong"],
        "within_route_comparison": route_comp,
        "gold_join_after_inference_only": True,
        "no_training_or_inference_threshold_proposed": True,
        "model_accepted": False,
        "notes": [
            "Compare ONLY rows with actual second-step AR logits; absent logits are not imputed.",
            "Raw AR top1 can differ from final output due to copy-net / guard overrides.",
            "Logit margin is not calibrated confidence, and a margin distribution alone does not prove quality gains.",
            "Step2 EOS rank measures early EOS, NOT EOS after generating token2.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frozen_s179_trace", type=Path)
    p.add_argument("frozen_s87_gold", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("S180 read-only comparison output must not be overwritten")
    report = compare(json.loads(a.frozen_s179_trace.read_text()),
                     json.loads(a.frozen_s87_gold.read_text()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "event": "S180_FROZEN_428_VS_18_TOP2_MARGIN_COMPARISON",
        "correct": report["frozen_correct_428"]["top1_minus_top2_logits"],
        "wrong_second": report["first_correct_second_wrong_18"]["top1_minus_top2_logits"],
        "correct_missing_logits": report["frozen_correct_428"]["no_actual_token2_logits"],
        "model_accepted": False,
    }), flush=True)


if __name__ == "__main__":
    main()
