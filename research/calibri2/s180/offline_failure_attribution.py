"""CALIBRI S180: offline-only attribution of frozen S179 456 trace rows.

Research read-only: join S87 TRAIN labels only AFTER the completed S179
inference artifact is immutable. No model imports/weights/selectors/routing
or parameter writes. Outputs SHA-only case summaries, never gold token ids.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(trace_archive: dict, s87_report: dict) -> dict:
    if (trace_archive.get("stage") != "S179_TRACE_ONLY"
            or trace_archive.get("number") != 456
            or trace_archive.get("labels_used") is not False):
        raise AssertionError("S180 requires immutable label-free S179 456 trace")
    if s87_report.get("stage") != "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT":
        raise AssertionError("S180 requires frozen S87 original TRAIN labels")
    cases = trace_archive.get("cases")
    labeled = s87_report.get("all_456_per_identity_results")
    if not isinstance(cases, list) or not isinstance(labeled, list):
        raise AssertionError("S180 source records missing")
    if len(cases) != 456 or len(labeled) != 456:
        raise AssertionError("S180 requires exactly 456 source examples")
    expected = {}
    for r in labeled:
        sha, ids = r["question_sha256"], r["target_ids"]
        if (sha in expected or not isinstance(ids, list) or len(ids) < 2
                or any(not isinstance(i, int) or i < 0 for i in ids)):
            raise AssertionError("S180 duplicated or invalid S87 labels")
        expected[sha] = ids
    observed = {}
    for row in cases:
        sha = row["prompt_sha256"]
        if (sha in observed or row.get("gold_seen_at_inference") is not False
                or "expected" in json.dumps(row).lower()
                or not isinstance(row.get("full_generated_ids"), list)):
            raise AssertionError("S180 label leakage or duplicated S179 trace")
        observed[sha] = row
    if set(observed) != set(expected):
        raise AssertionError("S180 S87-to-S179 hashed 456 cohort mismatch")
    flags = Counter()
    routes = Counter()
    guards = Counter()
    rank_count = Counter()
    misses = []
    for sha in sorted(observed):
        row = observed[sha]
        gen, gold = row["full_generated_ids"], expected[sha]
        if not gen or any(not isinstance(x, int) for x in gen):
            raise AssertionError("S180 malformed full decode")
        eos = gold[-1]
        if row["terminated_in_eos"] != (gen[-1] == eos):
            raise AssertionError("S180 EOS trace integrity failure")
        if gen == gold:
            continue
        first_wrong = gen[0] != gold[0]
        second_wrong_after_first_right = (
            not first_wrong and (len(gen) < 2 or gen[1] != gold[1]))
        early_eos = eos in gen[:2]
        missing_eos = gen[-1] != eos
        extra_or_short = len(gen) != len(gold)
        route = row["route"]
        routes[route] += 1
        for k, v in (("first_token_wrong", first_wrong),
                     ("second_wrong_after_first_right", second_wrong_after_first_right),
                     ("early_eos", early_eos),
                     ("missing_eos", missing_eos),
                     ("wrong_length", extra_or_short)):
            if v:
                flags[k] += 1
        for event in row["guard_events"]:
            guards[event] += 1
        conditional = row.get("conditional_step2")
        correct_second_in_top5 = None
        eos_rank_at_second_step = None
        if conditional is not None:
            if conditional.get("score_type") != "actual_ar_decoder_conditional_step2_logits":
                raise AssertionError("S180 cannot use surrogate span scores")
            top5 = conditional["top5_token_ids_and_logits"]
            correct_second_in_top5 = any(t[0] == gold[1] for t in top5)
            eos_rank_at_second_step = conditional["eos_rank"]
            rank_count["gold_second_in_top5" if correct_second_in_top5
                       else "gold_second_outside_top5"] += 1
        else:
            rank_count["no_conditional_step2"] += 1
        misses.append({
            "prompt_sha256": sha,
            "route": route,
            "first_token_wrong": first_wrong,
            "second_wrong_after_first_right": second_wrong_after_first_right,
            "early_eos": early_eos,
            "missing_eos": missing_eos,
            "wrong_length": extra_or_short,
            "correct_second_in_observed_step2_top5": correct_second_in_top5,
            "eos_rank_at_second_step": eos_rank_at_second_step,
            "guard_events": row["guard_events"],
        })
    if len(misses) != 28:
        raise AssertionError(f"S180 S176 428/456 source drift: {len(misses)} failures")
    return {
        "stage": "S180_OFFLINE_POSTHOC_SHA_ONLY",
        "source_s179_run_id": 43323981995,
        "original_train_count": 456,
        "full_eos_correct": 456 - len(misses),
        "incorrect_count": len(misses),
        "failure_flags_overlapping": dict(sorted(flags.items())),
        "failure_routes": dict(sorted(routes.items())),
        "failure_guard_events": dict(sorted(guards.items())),
        "observed_second_step_top5_coverage": dict(sorted(rank_count.items())),
        "incorrect_sha_only": misses,
        "gold_used_at_inference": False,
        "no_model_training_or_selection": True,
        "model_accepted": False,
        "Jessica_LIVE_untouched": True,
        "note": "Step-2 EOS ranks are BEFORE generating the second token, not EOS-after-token-2.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inference_trace", type=Path)
    p.add_argument("frozen_s87_gold", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("S180 posthoc results must not be overwritten")
    result = analyze(json.loads(a.inference_trace.read_text()),
                     json.loads(a.frozen_s87_gold.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "S180_POSTHOC_28_SHA_ONLY",
        "incorrect": result["incorrect_count"],
        "flags": result["failure_flags_overlapping"],
        "rank_stats": result["observed_second_step_top5_coverage"],
        "model_accepted": False,
    }), flush=True)


if __name__ == "__main__":
    main()
