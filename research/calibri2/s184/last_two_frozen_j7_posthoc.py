"""S184: POSTHOC-only identify 2 unresolved original TRAIN J7 errors after S182/S183.

Only immutable, already-generated artifacts enter this program. S87 labels
are used solely to score completed S179 traces: NEVER for inference, routing,
candidate selection, training, or generation. Output is SHA-only, without
question strings, predicted tokens, target IDs, or decoded reference answers.
No Jessica chat, LIVE, Render, Supabase, APK, or model touched.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from calibri2.s180.offline_failure_attribution import analyze as frozen_s180_analysis


def analyze(original_trace: dict, frozen_gold: dict,
            s182: dict, s183: dict) -> dict:
    base = frozen_s180_analysis(original_trace, frozen_gold)
    if (original_trace.get("labels_used") is not False
            or base["full_eos_correct"] != 428
            or base["incorrect_count"] != 28):
        raise AssertionError("S184 immutable original 428/456 frozen CPU provenance failed")
    if (s182.get("event") != "S182_REAL_CPU_J7_456_128_SHADOW_PROTECTED"
            or s182.get("source_s179_frozen_baseline_run") != 43323981995
            or s182.get("retention_passed") is not True
            or s182.get("model_accepted") is not False
            or s182.get("old_original_train_before") != 428
            or s182.get("old_original_train_after") != 454
            or s182.get("new_train128_actual_full_EOS_correct") != 128
            or s182.get("old_gains") != 26
            or s182.get("old_losses") != 0
            or s182.get("first_losses") != 0
            or s182.get("non_shadow_prediction_changes") != 0
            or s182.get("real_CPU_J7_decoder_invoked") is not True):
        raise AssertionError("S184 immutable S182 real CPU J7 no-forgetting gate failed")
    if (s183.get("event") != "S183_REAL_CPU_S176_VS_S182_PAIRED_SAME_CPU_SHADOW_J7"
            or s183.get("parent_S182_run") != 43331434088
            or s183.get("independent_speed_gate_pass") is not True
            or s183.get("model_accepted") is not False
            or s183.get("correct_456") != 454
            or s183.get("correct_new128") != 128
            or s183.get("old_losses") != 0
            or s183.get("matched_combined_speed_ratio", 0) < .9
            or s183.get("physical_bpp_including_benchmark_code", 1) > .7):
        raise AssertionError("S184 immutable S183 independent CPU/bpp contract failed")
    misses = {row["prompt_sha256"]: row for row in base["incorrect_sha_only"]}
    gained = s182.get("old_gain_sha_only")
    lost = s182.get("old_loss_sha_only")
    if (not isinstance(gained, list) or len(gained) != 26
            or len(set(gained)) != 26 or not set(gained).issubset(misses)
            or lost != []):
        raise AssertionError("S184 improved SHA provenance invalid or old gains not misses")
    remain = sorted(set(misses) - set(gained))
    if len(remain) != 2:
        raise AssertionError("S184 expected exactly 2 unresolved original frozen J7 examples")
    kinds = Counter()
    remaining_rows = []
    for sha in remain:
        row = misses[sha]
        first = bool(row["first_token_wrong"])
        second = bool(row["second_wrong_after_first_right"])
        eos = bool(row["missing_eos"] or row["early_eos"] or row["wrong_length"])
        family = ("first-token" if first else "second-token" if second
                  else "EOS-or-length" if eos else "other-full-sequence")
        kinds[family] += 1
        remaining_rows.append({
            "prompt_sha256": sha,
            "route_from_original_real_J7": row["route"],
            "failure_family_posthoc": family,
            "first_token_wrong": first,
            "second_wrong_after_first_right": second,
            "early_eos": bool(row["early_eos"]),
            "missing_eos": bool(row["missing_eos"]),
            "wrong_length": bool(row["wrong_length"]),
            "second_token_target_in_original_frozen_top5_posthoc":
                row["correct_second_in_observed_step2_top5"],
            "original_second_step_eos_rank": row["eos_rank_at_second_step"],
            "original_preexisting_guard_events": row["guard_events"],
        })
    report = {
        "event": "S184_LAST_TWO_SOURCE456_FROZEN_J7_POSTHOC_SHA_ONLY",
        "S179_run_id": 43323981995,
        "S182_success_run_id": 43331434088,
        "S183_success_run_id": 43332985210,
        "original_frozen_full_before": 428,
        "s182_actual_frozen_full_after": 454,
        "S182_gained_frozen_original": 26,
        "remaining_incorrect": len(remain),
        "original_train_size": 456,
        "S183_new128_protected_full": 128,
        "remaining_failure_families": dict(sorted(kinds.items())),
        "remaining_cases_SHA_only": remaining_rows,
        "no_labels_or_answers_enter_model_or_selector": True,
        "no_model_execution_or_training": True,
        "no_QA_TEST": True,
        "LIVE_Jessica_Render_Supabase_APK_unchanged": True,
        "model_accepted": False,
        "next_hypothesis": "Design separate prompt-only fail-closed research-only proposals for remaining family; test only on frozen TRAIN with zero forgetting, never lookup by SHA at inference.",
    }
    dump = json.dumps(report, sort_keys=True)
    for forbidden in ("target_ids", "full_generated_ids", "expected_ids",
                      "decoded_answer", "question_text"):
        if forbidden in dump:
            raise AssertionError("S184 posthoc report exposed gold/predicted answers")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("original_s179_trace", type=Path)
    p.add_argument("original_s87_gold", type=Path)
    p.add_argument("passed_s182_real_j7", type=Path)
    p.add_argument("passed_s183_speed", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("S184 immutable SHA-only output cannot be overwritten")
    inputs = [json.loads(x.read_text(encoding="utf-8")) for x in (
        args.original_s179_trace, args.original_s87_gold,
        args.passed_s182_real_j7, args.passed_s183_speed)]
    report = analyze(*inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2),
                           encoding="utf-8")
    print(json.dumps({
        "event": report["event"],
        "remaining_incorrect": report["remaining_incorrect"],
        "failure_families": report["remaining_failure_families"],
        "no_model_execution": True, "research_only": True,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
