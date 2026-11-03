"""Judge actual answers, meaningful generation and physical/runtime cost ONLY.

No KL, NLL, teacher output matching, or weights similarity in decision API.
Small heldout subsets are exploratory; better scores can be promoted only
after broader external correctness and qualitative response validation.
"""
from __future__ import annotations
from binary_geometry_v64 import EXPECTED_UNIQUE

BPP_LIMIT=.700
MIN_VALID_QUESTIONS=100
MIN_DIAGNOSTIC_PROMPTS=12
TARGET_DISTINCT=.25
MAX_TOKEN_DOMINANCE=.65
MIN_NATIVE_SPEED_RATIO=.90

def assess(qa,generation,physical_bytes,native_speed_ratio,
           native_speed_is_full_model,generation_prompts,
           meaningful_quality_checked=False,full_model_speed_checked=False):
    if physical_bytes<=0 or native_speed_ratio<=0:raise AssertionError("invalid physical/runtime measurements")
    bpp=8*physical_bytes/EXPECTED_UNIQUE
    a=qa["original"];b=qa["candidate"]
    baseline=generation["baseline"];candidate=generation["candidate"]
    checks={
        "all_archive_bytes_under_0p700bpp":bpp<=BPP_LIMIT,
        "more_correct_answers":qa["correct_answer_gain"]>0,
        "no_balanced_accuracy_regression":(
            b["balanced_accuracy"] is not None and a["balanced_accuracy"] is not None
            and b["balanced_accuracy"]>=a["balanced_accuracy"]),
        "diversity_improved":candidate["mean_distinct_ratio"]>baseline["mean_distinct_ratio"],
        "distinct_ratio_at_least_0p25":candidate["mean_distinct_ratio"]>=TARGET_DISTINCT,
        "repeat_dominance_at_most_0p65":candidate["mean_most_frequent_share"]<=MAX_TOKEN_DOMINANCE,
        "enough_independent_VALID_questions":a["count"]>=MIN_VALID_QUESTIONS,
        "enough_generation_prompts":generation_prompts>=MIN_DIAGNOSTIC_PROMPTS,
        "paired_correctness_evidence":qa["claim_statistically_proven"],
        "meaningful_coherent_response_quality_checked":meaningful_quality_checked,
        "actual_full_model_speed_checked":bool(native_speed_is_full_model and full_model_speed_checked),
        "actual_full_model_speed_at_least_90pct":bool(
            native_speed_is_full_model and full_model_speed_checked
            and native_speed_ratio>=MIN_NATIVE_SPEED_RATIO),
    }
    if (not checks["all_archive_bytes_under_0p700bpp"]
        or qa["correct_answer_gain"]<0
        or b["balanced_accuracy"] is not None and a["balanced_accuracy"] is not None
        and b["balanced_accuracy"]<a["balanced_accuracy"]
        or candidate["mean_distinct_ratio"]<baseline["mean_distinct_ratio"]):
        status="REJECT_QUALITY_REGRESSION"
    elif all(checks.values()):
        status="STAGE_FOR_HUMAN_QUALITY_REVIEW"
    else:
        status="PROMISING_BUT_NOT_YET_PROVEN" if (
            checks["more_correct_answers"] and checks["diversity_improved"]
            and checks["all_archive_bytes_under_0p700bpp"]) else "INSUFFICIENT_EVIDENCE"
    return {"status":status,"checks":checks,
        "all_source_bits_per_parameter":bpp,
        "decision_is_based_only_on_task_outcomes_diversity_speed_size":True,
        "no_distribution_distance_or_teacher_similarity_in_selection":True,
        "automatic_live_changes_authorized":False,
        "frozen_independent_TEST_not_accessed":True}
