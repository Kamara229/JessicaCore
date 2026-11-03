"""Do not spend an extra bit or sacrifice achieved speed for exploratory accuracy.

Actual original FP32 correct answer count is functional reference only.
PASS research candidate != release approval. Require qualitative QA review.
"""
from __future__ import annotations
from binary_geometry_v64 import EXPECTED_UNIQUE

MAX_BPP=.700
ACHIEVED_NATIVE_PARENT_BYTES=11229303
MIN_MATCHED_SPEED_RATIO=.98

def decisions(candidate,old,teacher,physical_bytes,parent_bytes,speed_ratio,baseline_generation,candidate_generation):
    if parent_bytes!=ACHIEVED_NATIVE_PARENT_BYTES:
        raise AssertionError("the locked physical CB59 parent was replaced")
    if physical_bytes!=parent_bytes:
        raise AssertionError("ZERO extra-byte operator rewrite contract breached")
    if not speed_ratio>0:raise AssertionError("missing genuine full native CPU speed")
    qa={}
    for d in ("BoolQ","SciQ"):
        value=candidate[d]["score"];baseline=old[d]["score"];fp32=teacher[d]["score"]
        qa[d]={"original_FP32_correct":fp32["correct"],
               "locked_CB59_correct":baseline["correct"],
               "new_candidate_correct":value["correct"],
               "gain_over_CB59":value["correct"]-baseline["correct"],
               "difference_from_FP32_actual_answers":value["correct"]-fp32["correct"],
               "balanced_accuracy_change":value["balanced_accuracy"]-baseline["balanced_accuracy"]}
    no_domain_loss=all(x["gain_over_CB59"]>=0 and x["balanced_accuracy_change"]>=0 for x in qa.values())
    fp32_preserved=all(x["difference_from_FP32_actual_answers"]>=0 for x in qa.values())
    generation_ok=(candidate_generation["mean_distinct_ratio"]>=baseline_generation["mean_distinct_ratio"]-0.005
      and candidate_generation["mean_most_frequent_share"]<=baseline_generation["mean_most_frequent_share"]+0.005)
    checks={"same_complete_archive_bytes":physical_bytes==parent_bytes,
        "all_in_bpp_under_original_ceiling":8*physical_bytes/EXPECTED_UNIQUE<=MAX_BPP,
        "no_labeled_domain_accuracy_regression_vs_CB59":no_domain_loss,
        "ground_truth_original_FP32_quality_preserved":fp32_preserved,
        "full_model_native_speed_preserved_at_98pct":speed_ratio>=MIN_MATCHED_SPEED_RATIO,
        "greedy_generation_did_not_regress":generation_ok,
        "greedy_distinct_ratio_at_least_0p25":candidate_generation["mean_distinct_ratio"]>=0.25,
        "source_answers_semantically_verified":False}
    return {"decision":"PROMISING_RESEARCH_CANDIDATE_NEEDS_SEMANTIC_REVIEW" if (
                       all(v for k,v in checks.items() if k!="source_answers_semantically_verified"))
                    else "NEEDS_MORE_QUALITY_WORK",
        "size":physical_bytes,"source_bits_per_original_param":8*physical_bytes/EXPECTED_UNIQUE,
        "full_model_speed_ratio":speed_ratio,"correct_answers":qa,"checks":checks,
        "no_new_Jessica_or_live_authorization":True,
        "original_official_independent_TEST_not_read":True}
