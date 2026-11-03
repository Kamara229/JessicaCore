"""Quality preserving compression gate: outcomes first, no FP32 imitation.

No release on a tiny sample or hard QA/generation regression. Source rate
and genuine end-to-end CPU decoding speed are hard safety constraints.
"""
from __future__ import annotations

MAX_BPP=.700
MIN_SPEED_RATIO=.90
MIN_DISTINCT=.25
MAX_REPETITION_SHARE=.65

def evaluate(qa,generation,rate,semantic_quality_verified=False):
    domains=("BoolQ","SciQ")
    if set(qa)!=set(domains):raise AssertionError("required real independent QA domains missing")
    if rate["source_all_in_bpp"]>MAX_BPP:
        return {"status":"REJECT_OVER_BUDGET","reason":"complete physical source above 0.700 bpw"}
    speed=generation["paired_full_model_greedy_decode_speed_ratio"]
    base=generation["original"];candidate=generation["candidate"]
    each={
        name:{"n":qa[name]["baseline"]["count"],
          "correct_delta":qa[name]["correct_answer_gain"],
          "accuracy_delta":qa[name]["accuracy_gain"],
          "balanced_accuracy_delta":qa[name]["rewritten"]["balanced_accuracy"]-
                                    qa[name]["baseline"]["balanced_accuracy"],
          "enough_samples":qa[name]["baseline"]["count"]>=40}
        for name in domains}
    pretrained_correct={name:qa[name].get("original_FP32_task_correctness_only",{}).get("correct") for name in domains}
    original_functional_quality_measured=all(v is not None for v in pretrained_correct.values())
    preserved_original_quality=original_functional_quality_measured and all(
        qa[name]["rewritten"]["correct"]>=pretrained_correct[name] for name in domains)
    metrics={"answer_gains":each,
        "original_FP32_actual_correct_answers":pretrained_correct,
        "original_functional_quality_measured":original_functional_quality_measured,
        "pretrained_answer_quality_preserved_on_all_domains":preserved_original_quality,
        "total_questions":sum(x["n"] for x in each.values()),
        "all_domains_at_least_no_regression":all(x["correct_delta"]>=0
                                 and x["balanced_accuracy_delta"]>=0 for x in each.values()),
        "overall_answer_gain":sum(x["correct_delta"] for x in each.values()),
        "generation_ratio_delta":candidate["mean_distinct_ratio"]-base["mean_distinct_ratio"],
        "actual_full_model_greedy_ratio":speed,
        "final_diversity":candidate["mean_distinct_ratio"],
        "final_repetition":candidate["mean_most_frequent_share"],
        "source_rate_bpp":rate["source_all_in_bpp"]}
    severe=(not metrics["all_domains_at_least_no_regression"] or
            metrics["generation_ratio_delta"]<-.005 or speed<MIN_SPEED_RATIO)
    ready=(metrics["total_questions"]>=160 and
           all(x["enough_samples"] for x in each.values()) and
           metrics["overall_answer_gain"]>0 and
           metrics["all_domains_at_least_no_regression"] and
           candidate["mean_distinct_ratio"]>=MIN_DISTINCT and
           candidate["mean_most_frequent_share"]<=MAX_REPETITION_SHARE and
           speed>=MIN_SPEED_RATIO and semantic_quality_verified and preserved_original_quality)
    status=("REJECT_FUNCTIONAL_REGRESSION" if severe else
           "STAGE_RESEARCH_CANDIDATE_FOR_REVIEW" if ready else
           "PROMISING_NEEDS_BETTER_BEHAVIORAL_QUALITY")
    return {"status":status,"metrics":metrics,
        "selection_uses_direct_functional_metrics_only":True,
        "semantic_responses_not_claimed_proven":not semantic_quality_verified,
        "requires_new_unopened_official_TEST_after_freeze":True,
        "automatic_live_Jessica_deploy_forbidden":True}
