"""Evidence-based automatic next repair action, not a manually chosen KL target.

Only propose isolated whole-block research trials; never modify Live Jessica.
"""
from __future__ import annotations

def plan(qa,generation,rate,decision):
    issues=[]
    for name in ("BoolQ","SciQ"):
        x=qa[name]
        if x["correct_answer_gain"]<0:
            issues.append({"priority":1,"problem":"functional_regression","domain":name,
                "lost_cases":x["regressed_examples"],"recovered_cases":x["recovered_examples"],
                "next_isolated_research_action":"replace_whole_compressed_block_operator_by_alternate_same_byte_encoding"})
        if x["rewritten"]["majority_answer_fraction"]>.90:
            issues.append({"priority":1,"problem":"answer_collapse","domain":name,
                "next_isolated_research_action":"test_attention_and_output_head_class_balanced_source_only_guard"})
    g=generation["candidate"]
    if g["mean_distinct_ratio"]<.25 or g["mean_most_frequent_share"]>.65:
        issues.append({"priority":1,"problem":"greedy_repetition",
            "observed_distinct":g["mean_distinct_ratio"],
            "observed_most_frequent_share":g["mean_most_frequent_share"],
            "next_isolated_research_action":"trace_KV_cache_and_attention_value_per_step_then_whole_block_substitution"})
    if generation["paired_full_model_greedy_decode_speed_ratio"]<.9:
        issues.append({"priority":1,"problem":"decode_speed_loss",
            "next_isolated_research_action":"profile_native_GEMV_hotpath_and_revert_slow_operator"})
    if rate["source_all_in_bpp"]>.7:
        issues.append({"priority":0,"problem":"source_over_budget",
            "next_isolated_research_action":"repack_insensitive_blocks_and_retest_full_source"})
    if not issues:
        issues.append({"priority":3,"problem":"insufficient_semantic_coverage",
            "next_isolated_research_action":"human_verify_factual_and_coherent_long_form_generation"})
    issues.sort(key=lambda x:(x["priority"],x["problem"],x.get("domain","")))
    return {"observed_ranked_incidents":issues,
        "next_action":issues[0]["next_isolated_research_action"],
        "all_future_corrections_budgeted_before_train":True,
        "can_open_autonomous_research_PR_only_after_separate_policy_check":True,
        "may_automatically_change_prod_model":False,
        "candidate_status":decision["status"]}
