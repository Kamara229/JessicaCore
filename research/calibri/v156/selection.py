"""Validate completed v155 exploratory evidence before third holdout opened.

Note v_proj was v154 diagnostic WORST, but happened to outperform CB59
on BOTH unseen v155 domains on 504 gold tokens each. It is still merely
a *selected hypothesis*; v156 is independent confirmation on new source.
"""
import json
from pathlib import Path
from .protocol import SELECTED_KEY,SECONDARY_KEY,DOMAINS,ARMS
from v154.protocol import CB153_SHA
from v117.protocol import ORIGINAL_CB59_SHA


def freeze_exploratory_evidence(path):
    report=json.loads(Path(path).read_text())
    if report.get("stage")!="v155_BEST_SINGLE_PAID_ORIGINAL_OPERATOR_UNSEEN_TWO_DOMAIN_VALID":
        raise ValueError("not actual completed v155 source validation")
    source=report.get("unchanged_frozen_original_complete_CB153",{})
    if source.get("whole_source_SHA256")!=CB153_SHA:
        raise ValueError("v155 original physical candidate bytes changed")
    decision=report.get("selection_FROM_prior_v154_eight_reused_prompts_ONLY",{})
    if decision.get("worst")!=SELECTED_KEY or decision.get("best")!=SECONDARY_KEY:
        raise ValueError("operator was posthoc changed after old v154 diagnostic")
    domains=report["one_operator_vs_original_CB59_and_all7"]
    if sorted(domains)!=sorted(("wiki_new_single_op_valid","news_new_single_op_valid")):
        raise ValueError("v155 tested another corpus")
    measurements=report["source_validation_true_NLL_KL_PPL_and_top1"]
    for domain in domains:
        neg=measurements["v154_worst_single"][domain]
        base=measurements["CB59_original"][domain]
        if (neg["mean_TRUE_next_token_NLL"]>=base["mean_TRUE_next_token_NLL"] or
                neg["teacher_to_candidate_KL"]>base["teacher_to_candidate_KL"]):
            raise ValueError("v_proj not genuinely positive on v155")
    return {
        "v_proj_was_worst_old_8_probe_candidate_but_positively_selected_on_v155":True,
        "actual_v155_sha":source["whole_source_SHA256"],
        "preselected_new_v156_operator":SELECTED_KEY,
        "diagnostic_original_hypothesis_operator":SECONDARY_KEY,
        "v155_504_next_tokens_each_of_two_domains":True,
        "new_third_VALID_not_used_for_v_proj_selection":True,
    }
