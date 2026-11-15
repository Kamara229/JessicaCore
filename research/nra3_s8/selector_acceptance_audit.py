"""S8 fail-closed acceptance audit: a green TRAIN pipeline is NOT a good router.

The first S8-v1 real CPU run chose GLOBAL for 63/66 actual words, and
S8b chose it for 66/66 despite falling TRAIN loss and unchanged physical
size. Both cases are **degenerate routing**, not adaptive 3way reasoning.
No output gate is ever allowed to change on this evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BRANCH_NAMES=("global_causal","local_recent_causal",
              "model_predicted_semantic_complementarity")


def audit_selector_training_only(report:dict)->dict:
    if (report.get("hypothesis_count")!=3 or
        report.get("extra_paid_trainable_parameters_added_for_parallel_hypotheses")!=0
        or report.get("output_gate")!=0 or
        report.get("candidate_accepted") is not False or
        report.get("original_unique_NRA3_parameters")!=6_106_154 or
        report.get("CB59_trained_physical_bytes")!=653_798 or
        report.get("original_5bit_code_indices_unchanged") is not True or
        report.get("old_1d_physical_source_unchanged") is not True or
        report.get("old_five_consumed_EVAL128_untouched") is not True or
        report.get("sixth_reserved_S3CI_EVAL128_untouched") is not True or
        report.get("selector_uses_no_inference_gold") is not True):
        raise ValueError("S8 cannot audit nonphysical or activated NRA source")
    observed=report.get("selector_train_prediction_counts")
    teacher=report.get("selector_train_actual_word_target_counts")
    if (not isinstance(observed,dict) or not isinstance(teacher,dict)
        or set(observed)!=set(BRANCH_NAMES)
        or set(teacher)!=set(BRANCH_NAMES)
        or any(type(v) is not int or v<0 for v in observed.values())
        or any(type(v) is not int or v<0 for v in teacher.values())
        or sum(teacher.values())!=sum(observed.values())
        or sum(teacher.values())<30):
        raise ValueError("S8 invalid registered weak TRAIN audit counts")
    reasons=[]
    n=sum(teacher.values())
    if any(observed[branch]<max(2,int(.08*n)) for branch in BRANCH_NAMES):
        reasons.append("collapsed_branch_usage_under_8pct")
    if (report.get("selector_train_accuracy_after_native",0)
            <=report.get("selector_train_accuracy_before_native",1)+.02):
        reasons.append("no_weak_teacher_train_accuracy_improvement_over_2pp")
    if (report.get("selector_train_ce_after_native",10)
            >=report.get("selector_train_ce_before_native",0)):
        reasons.append("weak_teacher_cross_entropy_not_improved")
    if (report.get("selector_real_independent_quality_gain_proven") is not True
            or report.get("independent_heldout_Russian_accuracy_measured") is not True
            or report.get("full_answer_EOS_unseen_generalization_measured") is not True):
        reasons.append("no_new_independent_grammar_semantic_full_EOS_evaluation")
    if (report.get("full_616_J7_candidate_protection_measured") is not True
            or report.get("actual_Calibri_GT_NRA_inclusive_bpp_certified") is not True
            or report.get("real_43M_plus_GT_CPU_speed_ratio_verified") is not True):
        reasons.append("no_end_to_end_J7_bpp_CPU_candidate_certificate")
    # Do NOT count heuristic TRAIN targets as actual semantic accuracy.
    return {
        "stage":"NRA3_S8_NO_PROMOTION_FAIL_CLOSED_PARALLEL_SELECTOR_AUDIT",
        "selector_version":report.get("selector_version",1),
        "research_result_technically_valid_training":True,
        "trained_on_ONLY_12_previously_seen_Russian_sentences":True,
        "weak_teacher_is_NOT_semantic_ground_truth":True,
        "router_branch_predictions":observed,
        "router_weak_training_labels":teacher,
        "total_train_word_labels":n,
        "selector_train_accuracy_before":report.get("selector_train_accuracy_before_native"),
        "selector_train_accuracy_after":report.get("selector_train_accuracy_after_native"),
        "selector_train_ce_before":report.get("selector_train_ce_before_native"),
        "selector_train_ce_after":report.get("selector_train_ce_after_native"),
        "research_acceptance":not reasons,
        "activation_allowed":False,
        "output_gate":0,
        "candidate_accepted":False,
        "reasons":reasons,
        "next_required":"new independent Russian UD TRAIN and untouched heldout; structurally identifiable learnable branch router before any active generation",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--in-report",type=Path,required=True)
    p.add_argument("--out-audit",type=Path,required=True)
    a=p.parse_args()
    if a.out_audit.exists():
        raise FileExistsError("S8 must not overwrite an audit result")
    record=json.loads(a.in_report.read_text(encoding="utf-8"))
    audit=audit_selector_training_only(record)
    if audit["research_acceptance"] or audit["activation_allowed"]:
        raise AssertionError("S8 research with no heldout must never promote")
    a.out_audit.write_text(json.dumps(audit,sort_keys=True,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps(audit,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
