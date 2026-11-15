"""S13 acceptance audit: TRAIN numerical fit is NOT Russian competence.

Although physical unit/CPU source is valid, independent evidence is
missing and the primary actual hard UPOS TRAIN accuracy REGRESSED.
Always fail CLOSED: no chat/LIVE output effect and no promoted candidate.

This audit does NOT open or re-score S12c already consumed EVAL16 or any
earlier EVAL128. It uses only pre-existing CI's TRAIN metrics and
architecture certification flags. No retuning/model selection occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def require(value,why):
    if not value:
        raise ValueError("NRA3 S13 fail-closed research acceptance "+why)


def audit(report:dict)->dict:
    require(type(report) is dict and
        report.get("stage")==
        "NRA3_S13_REAL_43M_S10_TRAIN72_1266_PAID_FP16_CAUSAL_RELATION_LINGUISTIC_READOUT_SHADOW",
        "wrong artifact or stage")
    require(report.get("S13_real_calibri43M_train72_rows")==72
            and report.get("S13_train_only_words")==408
            and report.get("S13_new_rank8_feedback_unique_parameters")==1266
            and report.get("S13_new_rank8_feedback_physical_bytes")==2581
            and report.get("S13_heldout16_S12c_now_CONSUMED_and_not_loaded") is True
            and report.get("S13_native_all_43_matrices_still_physically_packed_inference_C") is True
            and report.get("no_LIVE_Render_Supabase_APK_Jessica_chat") is True,
            "physical or non-EVAL source regression")
    require(report.get("output_gate")==0 and
            report.get("candidate_accepted") is False and
            report.get("activation_allowed") is False,
            "no candidate may enter active generation")
    head_categories=(
        ("UPOS","S13_original_direct_causal_UPOS_train_accuracy",
         "S13_feedback_direct_causal_UPOS_train_accuracy"),
        ("semantic_roles","S13_original_direct_causal_role_train_accuracy",
         "S13_feedback_direct_causal_role_train_accuracy"),
        ("past_only_syntax_head","S13_original_known_past_head_train_accuracy",
         "S13_feedback_known_past_head_train_accuracy"),
    )
    comparisons={}
    for label,orig,trained in head_categories:
        x=report[orig];y=report[trained]
        require(type(x["correct"]) is int and
                type(y["correct"]) is int and
                x["total"]==y["total"] and
                x["total"]>0 and
                0<=x["correct"]<=x["total"] and
                0<=y["correct"]<=x["total"],
                "accuracy source counts invalid")
        comparisons[label]={
            "baseline_correct":x["correct"],
            "trained_correct":y["correct"],
            "cases":x["total"],
            "delta_correct":y["correct"]-x["correct"],
            "delta_pp":100*(y["correct"]-x["correct"])/x["total"],
        }
    reasons=[]
    if comparisons["UPOS"]["delta_correct"]<0:
        reasons.append("hard_causal_UPOS_accuracy_REGRESSION_on_train72")
    if comparisons["semantic_roles"]["delta_correct"]<4:
        reasons.append("semantic_role_accuracy_weak_or_no_gain_on_train72")
    if comparisons["past_only_syntax_head"]["delta_correct"]<=0:
        reasons.append("no_improvement_in_causal_syntax_heads_on_train72")
    if report.get("S13_any_independent_new_eval_or_full_answer_EOS_measured") is not False:
        raise ValueError("S13 unexpectedly evaluated or modified protected source")
    reasons.extend((
        "independent_NEW_Russian_EVAL_after_S13_lock_not_run",
        "no_active_full_EOS_or_new_616_J7_candidate_certificate",
        "no_end_to_end_original43M_GT_NRA_CPU_at_least90percent_certificate",
        "no_full_model_inclusive_bpp_at_most_point7_certificate",
        "raw_NRA_plus_selector_plus_feedback_bpp_above_point7",
        "old_S12c_EVAL16_consumed_do_not_tune_or_rescore",
    ))
    require(float(report["S13_raw_area_plus_selector_plus_feedback_bpp"])>.7,
            "physical scope/bpp claim requires separate full physical audit")
    return {
        "stage":"NRA3_S13_NO_PROMOTION_RESEARCH_ACCEPTANCE_AUDIT",
        "comparison":comparisons,
        "research_acceptance":False,
        "activation_allowed":False,
        "candidate_accepted":False,
        "output_gate":0,
        "technical_physical_training_CI_success_is_NOT_model_quality_success":True,
        "CE_in_sample_decrease_does_not_override_hard_UPOS_regression":True,
        "candidate_regressed_UPOS":comparisons["UPOS"]["delta_correct"]<0,
        "S12c_consumed_eval16_touched_for_this_audit":False,
        "independent_new_eval_measured":False,
        "all_inclusive_bpp_certified":False,
        "full_43M_GT_NRA_CPU_certified":False,
        "active_616_J7_and_full_answer_EOS_certified":False,
        "reasons":reasons,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--report",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("S13 audit may not overwrite prior locked result")
    raw=args.report.read_bytes()
    r=json.loads(raw)
    result=audit(r)
    result["physical_trainonly_report_sha256"]=hashlib.sha256(raw).hexdigest()
    args.out.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False,
                                   indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
