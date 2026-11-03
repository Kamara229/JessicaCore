#!/usr/bin/env python3
"""CALIBRI v80: source-only TRAIN/VALID root-cause audit for rejected v79 QA adaptation.

Inspect immutable prior CI TRAIN/VALID audits, NO TEST labels, model weights,
fitted features or checkpoint selection. Verify paid source physical budget,
per-task accuracy and conditional answer CE, per-corpus KL, preregistered
safety gate decisions, run-to-run source reproducibility and fit-overfit
signs. This forensic diagnostic performs NO NEW OPTIMIZATION and makes no
claims of real TEST semantic success. Research-only, no production changes.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

STEPS=(0,5,10,18)
QA_TASKS=("sciq","boolq")
LANG_CORPORA=("wiki_valid","news_valid")
REQUIRED_COMPLETE_BYTES=11725487
UNIQUE_SOURCE=134515008
CE_TOL=1e-5
ALLOWED_MAX_KL_REGRESSION=0.25
MIN_VALID_GAIN=0.05

def get(path):
    result=json.loads(path.read_text())
    if not isinstance(result,dict):
        raise AssertionError(f"not a JSON object: {path}")
    return result

def infer(check,base):
    scores=check["TRAIN_VALID_QA"]
    KL=check["TRAIN_VALID_teacher_KL"]
    gains={k:float(scores[k]["accuracy"])-float(base["TRAIN_VALID_baseline_QA"][k]["accuracy"])
           for k in QA_TASKS}
    KLdelta={k:float(KL[k])-float(base["TRAIN_VALID_baseline_KL"][k])
             for k in LANG_CORPORA}
    cond_qa={k:float(scores[k]["conditioned_answer_CE"])-
              float(base["TRAIN_VALID_baseline_QA"][k]["conditioned_answer_CE"])
              for k in QA_TASKS}
    nonregress=all(z>=-1e-6 for z in gains.values())
    largegain=any(z>=MIN_VALID_GAIN-1e-6 for z in gains.values())
    stable=all(z<=ALLOWED_MAX_KL_REGRESSION+1e-6 for z in KLdelta.values())
    eligible=check["step"]>0 and nonregress and largegain and stable
    if eligible!=check["predeclared_promotion_eligible"]:
        raise AssertionError(f"preregistered eligibility mismatch at step {check['step']}")
    if nonregress!=check["valid_non_degrading_both_tasks"] or largegain!=check["valid_positive_at_least_one_5pp"] or stable!=check["valid_lang_no_0p25nat_regression_each"]:
        raise AssertionError("claimed gates inconsistent with actual TRAIN VALID measured metrics")
    reasons=[]
    if check["step"]==0:reasons.append("UNCHANGED_PARENT_CONTROL")
    if not nonregress:reasons.append("QA_REGRESSION")
    if not largegain:reasons.append("NO_QA_GAIN_AT_LEAST_5PP")
    if not stable:reasons.append("LANGUAGE_KL_REGRESSION_OVER_0P25")
    return {
        "step":check["step"],
        "QA_correct":{k:int(scores[k]["correct"]) for k in QA_TASKS},
        "QA_count":{k:int(scores[k]["total"]) for k in QA_TASKS},
        "VALID_accuracy":{k:float(scores[k]["accuracy"]) for k in QA_TASKS},
        "VALID_qa_accuracy_delta_vs_parent":gains,
        "VALID_answer_choice_CE_delta_vs_parent":cond_qa,
        "VALID_language_KL":{k:float(KL[k]) for k in LANG_CORPORA},
        "VALID_language_KL_delta_vs_parent":KLdelta,
        "failed_gates":reasons,
        "eligible":eligible,
        "sum_accuracy_gain":sum(gains.values()),
        "better_conditional_CE_but_not_better_QA":{
           k:cond_qa[k]<-CE_TOL and gains[k]<=1e-6 for k in QA_TASKS}
    }

def audit(folder,out):
    first=get(folder/"v79_train_first.json")
    second=get(folder/"v79_train_second.json")
    freeze=get(folder/"v79_FROZEN_before_TEST.json")
    for obj in (first,second):
        if int(obj["physical_trained_source_bytes"])!=REQUIRED_COMPLETE_BYTES:
            raise AssertionError("missing paid source or uncounted bytes")
        if abs(float(obj["all_in_bpw"])-8*REQUIRED_COMPLETE_BYTES/UNIQUE_SOURCE)>1e-12:
            raise AssertionError("incorrect all-inclusive physical bpw accounting")
        if obj["candidate_training_steps"]!=list(STEPS):
            raise AssertionError("experimental checkpoint set drifted")
        if not obj["TEST_QA_label_and_TEST_language_never_opened"]:
            raise AssertionError("TRAIN report claims TEST information was opened")
        if int(obj["TRAIN_gold_questions"])!=72 or int(obj["VALID_gold_questions"])!=28:
            raise AssertionError("source TRAIN/VALID task counts drifted")
    if first["TRAIN_VALID_baseline_QA"]!=second["TRAIN_VALID_baseline_QA"]:
        raise AssertionError("same-runner TRAIN VALID baseline not repeatable")
    if first["TRAIN_VALID_baseline_KL"]!=second["TRAIN_VALID_baseline_KL"]:
        raise AssertionError("same-runner TRAIN language validation not repeatable")
    if freeze["frozen_first_TRAIN_winner"]!=first["chosen_step_TRAIN_only"]:
        raise AssertionError("frozen winner inconsistent with audited TRAIN record")
    if freeze["second_TRAIN_winner"]!=second["chosen_step_TRAIN_only"]:
        raise AssertionError("second winner inconsistent with audited TRAIN record")
    if freeze["frozen_candidate_source_SHA"]!=first["chosen_source_SHA"]:
        raise AssertionError("frozen physical source SHA inconsistent")
    if freeze["same_host_bitwise_reproducibility"]["first_SHA"]!=first["chosen_source_SHA"] or freeze["same_host_bitwise_reproducibility"]["second_SHA"]!=second["chosen_source_SHA"]:
        raise AssertionError("source SHA mismatch with v79 reproducibility audit")
    if freeze["source_heldout_TEST_unseen_before_frozen"] is not True:
        raise AssertionError("prior freeze stage did not respect TEST boundary")
    tables={}
    for name,data in (("first",first),("second",second)):
        recs=[infer(x,data) for x in data["candidate_TRAIN_VALID_fixed_promotion_gates"]]
        if [r["step"] for r in recs]!=list(STEPS):
            raise AssertionError("missing optimizer checkpoints")
        actual_winner=max(
            (x for x in recs if x["eligible"]),
            key=lambda x:(x["sum_accuracy_gain"],-x["step"]),default=None)
        selected=actual_winner["step"] if actual_winner else 0
        if selected!=data["chosen_step_TRAIN_only"]:
            raise AssertionError("source selection diverges from original frozen rule")
        tables[name]=recs
    comparable=[]
    for rowa,rowb in zip(tables["first"],tables["second"]):
        if rowa["step"]!=rowb["step"]:raise AssertionError("step misalignment")
        comparable.append({
            "step":rowa["step"],
            "same_VALID_task_accuracy":rowa["VALID_accuracy"]==rowb["VALID_accuracy"],
            "max_abs_language_KL_between_same_host_runs":max(
                abs(rowa["VALID_language_KL"][k]-rowb["VALID_language_KL"][k])
                for k in LANG_CORPORA)
        })
    summary={
      "experiment":"CALIBRI_v80_POSTMORTEM_v79_QA_train_valid_only",
      "source_pr":209,"source_CI_run":41550969128,
      "NO_TEST_data_or_labels_opened_by_v80":True,
      "NO_source_retraining_or_checkpoint_choice_by_v80":True,
      "full_paid_real_source_bytes":REQUIRED_COMPLETE_BYTES,
      "all_in_original_unique_parameter_bpw":8*REQUIRED_COMPLETE_BYTES/UNIQUE_SOURCE,
      "source_count_original_unique_parameters":UNIQUE_SOURCE,
      "TRAIN_FIT_gold_questions":72,"TRAIN_VALID_gold_questions":28,
      "freeze_original_parent_sha":freeze["parent_pretrained_full_source_SHA"],
      "frozen_complete_source_sha":freeze["frozen_candidate_source_SHA"],
      "original_preregistered_train_winner_step":first["chosen_step_TRAIN_only"],
      "selected_changed_original_paid_model":first["did_QA_supervision_change_saved_source"],
      "QA_TRAIN_VALID_parent_scores":first["TRAIN_VALID_baseline_QA"],
      "language_TRAIN_VALID_parent_KL":first["TRAIN_VALID_baseline_KL"],
      "all_checkpoint_eligibility":tables,
      "same_host_per_checkpoint_comparisons":comparable,
      "same_host_bitwise_frozen_source_equal":freeze["same_host_bitwise_reproducibility"]["same_runner_identical_source_SHA"],
      "viable_updated_source_candidates":sum(int(z["eligible"]) for z in tables["first"]),
      "not_a_new_test_of_original_model_answers":True,
      "negative_results_must_remain_visible":True
    }
    if summary["original_preregistered_train_winner_step"]==0 and summary["viable_updated_source_candidates"]!=0:
        raise AssertionError("rejected despite eligible candidate")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"CALIBRI_v80_v79_TRAIN_VALID_failures_explained",
      "source_full_bpw":round(summary["all_in_original_unique_parameter_bpw"],9),
      "frozen_original_training_step":summary["original_preregistered_train_winner_step"],
      "eligible_post0_candidates":summary["viable_updated_source_candidates"],
      "checkpoints":[{
        "step":r["step"],"QA_deltas":r["VALID_qa_accuracy_delta_vs_parent"],
        "KL_deltas":r["VALID_language_KL_delta_vs_parent"],
        "failed_gates":r["failed_gates"]}
        for r in tables["first"]],
      "same_runner_SHA":summary["same_host_bitwise_frozen_source_equal"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=Path("input"))
    ap.add_argument("--out",type=Path,default=Path("results/v80_original_TRAIN_VALID_failure_audit.json"))
    args=ap.parse_args()
    audit(args.input,args.out)
