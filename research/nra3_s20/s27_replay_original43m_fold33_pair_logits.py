"""S27: genuine S22d/S26 fold 33 train-only, SHA-verified pair-logit audit.

WHY: physical S22c RAW negative+dative cue is present 8/8, but S24 and S26
predicted REJECTED 0/8 for one leave-construction-out pattern. Need to
discover which directed head, polarity margin, or predicted event gating
fails before inventing yet another adapter. Model forward receives ONLY
frozen original43M+GT word338, NEVER human graph gold/held row metadata.
Human synthetic TRAIN labels are used ONLY for training and offline
inspection; original 616 J7 and all independent EVAL remain untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import (
    PaidSemanticEventGraph,
)
from research.nra3_s19.s19f_real43m_train9_graph_targets import ROLE
from research.nra3_s18.s18c3_raw_russian_grammar_cues import FEATURES
from research.nra3_s19.s19h_true43m_train9_loo_diagnostics import take
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    build_s22_train256,splits,FROZEN_S22_TRAIN256_SHA256,
)
from research.nra3_s20.s23_real_s22d_40fold_failure_audit import (
    exact_real_report_audit,S22D_MODEL_SHA,
)
from research.nra3_s20.s22d_train256_grouped40_true43m_graph import (
    load_physical_training_source,SEED as BASE_SEED,
    new_graph_model as build_base,fixed_train_head as train_base,
    S19D_FULL_PHYSICAL_SHA,
)
from research.nra3_s20.s26_train256_event_grounded_matched40 import (
    SEED as EXPERIMENT_SEED,
    new_graph_model as build_trial, fixed_train_head as train_trial,
)
from research.nra3_s20.s25_real43m_rejected_raw_cue_alignment_audit import cue_audit
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    ROLE_RECIPIENT,PREV_NEG,LIKELY_DATIVE,
)

FOLD=33
HELD_CONSTRUCTION=25
EXPECTED_S26_FULL_FP16_SHA="4819c8b0dfc402bddf036f16e04f85486c51a9a6600af63374cf40a1f6c00951"


def require(ok,reason):
    if not ok:
        raise ValueError("S27 authentic trained fold33: "+reason)


def details(out,pred,arg):
    roles=out.role[pred,arg]
    polar=out.argument_polarity[pred,arg]
    return {
        "role_predicted_class_index":int(roles.argmax()),
        "role_RECIPIENT_probability":float(roles.softmax(-1)[ROLE_RECIPIENT]),
        "polarity_predicted_class_index":int(polar.argmax()),
        "polarity_REJECTED_probability":float(polar.softmax(-1)[2]),
        "role_RECIPIENT_logit":float(roles[ROLE_RECIPIENT]),
        "polarity_REJECTED_logit":float(polar[2]),
        "polarity_NONE_logit":float(polar[0]),
    }


def audit(args):
    raw,valid,gold,src=load_physical_training_source(
        args.features,args.gold,args.manifest)
    require(src["S22_sha_locked_synthetic_TRAIN256_graph_gold"]==
            FROZEN_S22_TRAIN256_SHA256,"source S22 training SHA drift")
    base_info=json.loads(args.baseline_report.read_text())
    trial_info=json.loads(args.trial_report.read_text())
    base_sha=hashlib.sha256(args.baseline_model.read_bytes()).hexdigest()
    trial_sha=hashlib.sha256(args.trial_model.read_bytes()).hexdigest()
    exact_real_report_audit(base_info,base_sha)
    require(base_sha==S22D_MODEL_SHA
            and trial_sha==EXPECTED_S26_FULL_FP16_SHA==
            trial_info["physical_FP16_candidate_SHA256"]
            and trial_info["baseline_same_split_genuine_S22d_model_SHA"]==S22D_MODEL_SHA
            and trial_info["S25_genuine_real_CPU_preposed_8_negative_dative_confirmed"]
            and trial_info["output_gate"]==0
            and not trial_info["candidate_accepted"]
            and not trial_info["activation_allowed"],
            "wrong genuine S26 40fold physical source or output gate")
    cues=cue_audit(raw,valid,gold)["preposed_source_rejected_8"]
    require(cues["negation_and_dative_both"]==8
            and cues["recipient_precedes_predicate"]==8,
            "actual 43M+GT source missing negation before fronted recipient")
    require(BASE_SEED==EXPERIMENT_SEED and FOLD==8+HELD_CONSTRUCTION,
            "matched baseline+trial folds diverge")
    axis,group,trids,heldids=splits()[FOLD]
    require(axis=="CONSTRUCTION" and group==HELD_CONSTRUCTION
            and len(heldids)==8 and len(trids)==248
            and all(int(i)%32==HELD_CONSTRUCTION for i in heldids),
            "protected fronted recipient construction fold geometry")
    require(args.parent.is_file() and hashlib.sha256(args.parent.read_bytes()).hexdigest()==
            S19D_FULL_PHYSICAL_SHA,
            "frozen genuine original 43M+GT S19d 9332 parent SHA drift")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frozen=PaidContextualEventScope.from_physical(args.parent.read_bytes())
    trraw,trvalid,trgold=(take(raw,trids),take(valid,trids),take(gold,trids))
    dvraw,dvvalid=(take(raw,heldids),take(valid,heldids))
    old,pold=train_base(build_base(frozen,BASE_SEED+FOLD),trraw,trvalid,trgold)
    new,pnew=train_trial(build_trial(frozen,EXPERIMENT_SEED+FOLD),trraw,trvalid,trgold)
    shaold=hashlib.sha256(pold).hexdigest()
    shanew=hashlib.sha256(pnew).hexdigest()
    expected_old=base_info["all40_fixed_TRAIN_internal_folds"][FOLD]["fold_paid_FP16_SHA256"]
    expected_new=trial_info["all40_fixed_TRAIN_internal_folds"][FOLD]["fold_paid_FP16_SHA256"]
    print("S27_BASELINE_FOLD33_SHA_REPLAY",shaold,
          "ORIGINAL_ACTUAL_RUN",expected_old,flush=True)
    print("S27_S26_FOLD33_SHA_REPLAY",shanew,
          "ORIGINAL_ACTUAL_RUN",expected_new,flush=True)
    require(shaold==expected_old,
            "original S22d fold 33 physical FP16 mismatch; no trusted pair-logit inference")
    require(shanew==expected_new,
            "S26 matched fold 33 physical FP16 mismatch; no trusted pair-logit inference")
    with torch.inference_mode():
        old_out=old(dvraw,dvvalid)
        new_out=new(dvraw,dvvalid)
        # Reference for exact neural residual difference only: call the
        # inherited base forward directly, on same real raw inputs.
        no_residual=PaidSemanticEventGraph.forward(new,dvraw,dvvalid)
        source_event_prob=1.-no_residual.event_status.softmax(-1)[...,0]
    graphs=build_s22_train256()
    records=[]
    for local,row_id in enumerate(heldids):
        g=graphs[row_id]
        rejected=[a for a in g.arguments if a.polarity=="REJECTED"]
        require(len(rejected)==1 and rejected[0].role=="RECIPIENT",
                "held construction should have one rejected recipient")
        a=rejected[0]
        predicate=next(e.predicate-1 for e in g.events if e.event_id==a.event_id)
        j=a.token-1
        require(j<predicate and int(gold["argument_polarity_TRAIN_ONLY"][row_id,predicate,j])==1
                and int(gold["predicate_argument_role_TRAIN_ONLY"][row_id,predicate,j])==
                ROLE.index("RECIPIENT"),"actual held REJECTED gold indexing drift")
        require(dvraw[local,j,320+PREV_NEG]==1.
                and dvraw[local,j,320+LIKELY_DATIVE]==1.,
                "genuine current 43M+GT negative+dative missing")
        records.append({
            "source_row_index_for_offline_error_only":row_id,
            "raw_text_SHA256":hashlib.sha256(g.text.encode()).hexdigest(),
            "TRAIN_gold_directed_predicate_index":predicate,
            "TRAIN_gold_rejected_recipient_index":j,
            "original_model_predicted_event_prob_at_true_verb":float(source_event_prob[local,predicate]),
            "baseline_S22d_pred_to_recipient":details_one(old_out,local,predicate,j),
            "trial_S26_pred_to_recipient":details_one(new_out,local,predicate,j),
            "trial_S26_arg_to_pred_INCORRECT_reverse":details_one(new_out,local,j,predicate),
            "trial_S26_without_scoped_residual":details_one(no_residual,local,predicate,j),
            "actual_S26_delta_recipient_role_logit":float(
                new_out.role[local,predicate,j,ROLE_RECIPIENT]-
                no_residual.role[local,predicate,j,ROLE_RECIPIENT]),
            "actual_S26_delta_REJECTED_polarity_logit":float(
                new_out.argument_polarity[local,predicate,j,2]-
                no_residual.argument_polarity[local,predicate,j,2]),
        })
    require(len(records)==8,"not all eight preposed TRAIN-internal cases verified")
    before=sum(x["baseline_S22d_pred_to_recipient"]["polarity_predicted_class_index"]==2
               for x in records)
    after=sum(x["trial_S26_pred_to_recipient"]["polarity_predicted_class_index"]==2
              for x in records)
    require(before==base_info["all40_fixed_TRAIN_internal_folds"][FOLD][
                "REJECTED_specific_held_TRAIN"]["TP"]
            and after==trial_info["all40_fixed_TRAIN_internal_folds"][FOLD][
                "REJECTED_specific_held_TRAIN"]["TP"],
            "replayed precise original physical fold disagrees with training report")
    return {
        "stage":"NRA3_S27_GENUINE_ORIGINAL43M_GT_MATCHED_FOLD33_REJECTED_LOGIT_DIAGNOSTIC",
        "model_forward_input":"ONLY original SHA-locked 43M+GT 338D raw embedding+surface bits",
        "original_S22d_full_FP16_SHA256":S22D_MODEL_SHA,
        "original_S26_full_FP16_SHA256":EXPECTED_S26_FULL_FP16_SHA,
        "replayed_exact_real_fold33_S22d_physical_sha256":shaold,
        "replayed_exact_real_fold33_S26_physical_sha256":shanew,
        "S26_3_trained_FP16_event_scoped_head_weights":[float(x) for x in
             new.event_scoped_recipient_weights.detach().cpu()],
        "preposed_polarity_TP_baseline":before,
        "preposed_polarity_TP_trial":after,
        "per_example_error_evidence":records,
        "held_gold_never_passed_to_neural_forward":True,
        "independent_EVAL_not_opened_or_replayed":True,
        "no_active_model_J7_bpp_CPU_certification":True,
        "no_production_edits":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }


def details_one(out,local,p,j):
    return details_pair(out.role[local,p,j],out.argument_polarity[local,p,j])


def details_pair(roles,polarity):
    return {
        "role_predicted_class_index":int(roles.argmax()),
        "role_RECIPIENT_probability":float(roles.softmax(-1)[ROLE_RECIPIENT]),
        "polarity_predicted_class_index":int(polarity.argmax()),
        "polarity_REJECTED_probability":float(polarity.softmax(-1)[2]),
        "role_RECIPIENT_logit":float(roles[ROLE_RECIPIENT]),
        "polarity_REJECTED_logit":float(polarity[2]),
        "polarity_NONE_logit":float(polarity[0]),
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("features","gold","manifest","parent","baseline-model","baseline-report",
                 "trial-model","trial-report","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    require(not args.out.exists(),"never overwrite source or prior audit")
    report=audit(args)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"NRA3_S27_ORIGINAL43M_GT_MATCHED_PREPOSED_FOLD33_S22D_VS_S26.json").write_text(
        json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("S27_PREPOSED_PAIR_LOGIT_DIAG",
          json.dumps([{
              "row":x["source_row_index_for_offline_error_only"],
              "event_p":round(x["original_model_predicted_event_prob_at_true_verb"],3),
              "base_recipient_p":round(x["baseline_S22d_pred_to_recipient"]["role_RECIPIENT_probability"],3),
              "S26_recipient_p":round(x["trial_S26_pred_to_recipient"]["role_RECIPIENT_probability"],3),
              "S26_rejected_p":round(x["trial_S26_pred_to_recipient"]["polarity_REJECTED_probability"],3),
              "S26_REJECTED_delta_logit":round(x["actual_S26_delta_REJECTED_polarity_logit"],3),
          } for x in report["per_example_error_evidence"]],ensure_ascii=False),flush=True)
    print("S27_VERIFIED_S26_FOLD33_TRAINED_LOCAL_WEIGHTS",
          json.dumps(report["S26_3_trained_FP16_event_scoped_head_weights"]),flush=True)
    print("S27_PREPOSED_PHYSICAL_FP16_FOLD_SHA_VERIFIED",flush=True)


if __name__=="__main__":
    main()
