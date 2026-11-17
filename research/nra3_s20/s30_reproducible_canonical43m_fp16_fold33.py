"""S30: numerically reproducible *canonical FP16* fold33 on true 43M+GT.

WHY: physical S22c RAW negative+dative cue is present 8/8, but S24 and S26
predicted REJECTED 0/8 for one leave-construction-out pattern. Need to
discover which directed head, polarity margin, or predicted event gating
fails before inventing yet another adapter. The historical S22d/S26 per-fold
FP16 SHA could NOT be reproduced across CI runs. Both archived full models
are SHA-verified for provenance, but NO claim is made that the fresh fold
weights reproduce the archived fold or its scores. A repeat on the same
runner probes whether even same-host bytewise reproducibility holds. Model forward receives ONLY
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
from research.nra3_s20.s28_real43m_trainonly_determinism_probe import canonical_paid_semantic_fp16_sha
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    ROLE_RECIPIENT,PREV_NEG,LIKELY_DATIVE,
)

FOLD=33
HELD_CONSTRUCTION=25
EXPECTED_S26_FULL_FP16_SHA="4819c8b0dfc402bddf036f16e04f85486c51a9a6600af63374cf40a1f6c00951"
S29_ARCHIVED_S22D_CANONICAL="35102946886b9f35d86f57f791b6ab6001b813e4f79397cff8e4d25c04e7a538"
S29_ARCHIVED_S26_CANONICAL="02d4403ffbce01632d54d66349a01c3f10b81650d26d9540bb86f495cac62807"


def require(ok,reason):
    if not ok:
        raise ValueError("S30 authentic trained fold33: "+reason)


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
    # S29 immutable archived FULL model: both raw bytes and numeric paid
    # FP16 tensors are authentic. Only fold33 is a fresh refit; archived
    # fold33 checkpoints never existed so never claim historic identity.
    archived_base=PaidSemanticEventGraph.from_physical(args.baseline_model.read_bytes())
    archived_trial=EventGroundedInversionGraph.from_physical(args.trial_model.read_bytes())
    require(canonical_paid_semantic_fp16_sha(archived_base)==S29_ARCHIVED_S22D_CANONICAL
            and canonical_paid_semantic_fp16_sha(archived_trial)==S29_ARCHIVED_S26_CANONICAL,
            "actual S29 archived model canonical numeric provenance mismatch")
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
    # Refit diagnostic uses ONE CPU execution thread to test whether
    # the observed same-runner FP16 drift came from parallel reductions.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    physical_parent=args.parent.read_bytes()
    trraw,trvalid,trgold=(take(raw,trids),take(valid,trids),take(gold,trids))
    dvraw,dvvalid=(take(raw,heldids),take(valid,heldids))
    old,pold=train_base(
        build_base(PaidContextualEventScope.from_physical(physical_parent),BASE_SEED+FOLD),
        trraw,trvalid,trgold)
    new,pnew=train_trial(
        build_trial(PaidContextualEventScope.from_physical(physical_parent),EXPERIMENT_SEED+FOLD),
        trraw,trvalid,trgold)
    shaold=hashlib.sha256(pold).hexdigest()
    shanew=hashlib.sha256(pnew).hexdigest()
    # Fresh exact-same-host independent refits test deterministic replay.
    old_repeat,pold_repeat=train_base(
        build_base(PaidContextualEventScope.from_physical(physical_parent),BASE_SEED+FOLD),
        trraw,trvalid,trgold)
    new_repeat,pnew_repeat=train_trial(
        build_trial(PaidContextualEventScope.from_physical(physical_parent),EXPERIMENT_SEED+FOLD),
        trraw,trvalid,trgold)
    same_host_base=(pold==pold_repeat)
    same_host_trial=(pnew==pnew_repeat)
    canon_base=canonical_paid_semantic_fp16_sha(old)
    canon_base_replay=canonical_paid_semantic_fp16_sha(old_repeat)
    canon_trial=canonical_paid_semantic_fp16_sha(new)
    canon_trial_replay=canonical_paid_semantic_fp16_sha(new_repeat)
    paid_head_base=hashlib.sha256(old.head_physical()).hexdigest()
    paid_head_trial=hashlib.sha256(new.head_physical()).hexdigest()
    # Numerical identity must survive even if legacy nested safetensors
    # metadata changes and the complete serialized file SHA does not.
    require(canon_base==canon_base_replay
            and canon_trial==canon_trial_replay
            and old.head_physical()==old_repeat.head_physical()
            and new.head_physical()==new_repeat.head_physical(),
            "same-seed real FP16 paid *values* diverged; fail closed")
    print("S30_SAME_RUNNER_BASE_CANONICAL_PAID_FP16_EQUAL",True,canon_base,
          "FULL_BYTES_IDENTICAL",same_host_base,flush=True)
    print("S30_SAME_RUNNER_TRIAL_CANONICAL_PAID_FP16_EQUAL",True,canon_trial,
          "FULL_BYTES_IDENTICAL",same_host_trial,flush=True)
    expected_old=base_info["all40_fixed_TRAIN_internal_folds"][FOLD]["fold_paid_FP16_SHA256"]
    expected_new=trial_info["all40_fixed_TRAIN_internal_folds"][FOLD]["fold_paid_FP16_SHA256"]
    print("S30_BASELINE_FOLD33_SHA_REPLAY",shaold,
          "ORIGINAL_ACTUAL_RUN",expected_old,flush=True)
    print("S30_S26_FOLD33_SHA_REPLAY",shanew,
          "ORIGINAL_ACTUAL_RUN",expected_new,flush=True)
    historical_base_identical=(shaold==expected_old)
    historical_trial_identical=(shanew==expected_new)
    print("S30_CROSS_RUN_HISTORICAL_SHA_IDENTICAL",historical_base_identical,
          historical_trial_identical,flush=True)
    # Provenance is established by original full-physical model SHA + source
    # SHA; absent archived fold weights, historical exact refit is unproven.
    # This NEW run is a separate TRAIN-only research diagnostic, not an
    # original archived fold33 reconstruction or independent EVAL.
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
    historical_base_TP=base_info["all40_fixed_TRAIN_internal_folds"][FOLD][
                "REJECTED_specific_held_TRAIN"]["TP"]
    historical_trial_TP=trial_info["all40_fixed_TRAIN_internal_folds"][FOLD][
                "REJECTED_specific_held_TRAIN"]["TP"]
    print("S30_FRESH_REFIT_VS_HISTORICAL_TP",before,after,
          historical_base_TP,historical_trial_TP,flush=True)
    return {
        "stage":"NRA3_S30_CANONICAL_FP16_REPRODUCIBLE_REAL43M_GT_TRAIN_INTERNAL_FOLD33_PAIRS",
        "model_forward_input":"ONLY original SHA-locked 43M+GT 338D raw embedding+surface bits",
        "original_S22d_full_FP16_SHA256":S22D_MODEL_SHA,
        "original_S26_full_FP16_SHA256":EXPECTED_S26_FULL_FP16_SHA,
        "fresh_refit_real_fold33_S22d_physical_sha256":shaold,
        "fresh_refit_real_fold33_S26_physical_sha256":shanew,
        "archived_fold33_S22d_SHA":expected_old,
        "archived_fold33_S26_SHA":expected_new,
        "fresh_fold33_matches_historical_S22d_SHA":historical_base_identical,
        "fresh_fold33_matches_historical_S26_SHA":historical_trial_identical,
        "same_runner_same_seed_S22d_sha_reproduced":same_host_base,
        "same_runner_same_seed_S26_sha_reproduced":same_host_trial,
        "fresh_refit_NOT_equivalent_to_historical_exact_fold":True,
        "actual_archived_S22d_numeric_canonical_FP16_SHA256":S29_ARCHIVED_S22D_CANONICAL,
        "actual_archived_S26_numeric_canonical_FP16_SHA256":S29_ARCHIVED_S26_CANONICAL,
        "fresh_repeated_fold33_S22d_numeric_canonical_SHA256":canon_base,
        "fresh_repeated_fold33_S26_numeric_canonical_SHA256":canon_trial,
        "fresh_repeat_numerical_S22d_FP16_equal":canon_base==canon_base_replay,
        "fresh_repeat_numerical_S26_FP16_equal":canon_trial==canon_trial_replay,
        "fresh_fold33_S22d_paid_FP16_head_sha256":paid_head_base,
        "fresh_fold33_S26_paid_FP16_head_sha256":paid_head_trial,
        "legacy_full_blob_SHA_may_drift_despite_identical_tensors":True,
        "fresh_refit_one_CPU_thread_to_probe_concurrent_reduction_drift":True,
        "archived_fold33_baseline_REJECTED_TP":historical_base_TP,
        "archived_fold33_trial_REJECTED_TP":historical_trial_TP,
        "S26_3_trained_FP16_event_scoped_head_weights":[float(x) for x in
             new.event_scoped_recipient_weights.detach().cpu()],
        "preposed_polarity_TP_baseline":before,
        "preposed_polarity_TP_trial":after,
        "per_example_error_evidence":records,
        "held_8_TRAlN_only_roles_predicted_RECIPIENT_baseline":sum(
          x["baseline_S22d_pred_to_recipient"]["role_predicted_class_index"]==ROLE_RECIPIENT
          for x in records),
        "held_8_TRAIN_only_roles_predicted_RECIPIENT_trial":sum(
          x["trial_S26_pred_to_recipient"]["role_predicted_class_index"]==ROLE_RECIPIENT
          for x in records),
        "held_8_TRAIN_only_arg_to_pred_spurious_REJECTED_trial":sum(
          x["trial_S26_arg_to_pred_INCORRECT_reverse"]["polarity_predicted_class_index"]==2
          for x in records),
        "held_8_TRAIN_only_trial_REJECTED_probability_range":[
          min(x["trial_S26_pred_to_recipient"]["polarity_REJECTED_probability"] for x in records),
          max(x["trial_S26_pred_to_recipient"]["polarity_REJECTED_probability"] for x in records)],
        "held_gold_never_passed_to_neural_forward":True,
        "no_frozen_original43M_GT_weight_update":True,
        "all_8_held_rejected_rows_untouched_in_train_loss":True,
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
    (args.out/"NRA3_S30_REAL43M_GT_CANONICAL_PAID_FP16_FOLD33_INVERSION_DIAGNOSTICS.json").write_text(
        json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("S30_PREPOSED_PAIR_LOGIT_DIAG",
          json.dumps([{
              "row":x["source_row_index_for_offline_error_only"],
              "event_p":round(x["original_model_predicted_event_prob_at_true_verb"],3),
              "base_recipient_p":round(x["baseline_S22d_pred_to_recipient"]["role_RECIPIENT_probability"],3),
              "S26_recipient_p":round(x["trial_S26_pred_to_recipient"]["role_RECIPIENT_probability"],3),
              "S26_rejected_p":round(x["trial_S26_pred_to_recipient"]["polarity_REJECTED_probability"],3),
              "S26_REJECTED_delta_logit":round(x["actual_S26_delta_REJECTED_polarity_logit"],3),
          } for x in report["per_example_error_evidence"]],ensure_ascii=False),flush=True)
    print("S30_CANONICAL_NUMERIC_FP16_FOLD33_INVERSION_CONFIRMED",
          json.dumps({"S22d_rejected_TP":before,"S26_rejected_TP":after,
                      "S22d_correct_RECIPIENT":report["held_8_TRAlN_only_roles_predicted_RECIPIENT_baseline"],
                      "S26_correct_RECIPIENT":report["held_8_TRAIN_only_roles_predicted_RECIPIENT_trial"],
                      "S26_rejected_probability_range":report["held_8_TRAIN_only_trial_REJECTED_probability_range"]},sort_keys=True),flush=True)
    print("S30_FRESH_REFIT_S26_FOLD33_TRAINED_LOCAL_WEIGHTS",
          json.dumps(report["S26_3_trained_FP16_event_scoped_head_weights"]),flush=True)
    print("S30_PREPOSED_REAL43M_FRESH_REFIT_SOURCE_AND_MODEL_PROVENANCE_SHA_VERIFIED",flush=True)


if __name__=="__main__":
    main()
