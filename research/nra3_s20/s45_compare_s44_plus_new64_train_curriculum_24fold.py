"""S45 physical CPU S44 vs S44+NEW64 scope curriculum, 24 matched folds.

S45 changes ONE planned dimension: append separately precommitted 64
TRAIN-only Russian event+negation/quote rows to every original S43
TRAIN128 grouped fold. Refit identical S44 shadow grammar, same seed,
120 AdamW steps, LR .012, loss weights, architecture, inference and
threshold. The old S43 held 16/8 rows remain excluded from the original
TRAIN128 training side; NEW64 are extra TRAIN examples only.

This is a developmental diagnostic on already-inspected synthetic
held S43 construction groups. Do not use S38 adversarial 32 or claim
real Calibri43M+GreenTunnel/NRA2 training, independent language
generalization, 616J7 preservation, bpp or CB59.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import model_weights_digest
from research.nra3_s20.s43_new128_multi_event_trainonly_source import (
    build_train128,train_partitions,train_source_sha,validate_source as validate_old,
    PATTERNS,
)
from research.nra3_s20.s43_train128_joint_predicate_source_24fold import (
    score_after_frozen_training,summation,
)
from research.nra3_s20.s44_train128_quote_calibrated_source_contrast_24fold import (
    fit_s44,S44_TRAIN_STEPS,S44_TRAIN_LR,S44_STAGE,
)
from research.nra3_s20.s45_separately_frozen_new64_scope_curriculum import (
    train64,source_digest,validate_source as validate_new,STAGE as CURRICULUM_STAGE,
)

STAGE="NRA3_S45_DISJOINT64_ADDITIONAL_TRAIN_CURRICULUM_PAIRED_S44_24FOLD"
PREVIOUS_SUCCESSFUL_S44_RUN=44249892263
TRAIN64_BLOB_SHA="3cc3ed61a9798e78dffbbad0692dd6f8789c59b8"
NEW64_FROZEN_COMMIT="8f1e139c4a8c433b1545aebcfb03cdf00dea426a"

def require(ok,message):
    if not bool(ok):
        raise ValueError("S45 only TRAIN scope curriculum graph: "+message)

def micro_summary(records,key,axis):
    folds=[r[key] for r in records if r["axis"]==axis]
    def stats(prefix):
        tp=sum(x[prefix+"_TP"] for x in folds)
        fp=sum(x[prefix+"_FP"] for x in folds)
        fn=sum(x[prefix+"_FN"] for x in folds)
        return {"TP":tp,"FP":fp,"FN":fn,
                "precision":tp/(tp+fp) if tp+fp else 0.0,
                "recall":tp/(tp+fn) if tp+fn else 0.0,
                "F1":2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0}
    return {
        "event_head":stats("event_source"),
        "REJECTED_links":stats("REJECTED_link"),
        "quote":stats("quote"),
        "multi_recipient_held_rows":sum(
           x["held_rows_requiring_one_source_to_multiple_recipient_links"]
           for x in folds),
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--previous-s44-report",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),"do not overwrite research reports")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    original=validate_old()
    extra=validate_new()
    require(original["count"]==128 and original["grouped_folds"]==24
            and original["source_sha256"]==train_source_sha(),
            "original S43 NEW128 source provenance drift")
    require(extra["stage"]==CURRICULUM_STAGE
            and extra["TRAIN_only_rows"]==64
            and extra["source_sha256"]==source_digest()
            and extra["disjoint_S43_TRAIN128_S39_TRAIN80_S22_TRAIN256"]
            and extra["no_S38_examples_loaded"],
            "separately precommitted source NEW64 provenance invalid")
    prior=json.loads(args.previous_s44_report.read_text(encoding="utf-8"))
    require(prior["stage"]==S44_STAGE
            and prior["previous_true_S43_completed_run_for_provenance"]==44247621131
            and prior["same_source_new128_SHA256"]==original["source_sha256"]
            and len(prior["all_24_ALREADY_INSPECTED_synthetic_TRAIN_development_folds"])==24
            and prior["output_gate"]==0 and not prior["candidate_accepted"],
            "physically completed S44 24fold report source not validated")
    prior_folds={z["fold"]:z for z in
                 prior["all_24_ALREADY_INSPECTED_synthetic_TRAIN_development_folds"]}
    require(set(prior_folds)==set(range(24)),"missing archived original S44 folds")
    rows=build_train128()
    curriculum=train64()
    require(len(curriculum)==64 and len(rows)==128,"TRAIN geometry changed")
    require(not {r.text for r in rows}&{x.text for x in curriculum},
            "NEW64 text leaks from original S43 NEW128")
    results=[]
    for fold,(axis,group,tr,held) in enumerate(train_partitions()):
        require((len(tr),len(held))==(
                  (112,16) if axis=="LEXICAL" else (120,8))
                and set(tr).isdisjoint(held)
                and set(tr)|set(held)==set(range(128)),
                "original 24 frozen TRAIN128 grouped folds changed")
        archived=prior_folds[fold]
        require(archived["axis"]==axis and archived["held_group_OFFLINE_ONLY"]==group
                and archived["TRAIN_rows"]==len(tr)
                and archived["held_TRAIN_internal_rows"]==len(held),
                "S44 original held TEST rows differ from S45 paired folds")
        base_train=[rows[i] for i in tr]
        extension_train=[*base_train,*curriculum]
        held_dev=[rows[i] for i in held]
        require(len(extension_train)-len(base_train)==64
                and not {z.text for z in held_dev}&
                        {z.text for z in extension_train},
                "S45 curriculum overlaps original grouped held text")
        # Baseline S44: exact prior original model architecture/loss.
        old,old_train_loss=fit_s44(base_train,fold)
        # One intended contrast: fixed SAME S44 fitting, more TRAIN rows.
        new,new_train_loss=fit_s44(
            extension_train,fold,allow_S45_new64=True)
        sha_old=model_weights_digest(old)
        sha_new=model_weights_digest(new)
        # Models completely fitted prior to touching any held examples.
        old_metrics=score_after_frozen_training(old,held_dev)
        new_metrics=score_after_frozen_training(new,held_dev)
        require(sha_old==model_weights_digest(old)
                and sha_new==model_weights_digest(new),
                "frozen S44/S45 CPU weights mutated by offline scoring")
        require(old_metrics["event_source_support"]==
                    new_metrics["event_source_support"]
                and old_metrics["REJECTED_link_support"]==
                    new_metrics["REJECTED_link_support"]
                and old_metrics["held_rows_requiring_one_source_to_multiple_recipient_links"]==
                    new_metrics["held_rows_requiring_one_source_to_multiple_recipient_links"],
                "synthetic held development gold changed during curriculum")
        archived_metrics=archived["S44_new_TRAIN_internal"]
        for key in ("event_source_support","REJECTED_link_support"):
            require(archived_metrics[key]==old_metrics[key],
                    "original S44 archived gold support mismatch")
        entry={
           "fold":fold,"axis":axis,"group_OFFLINE_ONLY":group,
           "group_name_OFFLINE_ONLY":
             "lexical_family" if axis=="LEXICAL" else PATTERNS[group],
           "original_TRAIN128_rows":len(base_train),
           "extra_NEW64_TRAIN_rows":len(curriculum),
           "total_training_rows_S45":len(extension_train),
           "held_original_S43_synthetic_TRAIN_dev_rows":len(held_dev),
           "held_disjoint_from_ALL_curriculum_training":True,
           "S44_CPU_weights_SHA256":sha_old,
           "S45_CPU_weights_SHA256":sha_new,
           "S44_cross_run_archive_same_numeric_weights":
                sha_old==archived["S44_CPU_weights_SHA256"],
           "S44_cross_run_archive_same_held_metrics":
                all(archived_metrics[k]==old_metrics[k]
                    for k in ("event_source_TP","event_source_FP","event_source_FN",
                              "REJECTED_link_TP","REJECTED_link_FP","REJECTED_link_FN",
                              "quote_TP","quote_FP","quote_FN")),
           "S44_same_run_TRAIN_internal":old_metrics,
           "S45_extended_TRAIN_internal":new_metrics,
           "S44_loss_ONLY_TRAIN":old_train_loss,
           "S45_loss_ONLY_TRAIN":new_train_loss,
           "S45_no_forward_gold_or_output_side_matching":True,
           "original_S38_control_NEVER_scored_or_trained":True,
           "output_gate":0,
        }
        results.append(entry)
        print("S45_EXTENDED_TRAIN_CURRICULUM_SAME_RUN_FOLD",
              json.dumps({"fold":fold,"axis":axis,"group":group,
                "S44_link_TP":old_metrics["REJECTED_link_TP"],
                "S44_link_FP":old_metrics["REJECTED_link_FP"],
                "S45_link_TP":new_metrics["REJECTED_link_TP"],
                "S45_link_FP":new_metrics["REJECTED_link_FP"],
                "S44_quote_FP":old_metrics["quote_FP"],
                "S45_quote_FP":new_metrics["quote_FP"]},
                sort_keys=True),flush=True)
    totals={axis:{
        "S44_same_run":micro_summary(results,"S44_same_run_TRAIN_internal",axis),
        "S45_with_extra_new64_TRAIN":micro_summary(
             results,"S45_extended_TRAIN_internal",axis),
    } for axis in ("LEXICAL","CONSTRUCTION")}
    focus={PATTERNS[group]:{
        "S44":next(z for z in results if z["axis"]=="CONSTRUCTION"
                   and z["group_OFFLINE_ONLY"]==group)["S44_same_run_TRAIN_internal"],
        "S45":next(z for z in results if z["axis"]=="CONSTRUCTION"
                   and z["group_OFFLINE_ONLY"]==group)["S45_extended_TRAIN_internal"],
    } for group in (1,3,4,5,6,7,8,9,10,11,12,13,15)}
    report={
      "stage":STAGE,
      "S44_previous_successful_CI_run":PREVIOUS_SUCCESSFUL_S44_RUN,
      "S45_NEW64_source_precommitted_git_blob_SHA":TRAIN64_BLOB_SHA,
      "S45_NEW64_source_precommitted_commit":NEW64_FROZEN_COMMIT,
      "original_S43_NEW128_source_sha256":original["source_sha256"],
      "S45_NEW64_TRAIN_source":extra,
      "S45_total_TRAIN_rows_per_fold":(176,184),
      "all_24_same_old_S43_SYNTHETIC_TRAIN_development_folds":results,
      "matched_F1_and_FP_ON_SYNTHETIC_TRAIN_dev":totals,
      "predeclared_mixed_event_nonlocal_negation_and_quote_development_diagnostics":focus,
      "one_precommitted_difference_only_extra_64_TRAIN_examples":True,
      "unchanged_S44_optimizer_objective_seed_steps_lr_inference":True,
      "same_120_TRAIN_steps":S44_TRAIN_STEPS,
      "same_optimizer_lr":S44_TRAIN_LR,
      "original_training_fold_and_held_disjoint_with_new64":True,
      "S38_adversarial_control_unseen_NOT_trained_scored_or_tuned":True,
      "S45_not_independent_validation_old_held_labels_already_examined":True,
      "S40_previously_handcrafted_russian_structure_priors_still_present":True,
      "original_Calibri43M_GT_NRA2_model_NOT_loaded_or_trained":True,
      "original_616J7_EVAL32_EVAL128_bpp_CB59_NOT_tested":True,
      "no_LIVE_Render_Supabase_Jessica_APK_changes":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    args.out.mkdir(parents=True,exist_ok=False)
    name="NRA3_S45_SAME_S44_WITH_DISJOINT64_ADDITIONAL_SCOPE_TRAIN_24FOLD.json"
    (args.out/name).write_text(json.dumps(report,ensure_ascii=False,
                            sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("S45_EXTRA64_TRAIN_S44_PAIRED_PHYSICAL_24FOLD_SUMMARY",
          json.dumps(totals,sort_keys=True),flush=True)
    print("S45_NOT_INDEPENDENT_EVAL_NOT_REAL43M_NRA2_NO_ACTIVATION",flush=True)

if __name__=="__main__":
    main()
