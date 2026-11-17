"""S31 research only: joint REJECTED+RECIPIENT TRAIN-side margin consistency.

S30 certified the genuine original43M+GT source, reproducible numeric
FP16 paid head, and persistent 0/8 inversions for BOTH S22d and S26.
S31 tests a *single preregistered* hypothesis: the rare negative-recipient
training gradients in the original five-head loss were drowned by dense
NONE pair supervision. Keep S26's 3052 learned FP16 head and frozen 9332
parent; change ONLY the TRAIN-only objective, not semantic model forward.

The new loss explicitly reinforces BOTH directed RECIPIENT and REJECTED
against alternate classes at true TRAIN-side pairs while discouraging
hard false candidate pairs with the SAME raw "не"+dative surface cue.
No TRAIN gold, roles, labels, or held-group data enter neural forward.
No independent EVAL, output_gate=0, no Calibri/NRA2/LIVE activation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19f_real43m_train9_graph_targets import ROLE
from research.nra3_s19.s19g_train9_real43m_graph import (
    fixed_multitask_loss,LR,STEPS,task_train_reuse_report,
)
from research.nra3_s19.s19h_true43m_train9_loo_diagnostics import take
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    splits,FROZEN_S22_TRAIN256_SHA256,
)
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    EventGroundedInversionGraph,PREV_NEG,LIKELY_DATIVE,ROLE_RECIPIENT,
)
from research.nra3_s20.s26_train256_event_grounded_matched40 import (
    load_physical_training_source,new_graph_model,
    rejected_specific_counts,source_only_counts,
    SEED,PAID_PARENT,PAID_GRAPH,PAID_TOTAL,
)
from research.nra3_s20.s23_real_s22d_40fold_failure_audit import (
    S22D_MODEL_SHA,
)
from research.nra3_s20.s25_real43m_rejected_raw_cue_alignment_audit import cue_audit
from research.nra3_s20.s28_real43m_trainonly_determinism_probe import (
    canonical_paid_semantic_fp16_sha,
)
from research.nra3_s20.s29_archived_original43m_canonical_fp16_identity import (
    FROZEN_ARCHIVES,
)
from research.nra3_s19.s19g_train9_real43m_graph import S19D_FULL_PHYSICAL_SHA

# One a priori design from S30 failure diagnosis. No search over margins,
# folds, loss multipliers, seeds, or checkpoints using held8 observations.
MARGIN=3.0
HARD_NEGATIVE_MARGIN=0.5
TRAIN_POSITIVE_COEFF=0.7
TRAIN_HARD_NEGATIVE_COEFF=0.1
FOLDS=(8+24,8+25,8+26,8+27,8+28,8+7,8+16,8+22)
TARGET_PREPOSED=33
PREDECLARED_STEPS=240
S29_S26_CANONICAL="02d4403ffbce01632d54d66349a01c3f10b81650d26d9540bb86f495cac62807"
S29_S22D_CANONICAL="35102946886b9f35d86f57f791b6ab6001b813e4f79397cff8e4d25c04e7a538"

def require(ok,msg):
    if not bool(ok):
        raise ValueError("S31 real43M TRAIN-only margin study: "+msg)


def pairwise_hard_negative_positive_margin_loss(out,train_x,valid,gold):
    """TRAIN supervision only; model forward already ran without gold.

    ROLE logits are 9 classes, ARG_POLARITY 3. Softplus hinge losses are
    label-aware only during TRAIN; no human oracle is fed to model.
    """
    b,w=valid.shape
    legal=(valid.unsqueeze(2)&valid.unsqueeze(1)) & (
        ~torch.eye(w,dtype=torch.bool,device=valid.device).unsqueeze(0))
    gold_role=gold["predicate_argument_role_TRAIN_ONLY"]
    rejected=gold["argument_polarity_TRAIN_ONLY"]==1
    gold_event=gold["event_status_TRAIN_ONLY"]>0
    correct=legal & rejected & (gold_role==ROLE_RECIPIENT)
    require(bool(correct.any()),"positive REJECTED recipient absent from TRAIN side")
    require(int((legal & rejected & (gold_role!=ROLE_RECIPIENT)).sum())==0,
            "new TRAIN objectives allowed only verified REJECTED recipient supervision")
    rlog=out.role
    plog=out.argument_polarity
    role_rest=torch.logsumexp(torch.cat(
        (rlog[...,:ROLE_RECIPIENT],rlog[...,ROLE_RECIPIENT+1:]),dim=-1),dim=-1)
    polar_rest=torch.logsumexp(plog[...,:2],dim=-1)
    role_margin=rlog[...,ROLE_RECIPIENT]-role_rest
    rejected_margin=plog[...,2]-polar_rest
    positive=(F.softplus(MARGIN-role_margin[correct])
              +F.softplus(MARGIN-rejected_margin[correct])).mean()
    raw_cue=(train_x[...,320+PREV_NEG]*train_x[...,320+LIKELY_DATIVE])>0.5
    # Hard negatives are supervised from TRAIN-side event source and
    # same surface cue. Pair direction *not* used in loss.
    hard=legal & gold_event.unsqueeze(2) & raw_cue.unsqueeze(1) & (~correct)
    require(not bool((hard & correct).any()),"hard neg includes true positive")
    negative=((F.softplus(HARD_NEGATIVE_MARGIN+role_margin[hard])
               +F.softplus(HARD_NEGATIVE_MARGIN+rejected_margin[hard])).mean()
              if bool(hard.any()) else out.role.sum()*0.)
    return positive,negative,{
       "TRAIN_only_true_rejected_recipient":int(correct.sum()),
       "TRAIN_only_hard_confusable_nonrecipient":int(hard.sum()),
    }


def fixed_train_S31(model,raw,valid,gold):
    require(STEPS==PREDECLARED_STEPS,"S31 training optimizer steps drift")
    optimizer=torch.optim.AdamW(list(model.semantic_params()),
                                lr=LR,weight_decay=.001)
    before=after=None
    sizes=None
    for t in range(PREDECLARED_STEPS):
        optimizer.zero_grad(set_to_none=True)
        out=model(raw,valid)
        base=fixed_multitask_loss(out,gold,valid)
        positive,negative,sizes=pairwise_hard_negative_positive_margin_loss(
            out,raw,valid,gold)
        loss=base+TRAIN_POSITIVE_COEFF*positive+TRAIN_HARD_NEGATIVE_COEFF*negative
        require(bool(torch.isfinite(loss)),"nonfinite S31 TRAIN-only margin loss")
        if t==0:before=float(loss.detach())
        loss.backward()
        require(all(p.grad is None for p in model.parent.parameters())
                and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                        for p in model.semantic_params()),
                "frozen original parent changed or semantic TRAIN gradients missing")
        torch.nn.utils.clip_grad_norm_(list(model.semantic_params()),1.)
        optimizer.step()
        if t==PREDECLARED_STEPS-1:after=float(loss.detach())
    paid=model.physical_bytes()
    physical=EventGroundedInversionGraph.from_physical(paid).eval()
    require(physical.head_physical()==model.head_physical()
            and canonical_paid_semantic_fp16_sha(physical)==
                canonical_paid_semantic_fp16_sha(model)
            and all(not p.requires_grad for p in physical.parameters())
            and physical.output_gate==0
            and not physical.candidate_accepted and not physical.activation_allowed,
            "actual 3052+9332 paid FP16 S31 physical roundtrip failed")
    return physical,paid,{"start_loss":before,"last_step_loss":after,**sizes}


def _held_rejected_role_recall(out,gold,valid):
    mask=(valid.unsqueeze(2)&valid.unsqueeze(1)) & (
        ~torch.eye(valid.shape[1],dtype=torch.bool).unsqueeze(0))
    positive=mask & (gold["argument_polarity_TRAIN_ONLY"]==1)
    pred_role=out.role.argmax(-1)==ROLE_RECIPIENT
    rejected=out.argument_polarity.argmax(-1)==2
    total=int(positive.sum())
    return {
       "held_REJECTED_total":total,
       "correct_RECIPIENT_at_gold_rejected_pairs":int((positive&pred_role).sum()),
       "correct_REJECTED_at_gold_rejected_pairs":int((positive&rejected).sum()),
       "jointly_correct_at_gold_rejected_pairs":int((positive&pred_role&rejected).sum()),
    }


def main():
    cli=argparse.ArgumentParser()
    for name in ("features","gold","manifest","s19d-parent",
                 "baseline-s22d-model","baseline-s26-model","baseline-s26-report","out"):
        cli.add_argument("--"+name,type=Path,required=True)
    args=cli.parse_args()
    require(not args.out.exists(),"never overwrite original FP16 archive/model outputs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    raw,valid,y,source=load_physical_training_source(
        args.features,args.gold,args.manifest)
    require(source["S22_sha_locked_synthetic_TRAIN256_graph_gold"]==
            FROZEN_S22_TRAIN256_SHA256,"S22c physical original TRAIN256 source drift")
    geometry=cue_audit(raw,valid,y)
    require(geometry["preposed_source_rejected_8"]["negation_and_dative_both"]==8,
            "S25 raw negative+dative preposed physical CPU geometry missing")
    b22=args.baseline_s22d_model.read_bytes()
    b26=args.baseline_s26_model.read_bytes()
    require(hashlib.sha256(b22).hexdigest()==FROZEN_ARCHIVES["S22d"]["sha"]
            and hashlib.sha256(b26).hexdigest()==FROZEN_ARCHIVES["S26"]["sha"],
            "S22d/S26 real archived model integrity SHA mismatch")
    import research.nra3_s19.s19g_paid_semantic_graph as graph
    old=graph.PaidSemanticEventGraph.from_physical(b22)
    baseline=EventGroundedInversionGraph.from_physical(b26)
    require(canonical_paid_semantic_fp16_sha(old)==S29_S22D_CANONICAL
            and canonical_paid_semantic_fp16_sha(baseline)==S29_S26_CANONICAL,
            "S29 canonical named-FP16 model provenance changed")
    oldreport=json.loads(args.baseline_s26_report.read_text())
    require(oldreport["physical_FP16_candidate_SHA256"]==FROZEN_ARCHIVES["S26"]["sha"]
            and len(oldreport["all40_fixed_TRAIN_internal_folds"])==40
            and oldreport["no_EVAL32_EVAL128_or_prior_spent_EVAL"]
            and oldreport["output_gate"]==0
            and not oldreport["candidate_accepted"],
            "actual S26 40fold synthetic TRAIN baseline report invalid")
    frozen_bytes=args.s19d_parent.read_bytes()
    require(hashlib.sha256(frozen_bytes).hexdigest()==S19D_FULL_PHYSICAL_SHA,
            "original physically paid 9332 FP16 S19d parent drift")
    all_splits=splits()
    require(len(all_splits)==40 and FOLDS[1]==TARGET_PREPOSED,
            "predeclared four rejection + unrelated controls split changed")
    records=[]
    for fold in FOLDS:
        axis,grp,tr,held=all_splits[fold]
        require(axis=="CONSTRUCTION" and len(held)==8 and len(tr)==248
                and set(tr).isdisjoint(held) and grp==fold-8,
                "S31 TRAIN-only fold partition contamination")
        trraw,trvalid,trgold=(take(raw,tr),take(valid,tr),take(y,tr))
        dvraw,dvvalid,dvgold=(take(raw,held),take(valid,held),take(y,held))
        model=new_graph_model(
            PaidContextualEventScope.from_physical(frozen_bytes),SEED+fold)
        candidate,archive,loss_diag=fixed_train_S31(model,trraw,trvalid,trgold)
        with torch.inference_mode():
            out=candidate(dvraw,dvvalid)
            tasks=source_only_counts(out,dvgold,dvvalid)
            rare=rejected_specific_counts(out,dvgold,dvvalid)
            roles=_held_rejected_role_recall(out,dvgold,dvvalid)
        original=oldreport["all40_fixed_TRAIN_internal_folds"][fold]
        require(original["fold_number"]==fold
                and original["held_group_index"]==grp
                and original["held_TRAIN_rows"]==8
                and rare["held_REJECTED_positive_support"]==
                    original["REJECTED_specific_held_TRAIN"]["held_REJECTED_positive_support"],
                "original S26 40fold same source fold mismatch")
        records.append({
            "fold":fold,"construction_index":grp,
            "fixed_original43m_s22c_source_SHA256":FROZEN_S22_TRAIN256_SHA256,
            "actual_training_side":248,"held_synthetic_TRAIN_rows":8,
            "gold_roles_NEVER_passed_to_neural_forward":True,
            "fixed_steps":PREDECLARED_STEPS,
            "physical_S31_paid_FP16_SHA256":hashlib.sha256(archive).hexdigest(),
            "canonical_S31_paid_FP16_numeric_SHA256":canonical_paid_semantic_fp16_sha(candidate),
            "S31_train_loss_diagnostic":loss_diag,
            "S31_held_REJECTED":rare,
            "S31_held_role_rejection_joint":roles,
            "S31_held_5heads":tasks,
            "S26_held_REJECTED":original["REJECTED_specific_held_TRAIN"],
            "S26_held_5heads":original["held_TRAIN_metrics"],
        })
        print("S31_REAL43M_TRAIN_FOLD_RESULT",
              json.dumps({"fold":fold,"grp":grp,
                          "S31_rejected_recall":rare["recall"],
                          "S26_rejected_recall":original["REJECTED_specific_held_TRAIN"]["recall"],
                          "S31_rejected_FP":rare["FP"],
                          "S31_joint":roles["jointly_correct_at_gold_rejected_pairs"],
                          "S31_held_relation_TP":tasks["event_relation"]["TP"]},
                          sort_keys=True),flush=True)
    require(len(records)==len(FOLDS),"preregistered fold coverage failed")
    preposed=next(x for x in records if x["fold"]==TARGET_PREPOSED)
    require(preposed["S31_held_REJECTED"]["held_REJECTED_positive_support"]==8
            and preposed["S26_held_REJECTED"]["TP"]==0,
            "known S30 physically observed preposed synthetic TRAIN case changed")
    result={
      "stage":"NRA3_S31_TRAIN_ONLY_RARE_PAIR_JOINT_MARGIN_SEMANTIC_FP16_ABLATION",
      "original_genuine_frozen43M_GT_source_S22c_sha256":FROZEN_S22_TRAIN256_SHA256,
      "original_S22d_archived_canonical_sha256":S29_S22D_CANONICAL,
      "original_S26_archived_canonical_sha256":S29_S26_CANONICAL,
      "S30_original43M_truth":"Both original models failed 0/8 preposed REJECTED; failure is role as well as polarity.",
      "preregistered_fixed_loss_margin":MARGIN,
      "preregistered_fixed_hardnegative_margin":HARD_NEGATIVE_MARGIN,
      "preregistered_positive_coefficient":TRAIN_POSITIVE_COEFF,
      "preregistered_hard_negative_coefficient":TRAIN_HARD_NEGATIVE_COEFF,
      "same_architecture_as_S26_zero_new_weights":True,
      "actual_paid_original_S26_total_FP16":PAID_TOTAL,
      "actual_paid_new_S31_total_FP16":PAID_TOTAL,
      "same_train256_only_four_REJECTED_and_four_control_folds":list(FOLDS),
      "per_fold_8_rows_synthetic_TRAIN_internal_NOT_independent":records,
      "preposed_rejected_TRAIN_only_8_fold33":preposed,
      "no_hyperparameter_or_checkpoint_selection_using_held_data":True,
      "no_natural_language_generalization_or_CPU_bpp_J7_claim":True,
      "original_616_J7_and_independent_EVAL_never_touched":True,
      "frozen_43M_GT_S19d_parent_no_grad":True,
      "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    args.out.mkdir(parents=True,exist_ok=False)
    destination=args.out/"NRA3_S31_REAL43M_GT_TRAIN256_FIXED8FOLD_RARE_PAIR_MARGIN_ABLATION.json"
    destination.write_text(
      json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("S31_REAL43M_PAIRED_PREPOSED",
      json.dumps({
          "S26_TP":preposed["S26_held_REJECTED"]["TP"],
          "S31_TP":preposed["S31_held_REJECTED"]["TP"],
          "S31_FP":preposed["S31_held_REJECTED"]["FP"],
          "S31_gold_role_correct":preposed["S31_held_role_rejection_joint"]["correct_RECIPIENT_at_gold_rejected_pairs"],
          "S31_joint_correct":preposed["S31_held_role_rejection_joint"]["jointly_correct_at_gold_rejected_pairs"],
          },sort_keys=True),flush=True)
    print("S31_TRAIN_ONLY_43M_GT_ACTUAL_S22C_TRUE_CPU_SHA_SAFE_COMPLETE",flush=True)


if __name__=="__main__":
    main()
