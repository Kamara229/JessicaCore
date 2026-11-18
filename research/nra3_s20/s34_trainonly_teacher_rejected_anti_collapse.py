"""S34: test a precommitted TRAIN-only anti-collapse REJECTED margin anchor.

S33 head-only precision reduced false REJECTED 8->4 but lost true positives
30->26 vs S32 and inverted-recipient 8/8->3/8 on synthetic TRAIN-internal
development folds. Retain EXACT S33 frozen-trunk/relation plan and same
eight grouped folds. Add a differentiable anchor from *fresh TRAIN-only*
S31 teacher margins for genuine REJECTED recipient pairs so the precision
objective cannot erase known rejection semantics without explicit cost.
The teacher uses TRAIN-side input only; no held group, gold labels or oracle
are fed to model forward. Fixed schedule no held-based checkpoint choice.
Original Calibri43M+GT/616 J7, independent EVAL and LIVE remain untouched.
""
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
from research.nra3_s20.s31_train256_rare_pair_margin_probe import fixed_train_S31

# One a priori design from S30 failure diagnosis. No search over margins,
# folds, loss multipliers, seeds, or checkpoints using held8 observations.
MARGIN=3.0
HARD_NEGATIVE_MARGIN=0.5
TRAIN_POSITIVE_COEFF=0.7
TRAIN_HARD_NEGATIVE_COEFF=0.1
TRAIN_ACCEPTED_RECIPIENT_COEFF=0.45
TRAIN_FALSE_PAIR_TOPK_COEFF=0.05
NOT_REJECTED_MARGIN=2.0
FALSE_PAIR_MARGIN=0.5
FALSE_PAIR_TOPK_PER_TEXT=2
S34_ONLY_HEAD_STEPS=64
S34_HEAD_FINETUNE_LR=0.003
S34_REJECTED_ANCHOR_COEFF=2.0
S34_REJECTED_TEACHER_FLOOR_DELTA=0.50
S34_TRUE_VS_FALSE_PAIR_RANK_COEFF=0.50
S34_TRAIN_RANK_MARGIN=1.0
S34_STAGE="S31_240_TRAIN_then_64_head_only_anti_collapse_anchor"
FOLDS=(8+24,8+25,8+26,8+27,8+28,8+7,8+16,8+22)
TARGET_PREPOSED=33
PREDECLARED_STEPS=240
S29_S26_CANONICAL="02d4403ffbce01632d54d66349a01c3f10b81650d26d9540bb86f495cac62807"
S29_S22D_CANONICAL="35102946886b9f35d86f57f791b6ab6001b813e4f79397cff8e4d25c04e7a538"

def require(ok,msg):
    if not bool(ok):
        raise ValueError("S34 real43M TRAIN-only margin study: "+msg)


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


def accepted_recipient_and_hard_false_pair_loss(out,valid,gold):
    """Separately penalize rejected class at TRAIN-side accepted recipients.

    Hard negative mining chooses the top 2 WRONG pairs based solely on
    predictions and TRAIN-side label exclusions; no heldout selection.
    Unrelated to gold scores at forward or inference.
    """
    b,w=valid.shape
    legal=(valid.unsqueeze(2)&valid.unsqueeze(1)) & (
        ~torch.eye(w,dtype=torch.bool,device=valid.device).unsqueeze(0))
    role_gold=gold["predicate_argument_role_TRAIN_ONLY"]
    polarity_gold=gold["argument_polarity_TRAIN_ONLY"]
    accepted_recipient=(legal & (role_gold==ROLE_RECIPIENT)
                        & (polarity_gold==0))
    require(bool(accepted_recipient.any()),
            "S34 TRAIN gold lacks accepted recipient contrasts")
    rejected_margin=(out.argument_polarity[...,2]-
                     torch.logsumexp(out.argument_polarity[...,:2],dim=-1))
    # An accepted recipient may co-occur with an explicitly REJECTED
    # alternative in the same sentence, so penalize only its own edge.
    accepted=F.softplus(
        NOT_REJECTED_MARGIN+rejected_margin[accepted_recipient]).mean()
    wrong=legal & (polarity_gold!=1) & (role_gold!=ROLE_RECIPIENT)
    hard_candidates=rejected_margin.masked_fill(~wrong,-1e8).reshape(b,-1)
    k=min(FALSE_PAIR_TOPK_PER_TEXT,hard_candidates.shape[1])
    top_values,top_indices=hard_candidates.topk(k,dim=1)
    top_legitimate=top_values>-1e7
    require(bool(top_legitimate.all()),
            "S34 true TRAIN side has insufficient non-recipient hard negatives")
    top=F.softplus(FALSE_PAIR_MARGIN+top_values).mean()
    require(bool(torch.isfinite(accepted)) and bool(torch.isfinite(top)),
            "S34 train-only negative losses not finite")
    return accepted,top,{
        "TRAIN_accepted_recipient_pairs":int(accepted_recipient.sum()),
        "TRAIN_other_nonrecipient_pairs":int(wrong.sum()),
        "TRAIN_per_sentence_top_false_pairs":k,
    }


def train_only_rejected_anti_collapse(out,teacher,valid,gold):
    """Protect TRAIN-labelled rare-positive directed REJECTED margins.

    teacher is frozen S31, run ON the same 248 TRAIN-side examples only.
    Forward receives raw word338 and valid mask; TRAIN gold is used only
    here for selecting the supervised loss terms, never network inputs.
    """
    w=valid.shape[1]
    legal=(valid.unsqueeze(2)&valid.unsqueeze(1)) & (
        ~torch.eye(w,device=valid.device,dtype=torch.bool).unsqueeze(0))
    role=gold["predicate_argument_role_TRAIN_ONLY"]
    pol=gold["argument_polarity_TRAIN_ONLY"]
    is_pos=legal & (role==ROLE_RECIPIENT) & (pol==1)
    require(bool(is_pos.any()),"TRAIN-side REJECTED recipient positives missing")
    def margin(pred):
        return pred.argument_polarity[...,2]-torch.logsumexp(
            pred.argument_polarity[...,:2],dim=-1)
    student=margin(out)
    with torch.no_grad():
        desired=margin(teacher).detach()
    # Exact floor protected only when student falls below the positive
    # TRAIN teacher margin (plus the fixed .5 safety margin).
    anchor=F.softplus(S34_REJECTED_TEACHER_FLOOR_DELTA+
                      desired[is_pos]-student[is_pos]).mean()
    # A differentiable listwise comparison against highest incorrect
    # candidate within the same TRAIN graph. No held fold scores or
    # held text enters this objective.
    wrong=legal & ~is_pos & (pol!=1)
    scores=student.masked_fill(~wrong,-1e6).flatten(start_dim=1)
    hardest=scores.max(dim=1).values
    positive_per_row=is_pos.flatten(start_dim=1).any(dim=1)
    positive_low=student.masked_fill(~is_pos,1e6).flatten(start_dim=1).min(dim=1).values
    require(bool(positive_per_row.any()),"no ranked rejected positives TRAIN-side")
    hard_for_pos=hardest[positive_per_row]
    true_for_pos=positive_low[positive_per_row]
    require(bool((hard_for_pos>-1e5).all()),"no comparable hard-negative TRAIN pairs")
    ranking=F.softplus(S34_TRAIN_RANK_MARGIN+hard_for_pos-true_for_pos).mean()
    require(bool(torch.isfinite(anchor) and torch.isfinite(ranking)),
            "S34 TRAIN-only ranking or anchor nonfinite")
    return anchor,ranking,{
        "TRAIN_only_supervised_rejected_pairs":int(is_pos.sum()),
        "TRAIN_only_ranked_texts":int(positive_per_row.sum()),
        "TRAIN_only_teacher_margin_reference":True,
    }


def fixed_train_S34(model,raw,valid,gold):
    """Refit exact S31 then constrained TRAIN-only precision calibration.

    The second phase can *only* change role/polarity logits; all relation,
    event and coreference scores must remain bitwise identical to S31.
    """
    require(STEPS==PREDECLARED_STEPS
            and S34_ONLY_HEAD_STEPS==64 and S34_HEAD_FINETUNE_LR==0.003,
            "precommitted S31+S34 optimizer schedule changed")
    s31,_,s31_train_diag=fixed_train_S31(model,raw,valid,gold)
    # Recovered S31 is an immutable original-header-digested FP16 model
    # while model source is true CPU43M+GT, and the active gateway is off.
    require(s31.output_gate==0 and not s31.candidate_accepted
            and not s31.activation_allowed
            and all(not p.requires_grad for p in s31.parameters()),
            "S31 physical checkpoint leaked trainable or active parameters")
    preserved_s31=EventGroundedInversionGraph.from_physical(s31.physical_bytes()).eval()
    initial_weights={n:p.detach().clone() for n,p in s31.named_parameters()}
    allowed={"event_scoped_recipient_weights"}
    allowed.update(f"role.{n}" for n,_ in s31.role.named_parameters())
    allowed.update(f"argument_polarity.{n}" for n,_ in s31.argument_polarity.named_parameters())
    require(allowed=={
        "event_scoped_recipient_weights",
        "role.weight","role.bias",
        "argument_polarity.weight","argument_polarity.bias",
    }, "only explicitly paid role/polarity scalar heads may update")
    for name,p in s31.named_parameters():
        p.requires_grad_(name in allowed)
    paid=[p for p in s31.parameters() if p.requires_grad]
    require(sum(p.numel() for p in paid)==
            (s31.role.weight.numel()+s31.role.bias.numel()
             +s31.argument_polarity.weight.numel()+s31.argument_polarity.bias.numel()
             +s31.event_scoped_recipient_weights.numel()),
            "all trainable weights must be restricted to five allowed tensors")
    optimizer=torch.optim.AdamW(paid,lr=S34_HEAD_FINETUNE_LR,weight_decay=.001)
    # S31 teacher logits are computed ONCE from genuine TRAIN-side raw338,
    # are not trainable, and no held data is inspected before stage2.
    with torch.inference_mode():
        teacher_logits=preserved_s31(raw,valid)
    starts=last=None
    diagnostics={}
    for t in range(S34_ONLY_HEAD_STEPS):
        optimizer.zero_grad(set_to_none=True)
        out=s31(raw,valid)
        base=fixed_multitask_loss(out,gold,valid)
        positive,negative,positive_stats=pairwise_hard_negative_positive_margin_loss(
            out,raw,valid,gold)
        accepted,top,precision_stats=accepted_recipient_and_hard_false_pair_loss(
            out,valid,gold)
        anchor,ranking,anchor_stats=train_only_rejected_anti_collapse(
            out,teacher_logits,valid,gold)
        loss=(base+TRAIN_POSITIVE_COEFF*positive
              +TRAIN_HARD_NEGATIVE_COEFF*negative
              +TRAIN_ACCEPTED_RECIPIENT_COEFF*accepted
              +TRAIN_FALSE_PAIR_TOPK_COEFF*top
              +S34_REJECTED_ANCHOR_COEFF*anchor
              +S34_TRUE_VS_FALSE_PAIR_RANK_COEFF*ranking)
        require(bool(torch.isfinite(loss)),"nonfinite head-only TRAIN loss")
        if t==0:starts=float(loss.detach())
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in paid)
                and all(p.grad is None for p in s31.parameters() if not p.requires_grad),
                "unexpected gradient to frozen original/relation/shared head")
        torch.nn.utils.clip_grad_norm_(paid,1.)
        optimizer.step()
        if t==S34_ONLY_HEAD_STEPS-1:
            last=float(loss.detach())
            diagnostics={**positive_stats,**precision_stats,**anchor_stats,
                         "TRAIN_last_anchor_loss":float(anchor.detach()),
                         "TRAIN_last_rank_loss":float(ranking.detach())}
    require(all(torch.equal(p.detach(),initial_weights[name])
                for name,p in s31.named_parameters() if name not in allowed),
            "S34 updated frozen original/shared node or relation weights")
    payload=s31.physical_bytes()
    physical=EventGroundedInversionGraph.from_physical(payload).eval()
    require(physical.head_physical()==s31.head_physical()
            and canonical_paid_semantic_fp16_sha(physical)==
                canonical_paid_semantic_fp16_sha(s31)
            and all(not p.requires_grad for p in physical.parameters())
            and physical.output_gate==0 and not physical.candidate_accepted
            and not physical.activation_allowed,
            "S34 12384 FP16 paid head roundtrip failed")
    return preserved_s31,physical,payload,{
        "stage1_exact_original_S31_train":s31_train_diag,
        "head_only_tensors_updated":sorted(allowed),
        "stage2_trainable_parameter_count":sum(p.numel() for p in paid),
        "head_only_steps":S34_ONLY_HEAD_STEPS,
        "head_only_lr":S34_HEAD_FINETUNE_LR,
        "stage2_train_loss_first":starts,"stage2_train_loss_last":last,
        **diagnostics,
    }


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
                 "baseline-s22d-model","baseline-s26-model","baseline-s26-report","baseline-s31-report","baseline-s32-report","baseline-s33-report","out"):
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
    previous_s31=json.loads(args.baseline_s31_report.read_text())
    require(previous_s31["stage"]==
             "NRA3_S31_TRAIN_ONLY_RARE_PAIR_JOINT_MARGIN_SEMANTIC_FP16_ABLATION"
            and previous_s31["original_genuine_frozen43M_GT_source_S22c_sha256"]==
                FROZEN_S22_TRAIN256_SHA256
            and previous_s31["same_train256_only_four_REJECTED_and_four_control_folds"]==list(FOLDS)
            and previous_s31["output_gate"]==0
            and not previous_s31["candidate_accepted"],
            "original completed S31 matched eight-fold baseline report drift")
    original_s31_folds={
        z["fold"]:z
        for z in previous_s31["per_fold_8_rows_synthetic_TRAIN_internal_NOT_independent"]
    }
    require(set(original_s31_folds)==set(FOLDS),
            "original S31 old eight held TRAIN folds not matched")
    previous_s32=json.loads(args.baseline_s32_report.read_text())
    require(previous_s32["stage"]==
            "NRA3_S32_TRAIN_ONLY_ACCEPTED_RECIPIENT_PRECISION_MARGIN_ABLATION"
            and previous_s32["original_genuine_frozen43M_GT_source_S22c_sha256"]==
                FROZEN_S22_TRAIN256_SHA256
            and previous_s32["same_train256_only_four_REJECTED_and_four_control_folds"]==list(FOLDS)
            and previous_s32["matched_S31_completed_action"]==44203704916
            and previous_s32["output_gate"]==0
            and not previous_s32["candidate_accepted"],
            "S32 already completed synthetic TRAIN development comparator invalid")
    original_s32_folds={
        z["fold"]:z
        for z in previous_s32["per_fold_8_rows_synthetic_TRAIN_internal_NOT_independent"]
    }
    require(set(original_s32_folds)==set(FOLDS),
            "S32 old eight training development fold IDs changed")
    previous_s33=json.loads(args.baseline_s33_report.read_text())
    require(previous_s33["stage"]==
            "NRA3_S33_12384_FP16_FRESH_S31_REFIT_HEAD_ONLY_PRESERVE_RELATION_TRAIN_ABLATION"
            and previous_s33["original_genuine_frozen43M_GT_source_S22c_sha256"]==
              FROZEN_S22_TRAIN256_SHA256
            and previous_s33["matched_S32_completed_action"]==44204827220
            and previous_s33["same_train256_only_four_REJECTED_and_four_control_folds"]==list(FOLDS)
            and previous_s33["output_gate"]==0
            and not previous_s33["candidate_accepted"],
            "authentic S33 prior TRAIN development report failed strict provenance")
    original_s33_folds={
        x["fold"]:x for x in
        previous_s33["per_fold_8_rows_synthetic_TRAIN_internal_NOT_independent"]
    }
    require(set(original_s33_folds)==set(FOLDS),
            "original S33 matched TRAIN-internal fold set changed")
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
                "S34 TRAIN-only fold partition contamination")
        trraw,trvalid,trgold=(take(raw,tr),take(valid,tr),take(y,tr))
        dvraw,dvvalid,dvgold=(take(raw,held),take(valid,held),take(y,held))
        model=new_graph_model(
            PaidContextualEventScope.from_physical(frozen_bytes),SEED+fold)
        s31_refit,candidate,archive,loss_diag=fixed_train_S34(
            model,trraw,trvalid,trgold)
        s31_prior=original_s31_folds[fold]
        s32_prior=original_s32_folds[fold]
        # Verify REAL refitted S31 numerically with its archived canonical
        # SHA; old raw serializer SHA is not a numerical fingerprint.
        # S28 proved same-runner numerical determinism but NOT cross-run
        # numerical equality across CI hosts or PyTorch build details.
        # Original S31 fold-specific numerical checkpoints were NOT saved,
        # so NEVER assert this fresh refit IS the archived S31 model.
        fresh_canonical=canonical_paid_semantic_fp16_sha(s31_refit)
        old_canonical=s31_prior["canonical_S31_paid_FP16_numeric_SHA256"]
        same_as_archived=(fresh_canonical==old_canonical)
        print("S34_ORIGINAL_S31_ARCHIVED_VS_FRESH_CANONICAL_FP16",fold,
              old_canonical,fresh_canonical,
              "IDENTICAL",same_as_archived,flush=True)
        with torch.inference_mode():
            old_out=s31_refit(dvraw,dvvalid)
            out=candidate(dvraw,dvvalid)
            for head in ("event_status","coreference","event_relation"):
                require(torch.equal(getattr(old_out,head),getattr(out,head)),
                        "S34 altered S31 event/coreference/relation predictions")
            prior_tasks=source_only_counts(old_out,dvgold,dvvalid)
            prior_rare=rejected_specific_counts(old_out,dvgold,dvvalid)
            historical_metrics_match=(
                prior_tasks==s31_prior["S31_held_5heads"] and
                prior_rare==s31_prior["S31_held_REJECTED"])
            print("S34_FRESH_S31_REFIT_HELD_VS_ARCHIVED_COUNTS",fold,
                  "IDENTICAL",historical_metrics_match,
                  "REL_TP_FRESH",prior_tasks["event_relation"]["TP"],
                  "REL_TP_ARCHIVED",s31_prior["S31_held_5heads"]["event_relation"]["TP"],
                  flush=True)
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
        prior_s31=s31_prior
        prior_s32=s32_prior
        require(prior_s31["construction_index"]==grp and
                prior_s31["S31_held_REJECTED"]["held_REJECTED_positive_support"]==
                  rare["held_REJECTED_positive_support"],
                "S34 matched S31 held TRAIN gold support differs")
        records.append({
            "S31_held_REJECTED":prior_s31["S31_held_REJECTED"],
            "S32_held_REJECTED":prior_s32["S32_held_REJECTED"],
            "S33_held_REJECTED":original_s33_folds[fold]["S33_held_REJECTED"],
            "S33_held_5heads":original_s33_folds[fold]["S33_held_5heads"],
            "S32_held_role_rejection_joint":prior_s32["S32_held_role_rejection_joint"],
            "S32_held_5heads":prior_s32["S32_held_5heads"],
            "S31_replayed_numerical_FP16_SHA256":fresh_canonical,
            "S31_archived_fold_numeric_FP16_SHA256":old_canonical,
            "fresh_S31_canonical_sha_matches_archived_S31":same_as_archived,
            "fresh_S31_held_counts_match_archived_S31":historical_metrics_match,
            "fresh_S31_held_5heads":prior_tasks,
            "fresh_S31_held_REJECTED":prior_rare,
            "unaffected_S31_event_coreference_relation_logits_exact_equal":True,
            "S31_held_role_rejection_joint":prior_s31["S31_held_role_rejection_joint"],
            "S31_held_5heads":prior_s31["S31_held_5heads"],
            "fold":fold,"construction_index":grp,
            "fixed_original43m_s22c_source_SHA256":FROZEN_S22_TRAIN256_SHA256,
            "actual_training_side":248,"held_synthetic_TRAIN_rows":8,
            "gold_roles_NEVER_passed_to_neural_forward":True,
            "fixed_steps":PREDECLARED_STEPS,
            "physical_S34_paid_FP16_SHA256":hashlib.sha256(archive).hexdigest(),
            "canonical_S34_paid_FP16_numeric_SHA256":canonical_paid_semantic_fp16_sha(candidate),
            "S34_train_loss_diagnostic":loss_diag,
            "S34_held_REJECTED":rare,
            "S34_held_role_rejection_joint":roles,
            "S34_held_5heads":tasks,
            "S26_held_REJECTED":original["REJECTED_specific_held_TRAIN"],
            "S26_held_5heads":original["held_TRAIN_metrics"],
        })
        print("S34_REAL43M_TRAIN_FOLD_RESULT",
              json.dumps({"fold":fold,"grp":grp,
                          "S34_rejected_recall":rare["recall"],
                          "S26_rejected_recall":original["REJECTED_specific_held_TRAIN"]["recall"],
                          "S31_rejected_recall":prior_s31["S31_held_REJECTED"]["recall"],
                          "S31_rejected_FP":prior_s31["S31_held_REJECTED"]["FP"],
                          "S34_rejected_FP":rare["FP"],
                          "S34_joint":roles["jointly_correct_at_gold_rejected_pairs"],
                          "S34_held_relation_TP":tasks["event_relation"]["TP"]},
                          sort_keys=True),flush=True)
    require(len(records)==len(FOLDS),"preregistered fold coverage failed")
    preposed=next(x for x in records if x["fold"]==TARGET_PREPOSED)
    require(preposed["S34_held_REJECTED"]["held_REJECTED_positive_support"]==8
            and preposed["S26_held_REJECTED"]["TP"]==0
            and preposed["S31_held_REJECTED"]["TP"]==7
            and preposed["S32_held_REJECTED"]["TP"]==8,
            "known S30 physically observed preposed synthetic TRAIN case changed")
    result={
      "stage":"NRA3_S34_TRAIN_ONLY_S31_TEACHER_ANTICOLLAPSE_HEADONLY_12384FP16",
      "original_genuine_frozen43M_GT_source_S22c_sha256":FROZEN_S22_TRAIN256_SHA256,
      "original_S22d_archived_canonical_sha256":S29_S22D_CANONICAL,
      "original_S26_archived_canonical_sha256":S29_S26_CANONICAL,
      "S30_original43M_truth":"Both original models failed 0/8 preposed REJECTED; failure is role as well as polarity.",
      "S31_original_held_8_truth":"S31 recovered preposed 7/8 but had 14 REJECTED false positives across all eight held TRAIN folds.",
      "S32_original_held_8_truth":"S32 recovered preposed 8/8 with eight FP overall but reduced held relation TP.",
      "stage1_S31_full_supervised_train_steps":PREDECLARED_STEPS,
      "stage2_only_role_polarity_3_residual_train_steps":S34_ONLY_HEAD_STEPS,
      "stage2_head_only_lr":S34_HEAD_FINETUNE_LR,
      "fresh_S31_relation_head_and_shared_trunk_bitwise_preserved":True,
      "original_S31_fold_checkpoint_not_archived_cross_run_numeric_nonidentity_possible":True,
      "matched_S32_completed_action":44204827220,
      "matched_S33_completed_action":44207183539,
      "S34_teacher_anti_collapse_coefficient":S34_REJECTED_ANCHOR_COEFF,
      "S34_teacher_REJECTED_floor_delta":S34_REJECTED_TEACHER_FLOOR_DELTA,
      "S34_rank_true_above_false_coefficient":S34_TRUE_VS_FALSE_PAIR_RANK_COEFF,
      "S34_rank_true_above_false_margin":S34_TRAIN_RANK_MARGIN,
      "strict_same_run_relation_logits_preserved":True,
      "synthetic_development_only_no_candidate_selection":True,
      "preregistered_fixed_loss_margin":MARGIN,
      "preregistered_fixed_hardnegative_margin":HARD_NEGATIVE_MARGIN,
      "preregistered_positive_coefficient":TRAIN_POSITIVE_COEFF,
      "preregistered_hard_negative_coefficient":TRAIN_HARD_NEGATIVE_COEFF,
      "S34_accepted_RECIPIENT_precision_coefficient":TRAIN_ACCEPTED_RECIPIENT_COEFF,
      "S34_top_false_pair_precision_coefficient":TRAIN_FALSE_PAIR_TOPK_COEFF,
      "S34_nonrejected_margin":NOT_REJECTED_MARGIN,
      "S34_false_pair_margin":FALSE_PAIR_MARGIN,
      "S34_top_false_pairs_per_sentence":FALSE_PAIR_TOPK_PER_TEXT,
      "matched_S31_completed_action":44203704916,
      "S31_is_prior_development_data_NOT_independent":True,
      "same_architecture_as_S26_zero_new_weights":True,
      "actual_paid_original_S26_total_FP16":PAID_TOTAL,
      "actual_paid_new_S34_total_FP16":PAID_TOTAL,
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
    destination=args.out/"NRA3_S34_REAL43M_GT_TRAIN256_8FOLD_RECALL_PRESERVE_PRECISION_ABLATION.json"
    destination.write_text(
      json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("S34_REAL43M_PAIRED_PREPOSED",
      json.dumps({
          "S26_TP":preposed["S26_held_REJECTED"]["TP"],
          "S31_archived_TP":preposed["S31_held_REJECTED"]["TP"],
          "S31_fresh_refit_TP":preposed["fresh_S31_held_REJECTED"]["TP"],
          "S31_fresh_refit_reln_TP":preposed["fresh_S31_held_5heads"]["event_relation"]["TP"],
          "S32_TP":preposed["S32_held_REJECTED"]["TP"],
          "S33_TP":preposed["S33_held_REJECTED"]["TP"],
          "S34_TP":preposed["S34_held_REJECTED"]["TP"],
          "S33_FP":preposed["S33_held_REJECTED"]["FP"],
          "S34_FP":preposed["S34_held_REJECTED"]["FP"],
          "S34_gold_role_correct":preposed["S34_held_role_rejection_joint"]["correct_RECIPIENT_at_gold_rejected_pairs"],
          "S34_joint_correct":preposed["S34_held_role_rejection_joint"]["jointly_correct_at_gold_rejected_pairs"],
          },sort_keys=True),flush=True)
    print("S34_ORIGINAL43M_GT_TEACHER_ANCHOR_NO_EVAL_RELATION_EXACT_PRESERVED",flush=True)


if __name__=="__main__":
    main()
