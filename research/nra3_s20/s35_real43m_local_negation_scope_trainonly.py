"""S35: genuine frozen43M+GT TRAIN-only local negation-scope decoder ablation.

The S34 teacher-margin paid FP16 model retained 30/32 rare REJECTED true
positives but hallucinated 20 false positives, mainly in held constructions
26 (REJECTED after "но не") and 28 (NEGATED predicate, NOT a rejected
recipient).  This preregistered *decoding-only* structural constraint
requires BOTH raw word-local [previous negative particle] and [dative
suffix likely] at the CANDIDATE RECIPIENT target token before the decoder
may emit REJECTED. It never uses the TRAIN gold, group number, template
identity, subject names, or event labels at inference. Other logits and
all paid FP16 weights remain bytewise untouched.

This is a tiny rule-based decoder proven ONLY on already-inspected
synthetic TRAIN templates, not evidence of natural Russian generalization.
A phrase like 'никому не передала' need not satisfy this local scope.
No independent EVAL128/32, no 616J7 replay and no LIVE changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import torch

from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import GraphLogits, PaidSemanticEventGraph
from research.nra3_s19.s19h_true43m_train9_loo_diagnostics import take
from research.nra3_s19.s19g_train9_real43m_graph import S19D_FULL_PHYSICAL_SHA
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    FROZEN_S22_TRAIN256_SHA256,PATTERNS,splits,
)
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    EventGroundedInversionGraph,PREV_NEG,LIKELY_DATIVE,
)
from research.nra3_s20.s26_train256_event_grounded_matched40 import (
    load_physical_training_source,new_graph_model,
    rejected_specific_counts,source_only_counts,SEED,PAID_TOTAL,
)
from research.nra3_s20.s34_trainonly_teacher_rejected_anti_collapse import (
    fixed_train_S34,FOLDS,TARGET_PREPOSED,PREDECLARED_STEPS,
    _held_rejected_role_recall,
)
from research.nra3_s20.s25_real43m_rejected_raw_cue_alignment_audit import cue_audit
from research.nra3_s20.s28_real43m_trainonly_determinism_probe import (
    canonical_paid_semantic_fp16_sha,
)
from research.nra3_s20.s29_archived_original43m_canonical_fp16_identity import (
    FROZEN_ARCHIVES,
)

S35_SCOPE_RULE="REJECTED_only_if_target_has_previous_NE_and_likely_DATIVE"
S35_FIXED_TARGET_GROUPS=(26,28)
S35_PRIOR_S34_RUN=44208569216
S35_S34_REPORT_STAGE="NRA3_S34_TRAIN_ONLY_S31_TEACHER_ANTICOLLAPSE_HEADONLY_12384FP16"


def require(ok,message):
    if not bool(ok):
        raise ValueError("S35 original43M+GT local negation decode TRAIN research: "+message)


def apply_local_ne_dative_scope(out:GraphLogits,raw:torch.Tensor,
                                valid:torch.Tensor)->GraphLogits:
    """Pure input-only postprocessor of REJECTED class, no TRAIN gold.

    Candidate pair is (event predicate index, recipient candidate index).
    Dative and immediately preceding НЕ belong to the recipient candidate
    itself. Does NOT encode labels, construction IDs or manual truth.
    The only affected prediction logit is the 3rd polarity class when
    the local raw18 cue is absent; original role, relation, coref, event,
    all non-rejected polarity logits remain numerically identical.
    """
    require(raw.ndim==3 and raw.shape[-1]==338 and valid.ndim==2
            and raw.shape[:2]==valid.shape and out.argument_polarity.shape[-1]==3,
            "true 338D raw grammar / label-free input geometry differs")
    has_ne=raw[...,320+PREV_NEG]>0.5
    likely_dative=raw[...,320+LIKELY_DATIVE]>0.5
    target_can_be_rejected=(valid & has_ne & likely_dative)
    can_use=valid.unsqueeze(2) & target_can_be_rejected.unsqueeze(1)
    b,w=valid.shape
    legal=can_use & ~torch.eye(w,dtype=torch.bool,device=valid.device).unsqueeze(0)
    amended=out.argument_polarity.clone()
    amended[...,2]=amended[...,2].masked_fill(~legal,-1e6)
    return GraphLogits(out.event_status,out.role,amended,
                       out.coreference,out.event_relation)


def verify_logits_unmodified(old:GraphLogits,new:GraphLogits,raw,valid):
    for head in ("event_status","role","coreference","event_relation"):
        require(torch.equal(getattr(old,head),getattr(new,head)),
                "structural decoding changed protected "+head+" logits")
    require(torch.equal(old.argument_polarity[...,:2],
                        new.argument_polarity[...,:2]),
            "decoder changed class NONE or ACCEPTED")
    # Local structural mask may ONLY alter REJECTED logits outside scope.
    can=(valid & (raw[...,320+PREV_NEG]>0.5)
         & (raw[...,320+LIKELY_DATIVE]>0.5))
    pair=can.unsqueeze(1).expand_as(old.argument_polarity[...,2])
    require(torch.equal(old.argument_polarity[...,2][pair],
                        new.argument_polarity[...,2][pair]),
            "decoder modified a legitimate local NEG+DATIVE candidate")
    require(bool((new.argument_polarity[...,2][~pair]==-1e6).all()),
            "out-of-scope REJECTED logits must be blocked")
    return True


def _sum_metrics(records,key):
    support=sum(r[key]["held_REJECTED_positive_support"] for r in records)
    tp=sum(r[key]["TP"] for r in records)
    fp=sum(r[key]["FP"] for r in records)
    fn=sum(r[key]["FN"] for r in records)
    require(support==32 and fn==support-tp,
            "known 4×8 rare-positive support or confusion counts inconsistent")
    return {"TP":tp,"FP":fp,"FN":fn,
            "support":support,"recall":tp/support,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def main():
    p=argparse.ArgumentParser()
    for a in ("features","gold","manifest","s19d-parent","s22d-model",
              "s26-model","baseline-s34-report","out"):
        p.add_argument("--"+a,type=Path,required=True)
    args=p.parse_args()
    require(not args.out.exists(),"never overwrite earlier source artifacts")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    raw,valid,y,src=load_physical_training_source(
        args.features,args.gold,args.manifest)
    require(src["S22_sha_locked_synthetic_TRAIN256_graph_gold"]==
            FROZEN_S22_TRAIN256_SHA256,
            "SHA-locked genuine original43M/GT S22c TRAIN256 source changed")
    observed=cue_audit(raw,valid,y)
    require(observed["source_SHA256"]==FROZEN_S22_TRAIN256_SHA256,
            "S25 authentic raw18 target-cue proof drift")
    original_s34=json.loads(args.baseline_s34_report.read_text())
    require(original_s34["stage"]==S35_S34_REPORT_STAGE
            and original_s34["original_genuine_frozen43M_GT_source_S22c_sha256"]==
                FROZEN_S22_TRAIN256_SHA256
            and original_s34["matched_S33_completed_action"]==44207183539
            and original_s34["output_gate"]==0
            and not original_s34["candidate_accepted"]
            and len(original_s34["per_fold_8_rows_synthetic_TRAIN_internal_NOT_independent"])==8,
            "prior S34 genuinely completed TRAIN-only report unverified")
    prior={z["fold"]:z for z in
           original_s34["per_fold_8_rows_synthetic_TRAIN_internal_NOT_independent"]}
    require(set(prior)==set(FOLDS),"prior S34 fold set changed")
    b22=args.s22d_model.read_bytes()
    b26=args.s26_model.read_bytes()
    require(hashlib.sha256(b22).hexdigest()==FROZEN_ARCHIVES["S22d"]["sha"]
            and hashlib.sha256(b26).hexdigest()==FROZEN_ARCHIVES["S26"]["sha"],
            "immutable original archived true43M paid FP16 integrity wrong")
    parent=args.s19d_parent.read_bytes()
    require(hashlib.sha256(parent).hexdigest()==S19D_FULL_PHYSICAL_SHA,
            "original frozen paid 9332 FP16 parent changed")
    archive_s22d=PaidSemanticEventGraph.from_physical(b22)
    archive_s26=EventGroundedInversionGraph.from_physical(b26)
    sha_old=canonical_paid_semantic_fp16_sha(archive_s22d)
    sha_new=canonical_paid_semantic_fp16_sha(archive_s26)
    require(sha_old==
            "35102946886b9f35d86f57f791b6ab6001b813e4f79397cff8e4d25c04e7a538"
            and sha_new==
            "02d4403ffbce01632d54d66349a01c3f10b81650d26d9540bb86f495cac62807",
            "original true43M paid numerical FP16 identity drift")
    all_splits=splits()
    require(len(all_splits)==40 and FOLDS==(32,33,34,35,36,15,24,30)
            and TARGET_PREPOSED==33 and PAID_TOTAL==12384,
            "precommitted S34 comparison and paid model size drift")
    records=[]
    for fold in FOLDS:
        axis,construction,tr,held=all_splits[fold]
        require(axis=="CONSTRUCTION" and construction==fold-8
                and len(tr)==248 and len(held)==8
                and set(tr).isdisjoint(held),
                "held synthetic TRAIN fold compromised")
        tx,tv,tg=take(raw,tr),take(valid,tr),take(y,tr)
        hx,hv,hg=take(raw,held),take(valid,held),take(y,held)
        fresh_model=new_graph_model(
            PaidContextualEventScope.from_physical(parent),SEED+fold)
        teacher,candidate,payload,train_diag=fixed_train_S34(
            fresh_model,tx,tv,tg)
        require(candidate.output_gate==0
                and not candidate.candidate_accepted
                and not candidate.activation_allowed,
                "research-only model unexpectedly activatable")
        before_sha=canonical_paid_semantic_fp16_sha(candidate)
        with torch.inference_mode():
            original=candidate(hx,hv)
            decoded=apply_local_ne_dative_scope(original,hx,hv)
            verify_logits_unmodified(original,decoded,hx,hv)
            base_r=rejected_specific_counts(original,hg,hv)
            scoped_r=rejected_specific_counts(decoded,hg,hv)
            base_tasks=source_only_counts(original,hg,hv)
            scoped_tasks=source_only_counts(decoded,hg,hv)
            base_roles=_held_rejected_role_recall(original,hg,hv)
            scoped_roles=_held_rejected_role_recall(decoded,hg,hv)
        require(canonical_paid_semantic_fp16_sha(candidate)==before_sha,
                "input-only decoder changed model paid FP16 tensor values")
        for k in ("event","role","coreference","event_relation"):
            require(base_tasks[k]==scoped_tasks[k],
                    "S35 changed paid event/relation/coreference/role metrics")
        require(base_r["held_REJECTED_positive_support"]==
                prior[fold]["S34_held_REJECTED"]["held_REJECTED_positive_support"],
                "prior S34 held TRAIN rare positive source differs")
        # Verify the structural rule does not erase ANY of the known
        # synthetic TRAIN positive gold targets. This is an offline
        # label-only diagnostic, not part of decoding or architecture.
        pos=(hg["argument_polarity_TRAIN_ONLY"]==1)
        ne=(hx[...,320+PREV_NEG]>0.5)
        dat=(hx[...,320+LIKELY_DATIVE]>0.5)
        allowed=(ne & dat).unsqueeze(1).expand_as(pos)
        require(not bool((pos & ~allowed).any()),
                "S35 selected local scope loses true positive REJECTED on TRAIN")
        records.append({
            "fold":fold,"held_construction_index":construction,
            "held_construction_display_only":PATTERNS[construction],
            "real_original43M_GT_source_sha256":FROZEN_S22_TRAIN256_SHA256,
            "original_S34_fresh_paid_FP16_canonical_sha256":before_sha,
            "archived_S34_perfold_canonical_sha256":
                prior[fold]["canonical_S34_paid_FP16_numeric_SHA256"],
            "fresh_S34_canonical_matches_archived":
                before_sha==prior[fold]["canonical_S34_paid_FP16_numeric_SHA256"],
            "historical_S34_metrics_match_fresh":
                base_r==prior[fold]["S34_held_REJECTED"],
            "S34_fresh_held_REJECTED":base_r,
            "S35_local_scope_held_REJECTED":scoped_r,
            "S34_archived_held_REJECTED":prior[fold]["S34_held_REJECTED"],
            "S34_fresh_held_5heads":base_tasks,
            "S35_scope_held_5heads":scoped_tasks,
            "S34_fresh_gold_rejected_pair_roles":base_roles,
            "S35_scope_gold_rejected_pair_roles":scoped_roles,
            "trained_source_side_rows":248,
            "held_source_side_synthetic_TRAIN_rows":8,
            "model_forward_received_only_raw338_valid":True,
            "gold_never_entered_local_scope_decoder":True,
            "physical_candidate_weight_sha_identical_before_after_decoder":True,
            "paid_43M_GT_frozen_and_relation_logits_unchanged":True,
            "original_j7_EVAL_never_replayed":True,
        })
        print("S35_REAL43M_GT_TRAINONLY_FOLD_COMPARISON",
              json.dumps({"fold":fold,"construction":PATTERNS[construction],
                          "S34_rejected_TP":base_r["TP"],
                          "S34_rejected_FP":base_r["FP"],
                          "S35_rejected_TP":scoped_r["TP"],
                          "S35_rejected_FP":scoped_r["FP"],
                          "S31_relation_TP_unchanged":
                            base_tasks["event_relation"]["TP"],
                          "cross_run_archive_same":
                            before_sha==prior[fold]["canonical_S34_paid_FP16_numeric_SHA256"]},
                         sort_keys=True),flush=True)
    rare_folds=[x for x in records if 24<=x["held_construction_index"]<=27]
    base_total=_sum_metrics(rare_folds,"S34_fresh_held_REJECTED")
    new_total=_sum_metrics(rare_folds,"S35_local_scope_held_REJECTED")
    original_fp_in_26_28=sum(z["S34_fresh_held_REJECTED"]["FP"] for z in records
                             if z["held_construction_index"] in S35_FIXED_TARGET_GROUPS)
    revised_fp_in_26_28=sum(z["S35_local_scope_held_REJECTED"]["FP"] for z in records
                            if z["held_construction_index"] in S35_FIXED_TARGET_GROUPS)
    report={
        "stage":"NRA3_S35_GENUINE_ORIGINAL43M_GT_LOCAL_NE_DATIVE_DECODE_FPFN_TRAINONLY",
        "original_real43M_GT_CPU_train256_graph_SHA":FROZEN_S22_TRAIN256_SHA256,
        "actual_original_S22d_archived_canonical_FP16_SHA":sha_old,
        "actual_original_S26_archived_canonical_FP16_SHA":sha_new,
        "original_S34_completed_run":S35_PRIOR_S34_RUN,
        "S35_rule_declared_before_running":S35_SCOPE_RULE,
        "rule_only_token_local_NE_and_DATIVE_not_pattern_id":True,
        "all_32_synthetic_TRAIN_positive_recipient_target_cues_present":True,
        "underlying_original_12384_FP16_paid_parameters_unchanged":True,
        "output_is_decoder_diagnostic_NOT_new_model_or_adopted_candidate":True,
        "eight_previously_inspected_synthetic_TRAIN_development_folds_only":list(FOLDS),
        "all_8_fold_details":records,
        "matched_synthetic_rare_four_fold_S34_fresh":base_total,
        "matched_synthetic_rare_four_fold_S35_scope":new_total,
        "target_26_28_S34_false_positives":original_fp_in_26_28,
        "target_26_28_S35_false_positives":revised_fp_in_26_28,
        "S35_scope_condition_no_loss_on_32_known_synthetic_TRAIN_positives":
            new_total["TP"]==base_total["TP"],
        "S35_scope_condition_fewer_26_28_FP":
            revised_fp_in_26_28<original_fp_in_26_28,
        "no_hyperparameter_or_checkpoint_selection_from_held_source":True,
        "not_generalized_language_negation_scope":True,
        "no_independent_eval32_128_or_old_616J7_replay":True,
        "no_active_43M_GT_NRA2_LIVE_Render_Supabase_Jessica_APK_changes":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    args.out.mkdir(parents=True,exist_ok=False)
    file=args.out/"NRA3_S35_GENUINE43M_GT_TRAIN256_LOCAL_NEGATION_DECODE_ABLATION.json"
    file.write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("S35_GENUINE43M_GT_PAIRED_TRAIN_ONLY_TOTALS",
          json.dumps({"S34_fresh":base_total,"S35_structural_decode":new_total,
                      "FP_26_28_before":original_fp_in_26_28,
                      "FP_26_28_after":revised_fp_in_26_28},sort_keys=True),flush=True)
    print("S35_TRAIN_ONLY_DIAGNOSTIC_FINISHED_NO_DECODER_ADOPTION",flush=True)


if __name__=="__main__":
    main()
