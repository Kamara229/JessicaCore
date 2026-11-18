"""S37: TRAIN-only source-verb shape gate after S35 local НЕ+dative scope.

S36 true43M+GT pair audit #44211688841 proves all 7 residual FP
in construction26 originate from conjunction "но" and correctly point
to the rejected recipient. This fixed input-only hypothesis tests whether
the SOURCE of a REJECTED pair has a raw18 past/future finite-verb shape.
It NEVER sees gold, annotated event source, held group IDs, lexical lists,
the connector "но" or any hard-coded construction during inference.

No training change: original S34 12384-paid-FP16 HEAD refitted on fixed
TRAIN248 side, immutable original43M+GT CPU word338 features. Held eight
synthetic TRAIN rows per fold are already inspected development data.
The rule does not establish general Russian grammar or independent EVAL.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from research.nra3_s18.s18c3_raw_russian_grammar_cues import FEATURES
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import GraphLogits,PaidSemanticEventGraph
from research.nra3_s19.s19g_train9_real43m_graph import S19D_FULL_PHYSICAL_SHA
from research.nra3_s19.s19h_true43m_train9_loo_diagnostics import take
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    FROZEN_S22_TRAIN256_SHA256,PATTERNS,splits,
)
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s26_train256_event_grounded_matched40 import (
    load_physical_training_source,new_graph_model,
    rejected_specific_counts,source_only_counts,SEED,PAID_TOTAL,
)
from research.nra3_s20.s34_trainonly_teacher_rejected_anti_collapse import (
    fixed_train_S34,FOLDS,
)
from research.nra3_s20.s35_real43m_local_negation_scope_trainonly import (
    apply_local_ne_dative_scope,verify_logits_unmodified,_sum_metrics,
)
from research.nra3_s20.s28_real43m_trainonly_determinism_probe import (
    canonical_paid_semantic_fp16_sha,
)
from research.nra3_s20.s29_archived_original43m_canonical_fp16_identity import FROZEN_ARCHIVES

PAST=FEATURES.index("past_verb_shape_possible")
FUTURE=FEATURES.index("finite_future_verb_shape_possible")
S37_RULE="source_has_raw18_past_or_finite_future_verb_shape"
S35_RUN=44210352020
S36_RUN=44211688841
S35_STAGE="NRA3_S35_GENUINE_ORIGINAL43M_GT_LOCAL_NE_DATIVE_DECODE_FPFN_TRAINONLY"


def require(ok,why):
    if not bool(ok):
        raise ValueError("S37 genuine43M+GT TRAIN-only source-verb diagnostic: "+why)


def apply_raw_source_verb_scope(out:GraphLogits,raw:torch.Tensor,valid:torch.Tensor)->GraphLogits:
    """Strict original raw18 source-token verb shape, no oracle/event labels."""
    b,w=valid.shape
    require(raw.shape==(b,w,338) and valid.dtype==torch.bool
            and out.argument_polarity.shape==(b,w,w,3),
            "invalid true CPU word338 and word-pair shape")
    is_verb=(valid & ((raw[...,320+PAST]>0.5) |
                       (raw[...,320+FUTURE]>0.5)))
    eligible=(is_verb.unsqueeze(2) & valid.unsqueeze(1)
              & ~torch.eye(w,dtype=torch.bool,device=valid.device).unsqueeze(0))
    revised=out.argument_polarity.clone()
    revised[...,2]=revised[...,2].masked_fill(~eligible,-1e6)
    return GraphLogits(out.event_status,out.role,revised,
                       out.coreference,out.event_relation)


def assert_only_ineligible_rejected_masked(before,after,raw,valid):
    b,w=valid.shape
    for head in ("event_status","role","coreference","event_relation"):
        require(torch.equal(getattr(before,head),getattr(after,head)),
                "S37 changed protected paid "+head+" logits")
    require(torch.equal(before.argument_polarity[...,:2],
                        after.argument_polarity[...,:2]),
            "S37 changed non-rejected polarity")
    source=(valid & ((raw[...,320+PAST]>0.5)|(raw[...,320+FUTURE]>0.5)))
    eligible=(source.unsqueeze(2) & valid.unsqueeze(1)
              & ~torch.eye(w,dtype=torch.bool,device=valid.device).unsqueeze(0))
    require(torch.equal(before.argument_polarity[...,2][eligible],
                        after.argument_polarity[...,2][eligible]),
            "legitimate raw source-verb REJECTED score changed")
    require(bool((after.argument_polarity[...,2][~eligible]==-1e6).all()),
            "ineligible source logits were not suppressed")
    return True


def main():
    p=argparse.ArgumentParser()
    for item in ("features","gold","manifest","s19d-parent","s22d-model",
                 "s26-model","baseline-s35-report","out"):
        p.add_argument("--"+item,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"no overwrites of immutable research artifacts")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    raw,valid,gold,manifest=load_physical_training_source(
        a.features,a.gold,a.manifest)
    require(manifest["S22_sha_locked_synthetic_TRAIN256_graph_gold"]==
            FROZEN_S22_TRAIN256_SHA256,"real original43M GT source SHA drift")
    original_parent=a.s19d_parent.read_bytes()
    require(hashlib.sha256(original_parent).hexdigest()==S19D_FULL_PHYSICAL_SHA,
            "original 9332 FP16 paid parent mutated")
    b22=a.s22d_model.read_bytes()
    b26=a.s26_model.read_bytes()
    require(hashlib.sha256(b22).hexdigest()==FROZEN_ARCHIVES["S22d"]["sha"]
            and hashlib.sha256(b26).hexdigest()==FROZEN_ARCHIVES["S26"]["sha"],
            "original real43M archived model full physical SHA drift")
    require(canonical_paid_semantic_fp16_sha(PaidSemanticEventGraph.from_physical(b22))==
            "35102946886b9f35d86f57f791b6ab6001b813e4f79397cff8e4d25c04e7a538"
            and canonical_paid_semantic_fp16_sha(EventGroundedInversionGraph.from_physical(b26))==
            "02d4403ffbce01632d54d66349a01c3f10b81650d26d9540bb86f495cac62807",
            "real archived numeric FP16 canonical provenance changed")
    historical=json.loads(a.baseline_s35_report.read_text())
    require(historical["stage"]==S35_STAGE
            and historical["original_real43M_GT_CPU_train256_graph_SHA"]==
                FROZEN_S22_TRAIN256_SHA256
            and historical["output_gate"]==0
            and not historical["candidate_accepted"]
            and len(historical["all_8_fold_details"])==8,
            "S35 completed research-only source metrics mismatch")
    previous={z["fold"]:z for z in historical["all_8_fold_details"]}
    require(set(previous)==set(FOLDS)
            and FOLDS==(32,33,34,35,36,15,24,30)
            and PAID_TOTAL==12384,"predeclared exact eight folds and paid FP16 size changed")
    folds=splits()
    require(len(folds)==40,"immutable S22 synthetic TRAIN geometry changed")
    results=[]
    for fold in FOLDS:
        axis,group,tr,held=folds[fold]
        require(axis=="CONSTRUCTION" and group==fold-8
                and len(tr)==248 and len(held)==8
                and set(tr).isdisjoint(held),
                "TRAIN-only construction split leakage")
        trained,new_paid,_,lossdiag=fixed_train_S34(
            new_graph_model(
                PaidContextualEventScope.from_physical(original_parent),
                SEED+fold),
            take(raw,tr),take(valid,tr),take(gold,tr))
        require(new_paid.output_gate==0 and not new_paid.candidate_accepted
                and not new_paid.activation_allowed,
                "unexpected S34 research model activation")
        hx,hv,hg=take(raw,held),take(valid,held),take(gold,held)
        initial_digest=canonical_paid_semantic_fp16_sha(new_paid)
        with torch.inference_mode():
            actual=new_paid(hx,hv)
            s35=apply_local_ne_dative_scope(actual,hx,hv)
            verify_logits_unmodified(actual,s35,hx,hv)
            gated=apply_raw_source_verb_scope(s35,hx,hv)
            assert_only_ineligible_rejected_masked(s35,gated,hx,hv)
            prior_metric=rejected_specific_counts(s35,hg,hv)
            new_metric=rejected_specific_counts(gated,hg,hv)
            t35=source_only_counts(s35,hg,hv)
            t37=source_only_counts(gated,hg,hv)
            for task in ("event","directed_role","coreference","event_relation"):
                require(t35[task]==t37[task],
                        "S37 changed event/role/relation/coreference predictions")
            legal=(hv.unsqueeze(1)&hv.unsqueeze(2)) & (
                ~torch.eye(hv.shape[1],dtype=torch.bool).unsqueeze(0))
            gold_positive=legal & (hg["argument_polarity_TRAIN_ONLY"]==1)
            possible_source=((hx[...,320+PAST]>0.5)|
                             (hx[...,320+FUTURE]>0.5))
            source_eligible=possible_source.unsqueeze(2).expand_as(gold_positive)
            annotated_true_positive_outside_verb_source=int(
                (gold_positive & ~source_eligible).sum())
        require(canonical_paid_semantic_fp16_sha(new_paid)==initial_digest,
                "S37 source-only rule changed actual 12384 paid FP16 tensors")
        require(prior_metric["held_REJECTED_positive_support"]==
                previous[fold]["S35_local_scope_held_REJECTED"]["held_REJECTED_positive_support"],
                "S35 archived vs current synthetic held TRAIN source inconsistent")
        results.append({
            "fold":fold,"held_construction_index":group,
            "held_construction_display_only":PATTERNS[group],
            "S35_same_run_held_REJECTED":prior_metric,
            "S37_verb_source_held_REJECTED":new_metric,
            "S35_archived_held_REJECTED":
                previous[fold]["S35_local_scope_held_REJECTED"],
            "S35_archived_numeric_fp16_same_as_fresh":
                previous[fold]["original_S34_fresh_paid_FP16_canonical_sha256"]==
                initial_digest,
            "S35_same_run_5heads":t35,"S37_verb_scope_5heads":t37,
            "S34_fresh_FP16_numeric_sha256":initial_digest,
            "TRAIN_gold_REJECTED_true_pairs_outside_raw_source_verb":
                annotated_true_positive_outside_verb_source,
            "TRAIN_side_rows":248,"held_synthetic_TRAIN_only_rows":8,
            "only_original_raw18_source_verb_shape_used_at_inference":True,
            "no_model_FP16_or_protected_graph_logits_changed":True,
            "source_labels_only_for_offline_report":True,
        })
        print("S37_TRUE43M_GT_TRAIN_ONLY_FOLD",
              json.dumps({"fold":fold,"pattern":PATTERNS[group],
                          "S35_TP":prior_metric["TP"],"S35_FP":prior_metric["FP"],
                          "S37_TP":new_metric["TP"],"S37_FP":new_metric["FP"],
                          "true_positives_outside_verb_shape":
                           annotated_true_positive_outside_verb_source},sort_keys=True),flush=True)
    rare=[z for z in results if 24<=z["held_construction_index"]<=27]
    before=_sum_metrics(rare,"S35_same_run_held_REJECTED")
    after=_sum_metrics(rare,"S37_verb_source_held_REJECTED")
    target_before=sum(z["S35_same_run_held_REJECTED"]["FP"]
                      for z in results if z["held_construction_index"]==26)
    target_after=sum(z["S37_verb_source_held_REJECTED"]["FP"]
                     for z in results if z["held_construction_index"]==26)
    other_before=sum(z["S35_same_run_held_REJECTED"]["FP"]
                     for z in results if z["held_construction_index"] not in (24,25,26,27))
    other_after=sum(z["S37_verb_source_held_REJECTED"]["FP"]
                    for z in results if z["held_construction_index"] not in (24,25,26,27))
    report={
      "stage":"NRA3_S37_TRUE43M_GT_RAW18_SOURCE_VERB_ELIGIBILITY_REJECTED_TRAIN8FOLD",
      "original_real43M_GT_s22c_TRAIN256_SHA256":FROZEN_S22_TRAIN256_SHA256,
      "S36_completed_original_pair_audit_run":S36_RUN,
      "S35_completed_original_8fold_run":S35_RUN,
      "rule_predeclared":S37_RULE,
      "source_morphological_cues_only_not_predicted_or_gold_event_type":True,
      "not_blacklisting_connector_NO_or_construction26":True,
      "same_original_12384_paid_FP16_params_preserved":True,
      "all_eight_PREVIOUSLY_EXAMINED_synthetic_TRAIN_folds":list(FOLDS),
      "folds":results,
      "rare_four_folds_S35_same_run":before,
      "rare_four_folds_S37_verb_scope":after,
      "construction26_before_FP":target_before,
      "construction26_after_FP":target_after,
      "other_construction_control_FP_S35":other_before,
      "other_construction_control_FP_S37":other_after,
      "new_scope_keeps_all_earlier_synthetic_TRAIN_true_REJECTED":after["TP"]==before["TP"],
      "new_scope_reduces_synthetic_TRAIN_FP":after["FP"]<before["FP"],
      "new_scope_no_new_FP_on_controls":other_after<=other_before,
      "all_synthetic_held_gold_REJECTED_pairs_have_raw_source_verb_shape":
          all(z["TRAIN_gold_REJECTED_true_pairs_outside_raw_source_verb"]==0 for z in results),
      "all_event_role_coref_relation_logits_bitwise_unchanged":True,
      "label_blind_model_and_decoder":True,
      "not_generalized_Russian_PoS_morphology":True,
      "not_independent_EVAL_or_candidate_validation":True,
      "original_616J7_bpp_CB59_not_tested":True,
      "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"NRA3_S37_REAL43M_GT_TRAIN256_8FOLD_SOURCE_VERB_GATE.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8")
    print("S37_TRUE43M_GT_SYNTHETIC_TRAIN_ONLY_TOTAL",
          json.dumps({"S35":before,"S37":after,"FP26_before":target_before,
                      "FP26_after":target_after,"controls_before":other_before,
                      "controls_after":other_after},sort_keys=True),flush=True)
    print("S37_TRAIN_ONLY_NO_MODEL_PROMOTION_NO_GOLD_INFERENCE",flush=True)


if __name__=="__main__":
    main()
