"""S36: true original43M+GT fold34 *TRAIN-internal* surviving REJECTED FP audit.

S35 #44210352020 SUCCESS showed 7 remaining false REJECTED directed
pairs on the 8 held synthetic TRAIN rows of S22 construction26
'rejected_positive_first'.  Train only the predeclared complementary
248 TRAIN rows with immutable S34 code and frozen S19d CPU graph input;
apply the S35 local-ne+dative output-only scope.  For EVERY leftover
false positive inspect the exact predicate source index, target word,
role, event confidence and original raw cues. Gold is consulted ONLY
after all predictions are computed; it NEVER reaches model or decoder.
No modification to weights, architecture, benchmark or live endpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    FEATURES,raw_surface_grammar,
)
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_train9_real43m_graph import S19D_FULL_PHYSICAL_SHA
from research.nra3_s19.s19g_paid_semantic_graph import PaidSemanticEventGraph
from research.nra3_s19.s19h_true43m_train9_loo_diagnostics import take
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    FROZEN_S22_TRAIN256_SHA256,PATTERNS,build_s22_train256,splits,
)
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    EventGroundedInversionGraph,PREV_NEG,LIKELY_DATIVE,ROLE_RECIPIENT,
)
from research.nra3_s20.s26_train256_event_grounded_matched40 import (
    SEED,PAID_TOTAL,load_physical_training_source,new_graph_model,
    rejected_specific_counts,source_only_counts,
)
from research.nra3_s20.s34_trainonly_teacher_rejected_anti_collapse import (
    fixed_train_S34,
)
from research.nra3_s20.s35_real43m_local_negation_scope_trainonly import (
    apply_local_ne_dative_scope,verify_logits_unmodified,
)
from research.nra3_s20.s28_real43m_trainonly_determinism_probe import (
    canonical_paid_semantic_fp16_sha,
)
from research.nra3_s20.s29_archived_original43m_canonical_fp16_identity import (
    FROZEN_ARCHIVES,
)

FOLD=34
HELD_CONSTRUCTION=26
S35_RUN=44210352020
S35_STAGE="NRA3_S35_GENUINE_ORIGINAL43M_GT_LOCAL_NE_DATIVE_DECODE_FPFN_TRAINONLY"


def require(ok,reason):
    if not bool(ok):
        raise ValueError("S36 real43M+GT fold34 TRAIN-only FP pair audit: "+reason)


def describe_fp_pairs(scoped,raw,valid,gold,held_ids,graph_rows):
    """Never uses TRAIN gold to select logits or to alter predictions.

    Offline diagnostics of held TRAIN-only directed (source,target) edges
    after model/scoped decoder already produced all scores; gold only
    supplies TP/FP/FN and reference labels in the resulting report.
    """
    b,w=valid.shape
    require(len(held_ids)==b==8 and raw.shape==(b,w,338),
            "real CPU word338 and predeclared held fold dimensions differ")
    legal=(valid.unsqueeze(1)&valid.unsqueeze(2)) & (
        ~torch.eye(w,dtype=torch.bool,device=valid.device).unsqueeze(0))
    pred=(scoped.argument_polarity.argmax(-1)==2) & legal
    truth=(gold["argument_polarity_TRAIN_ONLY"]==1) & legal
    rejected_prob=torch.softmax(scoped.argument_polarity,dim=-1)[...,2]
    role_prob=torch.softmax(scoped.role,dim=-1)[...,ROLE_RECIPIENT]
    event_prob=torch.softmax(scoped.event_status,dim=-1)
    counts=Counter()
    examples=[]
    for row in range(b):
        source_id=int(held_ids[row])
        graph=graph_rows[source_id]
        tokenized=raw_surface_grammar(graph.text)
        n=int(valid[row].sum())
        require(n==len(graph.tokens)==len(tokenized),
                "human annotated TRAIN text to real word338 alignment changed")
        for j in range(n):
            require(graph.tokens[j].form==tokenized[j].form and
                    all(int(raw[row,j,320+k])==int(tokenized[j].features[k])
                        for k in range(len(FEATURES))),
                    "physical source raw18 mismatches original input words")
        actual_events={e.predicate-1 for e in graph.events}
        gold_sources=[tuple(int(x) for x in pair) for pair in
                      torch.nonzero(truth[row],as_tuple=False).tolist()]
        predicted_pairs=torch.nonzero(pred[row],as_tuple=False).tolist()
        current=[]
        for src,tgt in predicted_pairs:
            is_tp=bool(truth[row,src,tgt])
            source_is_event=src in actual_events
            target_matches_gold_any=any(tgt==j for _,j in gold_sources)
            src_matches_gold_any=any(src==i for i,_ in gold_sources)
            if is_tp:
                reason="TRUE_REJECTED_EDGE"
                counts["TP"]+=1
            elif target_matches_gold_any and not source_is_event:
                reason="WRONG_NON_EVENT_SOURCE_TO_REAL_REJECTED_TARGET"
                counts["FP_wrong_non_event_source"]+=1
                counts["FP"]+=1
            elif target_matches_gold_any and source_is_event:
                reason="WRONG_EVENT_SOURCE_TO_REAL_REJECTED_TARGET"
                counts["FP_wrong_event_source"]+=1
                counts["FP"]+=1
            elif source_is_event:
                reason="WRONG_TARGET_ON_REAL_EVENT_SOURCE"
                counts["FP_wrong_target"]+=1
                counts["FP"]+=1
            else:
                reason="WRONG_SOURCE_AND_TARGET_NON_EVENT"
                counts["FP_wrong_both"]+=1
                counts["FP"]+=1
            current.append({
                "pair_source_token_index_0based":src,
                "pair_source_token":tokenized[src].form,
                "pair_source_is_annotated_event_OFFLINE_ONLY":source_is_event,
                "pair_source_predicted_event_class_index":
                    int(scoped.event_status[row,src].argmax()),
                "pair_source_predicted_event_confidence":
                    float(event_prob[row,src].max()),
                "pair_target_token_index_0based":tgt,
                "pair_target_token":tokenized[tgt].form,
                "pair_target_previous_NE_raw18":int(raw[row,tgt,320+PREV_NEG]),
                "pair_target_dative_suffix_possible_raw18":
                    int(raw[row,tgt,320+LIKELY_DATIVE]),
                "pair_target_REJECTED_probability":float(rejected_prob[row,src,tgt]),
                "pair_target_predicted_role_index":int(scoped.role[row,src,tgt].argmax()),
                "pair_target_predicted_RECIPIENT_probability":float(role_prob[row,src,tgt]),
                "pair_target_gold_is_real_rejected_OFFLINE_ONLY":
                    is_tp,
                "pair_target_matches_gold_rejected_word_OFFLINE_ONLY":
                    target_matches_gold_any,
                "source_matches_gold_rejected_predicate_OFFLINE_ONLY":
                    src_matches_gold_any,
                "offline_failure_category":reason,
            })
        false_pairs=[pair for pair in current if not pair[
            "pair_target_gold_is_real_rejected_OFFLINE_ONLY"]]
        false_negatives=int((truth[row] & ~pred[row]).sum())
        counts["FN"]+=false_negatives
        examples.append({
            "synthetic_TRAIN_original_source_row_id":source_id,
            "text_TRAIN_only":graph.text,
            "annotated_event_predicate_tokens_OFFLINE_ONLY":[
                {"index0":p,"word":tokenized[p].form} for p in sorted(actual_events)
            ],
            "true_REJECTED_pairs_OFFLINE_ONLY":[
                {"source_token_index0":i,"source":tokenized[i].form,
                 "target_token_index0":j,"target":tokenized[j].form}
                for i,j in gold_sources],
            "predicted_all_REJECTED_pairs":current,
            "false_positive_pairs":false_pairs,
            "false_negative_count":false_negatives,
        })
    require(counts["TP"]+counts["FN"]==8,"true held construction26 8 TP support changed")
    require(counts["FP"]==sum(len(e["false_positive_pairs"]) for e in examples),
            "pair-level false positives differ from raw list")
    return examples,dict(counts)


def main():
    cli=argparse.ArgumentParser()
    for k in ("features","gold","manifest","s19d-parent","s22d-model","s26-model",
              "baseline-s35-report","out"):
        cli.add_argument("--"+k,type=Path,required=True)
    a=cli.parse_args()
    require(not a.out.exists(),"never overwrite prior research outputs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    raw,valid,y,physical_src=load_physical_training_source(
        a.features,a.gold,a.manifest)
    require(physical_src["S22_sha_locked_synthetic_TRAIN256_graph_gold"]==
            FROZEN_S22_TRAIN256_SHA256,"real original43M+GT TRAIN256 raw source changed")
    parent_bytes=a.s19d_parent.read_bytes()
    require(hashlib.sha256(parent_bytes).hexdigest()==S19D_FULL_PHYSICAL_SHA,
            "original frozen S19d parent FP16 physical SHA changed")
    s22d=a.s22d_model.read_bytes()
    s26=a.s26_model.read_bytes()
    require(hashlib.sha256(s22d).hexdigest()==FROZEN_ARCHIVES["S22d"]["sha"]
            and hashlib.sha256(s26).hexdigest()==FROZEN_ARCHIVES["S26"]["sha"],
            "original full S22d/S26 FP16 archive hash not matched")
    require(canonical_paid_semantic_fp16_sha(
                PaidSemanticEventGraph.from_physical(s22d))==
            "35102946886b9f35d86f57f791b6ab6001b813e4f79397cff8e4d25c04e7a538"
            and canonical_paid_semantic_fp16_sha(
                EventGroundedInversionGraph.from_physical(s26))==
            "02d4403ffbce01632d54d66349a01c3f10b81650d26d9540bb86f495cac62807",
            "S29 original archived FP16 canonical numeric SHA changed")
    source_rows=build_s22_train256()
    require(len(source_rows)==256 and PATTERNS[HELD_CONSTRUCTION]=="rejected_positive_first",
            "original synthetic TRAIN pattern source order changed")
    all_folds=splits()
    axis,group,trids,heldids=all_folds[FOLD]
    require(len(all_folds)==40 and axis=="CONSTRUCTION"
            and group==HELD_CONSTRUCTION
            and len(trids)==248 and len(heldids)==8
            and set(trids).isdisjoint(heldids)
            and tuple(heldids)==tuple(26+32*idx for idx in range(8)),
            "original fixed held construction26 fold34 drift")
    previous=json.loads(a.baseline_s35_report.read_text())
    require(previous["stage"]==S35_STAGE
            and previous["original_real43M_GT_CPU_train256_graph_SHA"]==
                FROZEN_S22_TRAIN256_SHA256
            and previous["output_gate"]==0
            and not previous["candidate_accepted"]
            and len(previous["all_8_fold_details"])==8,
            "original completed S35 8-fold synthetic TRAIN report invalid")
    prev_fold=next(z for z in previous["all_8_fold_details"] if z["fold"]==FOLD)
    require(prev_fold["held_construction_index"]==HELD_CONSTRUCTION
            and prev_fold["S35_local_scope_held_REJECTED"]["TP"]==8
            and prev_fold["S35_local_scope_held_REJECTED"]["FP"]==7,
            "S35 previously recorded seven surviving FP not matched")
    x,v,g=take(raw,trids),take(valid,trids),take(y,trids)
    held_x,held_v,held_g=take(raw,heldids),take(valid,heldids),take(y,heldids)
    model=new_graph_model(
        PaidContextualEventScope.from_physical(parent_bytes),SEED+FOLD)
    _,candidate,_,train_diag=fixed_train_S34(model,x,v,g)
    require(candidate.output_gate==0 and not candidate.candidate_accepted
            and not candidate.activation_allowed and PAID_TOTAL==12384,
            "shadow model must remain inactive at 12384 paid FP16")
    sha_before=canonical_paid_semantic_fp16_sha(candidate)
    with torch.inference_mode():
        pred=candidate(held_x,held_v)
        scoped=apply_local_ne_dative_scope(pred,held_x,held_v)
        verify_logits_unmodified(pred,scoped,held_x,held_v)
        previous_unscoped=rejected_specific_counts(pred,held_g,held_v)
        scoped_metrics=rejected_specific_counts(scoped,held_g,held_v)
        full_old=source_only_counts(pred,held_g,held_v)
        full_new=source_only_counts(scoped,held_g,held_v)
        detailed,classification=describe_fp_pairs(
            scoped,held_x,held_v,held_g,heldids,source_rows)
    require(sha_before==canonical_paid_semantic_fp16_sha(candidate),
            "inspection changed original paid FP16 numerics")
    require(previous_unscoped["held_REJECTED_positive_support"]==8
            and scoped_metrics["held_REJECTED_positive_support"]==8,
            "held TRAIN support changed")
    require(scoped_metrics["TP"]==classification.get("TP",0)
            and scoped_metrics["FN"]==classification.get("FN",0)
            and scoped_metrics["FP"]==classification.get("FP",0),
            "pair diagnostic must exactly reconcile with REJECTED TP/FP/FN")
    for task in ("event","directed_role","coreference","event_relation"):
        require(full_old[task]==full_new[task],
                "S36 scope changes protected event/role/relation metrics")
    report={
        "stage":"NRA3_S36_REAL43M_GT_TRAIN_ONLY_SEVEN_FP_DIRECTED_PAIR_AUDIT",
        "source_original_s22c_real43M_GT_256SHA":FROZEN_S22_TRAIN256_SHA256,
        "s35_previous_run_id":S35_RUN,
        "prior_s35_completed_fold34_FP":7,
        "previous_s35_fold34_paid_fp16_numeric_sha":
            prev_fold["original_S34_fresh_paid_FP16_canonical_sha256"],
        "new_S36_fold34_paid_fp16_numeric_sha":sha_before,
        "cross_run_original_S34_fold34_numerically_identical":
            prev_fold["original_S34_fresh_paid_FP16_canonical_sha256"]==sha_before,
        "original_S34_train_diagnostics":train_diag,
        "held_synthetic_TRAIN_source_split":"fold34, pattern26, 248/8; previously examined development labels",
        "source_graph_labels_never_in_model_or_decoder_forward":True,
        "raw18_features_verified_against_exact_source_text":True,
        "original_s34_unscoped_rejected":previous_unscoped,
        "S36_new_same_run_scoped_rejected":scoped_metrics,
        "S35_previous_run_scoped_rejected":
            prev_fold["S35_local_scope_held_REJECTED"],
        "rejected_pair_level_confusion_by_failure_type":classification,
        "all_eight_TRAIN_held_rows_with_directed_pair_details":detailed,
        "original_event_role_coref_relation_numerically_unmodified":True,
        "paid_12384_FP16_numeric_model_unchanged":True,
        "no_held_label_or_text_fed_to_training":True,
        "no_new_model_candidate_or_decision_rule_selected":True,
        "no_independent_EVAL32_128_or_original_616J7_touched":True,
        "LIVE_Render_Supabase_Jessica_APK_Calibri_GT_NRA2_unchanged":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    path=a.out/"NRA3_S36_TRUE43M_GT_TRAIN256_FOLD34_SURVIVING_FP_PAIRS.json"
    path.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                    encoding="utf-8")
    print("S36_REAL43M_GT_PAIR_DIAGNOSTIC_COUNTS",
          json.dumps({"classifications":classification,
                      "current_scoped":scoped_metrics,
                      "previous_S35_FP":7,
                      "cross_run_same":report["cross_run_original_S34_fold34_numerically_identical"]},
                     sort_keys=True),flush=True)
    for e in detailed:
        print("S36_TRAIN_ONLY_PREDICATE_PAIR_ROW",
              json.dumps({"row":e["synthetic_TRAIN_original_source_row_id"],
                          "false_pairs":e["false_positive_pairs"],
                          "gold_REJECTED_pairs":e["true_REJECTED_pairs_OFFLINE_ONLY"]},
                         ensure_ascii=False,sort_keys=True),flush=True)
    print("S36_NO_FP16_WEIGHT_CHANGES_NO_MODEL_PROMOTION",flush=True)


if __name__=="__main__":
    main()
