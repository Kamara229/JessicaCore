"""S42 genuine physical S40 rerun + sparse graph matching vs S41 TRAIN80.

Experimental research-only comparison using identical 8 lexical and 10
construction TRAIN-grouped folds. All held templates and S41 FPs were
already inspected, so this is development analysis, NOT independent
evaluation. NO S38 challenge text is imported, forwarded, or scored.
The original 43M Calibri/GT and true NRA2 weights are not loaded.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from research.nra3_s20.s39_new80_trainonly_scope_graph_source import (
    build_train80,validate_source,partitions,source_digest,
)
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import model_weights_digest
from research.nra3_s20.s40_fit_train80_compositional_graph_18fold import (
    train_one_fold,build_batch,TRAIN_STEPS,TRAIN_LR,
)
from research.nra3_s20.s41_typed_graph_edge_completion import (
    structurally_complete_REJECTED,S41_ORIGINAL_S40_RUN,
)
from research.nra3_s20.s42_sparse_predicate_recipient_matching import (
    matching_links_from_s40,verify_readonly_matching,
    S42_LINK_FLOOR_LOGIT,S42_RULE,S42_BASELINE_S41_RUN,
)

def require(ok,why):
    if not bool(ok):raise ValueError("S42 S40 same-run TRAIN80 research matching: "+why)

def score_graph_predictions(outputs,valid,legal,ev_gold,quote_gold,link_gold):
    with torch.inference_mode():
        pred=(outputs.rejected_logits>=0.)&legal
        ref=(link_gold>.5)&legal
        tp=int((pred&ref).sum())
        fp=int((pred&~ref&legal).sum())
        fn=int((~pred&ref).sum())
        event_pred=(outputs.event_logits>=0.)&valid
        event_ref=(ev_gold>.5)&valid
        quote_pred=(outputs.quote_logits>=0.)&valid
        quote_ref=(quote_gold>.5)&valid
    return {"TP":tp,"FP":fp,"FN":fn,
            "reference_positive_support":int(ref.sum()),
            "event_TP":int((event_pred&event_ref).sum()),
            "event_FP":int((event_pred&~event_ref&valid).sum()),
            "event_FN":int((~event_pred&event_ref).sum()),
            "quote_TP":int((quote_pred&quote_ref).sum()),
            "quote_FN":int((~quote_pred&quote_ref).sum())}

def category_taxonomy(pred,valid,legal,ev_gold,links_gold,rows):
    """Offline ONLY. Source annotations used after frozen model+decoder."""
    predicted=(pred.rejected_logits>=0.)&legal
    positive=(links_gold>.5)&legal
    diag={"FP_non_gold_source":0,"FP_gold_source_wrong_recipient":0,
          "FP_negative_sentence":0,"FP_other":0,"FP_total":0,
          "derived_event_source_TP":0,"derived_event_source_FP":0,
          "derived_event_source_FN":0,"true_predicate_head_TP":0,
          "true_predicate_head_FP":0,"true_predicate_head_FN":0}
    token_rows=[]
    for batch,row in enumerate(rows):
        n=int(valid[batch].sum())
        original_source=row.source_predicate_token_index0
        original_target=row.potential_recipient_token_index0
        matches=torch.nonzero(predicted[batch,:n,:n],as_tuple=False).tolist()
        truth_pairs=torch.nonzero(positive[batch,:n,:n],as_tuple=False).tolist()
        pred_sources={a for a,b in matches}
        true_source_mask={int(x) for x in torch.nonzero(
            ev_gold[batch,:n]>.5,as_tuple=False).flatten()}
        for i,j in matches:
            if bool(positive[batch,i,j]):continue
            diag["FP_total"]+=1
            if not truth_pairs:
                diag["FP_negative_sentence"]+=1
            elif i not in true_source_mask:
                diag["FP_non_gold_source"]+=1
            elif j!=original_target:
                diag["FP_gold_source_wrong_recipient"]+=1
            else:
                diag["FP_other"]+=1
        actual_head={int(x) for x in torch.nonzero(
            pred.event_logits[batch,:n]>=0.,as_tuple=False).flatten()}
        diag["derived_event_source_TP"]+=len(pred_sources&true_source_mask)
        diag["derived_event_source_FP"]+=len(pred_sources-true_source_mask)
        diag["derived_event_source_FN"]+=len(true_source_mask-pred_sources)
        diag["true_predicate_head_TP"]+=len(actual_head&true_source_mask)
        diag["true_predicate_head_FP"]+=len(actual_head-true_source_mask)
        diag["true_predicate_head_FN"]+=len(true_source_mask-actual_head)
        token_rows.append({
            "TRAIN_only_text":row.text,
            "source_lexical_family_for_OFFLINE_report_only":row.family,
            "source_construction_group_for_OFFLINE_report_only":row.pattern,
            "reference_is_rejected_OFFLINE_ONLY":row.rejected_recipient,
            "reference_source_OFFLINE_ONLY":original_source,
            "reference_target_OFFLINE_ONLY":original_target,
            "predicted_sparse_directed_index_pairs":matches,
            "reference_directed_index_pairs_OFFLINE_ONLY":truth_pairs,
            "predicted_original_event_head_source_indices":sorted(actual_head),
            "derived_from_sparse_links_source_indices":sorted(pred_sources),
        })
    return diag,token_rows

def summarize(grouped):
    tp=sum(r["TP"] for r in grouped)
    fp=sum(r["FP"] for r in grouped)
    fn=sum(r["FN"] for r in grouped)
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "not_independent_EVAL":True}

def main():
    p=argparse.ArgumentParser()
    for key in ("baseline-s40-report","baseline-s41-report","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"no overwrite of prior research artifacts")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source=validate_source()
    require(source["TRAIN_rows"]==80 and source_digest()==source["sha256"]
            and source["S38_challenge_source_not_imported_or_used"],
            "fixed original S39 synthetic TRAIN80 source changed")
    previous40=json.loads(a.baseline_s40_report.read_text(encoding="utf-8"))
    previous41=json.loads(a.baseline_s41_report.read_text(encoding="utf-8"))
    require(previous40["stage"]=="NRA3_S40_COMPOSITIONAL_S39_TRAIN80_SYNTHETIC_18FOLD_NO_EVAL"
            and previous41["stage"]=="NRA3_S41_TRAIN80_READONLY_GENERIC_STRUCTURAL_EDGE_COMPLETION_18FOLD"
            and previous40["new_TRAIN80_source"]["sha256"]==source["sha256"]
            and previous41["new_TRAIN80_source"]["sha256"]==source["sha256"]
            and previous41["original_archived_S40_completed_run"]==S41_ORIGINAL_S40_RUN
            and not previous40["candidate_accepted"]
            and not previous41["candidate_accepted"]
            and previous40["output_gate"]==previous41["output_gate"]==0,
            "archived physical S40/S41 TRAIN80 provenance invalid")
    old41={r["fold"]:r for r in previous41["all18_strict_synthetic_TRAIN_grouped_folds"]}
    old40={r["fold"]:r for r in previous40["all18_strict_synthetic_TRAIN_grouped_folds"]}
    require(set(old41)==set(old40)==set(range(18)),
            "original S40/S41 physical matched 18 TRAIN groups unavailable")
    data=build_train80()
    all_rows=[]
    for fold,(axis,group,tr,held) in enumerate(partitions()):
        require(len(tr)==(70 if axis=="LEXICAL" else 72)
                and len(held)==(10 if axis=="LEXICAL" else 8)
                and not set(tr).intersection(held),
                "S42 exact original S39/40/41 frozen TRAIN folds drift")
        require(old41[fold]["axis"]==old40[fold]["axis"]==axis
                and old41[fold]["held_group"]==old40[fold]["held_group"]==group,
                "cross-stage baseline TRAIN grouping changed")
        train=[data[i] for i in tr]
        held_data=[data[i] for i in held]
        model,diag=train_one_fold(train,fold)
        original_fp16_sha=model_weights_digest(model)
        x,q,pair,structure,valid,legal,ev,quote_y,link_y=build_batch(held_data)
        # Held labels are used in metrics/taxonomy ONLY AFTER all three
        # predictions were generated from raw text and frozen model.
        with torch.inference_mode():
            original=model(x,q,pair,structure)
            s41=structurally_complete_REJECTED(original,structure)
            s42,match=matching_links_from_s40(original,structure)
            require(verify_readonly_matching(original,s42,match,structure),
                    "S42 structural matching invariant violated")
            m40=score_graph_predictions(original,valid,legal,ev,quote_y,link_y)
            m41=score_graph_predictions(s41,valid,legal,ev,quote_y,link_y)
            m42=score_graph_predictions(s42,valid,legal,ev,quote_y,link_y)
            taxonomy41,_=category_taxonomy(s41,valid,legal,ev,link_y,held_data)
            taxonomy42,pair_examples=category_taxonomy(s42,valid,legal,ev,link_y,held_data)
        require(model_weights_digest(model)==original_fp16_sha,
                "S42 inference mutated same-run physical S40 shadow weights")
        require(all(m40[k]==m41[k]==m42[k] for k in
                    ("event_TP","event_FP","event_FN","quote_TP","quote_FN")),
                "S42 modified original event and quotation neural head outputs")
        require(m40["reference_positive_support"]==
                    m41["reference_positive_support"]==
                    m42["reference_positive_support"]==
                    old41[fold]["held_TRAIN_internal_metrics"]
                    ["directed_rejected_gold_positive_support"],
                "S42 original held TRAIN positive support drift")
        require(m40["TP"]+m40["FN"]==m42["TP"]+m42["FN"],
                "S42 changed true held label support")
        require(taxonomy41["FP_total"]==m41["FP"]
                and taxonomy42["FP_total"]==m42["FP"],
                "S41/S42 FP pair-level taxonomy incomplete")
        record={
            "fold":fold,"axis":axis,"group":group,
            "original_S40_held_TRAIN_metrics":m40,
            "original_S41_held_TRAIN_metrics":m41,
            "S42_one_to_one_matched_held_TRAIN_metrics":m42,
            "S41_false_positive_taxonomy_OFFLINE_ONLY":taxonomy41,
            "S42_false_positive_taxonomy_OFFLINE_ONLY":taxonomy42,
            "S42_concrete_pairs_OFFLINE_ONLY":pair_examples,
            "TRAIN_rows":len(tr),"held_synthetic_TRAIN_rows":len(held),
            "TRAIN_and_held_disjoint":True,
            "physical_S40_shadow_weights_sha256":original_fp16_sha,
            "original_archived_S40_fp16_matches":original_fp16_sha==
                old40[fold]["fresh_model_weights_sha256"],
            "archived_S41_s40_fp16_matches":original_fp16_sha==
                old41[fold]["fresh_model_weights_sha256"],
            "three_head_forward_gold_blind":True,
            "event_quote_network_outputs_bitwise_unmodified":True,
            "no_model_weights_updated_by_decoder":True,
        }
        all_rows.append(record)
        print("S42_ORIGINAL_SHADOW_PHYSICAL_FOLD",
              json.dumps({"fold":fold,"axis":axis,"group":group,
                "S40_TP":m40["TP"],"S40_FP":m40["FP"],
                "S41_TP":m41["TP"],"S41_FP":m41["FP"],
                "S42_TP":m42["TP"],"S42_FP":m42["FP"],
                "derived_pred_source_TP":taxonomy42["derived_event_source_TP"],
                "original_pred_head_TP":taxonomy42["true_predicate_head_TP"]},
                sort_keys=True),flush=True)
    totals={}
    for axis in ("LEXICAL","CONSTRUCTION"):
        r=[rec for rec in all_rows if rec["axis"]==axis]
        totals[axis]={name:summarize([row[key] for row in r])
            for name,key in (("S40","original_S40_held_TRAIN_metrics"),
                             ("S41","original_S41_held_TRAIN_metrics"),
                             ("S42","S42_one_to_one_matched_held_TRAIN_metrics"))}
    target={}
    for group,tag in ((4,"short_passive"),(5,"infinitive_complement")):
        rec=next(r for r in all_rows if r["axis"]=="CONSTRUCTION" and r["group"]==group)
        target[tag]={
            "S40":rec["original_S40_held_TRAIN_metrics"],
            "S41":rec["original_S41_held_TRAIN_metrics"],
            "S42":rec["S42_one_to_one_matched_held_TRAIN_metrics"],
            "S42_event_source_diagnostic_only":
                rec["S42_false_positive_taxonomy_OFFLINE_ONLY"],
        }
    report={
        "stage":"NRA3_S42_SYNTHETIC_TRAIN80_SPARSE_SOURCE_TARGET_MATCHING_18FOLD_ABLATION",
        "source_sha256":source["sha256"],
        "previous_S40_completed_run":S41_ORIGINAL_S40_RUN,
        "previous_S41_completed_run":S42_BASELINE_S41_RUN,
        "source_grammar_rule_fixed_before_results":S42_RULE,
        "precommitted_selected_edge_minimum_logit":S42_LINK_FLOOR_LOGIT,
        "one_to_one_matching_is_hand_chosen_synthetic_prior_not_general_language":True,
        "S38_held32_source_not_loaded_touched_tuned_or_scored":True,
        "same_already_exposed_S39_80TRAIN_internal_folds_used":True,
        "unchanged_S40_learned_event_quote_heads":True,
        "same_run_physical_S40_weights_no_new_training_for_S42":True,
        "per_fold":all_rows,
        "matched_same_run_three_way_internal_only":totals,
        "passive_and_infinitive_construction_diagnostics":target,
        "all_results_synthetic_TRAIN_development_NOT_independent":True,
        "link_derived_source_metrics_NOT_trained_event_head":True,
        "no_real_Calibri43M_GT_or_original_NRA2_weights_loaded":True,
        "old_616J7_independent_EVAL32_128_bpp_CB59_not_tested":True,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    path=a.out/"NRA3_S42_TRAIN80_S40_S41_S42_MATCHED18_SPARSE_EDGE_SOURCE_AUDIT.json"
    path.write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
                    encoding="utf-8")
    print("S42_MATCHED18_TRAIN80_THREE_WAY_TOTALS",
          json.dumps(totals,sort_keys=True),flush=True)
    print("S42_PASSIVE_INFINITIVE_EVENT_HEAD_DIAGNOSTIC",
          json.dumps(target,sort_keys=True),flush=True)
    print("S42_NOT_INDEPENDENT_NO_REAL43M_OR_S38_OR_LIVE",flush=True)

if __name__=="__main__":
    main()
