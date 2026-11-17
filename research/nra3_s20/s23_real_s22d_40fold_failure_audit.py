"""S23 read-only audit of the *actual* S22d physical FP16 40-fold TRAIN report.

Reads the successful real model's physical SHA-verified JSON diagnostics;
makes NO model predictions, changes NO source weights, and never opens EVAL.
The 40-fold set is an internal synthetic TRAIN split, *not* independent EVAL.
Class-specific relation confusion CANNOT be inferred from task-level counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    FROZEN_S22_TRAIN256_SHA256, PATTERNS, build_s22_train256, splits,
    verify_coverage,
)

S22D_ORIGINAL_CI=44192186916
S22D_MODEL_SHA="bd56293aa6ada2411152f0ac528633315731c39de6e74a759704577defbadcbb"
S22C_ORIGINAL_CI=44191302703
S22B_ORIGINAL_CI=44189739539
S22D_ARTIFACT_ID=14376608685
S22D_STAGE="NRA3_S22D_REAL43M_GT_NEW256_SYNTHETIC_TRAIN_ONLY_GROUPED40_GRAPH"
TASKS=("event","directed_role","argument_polarity","coreference","event_relation")
REJECTED="rejected_specific"


def required(ok,description):
    if not ok:
        raise ValueError("S23 physical S22d diagnostics: "+description)


def f1(tp,fp,fn):
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.


def _counts(rows,key):
    return {name:sum(r[key][name] for r in rows)
            for name in ("TP","FP","FN")}


def exact_real_report_audit(report:dict, physical_sha:str):
    """Strict audit of actual physical model and forty precommitted folds."""
    m=report
    required(isinstance(m,dict) and m.get("stage")==S22D_STAGE,
             "wrong S22d stage")
    required(m.get("source_256_synthetic_graph_SHA256")==FROZEN_S22_TRAIN256_SHA256,
             "SHA-sealed S22 TRAIN graph source drift")
    required(physical_sha==S22D_MODEL_SHA==
             m.get("physical_FP16_candidate_SHA256"),"real FP16 candidate SHA differs")
    required(m.get("true_frozen_original43M_GT_from_source_S22b") is True
             and m.get("true_physical_256_SHAlocked_separate_model_vs_gold_S22c") is True
             and m.get("protected_original_616_J7_digest_unchanged_at_source_capture_ONLY") is True,
             "original frozen 43M/GT and protected J7 provenance failed")
    required(m.get("original_S19d_grammar_context_9332_FP16_frozen_sha")==
             "928f735a2018604db2ffe649799778df3805162b79a989a36e282dc3cb61c774",
             "frozen S19d parent SHA drift")
    required(m.get("full_physical_FP16_parameters")==12381
             and m.get("frozen_parent_FP16_parameters")==9332
             and m.get("trainable_graph_FP16_parameters")==3049
             and m.get("optimizer_scope")=="ONLY_SEMANTIC_HEAD_3049_FP16",
             "wrong physical learned adapter or forbidden original weight training")
    for flag in ("no_EVAL32_EVAL128_or_prior_spent_EVAL",
                 "synthetic_TRAIN_template_generalization_only_not_real_language_understanding",
                 "no_active_candidate_616_J7_certification",
                 "no_all_in_bpp_le07_certification",
                 "no_full_CPU_ge90pct_CB59_certification",
                 "original_Calibri_GT_LIVE_Render_Supabase_Jessica_APK_untouched",
                 "NRA2_live_event_graph_decoder_bridge_not_connected"):
        required(m.get(flag) is True,"safety/provenance fail: "+flag)
    required(m.get("output_gate")==0 and
             m.get("candidate_accepted") is False and
             m.get("activation_allowed") is False,
             "never allow S22d model to activate")
    rows=build_s22_train256()
    verify_coverage(rows)
    folds=m.get("all40_fixed_TRAIN_internal_folds")
    required(isinstance(folds,list) and len(folds)==40,
             "missing physical 40-fold TRAIN diagnostics")
    seen=set()
    per_group=[]
    for idx,(fold,original) in enumerate(zip(folds,splits())):
        axis,grp,train,held=original
        required(fold.get("fold_number")==idx and
                 fold.get("axis")==axis and fold.get("held_group_index")==grp and
                 (axis,grp) not in seen,
                 "wrong predeclared lexical/construct fold order")
        seen.add((axis,grp))
        required(fold.get("TRAIN_side_rows")==len(train) and
                 fold.get("held_TRAIN_rows")==len(held) and
                 fold.get("held_rows_never_in_gradient_checkpoint_or_model_selection") is True and
                 fold.get("original_true43M_GT_grammar_context_parent_frozen") is True and
                 fold.get("fixed_optimizer_steps")==240,
                 "data leakage, class grouping or frozen parent violation")
        gold_rel=Counter(link.relation for i in held
                         for link in rows[i].event_relations)
        gold_rej=sum(a.polarity=="REJECTED" for i in held
                     for a in rows[i].arguments)
        expected_count={
            "event":sum(len(rows[i].events) for i in held),
            "directed_role":sum(len(rows[i].arguments) for i in held),
            "argument_polarity":sum(len(rows[i].arguments) for i in held),
            "coreference":sum(len(rows[i].coreferences) for i in held),
            "event_relation":sum(len(rows[i].event_relations) for i in held),
        }
        observed=fold.get("held_TRAIN_metrics")
        required(isinstance(observed,dict) and set(observed)==set(TASKS),
                 "missing original S22d held task statistics")
        for task in TASKS:
            z=observed[task]
            required(all(type(z.get(x)) is int and z[x]>=0
                         for x in ("TP","FP","FN","TRAIN_INTERNAL_held_positive_support")),
                     "negative/missing TP FP FN")
            required(z["TRAIN_INTERNAL_held_positive_support"]==expected_count[task]
                     and z["TP"]+z["FN"]==expected_count[task],
                     "held gold count or tp/fn drift")
        reject=fold.get("REJECTED_specific_held_TRAIN")
        required(isinstance(reject,dict) and all(
            type(reject.get(x)) is int and reject[x]>=0 for x in
            ("TP","FP","FN","held_REJECTED_positive_support")),
            "no REJECTED specific exact counts")
        required(reject["held_REJECTED_positive_support"]==gold_rej
                 and reject["TP"]+reject["FN"]==gold_rej,
                 "REJECTED support/recall inconsistency")
        required(abs(reject["recall"]-
                     (reject["TP"]/gold_rej if gold_rej else 0.))<1e-12
                 and abs(reject["F1"]-f1(reject["TP"],reject["FP"],reject["FN"]))<1e-12,
                 "REJECTED physical report recall/F1 incorrect")
        # Real model's task micro-F1 is only a task-level statistic;
        # it is NOT per-class COORDINATION/CONDITION_RESULT/etc.
        relation=observed["event_relation"]
        per_group.append({
            "axis":axis,
            "precommitted_pattern_or_lexical_group":PATTERNS[grp] if axis=="CONSTRUCTION" else f"lexical_family_{grp}",
            "pattern_index":grp,
            "gold_event_relation_types":dict(gold_rel),
            "relation_task_micro_F1":f1(relation["TP"],relation["FP"],relation["FN"]),
            "relation_TP_FP_FN":_counts([{"counts":relation}],"counts"),
            "gold_rejected_support":gold_rej,
            "rejected_TP_FP_FN":_counts([{"counts":reject}],"counts"),
            "rejected_recall_if_supported":reject["recall"] if gold_rej else None,
            "no_independent_evaluation":True,
        })
    expected_total={"event":448,"directed_role":976,
                    "argument_polarity":976,"coreference":192,
                    "event_relation":192,"rejected_specific":32}
    axis_stats={}
    for axis,key in (("LEXICAL","lexical_8fold_TRAIN_internal_NOT_independent"),
                     ("CONSTRUCTION","syntax_32fold_TRAIN_internal_NOT_independent")):
        section=[f for f in folds if f["axis"]==axis]
        summary=m.get(key)
        required(len(section)==(8 if axis=="LEXICAL" else 32)
                 and isinstance(summary,dict) and set(summary)==set(TASKS)|{REJECTED},
                 "wrong physical grouped metric report")
        out={}
        for task in (*TASKS,REJECTED):
            source="REJECTED_specific_held_TRAIN" if task==REJECTED else "held_TRAIN_metrics"
            counts=_counts([f[source] for f in section],source) if False else {
                k:sum(f[source][k] for f in section) for k in ("TP","FP","FN")}
            actual=summary[task]
            required(all(actual[k]==counts[k] for k in ("TP","FP","FN")),
                     "pooled grouped metrics were not computed from fold counts")
            support_name=("held_REJECTED_positive_support" if task==REJECTED
                          else "TRAIN_INTERNAL_held_positive_support")
            required(actual[support_name]==expected_total[task],
                     "grouped gold support does not cover 256 new TRAIN rows exactly once")
            score_name=("F1" if task==REJECTED
                        else "TRAIN_INTERNAL_grouped_micro_F1_NOT_independent")
            required(abs(actual[score_name]-f1(**{
                "tp":counts["TP"],"fp":counts["FP"],"fn":counts["FN"]}))<1e-12,
                "pooled micro-F1 arithmetic drift")
            out[task]={**counts,"F1":actual[score_name]}
            if task==REJECTED:
                required(abs(actual["recall"]-
                             counts["TP"]/(counts["TP"]+counts["FN"]))<1e-12,
                         "pooled rejected recall mismatch")
                out[task]["recall"]=actual["recall"]
                out[task]["precision"]=actual["precision"]
        axis_stats[axis]=out
    syntax=axis_stats["CONSTRUCTION"]
    lexical=axis_stats["LEXICAL"]
    # Only observed internal synthetic scores are reported; the earlier S20d
    # 15.50% score was measured on 96 OTHER samples, not a comparable test.
    problems=sorted((x for x in per_group if x["axis"]=="CONSTRUCTION"),
                    key=lambda x:(x["relation_task_micro_F1"],
                                  x["pattern_index"]))
    rejection=sorted((x for x in per_group if x["axis"]=="CONSTRUCTION"
                     and x["gold_rejected_support"]>0),
                     key=lambda x:(x["rejected_recall_if_supported"],
                                   x["pattern_index"]))
    required(syntax["event_relation"]["TP"]==106
             and syntax["event_relation"]["FP"]==277
             and syntax["event_relation"]["FN"]==86,
             "not the actually successful S22d physical diagnostic")
    required(syntax[REJECTED]["TP"]==23
             and syntax[REJECTED]["FP"]==7
             and syntax[REJECTED]["FN"]==9
             and lexical[REJECTED]["TP"]==31,
             "not the 40-fold S22d REJECTED diagnostic")
    return {
        "stage":"NRA3_S23_REAL_S22D_SYNTH_TRAIN40_FOLD_DIAGNOSTIC_ONLY",
        "actual_S22d_run":S22D_ORIGINAL_CI,
        "actual_S22d_artifact_id":S22D_ARTIFACT_ID,
        "actual_S22d_FP16_sha256":S22D_MODEL_SHA,
        "new256_graph_gold_sha256":FROZEN_S22_TRAIN256_SHA256,
        "real_original43M_GT_S22b_source_run":S22B_ORIGINAL_CI,
        "physically_segregated_S22c_source_run":S22C_ORIGINAL_CI,
        "physical_FP16_verified_but_no_forward_or_optimizer":True,
        "not_independent_evaluation":True,
        "old_S20d_vs_S22d_not_a_matched_comparison":True,
        "label_level_relation_confusion_NOT_AVAILABLE_without_fresh_fold_predictions":True,
        "lexical_8fold_synthetic_TRAIN_only":lexical,
        "syntax_32fold_synthetic_TRAIN_only":syntax,
        "construction_folds_ordered_by_lowest_relation_micro_F1":problems,
        "REJECTED_construction_folds_ordered_by_lowest_recall":rejection,
        "NEXT_RESEARCH_requirement":"Save per-class relation confusion from strict synthetic TRAIN-internal folds, not spent EVAL.",
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
        "LIVE_Render_Supabase_Jessica_APK_and_original_Calibri_unchanged":True,
    }


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--report",type=Path,required=True)
    cli.add_argument("--model",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    required(args.report.is_file() and args.model.is_file(),"immutable S22d artifacts missing")
    digest=hashlib.sha256(args.model.read_bytes()).hexdigest()
    report=json.loads(args.report.read_text(encoding="utf-8"))
    audit=exact_real_report_audit(report,digest)
    required(not args.out.exists(),"must not overwrite old audit or physical model")
    args.out.mkdir(parents=True,exist_ok=False)
    target=args.out/"NRA3_S23_ACTUAL_S22D_40FOLD_STRUCTURAL_FAILURE_AUDIT.json"
    target.write_text(json.dumps(audit,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                      encoding="utf-8")
    print("S23_REAL_S22D_SYNTAX_RELATION_F1",
          audit["syntax_32fold_synthetic_TRAIN_only"]["event_relation"]["F1"],flush=True)
    print("S23_REAL_S22D_SYNTAX_REJECTED_RECALL",
          audit["syntax_32fold_synthetic_TRAIN_only"]["rejected_specific"]["recall"],flush=True)
    print("S23_LOWEST_SYNTACTIC_RELATION_FOLDS",
          json.dumps(audit["construction_folds_ordered_by_lowest_relation_micro_F1"][:6],
                     ensure_ascii=False,sort_keys=True),flush=True)
    print("S23_REJECTED_CONSTRUCTION_RECALL",
          json.dumps(audit["REJECTED_construction_folds_ordered_by_lowest_recall"],
                     ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
