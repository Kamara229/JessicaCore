"""S24 fixed real original43M+GT 256 TRAIN-only grouped graph learning.

40 predeclared TRAIN-internal folds: eight leave-lexical-family-out and
twelve leave-syntax-template-out. Each fold freshly initializes learned
five-head graph adapter; frozen S19d parent 9332-FP16 never trains.
Replays NO spent independent EVAL; no model outputs connected to NRA/LIVE.
Measures the severe previous S19h generalization issue under more
systematically populated (but still synthetic) semantic classes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import (
    full_parameter_count, semantic_parameter_count,
    STATUS, EVENT_RELATION,
)
from research.nra3_s20.s24_neural_role_local_negation_scope import ScopedRejectionGraph
from research.nra3_s20.s23_real_s22d_40fold_failure_audit import (
    exact_real_report_audit,S22D_MODEL_SHA,S22D_ORIGINAL_CI,
)
from research.nra3_s19.s19g_train9_real43m_graph import (
    fixed_multitask_loss, task_train_reuse_report,
    S19D_FULL_PHYSICAL_SHA, LR, STEPS,
)
from research.nra3_s19.s19h_true43m_train9_loo_diagnostics import take
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    FROZEN_S22_TRAIN256_SHA256,splits,EXPECTED,
)

SEED=320256
PAID_PARENT=9332
PAID_GRAPH=3052
PAID_TOTAL=12384


def require(ok,msg):
    if not bool(ok):
        raise ValueError("S24 original43M+GT 256 TRAIN grouped graph: "+msg)


def sha(f:Path):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def load_physical_training_source(raw_file:Path,gold_file:Path,manifest_file:Path):
    for p in (raw_file,gold_file,manifest_file):
        require(p.is_file(),"must download real S22c physically separated source")
    info=json.loads(manifest_file.read_text(encoding="utf-8"))
    require(info["stage"]=="NRA3_S22C_REAL_ORIGINAL43M_GT_NEW256_SEPARATE_GRAPH_TRAIN_SHARDS"
            and info["S22_sha_locked_synthetic_TRAIN256_graph_gold"]==FROZEN_S22_TRAIN256_SHA256
            and info["256_curated_synthetic_train_texts"]==256
            and info["S22_original_protected_616_J7_parent_unchanged"]
            and info["TRAIN_only_events"]==448
            and info["TRAIN_only_directed_role_links"]==976
            and info["TRAIN_only_rejected_recipient_links"]==32
            and info["TRAIN_only_coreference_edges"]==192
            and info["TRAIN_only_event_relations"]==192
            and info["strict_predeclared_lexical_and_syntax_TRAIN_folds"]==40
            and info["genuine_original_CPU_hidden_plus_plain_raw18_ONLY_as_model_inputs"]
            and info["human_graph_targets_NEVER_passed_to_original_or_adapter_forward"]
            and info["no_new_optimizer_no_model_scoring_or_quality_claim"]
            and info["no_consumed_EVAL32_EVAL128_S17c_S15c_S12c"]
            and info["new_active_candidate_616_J7_CPU_bpp_NOT_CERTIFIED"]
            and not info["candidate_accepted"] and not info["activation_allowed"]
            and info["output_gate"]==0
            and info["raw338_word_features_filename"]==raw_file.name
            and info["TRAIN_only_gold_filename"]==gold_file.name
            and sha(raw_file)==info["raw338_word_features_SHA256"]
            and sha(gold_file)==info["TRAIN_only_gold_SHA256"],
            "original real CPU protected parent/train256 source SHA or output gate drift")
    x=load_file(str(raw_file),device="cpu")
    y=load_file(str(gold_file),device="cpu")
    require(set(x)=={"raw_text_frozen_word338","valid_word_mask"}
            and set(y)=={"event_status_TRAIN_ONLY",
                         "predicate_argument_role_TRAIN_ONLY",
                         "argument_polarity_TRAIN_ONLY","coreference_TRAIN_ONLY",
                         "event_relation_TRAIN_ONLY"},
            "model input must not contain human roles or parser gold")
    raw=x["raw_text_frozen_word338"]
    valid=x["valid_word_mask"]
    require(raw.dtype==torch.float32 and raw.device.type=="cpu"
            and raw.shape[0]==EXPECTED and raw.shape[2]==338
            and valid.shape==raw.shape[:2] and valid.dtype==torch.bool
            and bool(valid[:,0].all())
            and bool((valid.long().cummin(1).values==valid.long()).all())
            and bool(torch.isfinite(raw).all())
            and bool((raw[~valid]==0).all())
            and bool(((raw[...,-18:]==0)|(raw[...,-18:]==1)).all()),
            "gold-free actual original CPU 338-D raw word input invalid")
    b,w=valid.shape
    allowed=(valid.unsqueeze(1)&valid.unsqueeze(2)) & (
        ~torch.eye(w,dtype=torch.bool).unsqueeze(0))
    require(y["event_status_TRAIN_ONLY"].shape==(b,w)
            and all(t.shape==(b,w,w) for name,t in y.items()
                    if name!="event_status_TRAIN_ONLY")
            and all(bool((t[~allowed]==-1).all()) for name,t in y.items()
                    if name!="event_status_TRAIN_ONLY")
            and int((y["event_status_TRAIN_ONLY"]>0).sum())==448
            and int((y["predicate_argument_role_TRAIN_ONLY"]>0).sum())==976
            and int((y["argument_polarity_TRAIN_ONLY"]==1).sum())==32
            and int((y["coreference_TRAIN_ONLY"]==1).sum())==192
            and int((y["event_relation_TRAIN_ONLY"]>0).sum())==192,
            "human TRAIN256 supervision/word valid indices drift")
    return raw,valid,y,info


def new_graph_model(parent:PaidContextualEventScope,seed:int):
    require(all(not p.requires_grad for p in parent.parameters())
            and parent.output_gate==0 and not parent.candidate_accepted,
            "original S19d 9332-FP16 contextual parent must remain frozen")
    torch.manual_seed(seed)
    m=ScopedRejectionGraph(parent).train()
    require(full_parameter_count(m)==PAID_TOTAL
            and semantic_parameter_count(m)==PAID_GRAPH
            and m.output_gate==0 and not m.candidate_accepted
            and not m.activation_allowed,"full physical 12381 FP16 ABI changed")
    return m


def fixed_train_head(model,raw,valid,gold):
    optimizer=torch.optim.AdamW(list(model.semantic_params()),
                                lr=LR,weight_decay=.001)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss=fixed_multitask_loss(model(raw,valid),gold,valid)
        require(bool(torch.isfinite(loss)),"nonfinite source-only TRAIN loss")
        loss.backward()
        require(all(p.grad is None for p in model.parent.parameters()),
                "training path updated frozen original Calibri grammar/context")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.semantic_params()),"untrained semantic head")
        torch.nn.utils.clip_grad_norm_(list(model.semantic_params()),1.)
        optimizer.step()
    payload=model.physical_bytes()
    physical=ScopedRejectionGraph.from_physical(payload).eval()
    require(physical.head_physical()==model.head_physical()
            and not any(p.requires_grad for p in physical.parameters())
            and physical.output_gate==0 and not physical.candidate_accepted
            and not physical.activation_allowed,
            "real physically billed FP16 roundtrip failed")
    return physical,payload


def source_only_counts(pred,truth,valid):
    """Rename S19g in-sample arithmetic to honest held TRAIN-internal scores."""
    raw=task_train_reuse_report(pred,truth,valid)
    return {task:{
        "TP":score["TP"],"FP":score["FP"],"FN":score["FN"],
        "TRAIN_INTERNAL_held_positive_support":score["TRAIN_ONLY_supervised_labels_nonNONE"],
    } for task,score in raw.items()}


def rejected_specific_counts(out,gold,valid):
    """Rare REJECTED label (polarity=1) only, with NONE/positive negatives.

    This is a separate per-class one-vs-rest calculation on a held
    TRAIN-internal fold; gold is NEVER a model forward input.
    """
    b,w=valid.shape
    legal=(valid.unsqueeze(1)&valid.unsqueeze(2)) & (
        ~torch.eye(w,dtype=torch.bool).unsqueeze(0))
    prediction=out.argument_polarity.argmax(-1)[legal] == 2
    truth=gold["argument_polarity_TRAIN_ONLY"][legal] == 1
    tp=int((prediction & truth).sum())
    fp=int((prediction & ~truth).sum())
    fn=int((~prediction & truth).sum())
    return {"TP":tp,"FP":fp,"FN":fn,"held_REJECTED_positive_support":int(truth.sum()),
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def aggregate(records,axis):
    group=[x for x in records if x["axis"]==axis]
    require(len(group)==(8 if axis=="LEXICAL" else 32),
            "grouped TRAIN-internal 256 total coverage changed")
    result={}
    for task in ("event","directed_role","argument_polarity","coreference","event_relation"):
        counts={key:sum(x["held_TRAIN_metrics"][task][key] for x in group)
                for key in ("TP","FP","FN","TRAIN_INTERNAL_held_positive_support")}
        tp,fp,fn=(counts[k] for k in ("TP","FP","FN"))
        result[task]={
            **counts,
            "TRAIN_INTERNAL_grouped_micro_F1_NOT_independent":
                2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        }
    rc={k:sum(x["REJECTED_specific_held_TRAIN"][k] for x in group)
        for k in ("TP","FP","FN","held_REJECTED_positive_support")}
    tp,fp,fn=(rc[k] for k in ("TP","FP","FN"))
    result["rejected_specific"]={**rc,
        "precision":tp/(tp+fp) if tp+fp else 0.,
        "recall":tp/(tp+fn) if tp+fn else 0.,
        "F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return result


def run_fixed_40_folds(raw,valid,y,parent):
    records=[]
    groups=splits()
    require(len(groups)==40,"S22 precommitted 40 TRAIN-only split contract")
    for foldno,(axis,group,train_ids,held_ids) in enumerate(groups):
        require(len(train_ids)+len(held_ids)==256
                and not set(train_ids).intersection(held_ids)
                and set(train_ids)|set(held_ids)==set(range(256)),
                "TRAIN-only lexical/construction split leakage")
        trraw,trvalid,trgold=(
            take(raw,train_ids),take(valid,train_ids),take(y,train_ids))
        dvraw,dvvalid,dvgold=(
            take(raw,held_ids),take(valid,held_ids),take(y,held_ids))
        # Fixed optimizer schedule was chosen BEFORE any S22 held TRAIN
        # group scores were computed. No fold checkpoint selection.
        model=new_graph_model(parent,SEED+foldno)
        trained,payload=fixed_train_head(model,trraw,trvalid,trgold)
        with torch.inference_mode():
            out=trained(dvraw,dvvalid)
            metrics=source_only_counts(out,dvgold,dvvalid)
            rare=rejected_specific_counts(out,dvgold,dvvalid)
        train_event=trgold["event_status_TRAIN_ONLY"][trvalid]
        held_event=dvgold["event_status_TRAIN_ONLY"][dvvalid]
        support={
            name:{"TRAIN_side":int((train_event==i).sum()),
                  "held_TRAIN":int((held_event==i).sum())}
            for i,name in enumerate(STATUS)}
        require(all(support[z]["TRAIN_side"]>0 for z in (
            "ASSERTED","NEGATED","CONDITIONAL","FUTURE_UNREALIZED"))
            and int((trgold["argument_polarity_TRAIN_ONLY"]==1).sum())>0,
            "rare class absent in S22 TRAIN-side grouped fold")
        relation_types={
            label:int((trgold["event_relation_TRAIN_ONLY"]==i).sum())
            for i,label in enumerate(EVENT_RELATION)}
        records.append({
            "axis":axis,"held_group_index":group,
            "fold_number":foldno,
            "TRAIN_side_rows":len(train_ids),
            "held_TRAIN_rows":len(held_ids),
            "held_rows_never_in_gradient_checkpoint_or_model_selection":True,
            "original_true43M_GT_grammar_context_parent_frozen":True,
            "fixed_optimizer_steps":STEPS,
            "TRAIN_event_status_support":support,
            "TRAIN_relation_label_support":relation_types,
            "TRAIN_rejected_recipient_support":int(
                (trgold["argument_polarity_TRAIN_ONLY"]==1).sum()),
            "held_TRAIN_rejected_recipient_support":int(
                (dvgold["argument_polarity_TRAIN_ONLY"]==1).sum()),
            "fold_paid_FP16_SHA256":hashlib.sha256(payload).hexdigest(),
            "held_TRAIN_metrics":metrics,
            "REJECTED_specific_held_TRAIN":rare,
        })
        print("S24_HELD_TRAIN_FOLD",axis,group,
              "EVENT",metrics["event"],"ROLE",metrics["directed_role"],
              flush=True)
    require(len(records)==40,"S22 complete 40-fold group schedule did not finish")
    return records,aggregate(records,"LEXICAL"),aggregate(records,"CONSTRUCTION")


def main():
    p=argparse.ArgumentParser()
    for item in ("features","gold","manifest","s19d-parent","baseline-s22d-report","baseline-s22d-model","out"):
        p.add_argument("--"+item,type=Path,required=True)
    args=p.parse_args()
    require(not args.out.exists(),"do not overwrite prior research models")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw,valid,y,source=load_physical_training_source(
        args.features,args.gold,args.manifest)
    parent_physical=args.s19d_parent.read_bytes()
    require(hashlib.sha256(parent_physical).hexdigest()==S19D_FULL_PHYSICAL_SHA,
            "original frozen S19d original43M GT physical FP16 SHA drift")
    parent=PaidContextualEventScope.from_physical(parent_physical)
    prior=json.loads(args.baseline_s22d_report.read_text(encoding="utf-8"))
    baseline_sha=hashlib.sha256(args.baseline_s22d_model.read_bytes()).hexdigest()
    baseline=exact_real_report_audit(prior,baseline_sha)
    require(baseline_sha==S22D_MODEL_SHA and
            baseline["actual_S22d_run"]==S22D_ORIGINAL_CI,
            "S24 must compare same frozen S22d real physical model")
    folds,lexical,syntax=run_fixed_40_folds(raw,valid,y,parent)
    previous=prior["all40_fixed_TRAIN_internal_folds"]
    require(len(previous)==len(folds)==40
            and all(a["axis"]==b["axis"] and a["held_group_index"]==b["held_group_index"]
                    and a["TRAIN_side_rows"]==b["TRAIN_side_rows"]
                    and a["held_TRAIN_rows"]==b["held_TRAIN_rows"]
                    for a,b in zip(previous,folds)),
            "matched paired 40-fold index/split contract drift")
    full_model=new_graph_model(parent,SEED+40)
    trained,paid=fixed_train_head(full_model,raw,valid,y)
    with torch.inference_mode():
        full_logits=trained(raw,valid)
        full_metrics=source_only_counts(full_logits,y,valid)
        full_rare=rejected_specific_counts(full_logits,y,valid)
    paired=[]
    for before,after in zip(previous,folds):
        prior_r=before["REJECTED_specific_held_TRAIN"]
        current_r=after["REJECTED_specific_held_TRAIN"]
        prior_rel=before["held_TRAIN_metrics"]["event_relation"]
        current_rel=after["held_TRAIN_metrics"]["event_relation"]
        paired.append({
          "axis":after["axis"],
          "held_group_index":after["held_group_index"],
          "held_REJECTED_support":current_r["held_REJECTED_positive_support"],
          "S22d_REJECTED_recall":prior_r["recall"],
          "S24_REJECTED_recall":current_r["recall"],
          "REJECTED_delta_recall":current_r["recall"]-prior_r["recall"],
          "S22d_relation_task_F1":2*prior_rel["TP"]/(2*prior_rel["TP"]+prior_rel["FP"]+prior_rel["FN"])
              if 2*prior_rel["TP"]+prior_rel["FP"]+prior_rel["FN"] else 0.,
          "S24_relation_task_F1":2*current_rel["TP"]/(2*current_rel["TP"]+current_rel["FP"]+current_rel["FN"])
              if 2*current_rel["TP"]+current_rel["FP"]+current_rel["FN"] else 0.,
        })
    args.out.mkdir(parents=True,exist_ok=False)
    name="NRA3_S24_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s24"
    (args.out/name).write_bytes(paid)
    require((args.out/name).read_bytes()==paid
            and ScopedRejectionGraph.from_physical(paid).head_physical()==trained.head_physical(),
            "final physical FP16 candidate failed roundtrip")
    record={
        "stage":"NRA3_S24_REAL43M_GT_NEW256_SYNTHETIC_TRAIN_ONLY_GROUPED40_GRAPH",
        "true_frozen_original43M_GT_from_source_S22b":True,
        "true_physical_256_SHAlocked_separate_model_vs_gold_S22c":True,
        "protected_original_616_J7_digest_unchanged_at_source_capture_ONLY":True,
        "source_256_synthetic_graph_SHA256":FROZEN_S22_TRAIN256_SHA256,
        "source_true43M_CPU_256_hidden_SHA256":source["S22b_original43M_GT_CPU_hidden320_FP16_sha"],
        "source_raw_input_sha":source["raw338_word_features_SHA256"],
        "source_human_gold_sha":source["TRAIN_only_gold_SHA256"],
        "optimizer_scope":"ONLY_SEMANTIC_HEAD_3052_FP16",
        "original_S19d_grammar_context_9332_FP16_frozen_sha":S19D_FULL_PHYSICAL_SHA,
        "full_physical_FP16_parameters":PAID_TOTAL,
        "frozen_parent_FP16_parameters":PAID_PARENT,
        "trainable_graph_FP16_parameters":PAID_GRAPH,
        "physical_FP16_candidate_SHA256":hashlib.sha256(paid).hexdigest(),
        "physical_FP16_filename":name,
        "all40_fixed_TRAIN_internal_folds":folds,
        "lexical_8fold_TRAIN_internal_NOT_independent":lexical,
        "syntax_32fold_TRAIN_internal_NOT_independent":syntax,
        "full256_TRAIN_resubstitution_NOT_generalization":full_metrics,
        "full256_TRAIN_rejected_specific_NOT_generalization":full_rare,
        "fixed_training_steps_per_fold_and_full":STEPS,
        "baseline_same_split_genuine_S22d_model_SHA":S22D_MODEL_SHA,
        "baseline_same_split_genuine_S22d_run":S22D_ORIGINAL_CI,
        "architecture_intervention":"Three-parameter order-invariant predicted-recipient local-negative polarity residual, no oracle",
        "strictly_matched_S22d_vs_S24_40fold_TRAIN_internal_pairs":paired,
        "held_out_preposed_rejected_fold_pair":paired[8+25],
        "S22d_REJECTED_syntax_32fold_recall":prior["syntax_32fold_TRAIN_internal_NOT_independent"]["rejected_specific"]["recall"],
        "S22d_relation_syntax_32fold_F1":prior["syntax_32fold_TRAIN_internal_NOT_independent"]["event_relation"]["TRAIN_INTERNAL_grouped_micro_F1_NOT_independent"],
        "S24_REJECTED_syntax_32fold_recall":syntax["rejected_specific"]["recall"],
        "S24_relation_syntax_32fold_F1":syntax["event_relation"]["TRAIN_INTERNAL_grouped_micro_F1_NOT_independent"],
        "S24_rejected_preposed_improved":paired[8+25]["REJECTED_delta_recall"]>0,
        "MODEL_NOT_SELECTED_ON_HELD_FOLDS":True,
        "no_EVAL32_EVAL128_or_prior_spent_EVAL":True,
        "synthetic_TRAIN_template_generalization_only_not_real_language_understanding":True,
        "no_active_candidate_616_J7_certification":True,
        "no_all_in_bpp_le07_certification":True,
        "no_full_CPU_ge90pct_CB59_certification":True,
        "original_Calibri_GT_LIVE_Render_Supabase_Jessica_APK_untouched":True,
        "NRA2_live_event_graph_decoder_bridge_not_connected":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
    (args.out/"NRA3_S24_REAL43M_GT_TRAIN256_GROUPED40_MATCHED_DIAGNOSTICS.json").write_text(
        json.dumps(record,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8")
    print("S24_TRAIN_INTERNAL_LEXICAL_8FOLD",json.dumps(lexical,sort_keys=True),flush=True)
    print("S24_TRAIN_INTERNAL_SYNTAX_32FOLD",json.dumps(syntax,sort_keys=True),flush=True)
    print("S24_TRUE43M_FP16_SHA",record["physical_FP16_candidate_SHA256"],flush=True)
    print("S24_REJECTED_PREPOSED_MATCHED",json.dumps(record["held_out_preposed_rejected_fold_pair"],sort_keys=True),flush=True)
    print("S24_SYNTAX_MATCHED",json.dumps({
        k:record[k] for k in ("S22d_REJECTED_syntax_32fold_recall",
           "S24_REJECTED_syntax_32fold_recall","S22d_relation_syntax_32fold_F1",
           "S24_relation_syntax_32fold_F1")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
