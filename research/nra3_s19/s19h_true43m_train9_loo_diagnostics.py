"""S19h: honest TRAIN-internal 9-fold leave-ONE-original-TRAIN-sentence-out.

This is an exploratory diagnostic on nine human-authored S19 TRAIN examples,
NOT a pristine independent eval (the sources already influenced architecture).
Every semantic head is reinitialized/retrained on eight *TRAIN* examples;
the ninth is inaccessible to optimizer, feature selection and checkpoint
selection. The 9332-FP16 frozen S19d parent came only from separate TRAIN96.

Zero hyperparameter tuning or model promotion on these nine fold results.
If this probe shows failure, collect new disjoint/train annotation diversity
rather than spend the old EVAL32 or overfit these diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import (
    PaidSemanticEventGraph, semantic_parameter_count, full_parameter_count,
    STATUS, ROLE, ARG_STATUS, COREF_STATUS, EVENT_RELATION,
)
from research.nra3_s19.s19g_train9_real43m_graph import (
    fixed_multitask_loss, load_segregated_real_source, task_train_reuse_report,
    STEPS, LR, SEED, S19D_FULL_PHYSICAL_SHA, S19F_REAL_SOURCE_SHA,
)
from research.nra3_s19.train_only_event_graph_source import TRAIN_ONLY_FROZEN_SHA256

LOO_FOLDS=9
BASELINE_S19G_FULL_TRAIN9_CI=44135054417
BASELINE_S19G_FULL_TRAIN_SHA="c741d754b9d7706685e179d3f40ff1e0486d33344b59ef5b51fe2e128317755e"


def require(ok,msg):
    if not bool(ok):
        raise ValueError("S19h TRAIN-internal nine-fold diagnosis: "+msg)


def take(x,ids):
    ix=torch.tensor(tuple(ids),dtype=torch.long)
    if isinstance(x,dict):
        return {k:v.index_select(0,ix) for k,v in x.items()}
    return x.index_select(0,ix)


def metric_from_train_scoring_alias(out,y,valid):
    """Reuse exact TP/FP/FN arithmetic; rename as held TRAIN-internal fold."""
    raw=task_train_reuse_report(out,y,valid)
    return {k:{
        "held_TRAIN_gold_nonNONE":v["TRAIN_ONLY_supervised_labels_nonNONE"],
        "TP":v["TP"],"FP":v["FP"],"FN":v["FN"],
    } for k,v in raw.items()}


def merge_counts(folds):
    names=("event","directed_role","argument_polarity","coreference","event_relation")
    result={}
    for task in names:
        totals={k:sum(row["per_head_heldout_TRAIN_only"][task][k] for row in folds)
                for k in ("TP","FP","FN","held_TRAIN_gold_nonNONE")}
        tp,fp,fn=totals["TP"],totals["FP"],totals["FN"]
        result[task]={
           **totals,
           "TRAIN_INTERNAL_LOO_micro_F1_NOT_independent":
             2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.,
        }
    return result


def run_9fold(raw,valid,y,frozen_parent):
    require(raw.shape[0]==9 and valid.shape[0]==9 and
            all(t.shape[0]==9 for t in y.values()),
            "only original nine TRAIN rows are accepted")
    require(all(not p.requires_grad for p in frozen_parent.parameters()),
            "all 9332 parent grammar/context params frozen")
    rows=[]
    for held in range(LOO_FOLDS):
        train_ids=tuple(i for i in range(9) if i!=held)
        held_id=(held,)
        require(len(train_ids)==8 and held not in train_ids
                and set(train_ids)|set(held_id)==set(range(9)),
                "leave-ONE-out split must have no held example in optimizer")
        tr_raw,tr_valid,tr_gold=(
            take(raw,train_ids),take(valid,train_ids),take(y,train_ids))
        dev_raw,dev_valid,dev_gold=(
            take(raw,held_id),take(valid,held_id),take(y,held_id))
        # Frozen learned S19d from separate original TRAIN96; every new
        # semantic graph head is freshly initialized on eight TRAIN rows.
        torch.manual_seed(SEED+held)
        model=PaidSemanticEventGraph(frozen_parent).train()
        require(full_parameter_count(model)==12381
                and semantic_parameter_count(model)==3049,
                "all 9 folds use identical full physical semantic head ABI")
        optimizer=torch.optim.AdamW(list(model.semantic_params()),
                                    lr=LR,weight_decay=.001)
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            score=model(tr_raw,tr_valid)  # NEVER DEV features in optimizer
            loss=fixed_multitask_loss(score,tr_gold,tr_valid)
            require(bool(torch.isfinite(loss)),
                    "nonfinite fold TRAIN-only loss")
            loss.backward()
            require(all(p.grad is None for p in model.parent.parameters()),
                    "frozen original43M grammar/context changed")
            torch.nn.utils.clip_grad_norm_(list(model.semantic_params()),1.)
            optimizer.step()
        # Only AFTER fixed optimizer schedule: reload real physical FP16.
        physical=model.physical_bytes()
        frozen=PaidSemanticEventGraph.from_physical(physical).eval()
        with torch.inference_mode():
            pred=frozen(dev_raw,dev_valid)
            held_metrics=metric_from_train_scoring_alias(
                pred,dev_gold,dev_valid)
        target_labels=tr_gold["event_status_TRAIN_ONLY"][tr_valid]
        held_labels=dev_gold["event_status_TRAIN_ONLY"][dev_valid]
        observed={
            name:{
                "train_side":int((target_labels==i).sum()),
                "held_TRAIN":int((held_labels==i).sum()),
            } for i,name in enumerate(STATUS)
        }
        rejected_in_train=int((tr_gold["argument_polarity_TRAIN_ONLY"]==1).sum())
        rejected_in_dev=int((dev_gold["argument_polarity_TRAIN_ONLY"]==1).sum())
        rows.append({
           "held_TRAIN_row_index_zero_based":held,
           "TRAIN_side_rows":len(train_ids),
           "held_TRAIN_rows":1,
           "no_held_row_in_any_optimizer_gradient":True,
           "same_fixed_steps_no_validation_checkpoint_selection":STEPS,
           "held_TRAIN_status_class_support":observed,
           "rejected_RECIPIENT_train_side_gold":rejected_in_train,
           "rejected_RECIPIENT_held_TRAIN_gold":rejected_in_dev,
           "full_paid_FP16_fold_SHA":hashlib.sha256(physical).hexdigest(),
           "per_head_heldout_TRAIN_only":held_metrics,
        })
        print("S19H_TRAIN_INTERNAL_LOO",held,
              "EVENT",held_metrics["event"],
              "ROLE",held_metrics["directed_role"],
              "REJECTED_HELD",rejected_in_dev,flush=True)
    require(len(rows)==9 and
            [x["held_TRAIN_row_index_zero_based"] for x in rows]==list(range(9))
            and sum(x["rejected_RECIPIENT_held_TRAIN_gold"] for x in rows)==1,
            "all original human TRAIN examples must be held once, including rejected role")
    return rows,merge_counts(rows)


def main():
    p=argparse.ArgumentParser()
    for name in ("features","gold","manifest","s19d-parent","out"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    require(not a.out.exists(),"do not overwrite any previous research artifact")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw,valid,y,meta=load_segregated_real_source(
        a.features,a.gold,a.manifest)
    blob=a.s19d_parent.read_bytes()
    require(hashlib.sha256(blob).hexdigest()==S19D_FULL_PHYSICAL_SHA,
            "S19d original frozen contextual parent drift")
    parent=PaidContextualEventScope.from_physical(blob)
    require(not any(p.requires_grad for p in parent.parameters()),
            "frozen S19d original weighted parent cannot train")
    fold_rows,aggregated=run_9fold(raw,valid,y,parent)
    a.out.mkdir(parents=True,exist_ok=False)
    report={
       "stage":"NRA3_S19H_GENUINE_ORIGINAL43M_GT_NEW9_TRAIN_INTERNAL_LOO_DIAGNOSTICS",
       "original_true_CPU_S19e":44132334034,
       "original_true_CPU_S19f_physical_graph_join":44133991642,
       "original_S19g_full9_train_only_smoke":BASELINE_S19G_FULL_TRAIN9_CI,
       "original_S19g_full9_train_FP16_SHA_NOT_loaded":BASELINE_S19G_FULL_TRAIN_SHA,
       "original_trueCPU_hidden320_sha":S19F_REAL_SOURCE_SHA,
       "sealed_human_9_TRAIN_gold_sha":TRAIN_ONLY_FROZEN_SHA256,
       "frozen_S19d_contextual_parent_full_fp16_sha":S19D_FULL_PHYSICAL_SHA,
       "number_fixed_group_disjoint_loocv_TRAIN_folds":LOO_FOLDS,
       "new_semantic_FP16_parameters_reinitialized_in_each_fold":3049,
       "parent_grammar_context_FP16_parameters_frozen":9332,
       "total_physical_FP16_params_per_fold":12381,
       "source_TRAIN_only_gold_blind_model_features":True,
       "no_GOLD_event_roles_rejection_or_coreference_in_input":True,
       "train_side_only_optimizer_and_fixed_steps":True,
       "held_TRAIN_only_all_nine_rows_seen_as_development_ONCE":True,
       "LOO_TRAIN_INTERNAL_NOT_pristine_or_independent_EVAL":True,
       "architecture_was_informed_by_previous_S19_TRAIN_source":True,
       "no_hyperparameter_selection_from_nine_LOO_scores":True,
       "per_fold_TRAIN_INTERNAL":fold_rows,
       "aggregate_TRAIN_INTERNAL_LOO_NOT_independent":aggregated,
       "no_consumed_EVAL32_EVAL128_S17c_S15c_S12c_loaded":True,
       "original_parent_616_J7_preserved_S19e_ONLY":True,
       "active_candidate_616_J7_NOT_CERTIFIED":True,
       "inclusive_bpp_le07_NOT_CERTIFIED":True,
       "total_CPU_ge90pct_CB59_NOT_CERTIFIED":True,
       "NRA2_graph_bridge_not_integrated":True,
       "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
       "no_LIVE_Render_Supabase_Jessica_APK_changes":True,
    }
    path=a.out/"NRA3_S19H_TRUE43M_GT_TRAIN9_LOO_NO_EVAL_REPORT.json"
    path.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n",
                    encoding="utf-8")
    print("S19H_TRAIN_INTERNAL_LOO_AGGREGATE",
          json.dumps(aggregated,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
