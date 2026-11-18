"""S43 actual CPU supervised TRAIN-only 24 grouped folds on 128 new Russian graphs.

Compare physically trained S40 vs new trainable S43 on EXACT same frozen
new TRAIN128, each initialised with same fixed fold seed and trained on
identical 112/120 TRAIN-side examples before any 16/8 held evaluation.

24 grouped synthetic TRAIN-internal holdouts are not independent EVAL.
All curated constructions are already reflected in inductive bias, and
results are not real Calibri43M/Green Tunnel/NRA2 integration.
S38 remains immutable and is not imported, scored, or used for tuning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import (
    raw_token_pair_features,model_weights_digest,
)
from research.nra3_s20.s40_compositional_predicate_negation_graph import (
    CompositionalNRA2S40,plain_scope_features,STRUCTURE_DIM,
)
from research.nra3_s20.s43_new128_multi_event_trainonly_source import (
    STAGE,build_train128,train_partitions,train_source_sha,validate_source,
    PATTERNS,
)
from research.nra3_s20.s43_joint_learned_predicate_source_adapter import (
    JointLearnedPredicateSourceS43,
    S43_TRAIN_EVENT_SOURCE_COEFF,S43_TRAIN_LINK_SOURCE_CONSISTENCY_COEFF,
    S43_TRAIN_HARD_NEGATIVE_COEFF,S43_BASELINE_S42_RUN,
)

S43_TRAIN_STEPS=120
S43_TRAIN_LR=.012
S43_SEED=431118
S43_PRED_THRESHOLD=0.0
S43_FOLDS=24

def require(ok,why):
    if not bool(ok):
        raise ValueError("S43 isolated NEW128 TRAIN-only predicate source research: "+why)

def batch_raw_plus_train_labels(rows):
    """TRAIN gold targets created ONLY in data/loss path, never forward."""
    fs=[raw_token_pair_features(r.text) for r in rows]
    gs=[plain_scope_features(r.text) for r in rows]
    b=len(rows)
    w=max(f[0].shape[0] for f in fs)
    pred=torch.zeros(b,w,16)
    quotation=torch.zeros(b,w,5)
    pair=torch.zeros(b,w,w,32)
    structure=torch.zeros(b,w,w,STRUCTURE_DIM)
    valid=torch.zeros(b,w,dtype=torch.bool)
    event_gold=torch.zeros(b,w)
    quote_gold=torch.zeros(b,w)
    rejected_gold=torch.zeros(b,w,w)
    for a,(row,feature,graph) in enumerate(zip(rows,fs,gs)):
        pp,qq,rr=feature
        n=pp.shape[0]
        require(n==len(raw_surface_grammar(row.text))
                and n==qq.shape[0]==rr.shape[0]==rr.shape[1]
                and graph.shape==(n,n,STRUCTURE_DIM)
                and n<=40,"S43 true raw text token alignment drift")
        pred[a,:n]=pp
        quotation[a,:n]=qq
        pair[a,:n,:n]=rr
        structure[a,:n,:n]=graph
        valid[a,:n]=True
        for src in row.event_sources:
            event_gold[a,src]=1.
        for tok in row.quoted_token_indices:
            quote_gold[a,tok]=1.
        for src,tgt in row.rejected_edges:
            rejected_gold[a,src,tgt]=1.
    legal=(valid.unsqueeze(1) & valid.unsqueeze(2)
           & ~torch.eye(w,dtype=torch.bool).unsqueeze(0))
    require(bool(legal.any()),"no legal directed token links")
    require(int(rejected_gold.sum())==sum(len(r.rejected_edges) for r in rows)
            and int(event_gold.sum())==sum(len(r.event_sources) for r in rows),
            "TRAIN label alignment failed")
    return (pred,quotation,pair,structure,valid,legal,
            event_gold,quote_gold,rejected_gold)

def train_one(model_kind,rows,fold):
    require(len(rows) in (112,120),"only new TRAIN128 grouped fold can be trained")
    require(model_kind in ("S40_BASELINE","S43_LEARNED"),"invalid fixed variant")
    torch.manual_seed(S43_SEED+fold)
    model=CompositionalNRA2S40() if model_kind=="S40_BASELINE" else JointLearnedPredicateSourceS43()
    require(model.output_gate==0 and not model.candidate_accepted
            and not model.activation_allowed,"original or shadow model activation forbidden")
    pred,quote,pair,structure,valid,legal,ey,qy,ry=batch_raw_plus_train_labels(rows)
    epos=(ey>.5)&valid
    eneg=(ey<.5)&valid
    qpos=(qy>.5)&valid
    qneg=(qy<.5)&valid
    ppos=(ry>.5)&legal
    pneg=(ry<.5)&legal
    require(bool(epos.any() and eneg.any() and ppos.any() and pneg.any()),
            "S43 TRAIN fold needs source/pair positive and negatives")
    optimizer=torch.optim.AdamW(model.parameters(),lr=S43_TRAIN_LR,weight_decay=.001)
    diag={}
    for step in range(S43_TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        out=model(pred,quote,pair,structure)
        positive_loss=F.softplus(-out.rejected_logits[ppos]).mean()
        negatives=F.softplus(out.rejected_logits[pneg])
        hard_count=min(negatives.numel(),6*int(ppos.sum()))
        negative_loss=negatives.topk(hard_count).values.mean()
        event_source_loss=(F.softplus(-out.event_logits[epos]).mean()
            +.3*F.softplus(out.event_logits[eneg]).mean())
        quote_loss=(
            F.softplus(-out.quote_logits[qpos]).mean() if bool(qpos.any())
            else 0.*out.quote_logits.sum()
        )+.3*F.softplus(out.quote_logits[qneg]).mean()
        if model_kind=="S43_LEARNED":
            # Source and pair consistency learnt ONLY on TRAIN-labelled
            # examples; gold is NEVER forwarded to event/relation heads.
            logits_source=out.event_logits.unsqueeze(2).expand_as(out.rejected_logits)
            correct_source=F.softplus(-logits_source[ppos]).mean()
            # Discourage false directed REJECTED links from sources for
            # which TRAIN supervision marks no predicate event.
            non_event_pairs=(eneg.unsqueeze(2)&legal)
            incorrect=F.softplus(out.rejected_logits[non_event_pairs])
            wrong_source=incorrect.topk(min(incorrect.numel(),
                                  max(1,4*int(ppos.sum())))).values.mean()
            consistency=correct_source+.5*wrong_source
            loss=(positive_loss
                  +S43_TRAIN_HARD_NEGATIVE_COEFF*negative_loss
                  +S43_TRAIN_EVENT_SOURCE_COEFF*event_source_loss
                  +.25*quote_loss
                  +S43_TRAIN_LINK_SOURCE_CONSISTENCY_COEFF*consistency)
        else:
            consistency=0.*positive_loss
            # Original S40 training structure with fixed matched LR/steps.
            loss=(positive_loss+negative_loss+.5*event_source_loss+
                  .25*quote_loss)
        require(bool(torch.isfinite(loss)),"nonfinite TRAIN loss")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),
                "neural source/link head missing TRAIN gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step in (0,S43_TRAIN_STEPS-1):
            diag["loss_initial" if step==0 else "loss_final"]=float(loss.detach())
    model.eval()
    return model,diag

def score_after_frozen_training(model,held):
    x,q,p,structure,valid,legal,ey,qy,ry=batch_raw_plus_train_labels(held)
    with torch.inference_mode():
        out=model(x,q,p,structure)
        event_pred=(out.event_logits>=S43_PRED_THRESHOLD)&valid
        event_real=(ey>.5)&valid
        link_pred=(out.rejected_logits>=S43_PRED_THRESHOLD)&legal
        link_real=(ry>.5)&legal
        quotes=(out.quote_logits>=S43_PRED_THRESHOLD)&valid
        ref_quotes=(qy>.5)&valid
        event_tp=int((event_pred&event_real).sum())
        event_fp=int((event_pred&~event_real&valid).sum())
        event_fn=int((~event_pred&event_real).sum())
        link_tp=int((link_pred&link_real).sum())
        link_fp=int((link_pred&~link_real&legal).sum())
        link_fn=int((~link_pred&link_real).sum())
        qtp=int((quotes&ref_quotes).sum())
        qfp=int((quotes&~ref_quotes&valid).sum())
        qfn=int((~quotes&ref_quotes).sum())
        links_from_non_event_sources=int((link_pred&~event_real.unsqueeze(2)&legal).sum())
        links_from_neural_non_event_sources=int((link_pred&~event_pred.unsqueeze(2)&legal).sum())
        recurrent_reference_sources=sum(len({i for i,j in r.rejected_edges}) <
                                        len(r.rejected_edges) for r in held)
        examples=[{
           "TRAIN_internal_text":r.text,
           "source_family_OFFLINE_ONLY":r.family,
           "construction_group_OFFLINE_ONLY":r.construction,
           "gold_event_source_indices_OFFLINE_ONLY":r.event_sources,
           "predicted_event_source_indices":torch.nonzero(event_pred[k],as_tuple=False).flatten().tolist(),
           "gold_REJECTED_edges_OFFLINE_ONLY":r.rejected_edges,
           "predicted_REJECTED_edges":torch.nonzero(link_pred[k],as_tuple=False).tolist(),
        } for k,r in enumerate(held)]
    return {
      "event_source_TP":event_tp,"event_source_FP":event_fp,"event_source_FN":event_fn,
      "event_source_support":int(event_real.sum()),
      "REJECTED_link_TP":link_tp,"REJECTED_link_FP":link_fp,"REJECTED_link_FN":link_fn,
      "REJECTED_link_support":int(link_real.sum()),
      "quote_TP":qtp,"quote_FP":qfp,"quote_FN":qfn,
      "links_from_offline_non_event_source":links_from_non_event_sources,
      "links_from_predicted_non_event_source":links_from_neural_non_event_sources,
      "held_rows_requiring_one_source_to_multiple_recipient_links":recurrent_reference_sources,
      "rows":examples,
    }

def summation(records,variant,axis):
    measures=[r[variant] for r in records if r["axis"]==axis]
    def calc(prefix):
        tp=sum(m[prefix+"_TP"] for m in measures)
        fp=sum(m[prefix+"_FP"] for m in measures)
        fn=sum(m[prefix+"_FN"] for m in measures)
        return {"TP":tp,"FP":fp,"FN":fn,
                "F1":2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.}
    return {"event_head":calc("event_source"),
            "REJECTED_links":calc("REJECTED_link"),
            "quote":calc("quote"),
            "known_source_multi_recipient_rows":
              sum(m["held_rows_requiring_one_source_to_multiple_recipient_links"] for m in measures)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    require(not args.out.exists(),"never overwrite S43 research outputs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source=validate_source()
    require(source["count"]==128 and source["grouped_folds"]==24
            and source["source_sha256"]==train_source_sha()
            and source["TRAIN_two_rejected_edges_per_sentence"]==32
            and source["S38_reserved_not_imported_or_used"],
            "new source TRAIN128 provenance or multiplicity invalid")
    data=build_train128()
    folds=train_partitions()
    records=[]
    for fold,(axis,group,tr,held) in enumerate(folds):
        require(len(tr)==(112 if axis=="LEXICAL" else 120)
                and len(held)==(16 if axis=="LEXICAL" else 8)
                and not set(tr)&set(held)
                and set(tr)|set(held)==set(range(128)),
                "S43 TRAIN-only lexical/construction fold split violated")
        train_rows=[data[i] for i in tr]
        held_rows=[data[i] for i in held]
        fitted40,train40=train_one("S40_BASELINE",train_rows,fold)
        fitted43,train43=train_one("S43_LEARNED",train_rows,fold)
        old_hash,new_hash=model_weights_digest(fitted40),model_weights_digest(fitted43)
        # Held TRAIN-development labels touched only here, after training.
        observed40=score_after_frozen_training(fitted40,held_rows)
        observed43=score_after_frozen_training(fitted43,held_rows)
        require(old_hash==model_weights_digest(fitted40)
                and new_hash==model_weights_digest(fitted43),
                "actual physical model weights changed during evaluation")
        require(observed40["REJECTED_link_support"]==
                observed43["REJECTED_link_support"]
                and observed40["event_source_support"]==
                observed43["event_source_support"]
                and len(observed40["rows"])==len(observed43["rows"])==len(held),
                "matched original new TRAIN128 holdout source changed")
        record={
            "fold":fold,"axis":axis,"group":group,"group_label_OFFLINE_ONLY":
                 ("lexical_family" if axis=="LEXICAL" else PATTERNS[group]),
            "TRAIN_rows":len(tr),"held_TRAIN_rows":len(held),
            "TRAIN_held_disjoint":True,
            "baseline_S40_physical_sha256":old_hash,
            "new_S43_physical_sha256":new_hash,
            "baseline_S40_loss":train40,
            "new_S43_loss":train43,
            "baseline_S40_held_TRAIN_internal":observed40,
            "S43_joint_learned_held_TRAIN_internal":observed43,
            "model_forward_is_gold_blind":True,
            "no_S42_one_to_one_or_admissible_edge_floor":True,
            "no_S38_control_example_in_train_or_held":True,
            "model_output_gate":0,
        }
        records.append(record)
        print("S43_NEW128_TRAIN_ONLY_PHYSICAL_FOLD",
              json.dumps({"fold":fold,"axis":axis,"group":group,
                           "S40_event_TP":observed40["event_source_TP"],
                           "S43_event_TP":observed43["event_source_TP"],
                           "S40_link_TP":observed40["REJECTED_link_TP"],
                           "S43_link_TP":observed43["REJECTED_link_TP"],
                           "S43_link_FP":observed43["REJECTED_link_FP"],
                           "S43_multi_recipient_rows":
                             observed43["held_rows_requiring_one_source_to_multiple_recipient_links"]},
                         sort_keys=True),flush=True)
    aggregate={axis:{
        "S40_shadow_baseline":summation(records,"baseline_S40_held_TRAIN_internal",axis),
        "S43_learned_predicate":summation(records,"S43_joint_learned_held_TRAIN_internal",axis)
      } for axis in ("LEXICAL","CONSTRUCTION")}
    specific={
        PATTERNS[g]:{
            "S40":next(r for r in records if r["axis"]=="CONSTRUCTION" and r["group"]==g)
                   ["baseline_S40_held_TRAIN_internal"],
            "S43":next(r for r in records if r["axis"]=="CONSTRUCTION" and r["group"]==g)
                   ["S43_joint_learned_held_TRAIN_internal"],
        }
        for g in (2,3,4,5,6,7,8,9,10,11,15)
    }
    report={
       "stage":"NRA3_S43_NEW128_TRAIN_ONLY_JOINT_LEARNED_PREDICATE_SOURCE_24FOLD",
       "NEW128_source":source,
       "NEW128_source_sha256":source["source_sha256"],
       "prior_S42_completed_run_for_motivating_failure_only":S43_BASELINE_S42_RUN,
       "original_S39_TRAIN80_and_256_not_reused":True,
       "original_S38_held32_control_NOT_loaded_or_used":True,
       "physically_trained_two_models_per_fold":True,
       "fixed_TRAIN_steps":S43_TRAIN_STEPS,"fixed_TRAIN_lr":S43_TRAIN_LR,
       "joint_TRAIN_event_loss_coefficient":S43_TRAIN_EVENT_SOURCE_COEFF,
       "joint_TRAIN_link_source_consistency_coefficient":
            S43_TRAIN_LINK_SOURCE_CONSISTENCY_COEFF,
       "fixed_logit_threshold":S43_PRED_THRESHOLD,
       "real_trained_source_head_not_heuristically_recovered_from_links":True,
       "original_S40_handcoded_eligibility_prior_still_inherited":True,
       "S42_one_to_one_greedy_matching_NOT_used":True,
       "S42_eligible_link_floor_NOT_used":True,
       "construction_groups_with_multiple_REJECTED_recipients_precommitted":True,
       "all_24_synthetic_TRAIN_only_grouped_folds":records,
       "TRAIN_internal_PAIRED_metrics":aggregate,
       "previously_predefined_source_kind_multi_event_diagnostics":specific,
       "not_statistically_blind_independent_EVAL":True,
       "not_original_real43M_GT_hidden320_or_NRA2_weight_integration":True,
       "protected_original_616J7_EVAL32_128_bpp_CB59_not_touched":True,
       "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
       "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    args.out.mkdir(parents=True,exist_ok=False)
    path=args.out/"NRA3_S43_NEW128_JOINT_PREDICATE_SOURCE_CONSISTENCY_24FOLD.json"
    path.write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
                    encoding="utf-8")
    print("S43_NEW128_PAIRED_24FOLD_EVENT_SOURCE_AND_MULTIEDGE_TRAIN_RESULTS",
          json.dumps(aggregate,sort_keys=True),flush=True)
    print("S43_NO_S38_NO_43M_NO_NRA2_LIVE_OR_ORIGINAL_WEIGHTS",flush=True)

if __name__=="__main__":
    main()
