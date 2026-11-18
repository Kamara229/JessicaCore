"""S41 paired TRAIN-only decoder ablation on physically fitted S40 graph.

Reproduce S40 original newTRAIN80 18 grouped fits (same seed/optimizer),
then compare in the SAME run without any additional fitting: plain S40
versus generic eligible-edge completion S41. S39/S40 prior physical
reports are READ ONLY; old S38 adversarial challenge stays excluded.
This is a rule-informed, already-inspected synthetic TRAIN-only
development dataset, not fresh blind validation or actual Calibri43M.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from research.nra3_s20.s39_new80_trainonly_scope_graph_source import (
    build_train80,validate_source,source_digest,partitions,STAGE,
)
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import (
    ShadowNRA2S39,raw_token_pair_features,labels_train_only,model_weights_digest,
)
from research.nra3_s20.s40_compositional_predicate_negation_graph import (
    CompositionalNRA2S40,plain_scope_features,STRUCTURE_DIM,
    PREDECLARED_BIAS_ALLOWED,PREDECLARED_BIAS_BLOCKED,
)
from research.nra3_s20.s41_typed_graph_edge_completion import (
    structurally_complete_REJECTED,S41_ELIGIBLE_LOWER_LOGIT,S41_RULE,
    S41_ORIGINAL_S40_RUN,
)

TRAIN_STEPS=96
TRAIN_LR=0.018
TRAIN_GLOBAL_SEED=391118
GROUP_FOLDS=18
TEST_THRESHOLD_FIXED_BEFORE_TRAIN=0.5

def require(ok,why):
    if not bool(ok):raise ValueError("S39 TRAIN-only linked adapter contract: "+why)

def build_batch(rows):
    raw_features=[raw_token_pair_features(row.text) for row in rows]
    train_labels=[labels_train_only(row) for row in rows]
    structure_features=[plain_scope_features(row.text) for row in rows]
    maxw=max(x[0].shape[0] for x in raw_features)
    b=len(rows)
    pred=torch.zeros(b,maxw,16)
    quote=torch.zeros(b,maxw,5)
    pair=torch.zeros(b,maxw,maxw,32)
    structure=torch.zeros(b,maxw,maxw,STRUCTURE_DIM)
    ev=torch.zeros(b,maxw)
    quote_y=torch.zeros(b,maxw)
    rejected=torch.zeros(b,maxw,maxw)
    valid=torch.zeros(b,maxw,dtype=torch.bool)
    for index,(f,ys) in enumerate(zip(raw_features,train_labels)):
        fp,fq,fl=f
        ye,yq,yp=ys
        n=fp.shape[0]
        require(n==fq.shape[0]==fl.shape[0]==fl.shape[1]
                and n==len(ye)==len(yq)==yp.shape[0]==yp.shape[1],
                "S39 raw/annotation tokenizer length divergence")
        pred[index,:n]=fp
        quote[index,:n]=fq
        pair[index,:n,:n]=fl
        structure[index,:n,:n]=structure_features[index]
        ev[index,:n]=ye
        quote_y[index,:n]=yq
        rejected[index,:n,:n]=yp
        valid[index,:n]=True
    n=maxw
    legal=(valid.unsqueeze(2)&valid.unsqueeze(1)
           & ~torch.eye(n,dtype=torch.bool).unsqueeze(0))
    require(bool(legal.any()),"no true candidate pairs")
    return pred,quote,pair,structure,valid,legal,ev,quote_y,rejected

def train_one_fold(train,fold_id):
    # No held group information, text or labels allowed in model training.
    require(len(train) in (70,72),"S39 training batch size drift")
    torch.manual_seed(TRAIN_GLOBAL_SEED+fold_id)
    model=CompositionalNRA2S40()
    optim=torch.optim.AdamW(model.parameters(),lr=TRAIN_LR,weight_decay=.001)
    xp,xq,xy,xstruct,valid,legal,ey,qy,ry=build_batch(train)
    positive=(ry>0.5)&legal
    negative=(ry<0.5)&legal
    require(int(positive.sum())>0 and int(negative.sum())>0,
            "S39 TRAIN fold needs positive+negative pair evidence")
    diagnostics={}
    for step in range(TRAIN_STEPS):
        optim.zero_grad(set_to_none=True)
        out=model(xp,xq,xy,xstruct)
        pos=F.softplus(-out.rejected_logits[positive]).mean()
        # Hard negative mining uses scores ONLY on the TRAIN batch.
        all_negative=F.softplus(out.rejected_logits[negative])
        k=min(int(positive.sum())*6,all_negative.numel())
        hard=all_negative.topk(k).values.mean()
        ep=(ey>0.5)&valid
        en=(ey<0.5)&valid
        event=F.softplus(-out.event_logits[ep]).mean()+.2*F.softplus(out.event_logits[en]).mean()
        qp=(qy>0.5)&valid
        qn=(qy<0.5)&valid
        qpos=F.softplus(-out.quote_logits[qp]).mean() if bool(qp.any()) else out.quote_logits.sum()*0.
        qneg=F.softplus(out.quote_logits[qn]).mean()
        quote_loss=qpos+.3*qneg
        loss=pos+hard+.5*event+.25*quote_loss
        require(bool(torch.isfinite(loss)),"nonfinite S39 TRAIN loss")
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()),"invalid S39 trainable gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if step in (0,TRAIN_STEPS-1):
            diagnostics["loss_start" if step==0 else "loss_last"]=float(loss.detach())
    model.eval()
    require(model.output_gate==0 and not model.candidate_accepted
            and not model.activation_allowed,"research-only shadow model activated")
    return model,diagnostics

def evaluate_after_training(model,rows,postprocess=False):
    xp,xq,xy,xstruct,valid,legal,ey,qy,ry=build_batch(rows)
    with torch.no_grad():
        y=model(xp,xq,xy,xstruct)
        if postprocess:
            y_old=y
            y=structurally_complete_REJECTED(y,xstruct)
            require(torch.equal(y_old.event_logits,y.event_logits)
                    and torch.equal(y_old.quote_logits,y.quote_logits),
                    "S41 decoding changed protected original learned event/quote logits")
            eligible=(xstruct[...,11]>0.5)
            require(torch.equal(y.rejected_logits[~eligible],
                                y_old.rejected_logits[~eligible]),
                    "S41 modified any uneligible directed edge logit")
            require(bool((y.rejected_logits[eligible]>=S41_ELIGIBLE_LOWER_LOGIT).all()),
                    "S41 did not protect known structurally eligible logits")
        r=torch.sigmoid(y.rejected_logits)>=TEST_THRESHOLD_FIXED_BEFORE_TRAIN
        r=r&legal
        positive=(ry>0.5)&legal
        tp=int((r&positive).sum())
        fp=int((r&(~positive)&legal).sum())
        fn=int(((~r)&positive).sum())
        events=(torch.sigmoid(y.event_logits)>=.5)&valid
        event_gold=(ey>0.5)&valid
        event_tp=int((events&event_gold).sum())
        event_fp=int((events&(~event_gold)&valid).sum())
        event_fn=int(((~events)&event_gold).sum())
        quotes=(torch.sigmoid(y.quote_logits)>=.5)&valid
        quote_gold=(qy>0.5)&valid
        quote_tp=int((quotes&quote_gold).sum())
        quote_fn=int(((~quotes)&quote_gold).sum())
    return {
        "directed_rejected_TP":tp,"directed_rejected_FP":fp,"directed_rejected_FN":fn,
        "directed_rejected_gold_positive_support":int(positive.sum()),
        "event_source_TP":event_tp,"event_source_FP":event_fp,"event_source_FN":event_fn,
        "quoted_token_TP":quote_tp,"quoted_token_FN":quote_fn,
        "example_count":len(rows),
        "threshold_precommitted":TEST_THRESHOLD_FIXED_BEFORE_TRAIN,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--baseline-s39-report",type=Path,required=True)
    parser.add_argument("--baseline-s40-report",type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),"never overwrite S39 research outputs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source=validate_source()
    require(source["sha256"]==source_digest()
            and source["TRAIN_rows"]==80 and source["internal_folds"]==18
            and source["S38_challenge_source_not_imported_or_used"],
            "S39 fixed new TRAIN source integrity missing")
    baseline=json.loads(args.baseline_s39_report.read_text())
    require(
        baseline["stage"]==
            "NRA3_S39_NEW80_TRAIN_ONLY_SHADOW_NRA2_GRAPH_ADAPTER_INTERNAL_18FOLD"
        and baseline["new_TRAIN80_source"]["sha256"]==source["sha256"]
        and len(baseline["all18_strict_synthetic_TRAIN_grouped_folds"])==18
        and baseline["output_gate"]==0
        and not baseline["candidate_accepted"],
        "S39 completed 18fold baseline or TRAIN80 source provenance invalid")
    prior_s39={x["fold"]:x for x in
               baseline["all18_strict_synthetic_TRAIN_grouped_folds"]}
    require(set(prior_s39)==set(range(18)),
            "S39 archived matched fold numbers incomplete")
    prior40=json.loads(args.baseline_s40_report.read_text())
    require(prior40["stage"]==
                "NRA3_S40_COMPOSITIONAL_S39_TRAIN80_SYNTHETIC_18FOLD_NO_EVAL"
            and prior40["new_TRAIN80_source"]["sha256"]==source["sha256"]
            and prior40["original_archived_S39_completed_run"]==44215938499
            and prior40["output_gate"]==0 and not prior40["candidate_accepted"]
            and len(prior40["all18_strict_synthetic_TRAIN_grouped_folds"])==18,
            "physical completed S40 development baseline invalid or source changed")
    prior_s40={x["fold"]:x for x in
               prior40["all18_strict_synthetic_TRAIN_grouped_folds"]}
    require(set(prior_s40)==set(range(18)),
            "original S40 18fold development set incomplete")
    rows=build_train80()
    folds=partitions()
    records=[]
    for i,(axis,group,tr,held) in enumerate(folds):
        require(len(held)==(10 if axis=="LEXICAL" else 8)
                and len(tr)==(70 if axis=="LEXICAL" else 72)
                and len(set(tr)|set(held))==80 and not set(tr).intersection(held),
                "S39 new TRAIN-only grouped internal fold leakage")
        subset_train=[rows[j] for j in tr]
        subset_held=[rows[j] for j in held]
        model,learning=train_one_fold(subset_train,i)
        # Model trained as actual S40: same seed, training examples and
        # optimizer. No extra training/update for S41; a PURE decoder.
        before_sha=model_weights_digest(model)
        metrics40=evaluate_after_training(model,subset_held,postprocess=False)
        metrics=evaluate_after_training(model,subset_held,postprocess=True)
        require(model_weights_digest(model)==before_sha,
                "S41 read-only decoder modified original S40 learned weights")
        require(metrics40["directed_rejected_gold_positive_support"]==
                    metrics["directed_rejected_gold_positive_support"],
                "paired original TRAIN targets changed")
        require(all(metrics40[z]==metrics[z]
                    for z in ("event_source_TP","event_source_FP",
                              "event_source_FN","quoted_token_TP","quoted_token_FN")),
                "S41 changed protected event and quote head performance")
        require(prior_s40[i]["held_TRAIN_internal_metrics"]
                ["directed_rejected_gold_positive_support"]==
                metrics40["directed_rejected_gold_positive_support"],
                "S40 source held TRAIN support drift")
        require(prior_s39[i]["axis"]==axis and
                prior_s39[i]["held_group"]==group and
                prior_s39[i]["TRAIN_side_examples"]==len(tr) and
                prior_s39[i]["TRAIN_internal_held_examples"]==len(held),
                "old S39 and new S40 synthetic TRAIN grouped splits drift")
        require(prior_s40[i]["axis"]==axis and
                prior_s40[i]["held_group"]==group and
                prior_s40[i]["TRAIN_side_examples"]==len(tr) and
                prior_s40[i]["TRAIN_internal_held_examples"]==len(held),
                "same S40 grouped TRAIN development fold absent")
        record={
            "S40_same_run_held_TRAIN_metrics":metrics40,
            "S40_original_archived_held_TRAIN_metrics":
                prior_s40[i]["held_TRAIN_internal_metrics"],
            "S40_prior_archived_numeric_fingerprint":
                prior_s40[i]["fresh_model_weights_sha256"],
            "S40_same_run_numeric_matches_archived":
                before_sha==prior_s40[i]["fresh_model_weights_sha256"],
            "S41_readonly_logits_and_protected_event_quote_preserved":True,
            "S39_original_completed_held_TRAIN_metrics":
                prior_s39[i]["held_TRAIN_internal_metrics"],
            "S39_original_completed_physical_numeric_SHA":
                prior_s39[i]["fresh_model_weights_sha256"],
            "fold":i,"axis":axis,"held_group":group,
            "TRAIN_side_examples":len(tr),"TRAIN_internal_held_examples":len(held),
            "TRAIN_and_held_disjoint":True,
            "fresh_model_weights_sha256":model_weights_digest(model),
            "train_loss":learning,
            "held_TRAIN_internal_metrics":metrics,
            "gold_never_used_in_neural_forward":True,
            "event_source_quote_rejected_three_heads_learned":True,
            "output_gate":0,
        }
        records.append(record)
        print("S41_ACTUAL_CPU_TRAIN_ONLY_FOLD",
              json.dumps({"fold":i,"axis":axis,"group":group,
                          "S40_TP":metrics40["directed_rejected_TP"],
                          "S40_FP":metrics40["directed_rejected_FP"],
                          "S41_TP":metrics["directed_rejected_TP"],
                          "S41_FP":metrics["directed_rejected_FP"],
                          "S41_FN":metrics["directed_rejected_FN"],
                          "sha_unchanged":record["fresh_model_weights_sha256"]},
                         sort_keys=True),flush=True)
    aggregated={}
    for axis in ("LEXICAL","CONSTRUCTION"):
        matches=[r["held_TRAIN_internal_metrics"] for r in records if r["axis"]==axis]
        tp=sum(x["directed_rejected_TP"] for x in matches)
        fp=sum(x["directed_rejected_FP"] for x in matches)
        fn=sum(x["directed_rejected_FN"] for x in matches)
        aggregated[axis]={
            "TP":tp,"FP":fp,"FN":fn,
            "TRAIN_internal_recall":tp/(tp+fn) if tp+fn else 0.,
            "TRAIN_internal_precision":tp/(tp+fp) if tp+fp else 0.,
            "TRAIN_internal_F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "not_independent_evaluation":True,
        }
    report={
        "stage":"NRA3_S41_TRAIN80_READONLY_GENERIC_STRUCTURAL_EDGE_COMPLETION_18FOLD",
        "original_archived_S39_completed_run":44215938499,
        "original_archived_S40_completed_run":S41_ORIGINAL_S40_RUN,
        "S41_precommitted_edge_floor_logit":S41_ELIGIBLE_LOWER_LOGIT,
        "S41_fixed_rule":S41_RULE,
        "S41_pure_output_postprocessor_no_extra_training":True,
        "original_true43M_12384FP16_not_loaded_or_modified":True,
        "S40_not_original_43M_or_12384_model_but_separate_shadow":True,
        "precommitted_grammar_prior_allowed":PREDECLARED_BIAS_ALLOWED,
        "precommitted_grammar_prior_blocked":PREDECLARED_BIAS_BLOCKED,
        "explicit_graph_features_quote_clauses_predicate_passive_infinitive_nonlocal_scope":True,
        "all_18_folds_same_exposed_development_as_S39":True,
        "same_train80_optimizer_and_fixed_seeds_as_S39":True,
        "new_TRAIN80_source":source,
        "original_256_TRAIN_disjoint":True,
        "untouched_S38_32_adversarial_reserved_for_later_control":True,
        "no_S38_text_or_label_loaded_in_training_loss_or_model_selection":True,
        "not_original43M_GT_inference_or_joint_trained_architecture":True,
        "paid_original_NRA2_Calibri43M_GT_616J7_untouched":True,
        "three_active_shadow_train_heads_predicate_quote_rejected":True,
        "trainable_adapter_parameters":sum(p.numel() for p in CompositionalNRA2S40().parameters()),
        "fixed_TRAIN_steps_per_fold":TRAIN_STEPS,
        "fixed_optimizer_lr":TRAIN_LR,
        "fixed_prediction_threshold":TEST_THRESHOLD_FIXED_BEFORE_TRAIN,
        "all18_strict_synthetic_TRAIN_grouped_folds":records,
        "summary_internal_only_by_fold_axis":aggregated,
        "candidate_accepted":False,"activation_allowed":False,"output_gate":0,
        "not_independent_EVAL32_EVAL128_or_616J7_tested":True,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
    }
    args.out.mkdir(parents=True,exist_ok=False)
    target=args.out/"NRA3_S41_TRAIN80_GENERIC_GRAPH_EDGE_COMPLETION_AGAINST_S40.json"
    paired40={}
    for axis in ("LEXICAL","CONSTRUCTION"):
        old=[z["S40_same_run_held_TRAIN_metrics"] for z in records if z["axis"]==axis]
        tp=sum(y["directed_rejected_TP"] for y in old)
        fp=sum(y["directed_rejected_FP"] for y in old)
        fn=sum(y["directed_rejected_FN"] for y in old)
        paired40[axis]={"TP":tp,"FP":fp,"FN":fn,
                        "F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    report["matched_same_run_original_S40"] = paired40
    target_constructions={}
    for ix,label in ((12,"short_passive"),(13,"infinitive_complement")):
        item=records[ix]
        require(item["axis"]=="CONSTRUCTION" and item["held_group"]==ix-8,
                "passive/infinitive held TRAIN fixed indices drift")
        target_constructions[label]={
            "S40_same_run":item["S40_same_run_held_TRAIN_metrics"],
            "S41_rule_completed":item["held_TRAIN_internal_metrics"],
        }
    report["fixed_passive_infinitive_construction_diagnostics"]=target_constructions
    target.write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
                      encoding="utf-8")
    print("S41_REAL_TRAIN80_GRAPH_COMPLETION_18FOLD_INTERNAL_RESULTS",
          json.dumps({"S40":paired40,"S41":aggregated},sort_keys=True),flush=True)
    print("S41_NO_43M_NO_S38_EVAL_NO_LIVE_NO_WEIGHT_UPDATE",flush=True)

if __name__=="__main__":
    main()
