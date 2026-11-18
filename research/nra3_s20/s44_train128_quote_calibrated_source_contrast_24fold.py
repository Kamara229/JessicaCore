"""S44 physical CPU TRAIN-only quote calibration + source/recipient contrast.

The genuinely learned S43 event+link graph on newly defined TRAIN128
regressed on held constructions, and its quotation classifier generated
590-696 false quote-token predictions in its synthetic fold aggregates.
This S44 experiment predeclares one TRAIN-loss-only alternative:
  - balance quotation false positive penalty relative to quoted tokens
  - balance false predicate source penalty
  - add TRAIN-only counterfactual hardest wrong-source→same-recipient
    and same-source→wrong-recipient pairwise ranking
  - preserve S43 exact model and same optimizer, 120 steps, 24 folds
No inference heuristic, no S42 one-to-one cap or positive edge floor.

S38 is reserved; NO S38 text or labels in train, loss, tuning or test.
These synthetic TRAIN128 held templates have already been inspected by
S43. This is not independent evaluation or real Calibri43M+GT integration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import model_weights_digest
from research.nra3_s20.s43_new128_multi_event_trainonly_source import (
    STAGE,build_train128,train_partitions,train_source_sha,validate_source,PATTERNS,
)
from research.nra3_s20.s43_joint_learned_predicate_source_adapter import (
    JointLearnedPredicateSourceS43,
    S43_TRAIN_EVENT_SOURCE_COEFF,S43_TRAIN_LINK_SOURCE_CONSISTENCY_COEFF,
    S43_TRAIN_HARD_NEGATIVE_COEFF,
)
from research.nra3_s20.s43_train128_joint_predicate_source_24fold import (
    batch_raw_plus_train_labels,train_one as fit_s43,
    score_after_frozen_training,summation,
    S43_SEED,S43_TRAIN_STEPS,S43_TRAIN_LR,S43_PRED_THRESHOLD,
)

S44_STAGE="NRA3_S44_TRAIN128_QUOTE_CALIBRATION_DIRECTED_SOURCE_CONTRAST_24FOLD"
S44_PREVIOUS_RUN=44247621131
S44_QUOTE_NEGATIVE_WEIGHT=2.0
S44_QUOTE_LOSS_COEFFICIENT=.8
S44_EVENT_NEGATIVE_WEIGHT=1.0
S44_WRONG_SOURCE_PAIR_MARGIN=1.0
S44_WRONG_RECIPIENT_PAIR_MARGIN=1.0
S44_PAIR_RANKING_COEFFICIENT=.6
S44_TRAIN_STEPS=S43_TRAIN_STEPS
S44_TRAIN_LR=S43_TRAIN_LR

def require(ok,why):
    if not bool(ok):
        raise ValueError("S44 isolated genuine TRAIN128 grammar research: "+why)

def directed_source_recipient_contrast(logits,reference,legal):
    """TRAIN-only gold supervision enters loss, NOT prediction/inference.

    Each true directed edge competes with incorrect sources to same
    recipient and incorrect recipients for the same source. Empty sets
    are skipped, all groups predetermined. This admits 2+ true targets
    per source; one-to-one matching is NEVER imposed.
    """
    positive=(reference>.5)&legal
    gold_positions=torch.nonzero(positive,as_tuple=False)
    require(len(gold_positions)>0,"S44 requires positive TRAIN links")
    source_losses=[]
    target_losses=[]
    for b,s,t in gold_positions.tolist():
        correct=logits[b,s,t]
        wrong_sources=legal[b,:,t] & ~positive[b,:,t]
        wrong_targets=legal[b,s,:] & ~positive[b,s,:]
        if bool(wrong_sources.any()):
            best_wrong=logits[b,:,t][wrong_sources].max()
            source_losses.append(
                F.relu(S44_WRONG_SOURCE_PAIR_MARGIN + best_wrong-correct))
        if bool(wrong_targets.any()):
            best_wrong=logits[b,s,:][wrong_targets].max()
            target_losses.append(
                F.relu(S44_WRONG_RECIPIENT_PAIR_MARGIN+best_wrong-correct))
    require(source_losses and target_losses,
            "no valid TRAIN wrong-source / wrong-recipient contrast")
    return torch.stack(source_losses).mean()+torch.stack(target_losses).mean()

def fit_s44(train_rows,fold):
    require(len(train_rows) in (112,120),
            "S44 only 112/120 NEW128 train fold rows")
    torch.manual_seed(S43_SEED+fold)
    model=JointLearnedPredicateSourceS43()
    require(model.output_gate==0 and not model.candidate_accepted
            and not model.activation_allowed,"S44 original shadow output gate changed")
    optim=torch.optim.AdamW(model.parameters(),lr=S44_TRAIN_LR,
                            weight_decay=.001)
    x,q,p,structure,valid,legal,ey,qy,ry=batch_raw_plus_train_labels(train_rows)
    epos=(ey>.5)&valid
    eneg=(ey<.5)&valid
    qpos=(qy>.5)&valid
    qneg=(qy<.5)&valid
    pos=(ry>.5)&legal
    neg=(ry<.5)&legal
    require(bool(epos.any() and eneg.any() and qpos.any() and qneg.any()
                 and pos.any() and neg.any()),
            "S44 needs both positive+negative TRAIN graph, quote and event examples")
    diagnostics={}
    for step in range(S44_TRAIN_STEPS):
        optim.zero_grad(set_to_none=True)
        out=model(x,q,p,structure)
        link_pos=F.softplus(-out.rejected_logits[pos]).mean()
        neg_loss=F.softplus(out.rejected_logits[neg])
        k=min(neg_loss.numel(),6*int(pos.sum()))
        link_neg=neg_loss.topk(k).values.mean()
        source_loss=(F.softplus(-out.event_logits[epos]).mean()
                     +S44_EVENT_NEGATIVE_WEIGHT*
                      F.softplus(out.event_logits[eneg]).mean())
        quote_loss=(F.softplus(-out.quote_logits[qpos]).mean()
                    +S44_QUOTE_NEGATIVE_WEIGHT*
                     F.softplus(out.quote_logits[qneg]).mean())
        source_at_true_edge=out.event_logits.unsqueeze(2).expand_as(
            out.rejected_logits)[pos]
        event_link_consistency=F.softplus(-source_at_true_edge).mean()
        wrong_source_mask=eneg.unsqueeze(2)&legal
        wrong_scores=F.softplus(out.rejected_logits[wrong_source_mask])
        hard_wrong_source=wrong_scores.topk(
            min(wrong_scores.numel(),max(1,4*int(pos.sum())))).values.mean()
        ranking=directed_source_recipient_contrast(out.rejected_logits,ry,legal)
        loss=(link_pos+S43_TRAIN_HARD_NEGATIVE_COEFF*link_neg
              +S43_TRAIN_EVENT_SOURCE_COEFF*source_loss
              +S44_QUOTE_LOSS_COEFFICIENT*quote_loss
              +S43_TRAIN_LINK_SOURCE_CONSISTENCY_COEFF*(
                  event_link_consistency+.5*hard_wrong_source)
              +S44_PAIR_RANKING_COEFFICIENT*ranking)
        require(bool(torch.isfinite(loss)),"nonfinite TRAIN-only loss")
        loss.backward()
        require(all(t.grad is not None and bool(torch.isfinite(t.grad).all())
                    for t in model.parameters()),
                "S44 missing trainable source, quote or link gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if step in (0,S44_TRAIN_STEPS-1):
            diagnostics["loss_initial" if step==0 else "loss_final"]=float(loss.detach())
            diagnostics["quote_loss_initial" if step==0 else "quote_loss_final"]=float(quote_loss.detach())
            diagnostics["pair_ranking_initial" if step==0 else "pair_ranking_final"]=float(ranking.detach())
    model.eval()
    return model,diagnostics

def summarize_fp_and_fn(r):
    """Metrics computed only after both models frozen on their TRAIN fold."""
    m=r["S44_new_TRAIN_internal"]
    return {
        "link_TP":m["REJECTED_link_TP"],
        "link_FP":m["REJECTED_link_FP"],
        "link_FN":m["REJECTED_link_FN"],
        "event_TP":m["event_source_TP"],
        "event_FP":m["event_source_FP"],
        "event_FN":m["event_source_FN"],
        "quoted_TP":m["quote_TP"],
        "quoted_FP":m["quote_FP"],
        "quoted_FN":m["quote_FN"],
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),"immutable research output overwrite blocked")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source=validate_source()
    require(source["count"]==128 and source["grouped_folds"]==24
            and source["source_sha256"]==train_source_sha()
            and source["S38_reserved_not_imported_or_used"]
            and source["TRAIN_two_rejected_edges_per_sentence"]==32,
            "prior S43 NEW128 frozen 24fold TRAIN graph changed")
    rows=build_train128()
    result=[]
    for fold,(axis,group,tr,held) in enumerate(train_partitions()):
        require(len(tr)==(112 if axis=="LEXICAL" else 120) and
                len(held)==(16 if axis=="LEXICAL" else 8) and
                not set(tr)&set(held) and set(tr)|set(held)==set(range(128)),
                "strict S43/S44 matched TRAIN grouping broken")
        train_rows=[rows[i] for i in tr]
        held_rows=[rows[i] for i in held]
        old,old_diag=fit_s43("S43_LEARNED",train_rows,fold)
        new,new_diag=fit_s44(train_rows,fold)
        old_sha=model_weights_digest(old)
        new_sha=model_weights_digest(new)
        # No held synthetic TRAIN text, labels, gold and group fed to training.
        old_metrics=score_after_frozen_training(old,held_rows)
        new_metrics=score_after_frozen_training(new,held_rows)
        require(model_weights_digest(old)==old_sha
                and model_weights_digest(new)==new_sha,
                "S44 held TRAIN metrics mutated actual CPU model weights")
        require(old_metrics["event_source_support"]==
                new_metrics["event_source_support"] and
                old_metrics["REJECTED_link_support"]==
                new_metrics["REJECTED_link_support"] and
                old_metrics["held_rows_requiring_one_source_to_multiple_recipient_links"]==
                new_metrics["held_rows_requiring_one_source_to_multiple_recipient_links"],
                "S44 gold reference set differs from original S43 same-run baseline")
        result.append({
           "fold":fold,"axis":axis,"held_group_OFFLINE_ONLY":group,
           "held_construction_name_OFFLINE_ONLY":
             PATTERNS[group] if axis=="CONSTRUCTION" else "lexical_family",
           "TRAIN_rows":len(tr),"held_TRAIN_internal_rows":len(held),
           "TRAIN_held_disjoint":True,
           "original_S43_CPU_weights_SHA256":old_sha,
           "S44_CPU_weights_SHA256":new_sha,
           "S43_TRAIN_loss":old_diag,
           "S44_TRAIN_loss":new_diag,
           "S43_same_run_TRAIN_internal":old_metrics,
           "S44_new_TRAIN_internal":new_metrics,
           "gold_labels_only_for_TRAIN_loss_or_OFFLINE_held_metrics":True,
           "S38_source_never_imported_used_or_tuned":True,
           "no_one_to_one_or_forced_positive_at_inference":True,
           "output_gate":0,
        })
        print("S44_SAME_RUN_TRAIN128_FOLD",
            json.dumps({"fold":fold,"axis":axis,"group":group,
                        "S43":{k:old_metrics[k] for k in
                          ("event_source_TP","REJECTED_link_TP",
                           "REJECTED_link_FP","quote_FP")},
                        "S44":{k:new_metrics[k] for k in
                          ("event_source_TP","REJECTED_link_TP",
                           "REJECTED_link_FP","quote_FP")}},sort_keys=True),flush=True)
    paired={
      axis:{
        "S43_same_run":summation(result,"S43_same_run_TRAIN_internal",axis),
        "S44_new_balanced_contrastive":summation(result,"S44_new_TRAIN_internal",axis)
      } for axis in ("LEXICAL","CONSTRUCTION")
    }
    diagnostics={PATTERNS[group]:{
        "S43":next(x for x in result if x["axis"]=="CONSTRUCTION" and
                   x["held_group_OFFLINE_ONLY"]==group)["S43_same_run_TRAIN_internal"],
        "S44":next(x for x in result if x["axis"]=="CONSTRUCTION" and
                   x["held_group_OFFLINE_ONLY"]==group)["S44_new_TRAIN_internal"]
      } for group in (2,3,4,5,6,7,8,9,10,11,15)}
    report={
      "stage":S44_STAGE,
      "previous_true_S43_completed_run_for_provenance":S44_PREVIOUS_RUN,
      "same_source_new128_SHA256":source["source_sha256"],
      "precommitted_hyperparams":{
        "quote_negative_weight":S44_QUOTE_NEGATIVE_WEIGHT,
        "quote_loss_coefficient":S44_QUOTE_LOSS_COEFFICIENT,
        "event_negative_weight":S44_EVENT_NEGATIVE_WEIGHT,
        "wrong_source_margin":S44_WRONG_SOURCE_PAIR_MARGIN,
        "wrong_recipient_margin":S44_WRONG_RECIPIENT_PAIR_MARGIN,
        "pair_ranking_coefficient":S44_PAIR_RANKING_COEFFICIENT,
        "steps":S44_TRAIN_STEPS,"lr":S44_TRAIN_LR,
      },
      "paired_CPU_training_two_models_per_each_of_24_folds":True,
      "fixed_SOURCE128_not_updated_with_stage_S44_experiment":True,
      "source_derived_only_from_original_Russian_plaintext_at_inference":True,
      "no_gold_in_original_S43_or_S44_model_forward":True,
      "no_output_side_one_to_one_matching_or_rule_link_floor":True,
      "S40_explicit_handcoded_grammar_priors_still_present":True,
      "S38_adversarial_32_NOT_train_eval_or_tuning":True,
      "all_24_ALREADY_INSPECTED_synthetic_TRAIN_development_folds":result,
      "TRAIN_only_paired_summary":paired,
      "predefined_multi_event_passive_quote_scope_group_diagnostics":diagnostics,
      "not_independent_model_generalization_or_original_43M_GT_NRA2":True,
      "protected_616J7_EVAL128_EVAL32_bpp_CB59_not_touched":True,
      "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
      "output_gate":0,"candidate_accepted":False,"activation_allowed":False
    }
    args.out.mkdir(parents=True,exist_ok=False)
    target=args.out/"NRA3_S44_TRAIN128_MATCHED_S43_S44_QUOTE_SOURCE_CONTRAST_24FOLD.json"
    target.write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
                      encoding="utf-8")
    print("S44_TRAIN128_SAME_RUN_QUOTE_SOURCE_CONTRAST_24FOLD",
          json.dumps(paired,sort_keys=True),flush=True)
    print("S44_NO_S38_NO_43M_NO_NRA2_NO_ACTIVATION",flush=True)

if __name__=="__main__":
    main()
