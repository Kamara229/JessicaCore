"""S19c TRAIN96 real-Calibri43M+GT event factuality learning.

Event truth is DIFFERENT from directed participant roles. Only three event
truth statuses are annotated in frozen S18 TRAIN96: ASSERTED / NEGATED /
FUTURE_UNREALIZED; an ordinary non-predicate token is NONE. No invented
CONDITION/COREFERENCE gold. Predicates are learned on ALL words, never
provided as oracle input. Grammar is trained on each fold TRAIN-side only
and frozen before event head training. No consumed held-out EVAL.
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from research.nra3_s18.real43m_train96_physical_source import (
    checked_real_train96, grouped_train96_split_contract, S18B_TRUE_CPU_RUN,
)
from research.nra3_s19.s19b_train96_true43m_grammar import (
    assemble_true43m_train96, subset, train_fixed as train_grammar_fixed, STEPS as GRAMMAR_STEPS,
    WordGrammarTrainData,
)
from research.nra3_s19.paid_grammar_parser import load_paid_fp16_bytes, trainable_parameter_count
from research.nra3_s19.s19c_paid_event_factuality import (
    PaidEventFactuality, STATUS, HEAD_PARAMETERS,
)

SEED = 319103
STEPS = 120
LR = 0.018
WEIGHTS = torch.tensor([.25,1.,1.5,1.5],dtype=torch.float32)
PARENT_GRAMMAR_SHA = "7341becd1693fa0de09d1bce8252e2f5f7a00ae96c255664112277b024bc3915"


def require(ok, msg):
    if not bool(ok):
        raise ValueError("S19c TRAIN96 TRUE original43M event scope: "+msg)


@dataclass(frozen=True)
class WordEventTargets:
    status: torch.Tensor  # [96,W] original TRAIN gold, no token gold in model input
    source_sha: str


def build_train_targets(samples, grammar: WordGrammarTrainData):
    require(len(samples)==96 and grammar.input_338.shape[0]==96,
            "source 96 trained-only rows")
    labels=torch.full_like(grammar.upos_gold,-1)
    status_counts={name:0 for name in STATUS}
    for i,sample in enumerate(samples):
        # Original precommitted S18 TRAIN96 semantic event modality annotations,
        # not inferred from model features or any consumed evaluation text.
        row=sample.row
        expected=row.event_modality
        require(expected in STATUS[1:],"unknown frozen S18 modality")
        for j,word in enumerate(row.words):
            y=STATUS.index(expected) if word.upos=="VERB" else 0
            labels[i,j]=y
            status_counts[STATUS[y]]+=1
    require(bool((labels[grammar.valid]>=0).all())
            and bool((labels[~grammar.valid]==-1).all())
            and int((labels[grammar.valid]>0).sum())==108
            and status_counts["NEGATED"]==12
            and status_counts["FUTURE_UNREALIZED"]==12
            and status_counts["ASSERTED"]==84
            and status_counts["NONE"]>400,
            "S18 physically locked EVENT modality 12 NEG 12 FUT 84 ASSERTED drift")
    return WordEventTargets(labels,grammar.source_sha)


def event_subset(t: WordEventTargets,row_ids):
    ids=torch.tensor(tuple(row_ids),dtype=torch.long)
    return WordEventTargets(t.status.index_select(0,ids),t.source_sha)


def fixed_event_training(frozen_grammar,grammar_batch,event_gold):
    require(frozen_grammar.output_gate==0 and not frozen_grammar.candidate_accepted
            and all(not p.requires_grad for p in frozen_grammar.parameters()),
            "grammatical parent is frozen before semantics")
    torch.manual_seed(SEED)
    model=PaidEventFactuality(frozen_grammar)
    require(model.output_gate==0 and model.candidate_accepted is False
            and model.activation_allowed is False
            and all(not p.requires_grad for p in model.grammar.parameters()),
            "event scope train-only, decoder remains disconnected")
    optim=torch.optim.AdamW(model.event_head.parameters(),lr=LR,weight_decay=0.0005)
    for step in range(STEPS):
        optim.zero_grad(set_to_none=True)
        logits=model(grammar_batch.input_338,grammar_batch.valid)
        require(logits.shape==(*grammar_batch.valid.shape,len(STATUS)),
                "word event logits must cover every unfiltered word")
        y=event_gold.status[grammar_batch.valid]
        loss=F.cross_entropy(logits[grammar_batch.valid],y,weight=WEIGHTS)
        require(bool(torch.isfinite(loss)),"event-head TRAIN loss nonfinite")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.event_head.parameters()) and
                all(p.grad is None for p in model.grammar.parameters()),
                "grammar gradients leaked / event gradients missing")
        torch.nn.utils.clip_grad_norm_(model.event_head.parameters(),1.)
        optim.step()
    paid=model.physical_bytes()
    loaded=PaidEventFactuality.from_physical(paid).eval()
    require(loaded.head_physical()==model.head_physical()
            and not any(p.requires_grad for p in loaded.parameters()),
            "FP16 physically paid joint frozen grammar+semantic head not roundtripped")
    return loaded,paid


def evaluate_frozen(model,grammar_batch,event_gold):
    with torch.inference_mode():
        score=model(grammar_batch.input_338,grammar_batch.valid)
        prediction=score.argmax(dim=-1)
    mask=grammar_batch.valid
    truth=event_gold.status
    predicted_event=(prediction>0)&mask
    gold_event=(truth>0)&mask
    correct=predicted_event&gold_event&(prediction==truth)
    tp=int(correct.sum())
    fp=int((predicted_event&~correct).sum())
    fn=int((gold_event&~correct).sum())
    per_status={}
    for idx,name in enumerate(STATUS[1:],1):
        pred=(prediction==idx)&mask
        gold=(truth==idx)&mask
        t=int((pred&gold).sum())
        f_p=int((pred&~gold).sum())
        f_n=int((gold&~pred).sum())
        per_status[name]=dict(TP=t,FP=f_p,FN=f_n,
            F1=(2*t/(2*t+f_p+f_n) if 2*t+f_p+f_n else None),
            TRAIN_internal_gold_support=int(gold.sum()))
    return {
      "all_raw_words":int(mask.sum()),
      "event_gold_count":int(gold_event.sum()),
      "correct_predicate_and_factuality_TP":tp,
      "false_predicate_or_wrong_factuality_FP":fp,
      "missed_or_wrong_factuality_FN":fn,
      "joint_predicate_factuality_precision":tp/(tp+fp) if tp+fp else 0.,
      "joint_predicate_factuality_recall":tp/(tp+fn) if tp+fn else 0.,
      "joint_predicate_factuality_F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
      "per_status":per_status,
    }


def aggregate(scores):
    tp=sum(z["correct_predicate_and_factuality_TP"] for z in scores)
    fp=sum(z["false_predicate_or_wrong_factuality_FP"] for z in scores)
    fn=sum(z["missed_or_wrong_factuality_FN"] for z in scores)
    n=sum(z["all_raw_words"] for z in scores)
    gold=sum(z["event_gold_count"] for z in scores)
    return {
       "all_raw_words":n,"event_gold_count":gold,
       "TP":tp,"FP":fp,"FN":fn,
       "joint_predicate_factuality_micro_F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
       "per_status":{name:{
           "TP":sum(z["per_status"][name]["TP"] for z in scores),
           "FP":sum(z["per_status"][name]["FP"] for z in scores),
           "FN":sum(z["per_status"][name]["FN"] for z in scores),
           "TRAIN_internal_gold_support":sum(z["per_status"][name]["TRAIN_internal_gold_support"]
                                             for z in scores),
       } for name in STATUS[1:]}
    }


def main():
    parser=argparse.ArgumentParser()
    for arg in ("s18b-hidden","s18b-report","s19b-full-grammar","out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),"do not overwrite previous trained artifacts")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    originals, report=checked_real_train96(args.s18b_hidden,args.s18b_report)
    grammar=assemble_true43m_train96(originals,report)
    targets=build_train_targets(originals,grammar)
    folds=grouped_train96_split_contract()
    require(S18B_TRUE_CPU_RUN==43910479658 and len(folds)==12
            and report["all_616_native_CPU_generated_token_digests_unchanged"],
            "parent frozen original Calibri43M 616 J7")
    record=[]
    for axis,ident,groups,train_ids,dev_ids in folds:
        require(set(train_ids).isdisjoint(dev_ids)
                and sorted([*train_ids,*dev_ids])==list(range(96)),
                "grouped TRAIN internal split leakage")
        # DO NOT load S19b full TRAIN model in any fold; all train-side only.
        local_grammar, grammar_physical=train_grammar_fixed(subset(grammar,train_ids))
        local_event, joint=fixed_event_training(
            local_grammar,subset(grammar,train_ids),event_subset(targets,train_ids))
        m=evaluate_frozen(local_event,subset(grammar,dev_ids),
                          event_subset(targets,dev_ids))
        record.append({
          "axis":axis,"fold_id":ident,"heldout_groups":list(groups),
          "train_rows":len(train_ids),"dev_rows":len(dev_ids),
          "grammar_trained_from_scratch_on_train_only":True,
          "no_dev_rows_ever_forwarded_to_grammar_or_event_optimizer":True,
          "paid_full_grammar_sha":hashlib.sha256(grammar_physical).hexdigest(),
          "physically_paid_joint_grammar_event_sha":hashlib.sha256(joint).hexdigest(),
          **m})
        print("S19C_TRAIN_INTERNAL_ONLY",ident,
              "event_joint_F1",m["joint_predicate_factuality_F1"],
              "event_gold",m["event_gold_count"],flush=True)
    # Full TRAIN96 source can use existing S19b pretrained FP16 parent,
    # but ONLY AFTER cross-validation has completed and been frozen.
    grammar_bytes=args.s19b_full_grammar.read_bytes()
    require(hashlib.sha256(grammar_bytes).hexdigest()==PARENT_GRAMMAR_SHA,
            "existing TRUE43M S19b full TRAIN96 physical SHA drift")
    full_grammar=load_paid_fp16_bytes(grammar_bytes)
    require(trainable_parameter_count(full_grammar)==6528,
            "S19b physical original 6528 FP16 grammar ABI drift")
    final,paid=fixed_event_training(full_grammar,grammar,targets)
    args.out.mkdir(parents=True,exist_ok=False)
    filename="NRA3_S19C_REAL43M_GT_TRAIN96_FROZEN_GRAMMAR_EVENT_SHADOW_FP16.n3s19c"
    (args.out/filename).write_bytes(paid)
    require((args.out/filename).read_bytes()==paid
            and PaidEventFactuality.from_physical(paid).head_physical()==final.head_physical(),
            "final joint FP16 paid artifact changed")
    full_metric=evaluate_frozen(final,grammar,targets)
    lexical=aggregate([x for x in record if x["axis"]=="lexical"])
    syntax=aggregate([x for x in record if x["axis"]=="syntax"])
    require(lexical["all_raw_words"]==syntax["all_raw_words"]==600
            and lexical["event_gold_count"]==syntax["event_gold_count"]==108,
            "TRAIN internal heldout coverage mismatch")
    summary={
      "stage":"NRA3_S19C_ORIGINAL43M_GT_TRAIN96_EVENT_FACTUALITY_SHADOW",
      "parent_original43m_gt_s18b_cpu_run":S18B_TRUE_CPU_RUN,
      "parent_grammar_s19b_success_CI_run":43934877964,
      "source_train96_sha":grammar.source_sha,
      "real_original43m_hidden_fp16_sha":grammar.hidden_sha,
      "physical_parent_grammar_sha":PARENT_GRAMMAR_SHA,
      "training_source_word_count":int(grammar.valid.sum()),
      "training_verb_events":108,
      "physically_paid_grammar_parameters":6528,
      "physically_paid_event_head_parameters":HEAD_PARAMETERS,
      "total_physical_fp16_parameters":6528+HEAD_PARAMETERS,
      "combined_paid_physical_bytes":len(paid),
      "combined_fp16_final_sha256":hashlib.sha256(paid).hexdigest(),
      "combined_fp16_filename":filename,
      "original_frozen_Calibri43M_GT_and_616_J7_unchanged_parent_only":True,
      "all_grammar_model_predictions_gold_blind":True,
      "all_non_predicate_words_in_event_predictions":True,
      "factuality_classes_covered_by_TRAIN96":list(STATUS),
      "S19_condition_or_coreference_event_labels_NOT_yet_learned":True,
      "S19_human_train9_event_graph_NOT_encoded_in_original43M_hidden":True,
      "S19b_grammar_training_steps_each_fold":GRAMMAR_STEPS,
      "S19c_event_head_training_steps_each_fold":STEPS,
      "CV_grouped_12_folds_each_grammar_retrained_from_train_rows_only":True,
      "TRAIN_internal_lexical_aggregate":lexical,
      "TRAIN_internal_syntax_aggregate":syntax,
      "TRAIN_internal_folds":record,
      "full96_resubstitution_metrics_NOT_generalization":full_metric,
      "NO_independent_or_spent_EVAL32_EVAL128_S17c_S15c_S12c_used":True,
      "new_candidate_616_active_J7_EOS_NOT_CERTIFIED":True,
      "inclusive_bpp_le07_NOT_CERTIFIED":True,
      "whole_CPU_speed_ge90pct_CB59_NOT_CERTIFIED":True,
      "strict_online_word_causality_NOT_PROVEN":True,
      "NRA2_Calibri_LIVE_output_gate":0,
      "candidate_accepted":False,
      "activation_allowed":False,
      "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
    }
    output=args.out/"NRA3_S19C_REAL43M_GT_TRAIN96_GROUPED12_EVENT_SHADOW_REPORT.json"
    output.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                      encoding="utf-8")
    print("S19C_TRAIN_INTERNAL_LEXICAL",json.dumps(lexical,sort_keys=True),flush=True)
    print("S19C_TRAIN_INTERNAL_SYNTAX",json.dumps(syntax,sort_keys=True),flush=True)
    print("S19C_FINAL_FP16_SHA",summary["combined_fp16_final_sha256"],flush=True)


if __name__=="__main__":
    main()
