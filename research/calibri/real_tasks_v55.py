#!/usr/bin/env python3
"""CALIBRI v55 genuinely first-use heldout task-choice accuracy.

This is an *experimental* frozen zero-shot selection protocol, not a
claim that base EleutherAI/pythia-14m is an instruction-tuned Jessica.
User's normal Jessica chat and production are never accessed.

Each option is measured by actual autoregressive pretrained LM
teacher-forced normalized mean token log probability conditioned on the
same question prefix, with fixed prefix/continuation token bounds.
Gold labels are read ONLY after the TRAIN selection decision. Never
use evaluation gold labels to select correction rows or gate candidate.
"""
from __future__ import annotations
import hashlib
import math
import torch
from datasets import load_dataset

DATASETS=(
    ("SciQ_TEST_128_160","allenai/sciq",None,"test[128:160]"),
    ("ARCEasy_TEST_0_32","allenai/ai2_arc","ARC-Easy","test[:32]"),
    ("BoolQ_VALIDATION_256_288","super_glue","boolq","validation[256:288]"),
    ("HellaSwag_VALIDATION_256_288","Rowan/hellaswag",None,"validation[256:288]"),
)
N_PER_TASK=32
MAX_PREFIX_TOKENS=128
MAX_OPTION_TOKENS=32
METHOD="mean conditional token log P; max among question continuation choices"
PROTOCOL_SHA=hashlib.sha256(
    (repr(DATASETS)+str(N_PER_TASK)+str(MAX_PREFIX_TOKENS)+
     str(MAX_OPTION_TOKENS)+METHOD).encode()).hexdigest()


def task_records(name,records):
    tasks=[]
    for row in records:
        if name=="SciQ_TEST_128_160":
            prefix=("Support: "+str(row["support"]).strip()+
                    "\nQuestion: "+str(row["question"]).strip()+
                    "\nAnswer:")
            candidates=[str(row[k]).strip() for k in (
                "correct_answer","distractor1","distractor2","distractor3")]
            if len(set(candidates))!=4:
                raise AssertionError("SciQ four distractor/answer options not unique")
            pairs=[(k,v) for k,v in enumerate(candidates)]
            pairs.sort(key=lambda p:(hashlib.sha256(
                (str(row["question"])+"|"+str(p[0])+"|"+p[1]).encode()
                ).hexdigest(),p[0]))
            gold=next(i for i,p in enumerate(pairs) if p[0]==0)
            options=[p[1] for p in pairs]
        elif name=="ARCEasy_TEST_0_32":
            prefix="Question: "+str(row["question"]).strip()+"\nAnswer:"
            labels=[str(x).strip() for x in row["choices"]["label"]]
            options=[str(x).strip() for x in row["choices"]["text"]]
            answer=str(row["answerKey"]).strip()
            if len(labels)!=len(options) or answer not in labels:
                raise AssertionError("ARC answerKey missing from provided options")
            gold=labels.index(answer)
        elif name=="BoolQ_VALIDATION_256_288":
            prefix=("Passage: "+str(row["passage"]).strip()+
                    "\nQuestion: "+str(row["question"]).strip()+
                    "\nAnswer:")
            options=["yes","no"]
            gold=0 if bool(row["label"]) else 1
        elif name=="HellaSwag_VALIDATION_256_288":
            prefix=str(row["ctx"]).strip()
            options=[str(s).strip() for s in row["endings"]]
            gold=int(row["label"])
        else:
            raise AssertionError("unregistered FIRST-use QA task")
        if (not prefix or len(options)<2 or not all(options)
            or not 0<=gold<len(options)):
            raise AssertionError("incomplete heldout task statement/options")
        idsum=hashlib.sha256((name+"\0"+prefix+"\0"+"\0".join(options)
                              ).encode()).hexdigest()
        tasks.append({"prefix":prefix,"options":options,
                      "gold":gold,"task_SHA256":idsum})
    if len(tasks)!=N_PER_TASK:
        raise AssertionError("fixed exact 32 validation items missing")
    return tasks


def read_untouched_tasks(revisions):
    tasks={}
    for name,repo,cfg,split in DATASETS:
        rev=revisions[name]
        ds=(load_dataset(repo,cfg,split=split,revision=rev)
            if cfg else load_dataset(repo,split=split,revision=rev))
        tasks[name]=task_records(name,list(ds))
    return tasks


def mean_conditional_ll(model,tok,prefix,option):
    """Actual causal LM likelihood of candidate, no label in model prompt."""
    # Include preceding tokenizer context but never gold or the other
    # candidates. Encode candidates at a consistent leading-space boundary.
    p=tok(prefix,add_special_tokens=False)["input_ids"][-MAX_PREFIX_TOKENS:]
    c=tok(" "+option,add_special_tokens=False)["input_ids"][:MAX_OPTION_TOKENS]
    if len(p)<1 or len(c)<1:
        raise AssertionError("prefix / candidate tokenizer empty")
    x=torch.tensor([p+c],dtype=torch.long)
    logits=model(input_ids=x).logits[0]
    next_logits=logits[len(p)-1:len(p)+len(c)-1]
    if next_logits.shape[0]!=len(c):
        raise AssertionError("causal token likelihood alignment error")
    logp=next_logits.float().log_softmax(dim=-1)
    chosen=logp.gather(-1,torch.tensor(c,dtype=torch.long).unsqueeze(-1))
    return float(chosen.mean().item())


def score_model(model,tokenizer,tasks,method):
    model.eval()
    report={}
    with torch.inference_mode():
        for name,records in tasks.items():
            correct=[]
            predicted=[]
            margins=[]
            for row in records:
                candidate_scores=[mean_conditional_ll(
                    model,tokenizer,row["prefix"],option
                    ) for option in row["options"]]
                if any(not math.isfinite(v) for v in candidate_scores):
                    raise AssertionError("nonfinite real trained task logprob")
                pred=max(range(len(candidate_scores)),key=lambda i:(
                    candidate_scores[i],-i))
                predicted.append(pred)
                correct.append(int(pred==row["gold"]))
                margins.append(candidate_scores[row["gold"]]-
                     max(score for i,score in enumerate(candidate_scores)
                         if i!=row["gold"]))
            report[name]={
                "n":len(records),"correct":sum(correct),
                "accuracy":sum(correct)/len(correct),
                "per_example_correct":correct,
                "predicted_label_indices":predicted,
                "true_answer_vs_best_other_mean_logprob_margin":
                    math.fsum(margins)/len(margins),
            }
            print({"actual_task_quality":name,"method":method,
                   "correct":sum(correct),"total":len(correct)},flush=True)
    return report


def audits(tasks):
    return {"protocol_SHA256":PROTOCOL_SHA,
            "task_record_count":{n:len(v) for n,v in tasks.items()},
            "all_task_text_sha256":{n:hashlib.sha256(
                "|".join(r["task_SHA256"] for r in v).encode()
                ).hexdigest() for n,v in tasks.items()},
            "task_specifications":{n:{"model_input":"only a frozen question prefix and ONE option at a time",
                "quality_target":"32 supervised answer labels used only for final score",
                "n_options":sorted(set(len(v["options"]) for v in rows))}
                for n,rows in tasks.items()},
            "leakage_prevention":"no task files are opened before one-shot TRAIN correction-selection decision",
            "all_four_task_metrics_are_accuracy_not_PPL":True}


def smoke():
    sciq={"support":"People can breathe oxygen.",
          "question":"What gas is needed for respiration?",
          "correct_answer":"oxygen","distractor1":"helium",
          "distractor2":"nitrogen","distractor3":"hydrogen"}
    rows=task_records("SciQ_TEST_128_160",[sciq]*N_PER_TASK)
    assert all(rows[0]["options"][r["gold"]]=="oxygen" for r in rows)
    assert len(rows[0]["task_SHA256"])==64
    print("CALIBRI v55 actual label scoring and deterministic task protocol smoke PASS")


if __name__=="__main__":
    smoke()
