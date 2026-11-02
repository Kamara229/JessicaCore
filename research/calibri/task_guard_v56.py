#!/usr/bin/env python3
"""CALIBRI v56: frozen TRAIN-only task answer correctness guard.

A task label can affect INT6 *model selection* only when it belongs
to these explicitly preregistered TRAIN partitions. New separate
heldout TEST/validation record windows are not loaded until selection
is completed and pinned, and cannot influence row-precision ranks.
"""
from __future__ import annotations
import hashlib
import math
import torch
from datasets import load_dataset
from real_tasks_v55 import mean_conditional_ll

TRAIN_DATA=(
    ("SciQ","allenai/sciq",None,"train[768:800]"),
    ("ARCEasy","allenai/ai2_arc","ARC-Easy","train[:32]"),
    ("BoolQ","super_glue","boolq","train[512:544]"),
    ("HellaSwag","Rowan/hellaswag",None,"train[:32]"),
)
UNTOUCHED_TEST=(
    ("SciQ","allenai/sciq",None,"test[160:224]"),
    ("ARCEasy","allenai/ai2_arc","ARC-Easy","test[32:96]"),
    ("BoolQ","super_glue","boolq","validation[288:352]"),
    ("HellaSwag","Rowan/hellaswag",None,"validation[288:352]"),
)
TRAIN_N=32
TEST_N=64
METHODS=("base_int6_pre","sse_cap_06pct","sse_cap_12pct","sse_cap_24pct")
TASKS=("SciQ","ARCEasy","BoolQ","HellaSwag")
PROTOCOL="conditional mean logprob of each answer; prompt <=128 tokens and continuation <=32; correct-gold absent from model input; fixed tokenization"
PROTOCOL_SHA=hashlib.sha256((str(TRAIN_DATA)+str(UNTOUCHED_TEST)+
                            PROTOCOL).encode()).hexdigest()


def task_records(task,rows,expected):
    result=[]
    for row in rows:
        if task=="SciQ":
            prefix=("Support: "+str(row["support"]).strip()+
                    "\nQuestion: "+str(row["question"]).strip()+
                    "\nAnswer:")
            choices=[str(row[k]).strip() for k in (
                "correct_answer","distractor1","distractor2","distractor3")]
            if len(set(choices))!=4:
                raise AssertionError("SciQ has duplicated answer choices")
            # Exactly the same preregistered deterministic shuffle as v55.
            shuffle=list(enumerate(choices))
            shuffle.sort(key=lambda p:(hashlib.sha256(
                 (str(row["question"])+"|"+str(p[0])+"|"+p[1]).encode()
                 ).hexdigest(),p[0]))
            gold=next(i for i,p in enumerate(shuffle) if p[0]==0)
            options=[p[1] for p in shuffle]
        elif task=="ARCEasy":
            prefix="Question: "+str(row["question"]).strip()+"\nAnswer:"
            labels=[str(x).strip() for x in row["choices"]["label"]]
            options=[str(x).strip() for x in row["choices"]["text"]]
            answer=str(row["answerKey"]).strip()
            if len(labels)!=len(options) or answer not in labels:
                raise AssertionError("ARCEasy answer key absent from labels")
            gold=labels.index(answer)
        elif task=="BoolQ":
            prefix=("Passage: "+str(row["passage"]).strip()+
                    "\nQuestion: "+str(row["question"]).strip()+
                    "\nAnswer:")
            options=["yes","no"]
            gold=0 if bool(row["label"]) else 1
        elif task=="HellaSwag":
            prefix=str(row["ctx"]).strip()
            options=[str(s).strip() for s in row["endings"]]
            gold=int(row["label"])
        else:
            raise AssertionError("unregistered real QA family")
        if (not prefix or len(options)<2 or
            not all(options) or not 0<=gold<len(options)):
            raise AssertionError("bad labeled task prompt/options")
        record_text=task+"\0"+prefix+"\0"+"\0".join(options)
        result.append({"prefix":prefix,"options":options,"gold":gold,
                       "record_sha256":hashlib.sha256(
                           record_text.encode()).hexdigest()})
    if len(result)!=expected:
        raise AssertionError(f"frozen {task} slice expected {expected} cases")
    return result


def read_tasks(definitions,revisions,expected):
    tasks={}
    for name,repo,cfg,split in definitions:
        rev=revisions[name]
        ds=(load_dataset(repo,cfg,split=split,revision=rev) if cfg
            else load_dataset(repo,split=split,revision=rev))
        tasks[name]=task_records(name,list(ds),expected)
    return tasks


def score_choices(model,tokenizer,tasks,method):
    model.eval()
    report={}
    with torch.inference_mode():
        for task,rows in tasks.items():
            correct=[]
            predictions=[]
            margins=[]
            for rec in rows:
                confidences=[mean_conditional_ll(
                    model,tokenizer,rec["prefix"],option)
                    for option in rec["options"]]
                if any(not math.isfinite(v) for v in confidences):
                    raise AssertionError("nonfinite candidate log likelihood")
                pred=max(range(len(confidences)),key=lambda i:(
                     confidences[i],-i))
                correct.append(int(pred==rec["gold"]))
                predictions.append(pred)
                margins.append(confidences[rec["gold"]]-max(
                    v for i,v in enumerate(confidences) if i!=rec["gold"]))
            report[task]={
                "n":len(rows),"correct":sum(correct),
                "accuracy":sum(correct)/len(rows),
                "per_item_correct":correct,
                "predicted_option_indices":predictions,
                "average_correct_minus_best_wrong_logprob":
                     math.fsum(margins)/len(rows)}
            print({"real_task":task,"method":method,
                   "correct":sum(correct),"n":len(rows)},flush=True)
    return report


def audit_source(tasks,definitions):
    return {
        "fixed_scoring_protocol_sha256":PROTOCOL_SHA,
        "label_usage":"TRAIN labels used solely for accept/reject of pretrained physical INT6 repaired model; heldout labels read only after gate",
        "task_split":{t:split for t,repo,cfg,split in definitions},
        "task_n":{name:len(rows) for name,rows in tasks.items()},
        "per_task_question_and_options_sha256":{
            name:hashlib.sha256("|".join(
                rec["record_sha256"] for rec in rows).encode()).hexdigest()
            for name,rows in tasks.items()}
    }


def task_guard(preapproved,train_results):
    """One fixed gate, frozen before any new heldout questions.

    Candidate is eligible only if it passes the unchanged v55 family
    bootstrap NLL gate AND all TRAIN-task safeguards:
       - total correct across four tasks at least one more than PRE
       - no training family loses >1 correct of 32
       - >=3 of 4 training families do not lose any correct.
    Rank eligible by overall additional correct, then original v55
    bootstrap 10th-percentile NLL gain, then smaller correction cap.
    Otherwise fall back to same physical plain INT6 PRE checkpoint.
    """
    baseline=train_results["base_int6_pre"]
    if set(preapproved["candidate_training_stability"])!=set(METHODS[1:]):
        raise AssertionError("missing independent NLL family stability scores")
    if set(train_results)!=set(METHODS) or any(
            set(train_results[k])!=set(TASKS) for k in METHODS):
        raise AssertionError("unregistered task-level training evidence")
    evidence={}
    for method in METHODS[1:]:
        gain={task:train_results[method][task]["correct"]-
                   baseline[task]["correct"] for task in TASKS}
        total=sum(gain.values())
        nonregress=sum(v>=0 for v in gain.values())
        passed_nll=preapproved["candidate_training_stability"][method][
                 "passes_frozen_heuristic_gate"]
        eligible=(passed_nll and total>=1 and min(gain.values())>=-1
                  and nonregress>=3)
        evidence[method]={
            "TRAIN_correct_delta_by_family":gain,
            "TRAIN_total_extra_correct":total,
            "TRAIN_nonregressing_families":nonregress,
            "passed_frozen_NLL_bootstrap":passed_nll,
            "eligible":eligible}
    valid=[method for method in METHODS[1:] if evidence[method]["eligible"]]
    chosen=(max(valid,key=lambda k:(
        evidence[k]["TRAIN_total_extra_correct"],
        preapproved["candidate_training_stability"][k][
            "bootstrap_tenth_percentile_NLL_gain"],
        -METHODS.index(k))) if valid else "base_int6_pre")
    return {
        "chosen":chosen,
        "rule":"v55 preregistered bootstrap NLL passed AND TRAIN 128 QA items: >=1 total extra correct, no task loses >1, >=3/4 tasks do not regress; else fallback plain physical PRE INT6",
        "method_order":list(METHODS),
        "TRAIN_only_evidence":evidence,
        "full_training_32x4_task_accuracy":train_results,
        "NLL_bootstrap_decision_as_independent_guard":preapproved["chosen"]
    }


def smoke():
    family={k:{"correct":10} for k in TASKS}
    scores={"base_int6_pre":family}
    for m in METHODS[1:]:
        scores[m]={k:{"correct":10} for k in TASKS}
    good={k:{"correct":10} for k in TASKS}
    good["SciQ"]={"correct":11}
    scores["sse_cap_06pct"]=good
    pre={"chosen":"sse_cap_06pct",
         "candidate_training_stability":{m:{
         "passes_frozen_heuristic_gate":m=="sse_cap_06pct",
         "bootstrap_tenth_percentile_NLL_gain":0.01}
         for m in METHODS[1:]}}
    assert task_guard(pre,scores)["chosen"]=="sse_cap_06pct"
    scores["sse_cap_06pct"]=family
    assert task_guard(pre,scores)["chosen"]=="base_int6_pre"
    print("CALIBRI v56 real QA TRAIN guard fallback and nonregression smoke PASS")


if __name__=="__main__":
    smoke()
