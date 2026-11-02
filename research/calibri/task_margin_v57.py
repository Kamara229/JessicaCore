#!/usr/bin/env python3
"""CALIBRI v57: frozen TRAIN-only paired answer-logprob margin stability.

Physical Pythia INT6 PRE and sparse FP16 correction candidates are
bit-identical to v56. We assess *continuous* gold-vs-best-wrong answer
log probability margin, not just 0/1 correct-count, in four genuinely
new TRAIN task partitions. No TEST labels are opened before choice.

The family-bootstrap lower-decile of FOUR family-mean differences is
a heuristic resampling stress test over four families; do not call this
a calibrated or population 90%-confidence bound.
"""
from __future__ import annotations
import hashlib,math,random,torch
from datasets import load_dataset
from real_tasks_v55 import mean_conditional_ll

METHODS=("base_int6_pre","sse_cap_06pct","sse_cap_12pct","sse_cap_24pct")
TASKS=("SciQ","ARCEasy","BoolQ","HellaSwag")
TRAIN_DEFS=(
    ("SciQ","allenai/sciq",None,"train[800:864]"),
    ("ARCEasy","allenai/ai2_arc","ARC-Easy","train[32:96]"),
    ("BoolQ","super_glue","boolq","train[544:608]"),
    ("HellaSwag","Rowan/hellaswag",None,"train[32:96]"),
)
BLIND_DEFS=(
    ("SciQ","allenai/sciq",None,"test[224:352]"),
    ("ARCEasy","allenai/ai2_arc","ARC-Easy","test[96:224]"),
    ("BoolQ","super_glue","boolq","validation[352:480]"),
    ("HellaSwag","Rowan/hellaswag",None,"validation[352:480]"),
)
TRAIN_N=64
BLIND_N=128
SEED=57
BOOT_DRAWS=1024
LOWER_DECILE_INDEX=102
MIN_POSITIVE_TASK_FAMILIES=3
MIN_MACRO_MARGIN_GAIN=0.001
MIN_LOWER_DECILE_GAIN=0.0005
MAX_NEGATIVE_SINGLE_FAMILY=-0.020
MAX_CORRECT_LOSS_PER_FAMILY=1

SCORING_SHA=hashlib.sha256((
  str(TRAIN_DEFS)+str(BLIND_DEFS)+
  "pretrained-Pythia; max 128 prefix tokens; 32 continuation tokens;"
  "option mean log P; correct margin versus best wrong option;"
  "SciQ option hash shuffle; model-independent; all gold labels held")
  .encode()).hexdigest()


def row_from_task(task,row):
    if task=="SciQ":
        prefix=("Support: "+str(row["support"]).strip()+
                "\nQuestion: "+str(row["question"]).strip()+
                "\nAnswer:")
        options=[str(row[k]).strip() for k in (
            "correct_answer","distractor1","distractor2","distractor3")]
        if len(set(options))!=4:
            raise AssertionError("SciQ duplicated answer")
        pairs=list(enumerate(options))
        pairs.sort(key=lambda p:(hashlib.sha256(
            (str(row["question"])+"|"+str(p[0])+"|"+p[1]).encode()
            ).hexdigest(),p[0]))
        gold=next(i for i,p in enumerate(pairs) if p[0]==0)
        options=[p[1] for p in pairs]
    elif task=="ARCEasy":
        prefix="Question: "+str(row["question"]).strip()+"\nAnswer:"
        labels=[str(x).strip() for x in row["choices"]["label"]]
        options=[str(x).strip() for x in row["choices"]["text"]]
        key=str(row["answerKey"]).strip()
        if len(labels)!=len(options) or key not in labels:
            raise AssertionError("ARC answer label does not exist")
        gold=labels.index(key)
    elif task=="BoolQ":
        prefix=("Passage: "+str(row["passage"]).strip()+
                "\nQuestion: "+str(row["question"]).strip()+
                "\nAnswer:")
        options=["yes","no"]
        gold=0 if bool(row["label"]) else 1
    elif task=="HellaSwag":
        prefix=str(row["ctx"]).strip()
        options=[str(x).strip() for x in row["endings"]]
        gold=int(row["label"])
    else:
        raise AssertionError("task not preregistered")
    if not prefix or len(options)<2 or not all(options) or not 0<=gold<len(options):
        raise AssertionError("incomplete task answer")
    h=hashlib.sha256((task+"\0"+prefix+"\0"+"\0".join(options)).encode()).hexdigest()
    return {"prompt":prefix,"options":options,"gold":gold,"sha256":h}


def load_fixed_tasks(defs,revisions,n):
    tasks={}
    for task,repo,config,split in defs:
        ds=(load_dataset(repo,config,revision=revisions[task],split=split)
            if config else load_dataset(repo,revision=revisions[task],split=split))
        tasks[task]=[row_from_task(task,row) for row in ds]
        if len(tasks[task])!=n:
            raise AssertionError(f"{task} frozen record count mismatch expected {n}")
    if set(tasks)!=set(TASKS):
        raise AssertionError("one task family was omitted")
    return tasks


def tasks_sha(tasks,defs):
    return {"frozen_task_scoring_SHA256":SCORING_SHA,
            "splits":{task:split for task,repo,config,split in defs},
            "items_per_task":{task:len(rows) for task,rows in tasks.items()},
            "prompt_and_answer_options_sha256":{task:hashlib.sha256(
                "|".join(r["sha256"] for r in rows).encode()).hexdigest()
                for task,rows in tasks.items()}}


def evaluate_real_task_margins(model,tok,tasks,name):
    model.eval()
    output={}
    with torch.inference_mode():
        for family,rows in tasks.items():
            correct=[]
            guessed=[]
            margins=[]
            for r in rows:
                scores=[mean_conditional_ll(model,tok,r["prompt"],option)
                        for option in r["options"]]
                if any(not math.isfinite(v) for v in scores):
                    raise AssertionError("nonfinite answer candidate likelihood")
                pred=max(range(len(scores)),key=lambda i:(scores[i],-i))
                correct.append(int(pred==r["gold"]))
                guessed.append(pred)
                margins.append(scores[r["gold"]]-max(
                    v for i,v in enumerate(scores) if i!=r["gold"]))
            output[family]={
                "n":len(rows),"correct":sum(correct),
                "accuracy":sum(correct)/len(rows),
                "per_item_correct":correct,
                "predicted_option_indices":guessed,
                "gold_logprob_margin_per_item":margins,
                "mean_gold_logprob_margin":math.fsum(margins)/len(rows)}
            print(f"CALIBRI v57 {name} {family}: {sum(correct)}/{len(rows)}",flush=True)
    return output


def frozen_margin_gate(bootstrap,train):
    if set(train)!=set(METHODS) or any(set(train[m])!=set(TASKS)
                                      for m in METHODS):
        raise AssertionError("missing task family TRAIN margin source")
    evidence={}
    source=train["base_int6_pre"]
    for method in METHODS[1:]:
        family={}
        changes={}
        for task in TASKS:
            a=source[task]["gold_logprob_margin_per_item"]
            b=train[method][task]["gold_logprob_margin_per_item"]
            if len(a)!=TRAIN_N or len(b)!=TRAIN_N:
                raise AssertionError("trained 64 answer margins required")
            family[task]=math.fsum(y-x for x,y in zip(a,b))/TRAIN_N
            changes[task]=train[method][task]["correct"]-source[task]["correct"]
        rng=random.Random(SEED)
        draws=[]
        for _ in range(BOOT_DRAWS):
            sample=[]
            for _ in TASKS:
                task=TASKS[rng.randrange(len(TASKS))]
                # Resample ONE of four equally-sized 16-question training
                # blocks, not an individual correlated answer token.
                block=rng.randrange(4)
                a=source[task]["gold_logprob_margin_per_item"]
                b=train[method][task]["gold_logprob_margin_per_item"]
                sample.append(math.fsum(b[i]-a[i] for i in range(
                    block*16,(block+1)*16))/16)
            draws.append(math.fsum(sample)/4)
        draws.sort()
        lower=draws[LOWER_DECILE_INDEX]
        macro=math.fsum(family.values())/len(family)
        positive=sum(v>0 for v in family.values())
        passed_bootstrap=bootstrap["candidate_training_stability"][method][
            "passes_frozen_heuristic_gate"]
        eligible=(passed_bootstrap and
                  positive>=MIN_POSITIVE_TASK_FAMILIES and
                  macro>=MIN_MACRO_MARGIN_GAIN and
                  min(family.values())>=MAX_NEGATIVE_SINGLE_FAMILY and
                  lower>=MIN_LOWER_DECILE_GAIN and
                  min(changes.values())>=-MAX_CORRECT_LOSS_PER_FAMILY)
        evidence[method]={
            "family_mean_relative_answer_margin_gain":family,
            "TRAIN_correct_change_by_family":changes,
            "positive_margin_families":positive,
            "equal_family_macro_margin_gain":macro,
            "worst_family_margin_gain":min(family.values()),
            "family_block_bootstrap_lower_decile_gain":lower,
            "family_block_bootstrap_samples":BOOT_DRAWS,
            "family_block_bootstrap_seed":SEED,
            "NLL_preapproval":passed_bootstrap,
            "eligible":eligible,
        }
    elig=[m for m in METHODS[1:] if evidence[m]["eligible"]]
    chosen=max(elig,key=lambda m:(
        evidence[m]["family_block_bootstrap_lower_decile_gain"],
        evidence[m]["equal_family_macro_margin_gain"],
        -METHODS.index(m))) if elig else "base_int6_pre"
    return {
        "chosen":chosen,
        "frozen_rule":"NLL-bootstrap-eligible; task margin gain >0 on 3/4 TRAIN families, macro >=.001, worst >=-.020, task block bootstrap lower decile >=.0005, no TRAIN family loses >1/64 correct, else PRE",
        "task_names":list(TASKS),"seed":SEED,
        "draws":BOOT_DRAWS,"lower_decile_index":LOWER_DECILE_INDEX,
        "TRAIN_margin_evidence":evidence,
        "full_64_each_task_TRAIN_metrics":train,
        "NLL_bootstrap_preapproval":bootstrap["chosen"],
        "is_not_statistical_population_CI":True,
    }


def smoke():
    baseline={}
    for name in TASKS:
        baseline[name]={"correct":16,"gold_logprob_margin_per_item":[0.]*64}
    improved={name:{"correct":16,"gold_logprob_margin_per_item":[.02]*64}
              for name in TASKS}
    allbad={name:{"correct":16,"gold_logprob_margin_per_item":[-.03]*64}
            for name in TASKS}
    train={"base_int6_pre":baseline,
           "sse_cap_06pct":improved,"sse_cap_12pct":allbad,
           "sse_cap_24pct":allbad}
    before={"chosen":"sse_cap_06pct",
            "candidate_training_stability":{m:{
                 "passes_frozen_heuristic_gate":m=="sse_cap_06pct"}
                 for m in METHODS[1:]}}
    assert frozen_margin_gate(before,train)["chosen"]=="sse_cap_06pct"
    train["sse_cap_06pct"]=baseline
    assert frozen_margin_gate(before,train)["chosen"]=="base_int6_pre"
    print("CALIBRI v57 paired task margin bootstrap TRAIN-only gate smoke PASS")


if __name__=="__main__":
    smoke()
