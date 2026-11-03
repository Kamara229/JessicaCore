"""Outcome-first: correctness by true answer plus paired uncertainty, never KL/NLL."""
from __future__ import annotations
import random

def class_report(rows):
    by_gold={}
    for gold,name in ((0,"yes"),(1,"no")):
        subset=[r for r in rows if r["gold"]==gold]
        by_gold[name]={"count":len(subset),"correct":sum(r["correct"] for r in subset),
            "recall":sum(r["correct"] for r in subset)/len(subset) if subset else None}
    count=sum(r["correct"] for r in rows)
    valid=[x["recall"] for x in by_gold.values() if x["recall"] is not None]
    return {"count":len(rows),"correct":count,"accuracy":count/len(rows),
            "by_gold":by_gold,"balanced_accuracy":sum(valid)/len(valid)
             if len(valid)==2 else None}

def compare_answers(rows):
    a=rows["original"];b=rows["candidate"]
    if len(a)!=len(b) or not a:raise AssertionError("unpaired answer grading")
    delta=[y["correct"]-x["correct"] for x,y in zip(a,b)]
    rng=random.Random(93000)
    n=len(delta)
    draws=sorted(sum(delta[rng.randrange(n)] for _ in range(n))/n
                 for _ in range(5000))
    return {"original":class_report(a),"candidate":class_report(b),
            "correct_answer_gain":sum(delta),
            "accuracy_gain":sum(delta)/n,
            "paired_bootstrap_95_interval":[draws[124],draws[4874]],
            "candidate_only_correct":sum(x["correct"]==0 and y["correct"]==1 for x,y in zip(a,b)),
            "original_only_correct":sum(x["correct"]==1 and y["correct"]==0 for x,y in zip(a,b)),
            "claim_statistically_proven":draws[124]>0}
