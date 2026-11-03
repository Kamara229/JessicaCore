"""Independent functional QA measures, disaggregated by domain and true answer.

Correctness and diverse useful generation are objectives. NO KL or NLL.
"""
from __future__ import annotations
import random

def summary(rows,num_classes):
    if not rows or num_classes<2:raise AssertionError("missing original task labels")
    counts=[0]*num_classes
    correct=[0]*num_classes
    predicted=[0]*num_classes
    for item in rows:
        g,p=item["gold"],item["pred"]
        if not (0<=g<num_classes and 0<=p<num_classes):
            raise AssertionError("label outside task")
        counts[g]+=1
        predicted[p]+=1
        correct[g]+=item["correct"]
        if item["correct"]!=int(g==p):raise AssertionError("incorrectly scored public original label")
    available=[correct[k]/counts[k] for k in range(num_classes) if counts[k]]
    return {"count":len(rows),"correct":sum(correct),"accuracy":sum(correct)/len(rows),
            "balanced_accuracy":sum(available)/len(available),
            "gold_counts":counts,"correct_by_gold":correct,"predicted_counts":predicted,
            "majority_answer_fraction":max(predicted)/len(rows)}

def paired(domain,a,b,num_classes):
    if len(a)!=len(b) or not a:raise AssertionError("unequal independent VALID questions")
    for x,y in zip(a,b):
        if (x["record_index"],x["prompt_SHA"],x["gold"])!=(
            y["record_index"],y["prompt_SHA"],y["gold"]):
            raise AssertionError("not paired identical original public questions")
    deltas=[y["correct"]-x["correct"] for x,y in zip(a,b)]
    r=random.Random(94000+len(a))
    count=len(deltas)
    draws=sorted(sum(deltas[r.randrange(count)] for _ in range(count))/count
                 for _ in range(3000))
    return {"domain":domain,"baseline":summary(a,num_classes),
            "rewritten":summary(b,num_classes),
            "correct_answer_gain":sum(deltas),
            "accuracy_gain":sum(deltas)/count,
            "paired_95_interval":[draws[74],draws[2924]],
            "regressed_examples":sum(x["correct"]==1 and y["correct"]==0 for x,y in zip(a,b)),
            "recovered_examples":sum(x["correct"]==0 and y["correct"]==1 for x,y in zip(a,b)),
            "statistical_evidence_gain":draws[74]>0,
            "paired_original_pinned_ground_truth":True}
