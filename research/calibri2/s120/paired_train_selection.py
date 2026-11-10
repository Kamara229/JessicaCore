"""S120 TRAIN-only deterministic counterfactual pair selection.

Operate ONLY on the original 456 authorized S87 historical TRAIN keys
and original S101 frozen greedy predictions. No QA, heuristics keyed by
unknown heldout examples, or owner-token answer lookup during inference.
"""
from __future__ import annotations
from collections import defaultdict


def select_train_counterfactual_pairs(rows,routes,original,
                                      *,limit:int=8):
    if limit<1:
        raise ValueError("Need positive number of original TRAIN pairs")
    if (len(rows)!=456 or set(rows)!=set(routes)
            or set(rows)!=set(original)):
        raise AssertionError("Only complete immutable original 456 TRAIN can be paired")
    grouped=defaultdict(dict)
    for key in rows:
        if not isinstance(key,str) or key.count("/")!=2:
            raise AssertionError("Unexpected S87 TRAIN pair key format")
        prefix,side=key.rsplit("/",1)
        if side not in ("left","right") or side in grouped[prefix]:
            raise AssertionError("Original S87 TRAIN side duplicated")
        grouped[prefix][side]=key
    if len(grouped)!=228 or any(set(p)!={"left","right"} for p in grouped.values()):
        raise AssertionError("Incomplete S87 counterfactual TRAIN pairing")
    by_family=defaultdict(list)
    first_bad=set()
    original_good=set()
    for key,row in rows.items():
        if original[key]==row["expected"]:
            original_good.add(key)
        if not original[key] or original[key][0]!=row["expected"][0]:
            first_bad.add(key)
    for ident,paired in sorted(grouped.items()):
        left,right=paired["left"],paired["right"]
        if (routes[left]!="S93" or routes[right]!="S93"
                or not ({left,right}&first_bad)):
            continue
        by_family[ident.split("/",1)[0]].append((left,right))
    selected=[]
    while len(selected)<limit and any(by_family.values()):
        for family in sorted(by_family):
            if len(selected)==limit:break
            if by_family[family]:
                selected.append(by_family[family].pop(0))
    if len(selected)!=limit:
        raise AssertionError("Insufficient original S93 paired first-token error TRAIN cases")
    if any(len(set(p))!=2 for p in selected):
        raise AssertionError("Pair duplicates same original TRAIN side")
    keys=[x for pair in selected for x in pair]
    if len(keys)!=2*limit or len(set(keys))!=2*limit:
        raise AssertionError("Repeated selected TRAIN identity across pairs")
    protected_counterparts=sorted(set(keys)&original_good)
    return {
        "pairs":tuple(tuple(pair) for pair in selected),
        "training_keys":tuple(keys),
        "first_wrong_keys":tuple(sorted(set(keys)&first_bad)),
        "protected_counterparts":tuple(protected_counterparts),
        "family_counts":{
            family:sum(k[0].split("/",1)[0]==family for k in selected)
            for family in sorted({p[0].split("/",1)[0] for p in selected})
        },
    }
