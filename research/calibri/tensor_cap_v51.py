#!/usr/bin/env python3
"""CALIBRI v51: deterministic per-tensor precision-budget concentration test.

Frozen source-only physical INT6 row-error scores from v49; no data-driven
re-ranking, no validation/TEST tuning. Keep 6.50bpw at-rest model budget.
Only vary the maximum fraction of available CR47 sidecar bytes per tensor:
6%, 12% (reproduce v49), 24%. Select first positive-SSE row from every
trained matrix, then greedily allocate remainder by identical source-only
quantization error / real physical storage cost.
"""
from __future__ import annotations
from collections import defaultdict
from deterministic_budget_v49 import choose as v49_choose,disk_bytes

CAPS=(("sse_cap_06pct",6),("sse_cap_12pct",12),("sse_cap_24pct",24))


def allocate_cap(scores,budget,cap_percent):
    if cap_percent not in (6,12,24) or budget<8192:
        raise ValueError("unregistered cap or insufficient physical bytes")
    ordered=sorted(scores,key=lambda k:(-(scores[k][0]/scores[k][1]),k[0],k[1]))
    by_tensor=defaultdict(list)
    for row in ordered:
        by_tensor[row[0]].append(row)
    selected=[]
    seen=set()
    per_tensor_bytes=defaultdict(int)
    overhead=81  # dynamically replaced below, actual parent/header/CRC/SHA sizes
    from sparse_recovery_v47 import HEADER,FOOTER_BYTES,TENSOR
    overhead=HEADER.size+FOOTER_BYTES
    cap=(budget*cap_percent)//100
    used=overhead
    def add(row):
        nonlocal used
        if row in seen:
            return
        name,i=row
        tensor_header=TENSOR.size+len(name.encode("utf-8")) if not per_tensor_bytes[name] else 0
        price=scores[row][1]+tensor_header
        if (used+price>budget or per_tensor_bytes[name]+price>cap):
            return
        used+=price
        per_tensor_bytes[name]+=price
        selected.append(row)
        seen.add(row)
    for name in sorted(by_tensor):
        add(by_tensor[name][0])
    for row in ordered:
        add(row)
    if not selected or disk_bytes(selected,scores)!=used or used>budget:
        raise AssertionError("physical patch rate overflow/under-count")
    if any(nbytes>cap for nbytes in per_tensor_bytes.values()):
        raise AssertionError("per-model tensor precision cap violated")
    return {"selection":selected,"rows":len(selected),
            "actual_patch_bytes_expected":used,
            "module_counts":{name:len([1 for a,b in selected if a==name])
                             for name in sorted(per_tensor_bytes)},
            "per_tensor_bytes":dict(sorted(per_tensor_bytes.items())),
            "cap_percent":cap_percent}


def plans_for_budget(scores,budget):
    plans={name:allocate_cap(scores,budget,pct) for name,pct in CAPS}
    # 12% must reproduce the previously independently measured v49
    # physical source-only method exactly (not merely approximate bytes).
    frozen=v49_choose(scores,budget)["sse_balanced_12pct"]
    d=plans["sse_cap_12pct"]
    if (d["selection"]!=frozen["selection"] or
        d["actual_patch_bytes_expected"]!=frozen["actual_patch_bytes_expected"] or
        d["module_counts"]!=frozen["module_counts"]):
        raise AssertionError("frozen 12-percent rate control changed")
    return plans


def smoke():
    import torch
    from deterministic_budget_v49 import candidates
    src={"a":torch.arange(100,dtype=torch.float32).reshape(10,10)/100,
         "b":torch.arange(180,dtype=torch.float32).reshape(15,12)/100}
    dst={name:(v*0.994).clone() for name,v in src.items()}
    scores,_=candidates(src,dst)
    plans=plans_for_budget(scores,10000)
    for plan in plans.values():
        if disk_bytes(plan["selection"],scores)!=plan["actual_patch_bytes_expected"]:
            raise AssertionError("synthetic real sidecar CRC/metadata byte mismatch")
    print("CALIBRI v51 deterministic 6/12/24-percent tensor budget smoke PASS")


if __name__=="__main__":
    smoke()
