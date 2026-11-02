#!/usr/bin/env python3
"""CALIBRI v49: CPU-reproducible INT6 row-precision budget allocation.

Pure original pretrained FP32 vs physically decoded INT6 weight error.
Compute exact-input FP32 source differences in Python binary64, stable
math.fsum, integer score quantization, deterministic module/row tie-break.
NO PPL, gradients, activation moments, external test texts or host-specific
matrix multiplications participate in row selection.

Two frozen alternatives:
  - SSE_GLOBAL: global error / real physical residual-row bytes
  - SSE_BALANCED_12PCT: first one high-SSE row per 2D trained tensor,
    then same global prioritization but maximum 12% of patch budget per
    tensor. This preserves layer diversity while charging full headers.
Both write genuine CR47 FP16 teacher-residual sidecars (<=6.5 total bpw).
"""
from __future__ import annotations

from collections import defaultdict
import math
from sparse_recovery_v47 import HEADER,TENSOR,ROW,FOOTER_BYTES

CAP_FRACTION_PER_TENSOR=0.12
SCORE_SCALE=10**15


def _score(w,q,i):
    """Bitwise reproducible float32 inputs, binary64 scalar error."""
    a=w[i].reshape(-1)
    b=q[i].reshape(-1)
    # Python float each input is an exact binary64 representation of FP32;
    # differences and math.fsum are deterministic independent of BLAS.
    return math.fsum((float(ai)-float(bi))**2 for ai,bi in zip(a,b))


def candidates(original,decoded):
    scores={}
    counts={}
    for name in sorted(original):
        w=original[name]
        if name not in decoded or w.ndim!=2 or not w.is_floating_point():
            continue
        q=decoded[name]
        if q.shape!=w.shape:
            raise AssertionError("original and physical INT6 dimensions mismatch")
        if not (w.isfinite().all() and q.isfinite().all()):
            raise AssertionError("nonfinite pretrained tensor or decoded INT6")
        m,n=w.shape
        counts[name]=m
        a=w.numpy()
        b=q.numpy()
        for i in range(m):
            cost=ROW.size+2*n
            raw=_score(a,b,i)
            # Unambiguous quantized fixed integer score. The integer
            # is computed solely from the two source-verified arrays.
            fixed=round(raw*SCORE_SCALE)
            if fixed>0:
                scores[(name,i)]=(fixed,cost)
    if not scores:
        raise AssertionError("source pretrained INT6 had no quantization loss")
    return scores,counts


def disk_bytes(rows,scores):
    if not rows:
        return 0
    modules=set(name for name,i in rows)
    return (HEADER.size+FOOTER_BYTES+
            sum(TENSOR.size+len(name.encode("utf-8")) for name in modules)+
            sum(scores[key][1] for key in rows))


def choose(scores,all_budget_bytes):
    if all_budget_bytes<8192:
        raise ValueError("not enough model byte budget to store FP16 corrections")
    # Stable fixed order independent of OS/NumPy/BLAS.
    ordered=sorted(scores, key=lambda k:(
        -(scores[k][0]/scores[k][1]),k[0],k[1]))
    per_tensor=defaultdict(list)
    for row in ordered:
        per_tensor[row[0]].append(row)
    plans={}
    for mode in ("sse_global","sse_balanced_12pct"):
        selected=[]
        selected_set=set()
        per_module_bytes=defaultdict(int)
        used=HEADER.size+FOOTER_BYTES
        cap=int(all_budget_bytes*CAP_FRACTION_PER_TENSOR)
        def add(row):
            nonlocal used
            name,i=row
            if row in selected_set:
                return
            header=(TENSOR.size+len(name.encode("utf-8"))
                    if per_module_bytes[name]==0 else 0)
            price=scores[row][1]+header
            if used+price>all_budget_bytes:
                return
            if (mode=="sse_balanced_12pct" and
                per_module_bytes[name]+price>cap):
                return
            used+=price
            per_module_bytes[name]+=price
            selected.append(row)
            selected_set.add(row)

        if mode=="sse_balanced_12pct":
            # One high-impact correction in every 2D trained tensor;
            # skip if individual row cannot fit tensor cap.
            for name in sorted(per_tensor):
                add(per_tensor[name][0])
        for row in ordered:
            add(row)
        if not selected or disk_bytes(selected,scores)!=used:
            raise AssertionError("physical row budget bookkeeping mismatch")
        if used>all_budget_bytes:
            raise AssertionError("trained residual rate overflow")
        plans[mode]={
            "selection":selected,
            "rows":len(selected),
            "actual_patch_bytes_expected":used,
            "module_counts":{n:sum(r[0]==n for r in selected)
                             for n in sorted(per_module_bytes)},
            "per_tensor_bytes":dict(sorted(per_module_bytes.items())),
        }
    return plans


def smoke():
    import numpy as np
    import torch
    a={"layer_a":torch.arange(48,dtype=torch.float32).reshape(8,6)/64,
       "layer_b":torch.arange(120,dtype=torch.float32).reshape(10,12)/128}
    b={name:(t*0.99).clone() for name,t in a.items()}
    s,_=candidates(a,b)
    p=choose(s,10000)
    for mode,r in p.items():
        if disk_bytes(r["selection"],s)!=r["actual_patch_bytes_expected"]:
            raise AssertionError("nonexact real-sidecar size in synthetic smoke")
    print("CALIBRI v49 deterministic INT6 physical row score/allocator smoke PASS")


if __name__=="__main__":
    smoke()
