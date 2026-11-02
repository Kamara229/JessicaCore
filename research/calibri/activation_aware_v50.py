#!/usr/bin/env python3
"""CALIBRI v50: TRAIN-A activation-weighted, exact-byte INT6 row reallocation.

Keeps v49 balanced row quotas PER TENSOR unchanged, including all non-MLP
rows. For each of 12 real Pythia MLP matrices, move its exact count of
FP16 error-repair rows to those minimizing a diagonal approximation to
activation-weighted output perturbation. No test or TRAIN B selection.
Predeclared conditioning: normalize input moments by their median, clip
to [0.25,4], quantize to nearest 1/16. No BLAS or gradients for row
ranking, deterministic id tie-break and exact full storage accounting.
"""
from __future__ import annotations
from collections import Counter
import math
import numpy as np
from deterministic_budget_v49 import disk_bytes


def activation_matched_rows(original,decoded,captures,balanced_rows,scores):
    quotas=Counter(name for name,row in balanced_rows)
    result=[(name,row) for name,row in balanced_rows if name not in {
        module+".weight" for module in captures}]
    diagnostics={}
    for module in sorted(captures):
        key=module+".weight"
        k=quotas.get(key,0)
        if k==0:
            diagnostics[key]={"selected":0,"changed":0}
            continue
        w=original[key]
        q=decoded[key]
        x=captures[module]
        if (w.ndim!=2 or q.shape!=w.shape or x.ndim!=2 or
            x.shape[1]!=w.shape[1]):
            raise AssertionError("activation-matched trained MLP tensor geometry mismatch")
        arr=x.detach().cpu().numpy().astype(np.float64)
        moment=np.mean(arr*arr,axis=0,dtype=np.float64)
        median=float(np.median(moment))
        if not math.isfinite(median) or median<=0:
            raise AssertionError("TRAIN-A activation second moment invalid")
        relative=np.clip(moment/median,0.25,4.)
        quantized=np.rint(relative*16.0).astype(np.int64)
        quantized=np.maximum(quantized,1)
        # Only 12 source-trained MLP projections are affected; input
        # weight per coordinate coarse-rounded to fixed sixteenths.
        delta=(w.detach().cpu().numpy().astype(np.float64)-
               q.detach().cpu().numpy().astype(np.float64))
        row_scores=np.sum(delta*delta*quantized[np.newaxis,:],
                          axis=1,dtype=np.float64)
        ranked=sorted(range(w.shape[0]),key=lambda i:(
            -round(float(row_scores[i])*10**14),i))
        chosen=[(key,int(i)) for i in ranked[:k]]
        original_local=set((name,row) for name,row in balanced_rows if name==key)
        result.extend(chosen)
        diagnostics[key]={
            "selected":k,
            "changed":len(set(chosen)-original_local),
            "weight_levels":len(set(int(v) for v in quantized.tolist())),
            "quantized_activation_moment_min":int(quantized.min()),
            "quantized_activation_moment_max":int(quantized.max())}
    if len(result)!=len(balanced_rows) or len(result)!=len(set(result)):
        raise AssertionError("different number of repaired trained source rows")
    a=disk_bytes(balanced_rows,scores)
    b=disk_bytes(result,scores)
    if a!=b:
        raise AssertionError("MLP reallocation changed exact physical CR47 byte rate")
    if Counter(name for name,row in result)!=quotas:
        raise AssertionError("reallocation changed per-source module row counts")
    return sorted(result),{"total_repaired_rows":len(result),
                           "exact_match_physical_bytes":a,
                           "same_tensor_row_quota":True,
                           "per_MLP_allocation_diagnostics":diagnostics}


def smoke():
    import torch
    w={"mlp.weight":torch.tensor([[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6]],
                                  dtype=torch.float32)}
    q={"mlp.weight":w["mlp.weight"]*0.9}
    xa={"mlp":torch.tensor([[0.1,2.0],[0.2,2.0],[0.1,2.0],[0.2,2.0]])}
    scores={(k,i):(100,6+2*v.shape[1]) for k,v in w.items()
            for i in range(v.shape[0])}
    selected=[("mlp.weight",0),("mlp.weight",1)]
    out,meta=activation_matched_rows(w,q,xa,selected,scores)
    assert len(out)==2 and meta["exact_match_physical_bytes"]==disk_bytes(selected,scores)
    print("CALIBRI v50 physical exact-byte MLP activation quota smoke PASS")


if __name__=="__main__":
    smoke()
