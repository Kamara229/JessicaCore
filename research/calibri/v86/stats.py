"""Paired immutable question statistics; no learned source selection."""
from __future__ import annotations
import math
import numpy as np


def paired_agreement(base:list,adaptive:list):
    a=np.asarray(base,dtype=np.int64)
    b=np.asarray(adaptive,dtype=np.int64)
    if a.shape!=b.shape or a.ndim!=1 or not len(a):
        raise AssertionError("unpaired original source questions")
    old=int(((a==1)&(b==0)).sum())
    new=int(((a==0)&(b==1)).sum())
    n=old+new
    p=min(1.0,2*sum(math.comb(n,k) for k in
        range(min(old,new)+1))/(2**n)) if n else 1.0
    rng=np.random.default_rng(86000)
    changes=(b-a).astype(np.float64)
    draws=rng.integers(0,len(a),size=(5000,len(a)))
    lo,hi=np.quantile(changes[draws].mean(1),[.025,.975])
    return {"base_correct":int(a.sum()),
            "adaptive_correct":int(b.sum()),
            "paired_cases":len(a),
            "base_only_correct":old,
            "adaptive_only_correct":new,
            "exact_two_sided_McNemar_p":float(p),
            "accuracy_change":float(changes.mean()),
            "paired_bootstrap_95":[float(lo),float(hi)]}
