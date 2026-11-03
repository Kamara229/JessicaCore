"""Frozen before TEST paired estimation; TEST data only observed, never tuned."""
from __future__ import annotations
import numpy as np
from v86.stats import paired_agreement

def paired_kl(a,b,metric):
    if len(a)!=len(b) or not a:
        raise AssertionError("unpaired original TEST context windows")
    x=np.asarray([row[metric] for row in a],dtype=np.float64)
    y=np.asarray([row[metric] for row in b],dtype=np.float64)
    delta=y-x
    rng=np.random.default_rng(89000)
    ix=rng.integers(0,len(delta),size=(5000,len(delta)))
    low,high=np.quantile(delta[ix].mean(-1),[.025,.975])
    return {"mean_change":float(delta.mean()),
            "paired_95_bootstrap_interval":[float(low),float(high)],
            "original_independent_windows":len(delta),
            "confidence_interval_crosses_zero":bool(low<=0<=high)}
