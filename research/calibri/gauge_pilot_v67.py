#!/usr/bin/env python3
"""CALIBRI v67: synthetic gauge-equivalent V/O paired operator, fixed-rate K32V8.

This is ONLY a synthetic numerical falsification pilot, NOT pretrained-LLM evidence,
a complete 135M archive, or a native compressed inference benchmark.
"""
import json
import math
import sys
import numpy as np

D, K, G, SEED = 576, 32, 8, 6701

def codec(w, seed):
    rng = np.random.default_rng(seed)
    m,n=w.shape
    assert n%G==0
    scales = np.maximum(np.abs(w).max(axis=1),1e-10).astype(np.float16)
    blocks = (w/scales.astype(np.float32)[:,None]).reshape(-1,G).astype(np.float32)
    sample = blocks[rng.choice(len(blocks),min(1536,len(blocks)),replace=False)]
    centers = sample[rng.choice(len(sample),K,replace=False)].copy()
    norms = np.sum(blocks**2,axis=1)
    for _ in range(5):
        distance = norms[:,None]+np.sum(centers**2,axis=1)[None,:]-2*blocks@centers.T
        indices = np.argmin(distance,axis=1)
        new = np.zeros_like(centers)
        counts = np.bincount(indices,minlength=K)
        np.add.at(new,indices,blocks)
        keep = counts>0
        new[keep] /= counts[keep,None]
        new[~keep] = centers[~keep]
        centers=new
    centers=centers.astype(np.float16)
    c=centers.astype(np.float32)
    distance=norms[:,None]+np.sum(c**2,axis=1)[None,:]-2*blocks@c.T
    codes=np.argmin(distance,axis=1)
    q=(c[codes].reshape(m,n)*scales.astype(np.float32)[:,None])
    # Each source local tensor: ceil(5 bits per 8 weights), FP16 K32x8,
    # FP16 scale per row, fixed 64-byte descriptor. Not a whole-model file.
    physical=int(math.ceil(len(blocks)*5/8))+2*K*G+2*m+64
    return q,physical

def rel_response_error(x,v,o,target):
    out=(x@v.T)@o.T
    return float(np.linalg.norm(out-target)/max(float(np.linalg.norm(target)),1e-12))

def pilot(seed=SEED, spread=2.3):
    rng=np.random.default_rng(seed)
    v0=(rng.normal(size=(D,D))/math.sqrt(D)).astype(np.float32)
    o0=(rng.normal(size=(D,D))/math.sqrt(D)).astype(np.float32)
    a=np.exp(rng.uniform(-spread,spread,D)).astype(np.float32)
    # These artificial factors produce EXACTLY the same full-precision O @ V.
    v=a[:,None]*v0
    o=o0/a[None,:]
    # No teacher, task labels, validation, or activation data used to choose b.
    xvalid=rng.normal(size=(128,D)).astype(np.float32)
    original=(xvalid@v.T)@o.T
    b=np.sqrt(np.linalg.norm(o.astype(np.float64),axis=0)/
              np.maximum(np.linalg.norm(v.astype(np.float64),axis=1),1e-12))
    b=np.clip(b,1/16,16).astype(np.float32)
    variants={"identity":np.ones(D,dtype=np.float32),
              "balance_L2":b,
              "half_balance":np.sqrt(b).astype(np.float32),
              "inverse_balance_negative_control":(1/b).astype(np.float32)}
    reports=[]
    for name,d in variants.items():
        vg=(v*d[:,None]).astype(np.float32)
        og=(o/d[None,:]).astype(np.float32)
        exact=rel_response_error(xvalid,vg,og,original)
        qv,bytes_v=codec(vg,17)
        qo,bytes_o=codec(og,29)
        err=rel_response_error(xvalid,qv,qo,original)
        reports.append({"gauge":name,
                        "exact_equivalence_rel_error":round(exact,8),
                        "quantized_heldout_response_rel_error":round(err,6),
                        "physical_local_pair_bytes":bytes_v+bytes_o,
                        "physical_local_pair_bpw":round(8*(bytes_v+bytes_o)/(2*D*D),6)})
    assert all(z["physical_local_pair_bytes"]==reports[0]["physical_local_pair_bytes"] for z in reports)
    assert max(z["exact_equivalence_rel_error"] for z in reports)<1e-5
    return {"scope":"synthetic two-linear-layer only, NOT a trained neural model",
            "seed":seed,"spread":spread,"dimension":D,
            "codebook_entries":K,"vector_dim":G,"heldout_samples":128,
            "cases":reports,
            "balance_vs_identity_ratio":round(reports[1]["quantized_heldout_response_rel_error"]/
                                              reports[0]["quantized_heldout_response_rel_error"],6),
            "no_gauge_parameters_stored":True,"not_whole_model_0_7_claim":True}

if __name__=="__main__":
    spread=float(sys.argv[1]) if len(sys.argv)>1 else 2.3
    print(json.dumps(pilot(spread=spread),indent=2))
