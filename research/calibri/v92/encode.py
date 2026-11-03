"""Rewrite one *entire* compressed 2D block using same 5-bit native ABI and bytes.
No sidecar, no dense runtime weights; optimize all codes and shared 8D centroids.
TRAIN-only input sensitivity weights, strictly disjoint VALID ranking.
"""
from __future__ import annotations
import numpy as np
from direct_codebook_v59 import VECTOR,K,pack5,unpack5

def weighted_distortion(w,indices,centroids,scales,diag):
    m,n=w.shape;g=n//VECTOR
    raw=w.reshape(m,g,VECTOR).astype(np.float64)
    decoded=centroids[indices.reshape(m,g)].astype(np.float64)*scales[:,None,None]
    val=diag.reshape(1,g,VECTOR)
    error=raw-decoded
    total=(error**2*val).sum()
    denom=(raw**2*val).sum()
    return float(total/max(denom,1e-30))

def reencode_8d(w,original_handle,train_diag,valid_diag,iterations=4):
    weight=np.asarray(w,dtype=np.float32)
    if (weight.ndim!=2 or tuple(weight.shape)!=tuple(original_handle.shape)
        or weight.shape[1]%VECTOR or not np.isfinite(weight).all()):
        raise AssertionError("full block pretrained geometry mismatch")
    if iterations<1 or iterations>8:raise AssertionError("bounded rewrite steps required")
    m,n=weight.shape;g=n//VECTOR
    original_scales=np.asarray(original_handle.row_scales,dtype=np.float32).copy()
    if np.any(original_scales<=0):raise AssertionError("invalid per-row original FP16 scales")
    train=np.clip(np.asarray(train_diag,dtype=np.float64),1e-8,None)
    valid=np.clip(np.asarray(valid_diag,dtype=np.float64),1e-8,None)
    if train.shape!=(n,) or valid.shape!=(n,) or not np.isfinite(train).all() or not np.isfinite(valid).all():
        raise AssertionError("invalid independent TRAIN/VALID activation-energy geometry")
    # Source reference for in-place reconstruction: no extra degrees of freedom,
    # no new per-position metadata or paid storage.
    centers=np.asarray(original_handle.centroids,dtype=np.float32).copy()
    scales=original_scales
    x=(weight.reshape(m,g,VECTOR)/scales[:,None,None]).reshape(-1,VECTOR)
    # Only 32 codebook entries x vector width 8. Per-column weighted least squares.
    wgt=np.broadcast_to(train.reshape(1,g,VECTOR),(m,g,VECTOR)).reshape(-1,VECTOR).copy()
    wgt*=np.repeat(scales.astype(np.float64)**2,g)[:,None]
    original_idx=unpack5(original_handle.packed.tobytes(),m*g)
    original_val=weighted_distortion(weight,original_idx,centers,scales,valid)
    best=(original_val,original_idx,centers.astype("<f2").copy())
    for _ in range(iterations):
        index=np.empty(len(x),dtype=np.uint8)
        for start in range(0,len(x),1024):
            end=min(start+1024,len(x))
            diff=x[start:end,None,:].astype(np.float64)-centers[None,:,:]
            distance=np.sum(diff**2*wgt[start:end,None,:],axis=-1)
            index[start:end]=np.argmin(distance,axis=1).astype(np.uint8)
        numerator=np.zeros((K,VECTOR),dtype=np.float64)
        denominator=np.zeros((K,VECTOR),dtype=np.float64)
        np.add.at(numerator,index,x.astype(np.float64)*wgt)
        np.add.at(denominator,index,wgt)
        ok=denominator>0
        updated=centers.astype(np.float64)
        updated[ok]=numerator[ok]/denominator[ok]
        centers=np.asarray(np.clip(updated,-60000,60000),dtype="<f2").astype(np.float32)
        if not np.isfinite(centers).all():
            raise AssertionError("nonfinite rewritten codebook")
        # Reassign all positions after codebook update on *TRAIN* metric.
        for start in range(0,len(x),1024):
            end=min(start+1024,len(x))
            diff=x[start:end,None,:].astype(np.float64)-centers[None,:,:]
            index[start:end]=np.argmin(np.sum(diff**2*wgt[start:end,None,:],axis=-1),axis=1)
        val=weighted_distortion(weight,index,centers,scales,valid)
        if val<best[0]:
            best=(val,index.copy(),centers.astype("<f2").copy())
    return {"indices":best[1],"centroids":best[2],
            "row_scales":scales.astype("<f2").copy(),
            "baseline_VALID_diagonal_weighted_MSE":original_val,
            "rewrite_VALID_diagonal_weighted_MSE":best[0],
            "validation_gain":original_val-best[0],
            "rewrite_attempted":True,
            "change_accepted_by_LOCAL_metric":best[0]<original_val,
            "physical_segment_bytes":len(pack5(best[1]))+
                len(scales.astype("<f2").tobytes())+len(best[2].tobytes())}
