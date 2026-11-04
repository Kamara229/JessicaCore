"""Source-only fixed-rate Lloyd optimization; no language labels, teacher logits.

Original first layer dictionary was fitted on only 4096 equally spaced vectors.
Here compare original code assignments to identical-byte-budget versions fitted
on disjoint 98304 source vector samples. Fixed heldout vector groups quantify
weight distortion only; they are not an independent language-quality TEST.
"""
import numpy as np
from direct_codebook_v59 import _nearest_integer_128


def lloyd(vectors, initial, *, energy=None, passes=8):
    x=np.asarray(vectors,dtype=np.float32)
    centers=np.asarray(initial,dtype="<f2").astype(np.float32).copy()
    if x.ndim!=2 or centers.ndim!=2 or x.shape[1]!=centers.shape[1]:
        raise ValueError("incompatible vector/centroid dimension")
    if len(centers)<2 or not 1<=passes<=16:
        raise ValueError("invalid codebook geometry")
    energy=np.ones(len(x),dtype=np.float64) if energy is None else np.asarray(
        energy,dtype=np.float64)
    if energy.shape!=(len(x),) or np.any(energy<=0) or not np.isfinite(energy).all():
        raise ValueError("invalid row-energy weighting")
    for _ in range(passes):
        assigned=_nearest_integer_128(x,centers)
        counts=np.bincount(assigned,weights=energy,minlength=len(centers))
        sums=np.zeros_like(centers,dtype=np.float64)
        np.add.at(sums,assigned,x.astype(np.float64)*energy[:,None])
        keep=counts>0
        centers[keep]=(sums[keep]/counts[keep,None]).astype("<f2").astype(np.float32)
        if not np.isfinite(centers).all():raise ValueError("nonfinite codebook")
    return centers.astype("<f2").astype(np.float32)


def product_centers(vectors,centers,passes=8):
    """Cartesian 8×4 4D dictionaries, encoded in the SAME 5bits/8 weights."""
    x=np.asarray(vectors,dtype=np.float32)
    base=np.asarray(centers,dtype=np.float32)
    if x.shape[1]!=8 or base.shape!=(32,8):raise ValueError("not V8/K32")
    left=lloyd(x[:,:4],base[np.linspace(0,31,8,dtype=np.int64),:4],passes=passes)
    right=lloyd(x[:,4:],base[np.linspace(0,31,4,dtype=np.int64),4:],passes=passes)
    combined=np.vstack([np.concatenate((a,b)) for a in left for b in right])
    if combined.shape!=(32,8):raise AssertionError("product dictionary lost exact rate")
    return combined.astype("<f2").astype(np.float32)


def score(actual,normalized,scales,centers,*,source_codes=None):
    centroids=np.asarray(centers,dtype=np.float32)
    if centroids.shape!=(32,8):raise ValueError("wrong physical centroid geometry")
    codes=(_nearest_integer_128(normalized,centroids)
           if source_codes is None else np.asarray(source_codes,dtype=np.int64))
    if codes.shape!=(len(actual),) or np.any(codes<0) or np.any(codes>31):
        raise ValueError("invalid literal 5-bit code assignment")
    reconstruct=centroids[codes]*scales[:,None]
    err=actual.astype(np.float64)-reconstruct.astype(np.float64)
    ref=actual.astype(np.float64)
    ratio=float(np.sqrt((err*err).sum()/max((ref*ref).sum(),1e-14)))
    return {
        "relative_weight_RMSE":ratio,
        "normalized_vector_MSE":float(np.mean(np.sum(
            (normalized.astype(np.float64)-centroids[codes])**2,axis=1))),
        "max_abs_reconstruction_error":float(np.max(np.abs(err))),
        "distinct_5bit_codes_on_valid":int(len(np.unique(codes))),
    }
