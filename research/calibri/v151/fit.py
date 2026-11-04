"""Second-rank source-only contextual residual fit, int3 output coding.

Fit exactly one stage after the existing physically paid CB149 first rank,
to ORIGINAL FP32 teacher logits on the previously frozen 504 TRAIN FIT
next-token positions. No v149 VALID, QA, v89 TEST or v151 VALID leakage.
Rank-two is ADDITIVE, not a refit of the first rank or original source.
"""
import numpy as np
from .protocol import (RANK2_POWER_ITERS,RANK2_RIDGE,
                       RANK2_OUTLIER_QUANTILE,TRAIN_HIDDEN,VOCAB)


def fit_second_rank(hidden, teacher, prior):
    x=np.asarray(hidden,dtype=np.float32)
    ref=np.asarray(teacher,dtype=np.float32)
    base=np.asarray(prior,dtype=np.float32)
    if (x.ndim!=2 or ref.ndim!=2 or base.shape!=ref.shape or
            x.shape[0]!=ref.shape[0] or x.shape[0]<16 or
            x.shape[1]<4 or ref.shape[1]<8):
        raise ValueError("teacher and paid residual source FIT geometry mismatch")
    if not (np.isfinite(x).all() and np.isfinite(ref).all() and np.isfinite(base).all()):
        raise ValueError("nonfinite trained source data")
    error=ref-base
    error-=error.mean(axis=1,keepdims=True)
    error-=error.mean(axis=0,keepdims=True)
    xcenter=x.mean(0,dtype=np.float64).astype(np.float32)
    x0=x-xcenter

    v=error[0].copy()
    if np.linalg.norm(v)<1e-8:
        raise ValueError("insufficient contextual teacher error")
    v=v/np.linalg.norm(v)
    for _ in range(RANK2_POWER_ITERS):
        v=error.T@(error@v)
        norm=float(np.linalg.norm(v))
        if norm<1e-9 or not np.isfinite(norm):
            raise ValueError("degenerate second-rank teacher residual")
        v/=norm
    if v[int(np.argmax(np.abs(v)))]<0:
        v=-v
    target=error@v
    a=x0.astype(np.float64)
    cov=a.T@a
    regularization=RANK2_RIDGE*np.trace(cov)/cov.shape[0]
    if regularization<=0 or not np.isfinite(regularization):
        raise ValueError("noninvertible fixed-ridge input covariance")
    cov.flat[::cov.shape[0]+1]+=regularization
    u=np.linalg.solve(cov,a.T@target.astype(np.float64)).astype(np.float32)
    bias=-float(xcenter.astype(np.float64)@u.astype(np.float64))
    scale=float(np.quantile(np.abs(v),RANK2_OUTLIER_QUANTILE))/3.
    if not np.isfinite(scale) or scale<=0:
        raise ValueError("cannot encode 3-bit output direction")
    symbols=np.rint(np.clip(v/scale,-3,3)).astype(np.int8)
    u16=u.astype(np.float16)
    b16=np.float16(bias)
    if not np.isfinite(u16).all() or not np.isfinite(b16):
        raise ValueError("FP16 second-rank hidden mapper overflow")
    predicted=(x@u16.astype(np.float32)+np.float32(b16))[:,None]*(symbols.astype(np.float32)*np.float32(scale))[None,:]
    before=float(np.mean(error.astype(np.float64)**2))
    after=float(np.mean((error.astype(np.float64)-predicted.astype(np.float64))**2))
    return u16,b16,symbols,np.float32(scale),{
        "FIT_residual_MSE_before_second_rank":before,
        "FIT_residual_MSE_after_second_rank":after,
        "FIT_MSE_better":after<before,
        "no_refitting_CB149_or_pretrained_weights":True,
        "FIT_output_code_symbol_count":len(set(symbols.tolist())),
        "FIT_int3_outlier_clipped_fraction":float(np.mean(np.abs(v)>3*scale)),
        "FIT_teacher_logits_may_not_restore_semantics":True,
        "fit_tokens":len(x),
    }


def fit_strict(hidden,teacher,base):
    if (np.shape(hidden)[1:]!=(TRAIN_HIDDEN,) or
            np.shape(teacher)[1:]!=(VOCAB,) or
            np.shape(base)[1:]!=(VOCAB,)):
        raise ValueError("not original 134.5M source hidden and tied vocab")
    return fit_second_rank(hidden,teacher,base)
