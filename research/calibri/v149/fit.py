"""Deterministic rank1 teacher residual learned ONLY on source TRAIN FIT.

After removing input-invariant vocabulary residuals already addressed by the
CB147 paid static bias, approximate remaining context-dependent logits by
    delta_logit(token, word) = (last_hidden @ u + intercept) * v[word]
with tiny FP16 u and physically paid int4 v. Selection passes=8,
ridge=20% average feature variance, and quantile=.997 are PREDECLARED;
there is NO fitting or re-selection against VALID, QA or independent TEST.
"""
import numpy as np

from .protocol import (POWER_ITERS,RIDGE_FRACTION,OUTLIER_QUANTILE,
                       TRAIN_HIDDEN,VOCAB)


def fit_rank_one(hidden,original_teacher_logits,compact_logits,
                 *, passes=POWER_ITERS,ridge=RIDGE_FRACTION,quantile=OUTLIER_QUANTILE):
    x=np.asarray(hidden,dtype=np.float32)
    teacher=np.asarray(original_teacher_logits,dtype=np.float32)
    compact=np.asarray(compact_logits,dtype=np.float32)
    if (x.ndim!=2 or teacher.ndim!=2 or compact.shape!=teacher.shape
            or x.shape[0]!=teacher.shape[0] or x.shape[0]<16
            or teacher.shape[1]<8 or x.shape[1]<4):
        raise ValueError("incompatible hidden and original teacher/student TRAIN tokens")
    if not (np.isfinite(x).all() and np.isfinite(teacher).all()
            and np.isfinite(compact).all()):
        raise ValueError("nonfinite teacher or compact source logits")
    if passes!=8 or not .0<ridge<=1 or not .9<=quantile<1:
        raise ValueError("unregistered fit protocol")

    gap=teacher-compact
    # Remove softmax-irrelevant per-token logit shifts and mean residual per
    # vocabulary token. The static int8 paid prior is already frozen.
    gap-=gap.mean(axis=1,keepdims=True)
    gap-=gap.mean(axis=0,keepdims=True)
    xcenter=x.mean(axis=0,dtype=np.float64).astype(np.float32)
    x0=x-xcenter

    # Deterministic power iteration for dominant contextual residual
    # direction, without materializing 49152x49152 covariance.
    v=gap[0].copy()
    if float(np.linalg.norm(v))<1e-9:
        raise ValueError("no contextual teacher-source residual")
    v/=np.linalg.norm(v)
    for _ in range(passes):
        coeff=gap@v
        v=gap.T@coeff
        magnitude=float(np.linalg.norm(v))
        if not np.isfinite(magnitude) or magnitude<1e-9:
            raise ValueError("rank-one context direction is degenerate")
        v/=magnitude
    if v[int(np.argmax(np.abs(v)))]<0:
        v=-v

    y=gap@v
    x64=x0.astype(np.float64)
    y64=y.astype(np.float64)
    gram=x64.T@x64
    reg=float(ridge*np.trace(gram)/x64.shape[1])
    if not np.isfinite(reg) or reg<=0:
        raise ValueError("hidden TRAIN sources provide no rank-one regressor")
    gram.flat[::gram.shape[0]+1]+=reg
    coef=np.linalg.solve(gram,x64.T@y64).astype(np.float32)
    intercept=-float(np.dot(xcenter.astype(np.float64),
                            coef.astype(np.float64)))
    if not (np.isfinite(coef).all() and np.isfinite(intercept)):
        raise ValueError("invalid fitted hidden contextual projection")

    gain=float(np.quantile(np.abs(v),quantile))/7.
    if not np.isfinite(gain) or gain<=0:
        raise ValueError("invalid 4-bit output scale")
    q=np.clip(np.rint(v/gain),-7,7).astype(np.int8)
    # Physical projection is already quantized to FP16 here; no uncounted
    # FP32 trainable source enters native runtime or archive.
    paid_u=coef.astype("<f2")
    paid_offset=np.float16(intercept)
    if not np.isfinite(paid_u).all() or not np.isfinite(paid_offset):
        raise ValueError("paid FP16 projection overflow")
    expected=(x@paid_u.astype(np.float32)+np.float32(paid_offset))[:,None]*(
        q.astype(np.float32)*np.float32(gain))[None,:]
    before=float(np.mean(gap.astype(np.float64)**2))
    after=float(np.mean((gap.astype(np.float64)-expected.astype(np.float64))**2))
    measurement={
        "source_FIT_context_residual_mean_squared_before":before,
        "source_FIT_context_residual_mean_squared_after_paid_rank1":after,
        "source_FIT_context_MSE_decreased":bool(after<before),
        "TRAIN_contextual_output_rank":1,
        "teacher_first_logit_bias_centered_out":True,
        "predeclared_power_passes":passes,
        "predeclared_ridge_fraction":ridge,
        "predeclared_int4_quantile":quantile,
        "output_vector_code_symbols_used":int(len(np.unique(q))),
        "output_vector_clipped_fraction":float(np.mean(np.abs(v)>7*gain)),
        "output_vector_scale":gain,
        "fully_quantized_student_projection":True,
    }
    return paid_u,paid_offset,q,np.float32(gain),measurement


def fit_strict_135m(hidden,teacher,compact):
    if (np.shape(hidden)[1:]!=(TRAIN_HIDDEN,) or
        np.shape(teacher)[1:]!=(VOCAB,) or
        np.shape(compact)[1:]!=(VOCAB,)):
        raise ValueError("original target model vocabulary geometry changed")
    return fit_rank_one(hidden,teacher,compact)
