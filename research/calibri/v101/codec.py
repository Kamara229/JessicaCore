"""Read-only EXACT 5-bit index roundtrip and native C head decoder parity.

The original trained dense FP32 matrix is used only as an external
reference for scientific quality analysis (not stored in compressed model).
Each 256-row compressed chunk is restored into TEMPORARY Python scratch to
measure encoder fidelity and C runtime parity, then discarded.
"""
from __future__ import annotations
import math
import numpy as np
from direct_codebook_v59 import pack5,unpack5,VECTOR,K
from .spec import ROW_CHUNK,EXPLICIT_ROW_INDICES,MAX_NATIVE_VS_PYTHON_ABS_DIFF

def decode_chunk(handle,start,end):
    m,n=handle.shape
    g=n//VECTOR
    if not (0<=start<end<=m and n%VECTOR==0 and g%8==0):
        raise AssertionError("unexpected unaligned packed vocabulary head")
    bpr=g*5//8
    source=handle.packed[start*bpr:end*bpr]
    if len(source)!=(end-start)*bpr:
        raise AssertionError("wrong physical row byte offsets")
    ix=unpack5(source.tobytes(),(end-start)*g)
    if pack5(ix)!=source.tobytes():
        raise AssertionError("5-bit logical encode/decode mismatch in original physical archive")
    if not (int(ix.min())>=0 and int(ix.max())<K):
        raise AssertionError("invalid source codebook index")
    recovered=(handle.centroids[ix].reshape(end-start,n)*
               handle.row_scales[start:end,None])
    if recovered.shape!=(end-start,n) or not np.isfinite(recovered).all():
        raise AssertionError("decoded head source has invalid shape or NaN/Inf")
    return recovered,ix

def exact_streamed_audit(handle,teacher_head_weight,activations,native_head_logits):
    if len(handle.shape)!=2 or tuple(teacher_head_weight.shape)!=handle.shape:
        raise AssertionError("original teacher/head codebook shape mismatch")
    m,n=handle.shape
    vectors=np.ascontiguousarray(np.asarray(activations,dtype=np.float32))
    if vectors.ndim!=2 or vectors.shape[1]!=n or not np.isfinite(vectors).all():
        raise AssertionError("invalid genuine final-norm CPU inputs")
    native=np.asarray(native_head_logits,dtype=np.float32)
    if native.shape!=(len(vectors),m):
        raise AssertionError("native GEMV full-vocabulary wrong shape")
    exact=np.asarray(teacher_head_weight.detach().cpu().numpy(),dtype=np.float32)
    if exact.shape!=(m,n) or not np.isfinite(exact).all():
        raise AssertionError("teacher FP32 pretrained source missing")
    decoded_logits=np.empty_like(native)
    samples={};weight_sq=0.0;error_sq=0.0;row_cos_sum=0.0
    row_cos_below_half=0;rows_scanned=0;code_counts=np.zeros(K,dtype=np.int64)
    for begin in range(0,m,ROW_CHUNK):
        finish=min(begin+ROW_CHUNK,m)
        q,codes=decode_chunk(handle,begin,finish)
        w=exact[begin:finish]
        q64=q.astype(np.float64);w64=w.astype(np.float64)
        diff=w64-q64
        weight_sq+=float(np.sum(w64*w64))
        error_sq+=float(np.sum(diff*diff))
        wn=np.sqrt(np.sum(w64*w64,axis=1))
        qn=np.sqrt(np.sum(q64*q64,axis=1))
        cos=np.sum(w64*q64,axis=1)/np.maximum(wn*qn,1e-18)
        row_cos_sum+=float(cos.sum())
        row_cos_below_half+=int(np.sum(cos<.5))
        code_counts+=np.bincount(codes,minlength=K)
        rows_scanned+=len(cos)
        decoded_logits[:,begin:finish]=vectors@q.T
        for idx in EXPLICIT_ROW_INDICES:
            if begin<=idx<finish:
                samples[idx]=q[idx-begin].copy()
    if rows_scanned!=m or set(samples)!={k for k in EXPLICIT_ROW_INDICES if k<m}:
        raise AssertionError("head did not decode every original trained row")
    discrepancy=np.abs(native-decoded_logits)
    maxdiff=float(discrepancy.max())
    rmse=float(np.sqrt(np.mean(discrepancy.astype(np.float64)**2)))
    if not (maxdiff<=MAX_NATIVE_VS_PYTHON_ABS_DIFF and rmse<0.001):
        raise AssertionError("NATIVE C GEMV differs from explicit original-head Python decoder")
    if any(int(np.argmax(native[i]))!=int(np.argmax(decoded_logits[i])) for i in range(len(vectors))):
        raise AssertionError("C native decoded-head top1 differs from Python exact source")
    return {
        "shape":[m,n],
        "same_source_full_5bit_index_roundtrip_pass":True,
        "all_physical_rows_scanned":m,
        "row_code_usage_counts":code_counts.tolist(),
        "mean_teacher_vs_decoded_row_cosine":row_cos_sum/m,
        "fraction_weight_rows_cosine_below_0p5":row_cos_below_half/m,
        "original_FP32_to_CB59_head_weight_relative_RMSE":
            math.sqrt(error_sq/max(weight_sq,1e-30)),
        "native_C_vs_explicit_Python_decoder_max_abs_logit_diff":maxdiff,
        "native_C_vs_explicit_Python_decoder_RMSE":rmse,
        "C_vs_Python_full_vocab_top1_all_match":True,
        "selected_full_decoded_rows":samples,
        "decoded_logits":decoded_logits
    }
