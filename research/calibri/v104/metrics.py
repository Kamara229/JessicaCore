"""Native banked decoder C-vs-Python parity and original FP32 source fidelity."""
from __future__ import annotations
import math
import numpy as np
from direct_codebook_v59 import unpack5
from .spec import BANKS,VOCAB,DIM,ROW_PART
MAX_ABS_LOGIT=0.003

def decode_rows(h,banks,start,stop):
    m,n=h.shape
    if (m,n)!=(VOCAB,DIM) or not (0<=start<stop<=m):
        raise AssertionError("unexpected actual original trained output head geometry")
    group=n//8
    row_bytes=group*5//8
    if group%8:raise AssertionError("5-bit row boundary not byte aligned")
    source=h.packed[start*row_bytes:stop*row_bytes].tobytes()
    codes=unpack5(source,(stop-start)*group)
    indices=np.arange(start,stop)//ROW_PART
    q=np.asarray(banks,dtype=np.float32)[indices[:,None],codes.reshape(stop-start,group)]
    restored=q.reshape(stop-start,n)*h.row_scales[start:stop,None]
    if not np.isfinite(restored).all():raise AssertionError("banked decoder nonfinite")
    return restored.astype(np.float32)

def full_head_evidence(h,banks,reference_weights,input_vector,native_logits):
    exact=reference_weights.detach().cpu().numpy().astype(np.float32)
    x=np.ascontiguousarray(input_vector,dtype=np.float32)
    observed=np.asarray(native_logits,dtype=np.float32)
    if (exact.shape!=(VOCAB,DIM) or x.ndim!=2 or x.shape[1]!=DIM or
        observed.shape!=(x.shape[0],VOCAB)):
        raise AssertionError("wrong pretrained teacher/native geometry")
    source_energy=0.;error_energy=0.;maximum_difference=0.
    matched_top1=[]
    python_logits=np.empty_like(observed)
    for begin in range(0,VOCAB,256):
        end=min(begin+256,VOCAB)
        restored=decode_rows(h,banks,begin,end)
        teacher_chunk=exact[begin:end].astype(np.float64)
        sqdiff=teacher_chunk-restored.astype(np.float64)
        source_energy+=float(np.sum(teacher_chunk**2))
        error_energy+=float(np.sum(sqdiff**2))
        python_logits[:,begin:end]=x@restored.T
        maximum_difference=max(maximum_difference,float(
            np.max(np.abs(observed[:,begin:end]-python_logits[:,begin:end]))))
    if maximum_difference>MAX_ABS_LOGIT:
        raise AssertionError("CB104 direct C kernel does not match banked Python decoding")
    for i in range(x.shape[0]):
        matched_top1.append(int(observed[i].argmax())==int(python_logits[i].argmax()))
    if not all(matched_top1):
        raise AssertionError("banked C-vs-Python output head top1 differs")
    return {"whole_original_model_head_rows_checked":VOCAB,
        "genuine_pretrained_original_head_weight_relative_RMSE":
            math.sqrt(error_energy/max(1e-30,source_energy)),
        "native_C_vs_Python_max_abs_full_vocab_logit_difference":maximum_difference,
        "native_C_vs_Python_full_vocab_all_top1_equal":True,
        "dense_pretrained_decoder_only_temporary_diagnostic_scratch":True}
