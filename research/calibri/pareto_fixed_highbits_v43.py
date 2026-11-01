"""CALIBRI v43: per-row fixed INT4/5/6 physical bytes and TRAIN-only calibration.

Keep CF15 on-disk original signed codebook, per-row FP16 scales,
CRC32 and byte count EXACT. Compare the actual file bytes to the
uncalibrated standard physical control. This uses only frozen WikiText
TRAIN-A second moments; not task validation PPL or new corpus labels.

v35 scale factor grid is frozen, not fitted on test data.
"""
from __future__ import annotations
import struct
import zlib
import numpy as np

from fixedbit_v15 import HEADER,encode as raw_encode,decode as raw_decode
from calibrated_rows_v35 import FACTORS

SUPPORTED=(4,5,6)


def trained_fixedbit(w,train_moments,bits):
    w=np.asarray(w,np.float32)
    moment=np.asarray(train_moments,np.float32)
    if bits not in SUPPORTED or w.ndim!=2 or moment.shape!=(w.shape[1],):
        raise ValueError("expected 4/5/6 physical bit baseline, genuine train second moments")
    if not np.isfinite(w).all() or not np.isfinite(moment).all() or np.min(moment)<0:
        raise ValueError("training moments must be finite nonnegative")
    m,n=w.shape
    b=raw_encode(w,bits)
    original=raw_decode(b)
    qbytes=(w.size*bits+7)//8
    previous_scales=np.frombuffer(
        b,dtype="<f2",count=m,offset=HEADER.size+qbytes).astype(np.float32)
    bound=(1<<(bits-1))-1
    selected_scale=np.empty(m,dtype="<f2")
    selected_codes=np.empty((m,n),np.int16)
    start_total=0.
    new_total=0.
    changed=0
    for row in range(m):
        v=w[row]
        before=original[row]
        old=float(previous_scales[row])
        start_error=float(np.sum(moment*np.square(v-before),dtype=np.float64))
        choices={old}
        maximum=float(np.max(np.abs(v)))/bound
        for ref in (maximum,old):
            for factor in FACTORS:
                val=float(np.float16(ref*factor))
                if 0<val<65504 and np.isfinite(val):
                    choices.add(val)
        best_score=start_error
        best_s=old
        best_q=np.clip(np.rint(v/old),-bound,bound).astype(np.int16)
        for scale in sorted(choices):
            if scale==old:
                continue
            q=np.clip(np.rint(v/scale),-bound,bound).astype(np.int16)
            error=float(np.sum(moment*np.square(v-q.astype(np.float32)*scale),dtype=np.float64))
            if error<best_score-1e-15*max(1.0,best_score):
                best_score=error
                best_s=scale
                best_q=q
        selected_scale[row]=np.float16(best_s)
        selected_codes[row]=best_q
        start_total+=start_error
        new_total+=best_score
        if best_s!=old or not np.array_equal(best_q,np.clip(
                np.rint(v/old),-bound,bound).astype(np.int16)):
            changed+=1
    # CF15 exact bitstream format: single sign-magnitude two's
    # complement signed B-bit stream followed by FP16 row scales.
    symbols=(selected_codes.reshape(-1)&((1<<bits)-1)).astype(np.uint8)
    packed=np.packbits(
        ((symbols[:,None]>>np.arange(bits-1,-1,-1,dtype=np.uint8))&1).ravel()
    ).tobytes()
    if len(packed)!=qbytes:raise AssertionError("incorrect actual packed bit length")
    payload=packed+selected_scale.tobytes()
    result=HEADER.pack(b"CF15",m,n,bits,0,0,zlib.crc32(payload))+payload
    restored=raw_decode(result)
    if len(result)!=len(b) or restored.shape!=w.shape or not np.isfinite(restored).all():
        raise AssertionError("calibration mutated physical storage or failed checksum")
    if new_total>start_total+1e-8*max(start_total,1):
        raise AssertionError("TRAIN diagonal second-moment approximation worsened")
    return b,result,{
        "bitwidth":bits,
        "original_bytes":len(b),
        "calibrated_same_bytes":len(result),
        "TRAIN_diagonal_weighted_MSE_before":start_total,
        "TRAIN_diagonal_weighted_MSE_after":new_total,
        "rows_calibrated_changed":changed,
        "same_3p5_v35_scale_factors":list(FACTORS),
        "method":"frozen TRAIN-A second-moment-weighted FP16 scale grid; no validation, no TRAIN-B nor external text",
    }


def smoke():
    rng=np.random.default_rng(43)
    w=rng.normal(0,0.02,(32,128)).astype(np.float32)
    moment=np.linspace(0.1,1.2,128,dtype=np.float32)
    for bits in SUPPORTED:
        original,recal,metric=trained_fixedbit(w,moment,bits)
        if len(original)!=len(recal):
            raise AssertionError("changing weight rate without permission")
        damaged=bytearray(recal);damaged[-1]^=1
        try:raw_decode(damaged)
        except ValueError:pass
        else:raise AssertionError("failed physical stream CRC corruption detection")
        if metric["TRAIN_diagonal_weighted_MSE_after"]>metric["TRAIN_diagonal_weighted_MSE_before"]*1.0000001:
            raise AssertionError("wrong train calibration objective")
    print("CALIBRI v43 strict signed fixed 4/5/6 physical TRAIN-only calibration smoke PASS",flush=True)


if __name__=="__main__":
    smoke()
