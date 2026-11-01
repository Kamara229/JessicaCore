"""CALIBRI v35: train-only covariance-aware rounding and FP16 row scales.

The 3.5-bpw physical CT33 codec and row bit allocations are FROZEN from
v34; only rounding and each row's existing two-byte FP16 scale can change.
Train-layer activations optimize actual linear-output SSE, not held-out PPL.
This is an ablation of known post-training quantization principles, not a
novel model architecture, end-to-end speed result or full-model codec.
"""
from __future__ import annotations
import struct
import zlib
import numpy as np
from direct_tiles_v33 import HEADER,MAGIC,encode as base_encode,decode,parse
from multidim_v32 import packed_symbols,unpacked_symbols


FACTORS=(0.55,0.65,0.75,0.825,0.9,0.95,1.0,1.05,1.10,1.175,1.25,1.375,1.50)

def calibrated_encode(weights,train_activations,baseline_blob):
    w=np.asarray(weights,np.float32)
    x=np.asarray(train_activations,np.float32)
    m,n,br,bc,blocks=parse(baseline_blob)
    if (w.shape!=(m,n) or br!=1 or bc!=n or len(blocks)!=m
        or x.ndim!=2 or x.shape[1]!=n or x.shape[0]<64
        or not np.isfinite(w).all() or not np.isfinite(x).all()):
        raise ValueError("expected genuine train activations and v34 rowwise physical baseline")
    payload=bytearray()
    total_base=0.0
    total_new=0.0
    changed=0
    improved=0
    bit_alloc=[]
    for row,(i,j,bits,old_scale,old_stream) in enumerate(blocks):
        if (i!=row or j!=0 or bits not in (3,4) or old_scale<=0):
            raise AssertionError("row bit allocation changed before calibration")
        limit=(1 << (bits-1))-1
        v=w[row]
        # The base q and FP16 scale MUST remain a candidate, so train
        # objective cannot deteriorate because of this row calibration.
        original_q=unpacked_symbols(old_stream,n,bits).astype(np.int16)
        old_recon=original_q.astype(np.float32)*old_scale
        delta0=v-old_recon
        base_y=x@delta0
        loss0=float(np.dot(base_y,base_y))
        if not np.isfinite(loss0):
            raise AssertionError("nonfinite source calibration loss")
        options=[(old_scale,original_q)]
        scale_candidates={float(old_scale)}
        upper=float(np.max(np.abs(v)))/limit
        for f in FACTORS:
            for ref in (old_scale,upper):
                s=float(np.float16(ref*f))
                if 0<s<65504 and np.isfinite(s):
                    scale_candidates.add(s)
        for s in sorted(scale_candidates):
            if s==old_scale:
                continue
            q=np.clip(np.rint(v/s),-limit,limit).astype(np.int16)
            options.append((s,q))
        # Batched GEMM computes EXACT output-error proxy X·(W-Q),
        # including real train activation cross-feature covariances,
        # rather than independent diagonal per-coordinate MSE.
        d=np.stack([(v-q.astype(np.float32)*s) for s,q in options],axis=0)
        y=x@d.T
        losses=np.sum(np.square(y,dtype=np.float64),axis=0,dtype=np.float64)
        # Include the precise original loss; avoid FP32 ordering
        # artifacts causing a false claimed improvement.
        losses[0]=loss0
        best=int(np.argmin(losses))
        new_loss=float(losses[best])
        if new_loss>loss0+1e-5*max(loss0,1.0):
            raise AssertionError("train-calibrated row error worsened")
        s,q=options[best]
        if best and (s!=old_scale or not np.array_equal(q,original_q)):
            changed+=1
        if new_loss<loss0-1e-8:
            improved+=1
        payload.extend(struct.pack("<Be",bits,s))
        payload.extend(packed_symbols(q,bits))
        total_base+=loss0
        total_new+=new_loss
        bit_alloc.append(bits)
    raw=bytes(payload)
    calibrated=HEADER.pack(MAGIC,m,n,br,bc,len(raw),zlib.crc32(raw))+raw
    if len(calibrated)!=len(baseline_blob):
        raise AssertionError("physical bytes MUST exactly match uncalibrated 3.5bpw stream")
    if decode(calibrated).shape!=w.shape:
        raise AssertionError("physical calibrated decoder geometry invalid")
    return calibrated,{
        "bits_are_identical_per_row":True,
        "row_bitwidths":bit_alloc,
        "rows_changed":changed,
        "rows_with_train_SSE_improvement":improved,
        "train_output_SSE_baseline":total_base,
        "train_output_SSE_calibrated":total_new,
        "train_output_SSE_ratio":total_new/max(total_base,1e-40),
        "train_activation_rows":x.shape[0],
        "scale_grid_factors":list(FACTORS),
        "calibration_rule":"actual train-layer X @ (W-Q) output error, per-row fixed bitwidth & two-byte FP16 scale; no validation or test data used",
    }


def smoke():
    rng=np.random.default_rng(35)
    w=(rng.standard_normal((128,512))*0.02).astype(np.float32)
    x=rng.standard_normal((80,512)).astype(np.float32)
    v=(x*x).mean(0)
    base,meta=base_encode(w,v,28672,br=1,bc=512)
    new,info=calibrated_encode(w,x,base)
    if (len(base)!=len(new) or
        info["train_output_SSE_calibrated"]>info["train_output_SSE_baseline"]*(1+1e-7) or
        meta["tiles_4bit"]!=info["row_bitwidths"].count(4)):
        raise AssertionError("train calibration strict physical budget/bit assignment not preserved")
    altered=bytearray(new);altered[-1]^=1
    try:decode(altered)
    except ValueError:pass
    else:raise AssertionError("CRC failed to catch modified byte")
    print("v35 unchanged 3.5bpw bytes, train-output calibration, row bits and CRC smoke PASS",flush=True)


if __name__=="__main__":
    smoke()
