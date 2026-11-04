"""TRAIN-only functional feedback on true original and compact-student signals."""
from __future__ import annotations
import hashlib
import struct
import zlib
import numpy as np
import torch
from v112.archive import Archive,HEADER,TRAILER,CODEBOOK_BYTES
from .protocol import SHRINKAGE,MIN_MULTIPLIER,MAX_MULTIPLIER

@torch.inference_mode()
def capture(model,tok,source,texts):
    """Return ordered per-layer input activations from a genuine full forward."""
    results={name:[] for name in source}
    handles=[]
    try:
        for name in source:
            module=model.get_submodule(name.removesuffix(".weight"))
            def hook(_module, inputs, *, key=name):
                x=inputs[0].detach().reshape(-1,inputs[0].shape[-1]).cpu().float().numpy()
                results[key].append(x.copy())
            handles.append(module.register_forward_pre_hook(hook))
        for prompt in texts:
            ids=tok(prompt,return_tensors="pt")["input_ids"]
            model(input_ids=ids,use_cache=False)
        if any(len(v)!=len(texts) for v in results.values()):
            raise AssertionError("missing real activation capture")
        return {name:np.concatenate(rows,axis=0) for name,rows in results.items()}
    finally:
        for h in handles:
            h.remove()

def gain_for_row(expected,estimated):
    """Least-squares scalar projection; shrink/clamp fixed before VALID."""
    e=np.asarray(expected,dtype=np.float64)
    p=np.asarray(estimated,dtype=np.float64)
    if e.shape!=p.shape or e.ndim!=2:
        raise ValueError("FIT targets do not match predicted activations")
    numer=np.sum(e*p,axis=0)
    denom=np.sum(p*p,axis=0)
    ls=numer / np.maximum(denom,1e-16)
    correction=1.0+SHRINKAGE*(ls-1.0)
    return np.clip(correction,MIN_MULTIPLIER,MAX_MULTIPLIER).astype(np.float32)

def recalibrate(baseline,original_weights,teacher_inputs,active_inputs):
    """Same-byte archive rewrite: ONLY row FP16 scales may change.

    Expected output: W_FP32 * x_FP32_teacher.
    Model output: encoded W * x_student (or teacher if control chosen).
    All activation data supplied here must originate from FIT only.
    """
    a=Archive(baseline)
    if set(original_weights)!=set(a.records) or set(teacher_inputs)!=set(a.records) or set(active_inputs)!=set(a.records):
        raise ValueError("missing original or FIT signal for a physical source matrix")
    payload=bytearray(a.payload)
    diagnostics={}
    for name,w in original_weights.items():
        teacher=teacher_inputs[name]
        student=active_inputs[name]
        if teacher.shape!=student.shape:
            raise ValueError("different token origins for paired FIT streams")
        target=teacher @ w.T
        reconstructed=a.decode(name)
        current=student @ reconstructed.T
        gain=gain_for_row(target,current)
        rec,scale,_=a._parts(name)
        new=(np.asarray(scale,dtype=np.float32)*gain).astype("<f2")
        if not np.isfinite(new).all():
            raise FloatingPointError("invalid trained physical FP16 scale")
        origin=float(np.square(target-current,dtype=np.float64).sum())
        corrected=student @ (reconstructed*(new.astype(np.float32)/np.where(
            np.abs(scale.astype(np.float32))<1e-20,1,scale.astype(np.float32)))[:,None]).T
        after=float(np.square(target-corrected,dtype=np.float64).sum())
        payload[rec["scale_at"]:rec["scale_at"]+rec["scale_bytes"]]=new.tobytes()
        diagnostics[name]={"relative_fit_sse":after/max(origin,1e-16),
                           "negative_corrected_scales":int(np.sum(new<0))}
    mlen=HEADER.unpack_from(baseline)[1]
    prefix=baseline[:HEADER.size+mlen]
    raw=prefix+payload
    with_crc=raw+struct.pack("<I",zlib.crc32(raw))
    result=with_crc+hashlib.sha256(with_crc).digest()
    if len(result)!=len(baseline):
        raise AssertionError("feedback changed physical bytes")
    new_archive=Archive(result)
    if not np.array_equal(a.books,new_archive.books):
        raise AssertionError("codebooks changed without being paid")
    for name in a.records:
        _,_,x=a._parts(name)
        _,_,y=new_archive._parts(name)
        if bytes(x)!=bytes(y):
            raise AssertionError("unpaid changes in packed 5-bit index stream")
    return result,diagnostics
