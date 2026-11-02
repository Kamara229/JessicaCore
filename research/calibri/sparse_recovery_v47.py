#!/usr/bin/env python3
"""CALIBRI v47 compact trained residual-row PRE/POST correction sidecar.

Sparse residual rows are selected on WikiText TRAIN-A using a full-model
compressed-teacher NLL gradient; the independently signed teacher residual
(W_original-W_INT6) is stored as FP16 and added only ONCE on cold load.
During autoregressive forward there is NO Python hook or unpacking overhead.
All sidecar bytes, tensor metadata, CRC and SHA count toward true storage.
"""
from __future__ import annotations
import hashlib
import struct
import zlib
from collections import Counter
import numpy as np
import torch

MAGIC=b"CR47"
HEADER=struct.Struct("<4sB32sHI")  # magic, schema, parent CB46 SHA, tensor count, row count
TENSOR=struct.Struct("<HII")        # name byte len, output rows, input columns
ROW=struct.Struct("<HI")           # module index, row index
FOOTER_BYTES=36


def _checksum(data):
    data=bytes(data)
    pre=data+struct.pack("<I",zlib.crc32(data))
    return pre+hashlib.sha256(pre).digest()


def encode_sparse_patch(original,compressed,choices,parent_sha):
    """Persist exact array geometry and float16 *residual* (not original W)."""
    keys=sorted(set(name for name,row in choices))
    if len(keys)>65535:
        raise ValueError("too many distinct tensors")
    if len(parent_sha)!=64:
        raise ValueError("requires physical pretrained parent checkpoint SHA256")
    items=sorted(set(choices),key=lambda z:(z[0],z[1]))
    if len(items)!=len(choices):
        raise ValueError("duplicate correction row")
    body=bytearray(HEADER.pack(MAGIC,1,bytes.fromhex(parent_sha),len(keys),len(items)))
    index={k:i for i,k in enumerate(keys)}
    for name in keys:
        if name not in original or name not in compressed:
            raise ValueError("unknown original trained source tensor")
        w=original[name]
        q=compressed[name]
        if w.ndim!=2 or q.shape!=w.shape:
            raise ValueError("sparse correction requires original trained 2D W")
        m,n=tuple(w.shape)
        if m>0xFFFFFFFF or n>0xFFFFFFFF:
            raise ValueError("source tensor too large")
        raw=name.encode("utf-8")
        if not (0<len(raw)<65536):
            raise ValueError("source parameter name too long")
        body.extend(TENSOR.pack(len(raw),m,n))
        body.extend(raw)
    for name,row in items:
        w=original[name]
        q=compressed[name]
        if not (0<=row<w.shape[0]):
            raise ValueError("out of range pretrained row")
        diff=(w[row].float()-q[row].float()).numpy().astype(np.float32)
        if not np.isfinite(diff).all() or np.max(np.abs(diff))>65000:
            raise ValueError("invalid FP16 teacher correction")
        body.extend(ROW.pack(index[name],row))
        body.extend(diff.astype("<f2").tobytes())
    return _checksum(body)


def parse_patch(data,parent_sha,expected_shapes):
    if len(data)<HEADER.size+FOOTER_BYTES:
        raise ValueError("short CR47 sidecar")
    magic,version,base,nmod,nrows=HEADER.unpack_from(data)
    if magic!=MAGIC or version!=1 or base.hex()!=parent_sha:
        raise ValueError("wrong parent pretrained model or correction schema")
    if hashlib.sha256(data[:-32]).digest()!=data[-32:]:
        raise ValueError("corrupt patch SHA256")
    if zlib.crc32(data[:-36])!=struct.unpack_from("<I",data,len(data)-36)[0]:
        raise ValueError("corrupt patch CRC")
    if not (1<=nmod<=300 and 1<=nrows<=200000):
        raise ValueError("invalid CR47 row or module count")
    off=HEADER.size
    end=len(data)-36
    names=[]
    for _ in range(nmod):
        if off+TENSOR.size>end:
            raise ValueError("truncated patch manifest")
        nl,m,n=TENSOR.unpack_from(data,off)
        off+=TENSOR.size
        if not (0<nl<=1024 and off+nl<=end):
            raise ValueError("bad tensor name")
        name=data[off:off+nl].decode("utf-8")
        off+=nl
        if name in names or expected_shapes.get(name)!=(m,n):
            raise ValueError("wrong trained tensor shape or duplicate source name")
        names.append(name)
    seen=set()
    patch=[]
    for _ in range(nrows):
        if off+ROW.size>end:
            raise ValueError("truncated patch row")
        mid,row=ROW.unpack_from(data,off)
        off+=ROW.size
        if mid>=len(names):
            raise ValueError("unknown patch tensor index")
        name=names[mid]
        m,n=expected_shapes[name]
        if row>=m or (name,row) in seen or off+2*n>end:
            raise ValueError("duplicate, truncated or out-of-range patch")
        seen.add((name,row))
        delta=np.frombuffer(data,dtype="<f2",count=n,offset=off).astype(np.float32).copy()
        off+=2*n
        if not np.isfinite(delta).all():
            raise ValueError("nonfinite patch row")
        patch.append((name,row,delta))
    if off!=end or len(seen)!=nrows:
        raise ValueError("trailing patch bytes")
    return patch


def apply_patch(model,path,parent_sha):
    """Add learned sparse residual once to fully decoded physical INT6 model.

    Cold load only: no extra runtime layer or weight scratch retained.
    """
    data=__import__("pathlib").Path(path).read_bytes()
    state=model.state_dict()
    shapes={k:tuple(v.shape) for k,v in state.items() if v.ndim==2}
    rows=parse_patch(data,parent_sha,shapes)
    with torch.no_grad():
        for name,i,delta in rows:
            dst=state[name]
            dst[i,:].add_(torch.from_numpy(delta).to(device=dst.device,dtype=dst.dtype))
    return {"patch_file_bytes":len(data),"patch_sha256":hashlib.sha256(data).hexdigest(),
            "physical_patched_rows":len(rows),"runtime_extra_matmul_ops_per_token":0}


def expected_overhead(choices):
    bymodule={}
    for name,row,n in choices:
        bymodule[name]=n
    return (HEADER.size+FOOTER_BYTES+
            sum(TENSOR.size+len(k.encode("utf-8")) for k in bymodule)+
            sum(ROW.size+2*n for name,row,n in choices))


def rank_rows(original,decoded,grads):
    """Return full-model TRAIN-A loss-gradient and squared-error row scores.

    score_gradient = max(0, -sum(dLoss/dQ_row * (W-Q)_row));
    score_sse = sum((W-Q)_row**2).
    Both measured from original trained weights and training-only full-model
    NLL gradient; neither sees evaluation corpus. Sorting is stable by names.
    """
    results={}
    for name in sorted(original):
        w=original[name]
        if w.ndim!=2 or name not in decoded:
            continue
        q=decoded[name]
        if q.shape!=w.shape:
            raise AssertionError("trained compressed tensor geometry mismatch")
        delta=(w.float()-q.float()).numpy().astype(np.float64)
        energy=np.sum(delta*delta,axis=1,dtype=np.float64)
        grad=grads.get(name)
        if grad is not None:
            if grad.shape!=w.shape:
                raise AssertionError("wrong full-model gradient source tensor")
            direction=-np.sum(grad.astype(np.float64)*delta,axis=1,dtype=np.float64)
        else:
            direction=np.zeros(w.shape[0],dtype=np.float64)
        for i in range(w.shape[0]):
            results[(name,i)]={
                "n":int(w.shape[1]),
                "cost":ROW.size+2*int(w.shape[1]),
                "gradient_gain":float(max(0.,direction[i])),
                "sse":float(energy[i])}
    return results


def pick_budget(scores,max_total_bytes,parent_size):
    """TRAIN-only globally ranked row patches under physical total cap."""
    methods=("gradient_global","sse_global")
    ranked={}
    for method in methods:
        key="gradient_gain" if method=="gradient_global" else "sse"
        ordered=sorted(scores,key=lambda id:(
            -scores[id][key]/scores[id]["cost"],id[0],id[1]))
        chosen=[]
        spent=0
        modules=set()
        for name,row in ordered:
            score=scores[(name,row)][key]
            if score<=0:
                break
            n=scores[(name,row)]["n"]
            added=ROW.size+2*n
            newmod=(TENSOR.size+len(name.encode("utf-8"))) if name not in modules else 0
            if parent_size+HEADER.size+FOOTER_BYTES+spent+added+newmod>max_total_bytes:
                continue
            chosen.append((name,row))
            spent+=added+newmod
            modules.add(name)
        ranked[method]=chosen

    # Exact-size matched control: same per-module patch counts and each
    # module's same row width/byte size, but choose max SSE rows. This
    # isolates row priority rather than moving bytes among layers.
    counts=Counter(name for name,row in ranked["gradient_global"])
    matched=[]
    for name,count in sorted(counts.items()):
        local=sorted((k for k in scores if k[0]==name),
                     key=lambda id:(-scores[id]["sse"],id[1]))
        matched.extend(local[:count])
    ranked["sse_module_size_matched"]=matched
    if expected_overhead((n,r,scores[(n,r)]["n"]) for n,r in matched)!=expected_overhead(
        (n,r,scores[(n,r)]["n"]) for n,r in ranked["gradient_global"]):
        raise AssertionError("control sidecar physical bytes not matched")
    if not ranked["gradient_global"] or not ranked["sse_global"]:
        raise AssertionError("TRAIN gradient correction did not propose a nonempty sidecar")
    return ranked
