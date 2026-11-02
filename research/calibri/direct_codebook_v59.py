#!/usr/bin/env python3
"""CALIBRI CB59: actual <=0.70 bpw complete pretrained UNIQUE state.

EXPERIMENTAL rate-first 5-bit-per-8-vector K=32 codebook indices, a
literal 0.625 bit / 2D source weight plus physical row scales, per-tensor
FP16 codebooks, raw small 1D tensors, metadata, physical CRC and SHA256.
Shared source aliases are stored by name and never counted twice.

Compressed matrices are *not* decoded into a dense FP32 model at runtime:
a separate C direct lookup-GEMV kernel consumes these 5-bit groups.
Embedding lookup decodes ONLY requested rows, never entire weight matrix.
Quality at <0.70bpw is NOT GUARANTEED, especially on a dense model.
"""
from __future__ import annotations
import hashlib,json,math,struct,zlib
from dataclasses import dataclass
import numpy as np
import torch
from shared_checkpoint_v58 import split_aliases

MAGIC=b"CB59"
HEADER=struct.Struct("<4sBII")
FOOTER=36
VECTOR=8
K=32
BITS=5
DIMS=8
SAMPLES=4096
KMEANS_ITERS=5


def pack5(ids):
    ids=np.asarray(ids,dtype=np.uint8).reshape(-1)
    if np.any(ids>=K):
        raise ValueError("5-bit code index overflow")
    n=len(ids)
    pad=(-n)%8
    if pad:
        ids=np.pad(ids,(0,pad))
    blocks=ids.reshape(-1,8).astype(np.uint64)
    word=np.zeros(len(blocks),dtype=np.uint64)
    for k in range(8):
        word|=blocks[:,k] << np.uint64(k*5)
    packed=np.column_stack([(word >> np.uint64(8*j))&255
                            for j in range(5)]).astype(np.uint8).tobytes()
    if len(packed)!=(n*BITS+7)//8:
        # 8-index groups pad to byte/5-byte boundary: this comparison
        # only holds if n is divisible by eight.
        if n%8==0:
            raise AssertionError("5-bit source index packing wrong")
    return packed


def unpack5(data,n):
    packed=np.frombuffer(data,dtype=np.uint8)
    if len(data)!=(n*BITS+7)//8:
        raise ValueError("wrong 5-bit logical index length")
    idx=np.arange(n,dtype=np.uint64)
    bit=idx*BITS
    byte=(bit//8).astype(np.int64)
    shift=bit%8
    low=packed[byte].astype(np.uint16)
    high=np.zeros_like(low)
    has=byte+1<len(packed)
    high[has]=packed[byte[has]+1]
    return np.asarray(((low|(high<<8)) >> shift)&31,dtype=np.uint8)


def row_scales(w):
    a=np.asarray(w,dtype=np.float32)
    # Per-row RMS, finite float32. FP16 representable source
    # and deterministic correction of FP16 rounded row scales.
    result=np.sqrt(np.mean(np.square(a.astype(np.float64)),axis=1))
    result=np.clip(result,2**-14,60000).astype("<f2")
    if not np.all(np.isfinite(result)):
        raise ValueError("invalid compressed per-row scale")
    return result


def _source_sample(w,scales,sample_count=SAMPLES):
    m,n=w.shape
    g=n//VECTOR
    total=m*g
    ids=np.linspace(0,total-1,min(sample_count,total),dtype=np.int64)
    rows=ids//g
    group=ids%g
    sampled=np.stack([w[row,8*col:8*col+8].astype(np.float32)/
                      np.float32(scales[row]) for row,col in zip(rows,group)])
    return sampled


def _centroids(sample):
    # Deterministic fixed starting positions, then 5 Lloyd passes.
    # This is SOURCE-ONLY codebook fitting; no test/validation samples.
    if not np.all(np.isfinite(sample)):
        raise ValueError("nonfinite source group samples")
    starters=np.linspace(0,len(sample)-1,K,dtype=np.int64)
    centers=sample[starters].copy()
    for _ in range(KMEANS_ITERS):
        similarity=np.dot(sample,centers.T)
        distances=(np.sum(sample*sample,axis=1,keepdims=True)-
                   2*similarity+
                   np.sum(centers*centers,axis=1)[None,:])
        ids=np.argmin(distances,axis=1).astype(np.int32)
        count=np.bincount(ids,minlength=K)
        updates=np.zeros((K,VECTOR),dtype=np.float64)
        np.add.at(updates,ids,sample.astype(np.float64))
        occupied=count>0
        centers[occupied]=(updates[occupied]/count[occupied,None]).astype(np.float32)
    q=centers.astype("<f2")
    if not np.all(np.isfinite(q)):
        raise ValueError("FP16 original-weight centroid overflow")
    return q


def compress_matrix(w,blockrows=256):
    w=np.asarray(w,dtype=np.float32)
    if w.ndim!=2 or w.shape[1]%VECTOR or not np.isfinite(w).all():
        raise ValueError("CB59 currently needs finite FP32 2D matrix with ncols divisible by 8")
    m,n=w.shape
    scales=row_scales(w)
    centers=_centroids(_source_sample(w,scales))
    c=centers.astype(np.float32)
    center_norm=np.sum(c*c,axis=1)
    g=n//VECTOR
    chunks=[]
    lost=0.
    original_energy=0.
    for start in range(0,m,blockrows):
        end=min(m,start+blockrows)
        q=w[start:end]
        scalar=scales[start:end].astype(np.float32)
        vectors=q.reshape(-1,VECTOR)/np.repeat(scalar,g)[:,None]
        dot=np.dot(vectors,c.T)
        distances=(np.sum(vectors*vectors,axis=1,keepdims=True)-
                   2*dot+center_norm[None,:])
        choice=np.argmin(distances,axis=1).astype(np.uint8)
        # Real float16 (stored) centroids & row scale in reconstruction:
        decoded=c[choice].reshape(end-start,g,VECTOR)
        decoded=decoded*scalar[:,None,None]
        diff=q.reshape(end-start,g,VECTOR)-decoded
        lost+=float(np.sum(diff.astype(np.float64)**2))
        original_energy+=float(np.sum(q.astype(np.float64)**2))
        chunks.append(pack5(choice))
    if not original_energy>0:
        raise AssertionError("zero-only pretrained matrix not meaningful as source")
    return {"packed":b"".join(chunks),
            "row_scales":scales.tobytes(),
            "centroids":centers.tobytes(),
            "num_vectors":m*g,
            "group_size":VECTOR,
            "codebook_size":K,
            "relative_RMSE":math.sqrt(lost/original_energy),
            "squared_error":lost,
            "source_squared_energy":original_energy,
            "actual_5bit_bytes":sum(len(chunk) for chunk in chunks)}


def encode(source):
    """Real CB59 physical full checkpoint with explicit complete aliases."""
    unique,aliases=split_aliases(source)
    body=bytearray()
    desc={}
    diagnostics={}
    for name in sorted(unique):
        t=unique[name]
        w=t.cpu().numpy()
        if w.ndim==2 and t.is_floating_point():
            built=compress_matrix(w)
            segments={}
            for part in ("packed","row_scales","centroids"):
                raw=built[part]
                segments[part]=[len(body),len(raw)]
                body.extend(raw)
            desc[name]={"shape":list(w.shape),"dtype":"CB59_K32V8",
                        "parts":segments}
            diagnostics[name]={k:built[k] for k in (
                "relative_RMSE","squared_error","source_squared_energy",
                "num_vectors","actual_5bit_bytes")}
        else:
            if t.dtype!=torch.float32:
                raise ValueError("only source FP32 floating 1D raw tensors supported")
            raw=w.astype("<f4").tobytes()
            desc[name]={"shape":list(w.shape),"dtype":"f4",
                        "parts":{"raw":[len(body),len(raw)]}}
            body.extend(raw)
    manifest={
        "format":"CB59 direct-resident experimental codebook weights",
        "version":1,
        "trained_unique_keys":sorted(unique),
        "full_original_state_keys":sorted(source),
        "source_alias_keys":dict(sorted(aliases.items())),
        "original_unique_parameter_count":sum(t.numel() for t in unique.values()),
        "tensor_descriptors":desc,
        "vector_length":VECTOR,"codebook_entries":K,"bits_per_code":BITS,
        "all_weight_2d_physically_compressed":True,
        "quality_not_guaranteed":True,
    }
    meta=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode("utf-8")
    if len(meta)>5_000_000 or len(body)>300_000_000:
        raise AssertionError("CB59 source manifest/body exceeds parser bounds")
    start=HEADER.pack(MAGIC,1,len(meta),len(body))
    unprotected=start+meta+bytes(body)
    with_crc=unprotected+struct.pack("<I",zlib.crc32(unprotected))
    blob=with_crc+hashlib.sha256(with_crc).digest()
    energy=math.fsum(x["source_squared_energy"] for x in diagnostics.values())
    error=math.fsum(x["squared_error"] for x in diagnostics.values())
    diagnostics_out={
        "full_source_weighted_relative_RMSE":math.sqrt(error/energy),
        "full_source_intact_weight_matrices":len(diagnostics),
        "source_squared_error":error,
        "source_squared_energy":energy,
        "largest_relative_matrix_RMSE":sorted(
            ((name,d["relative_RMSE"]) for name,d in diagnostics.items()),
            key=lambda x:-x[1])[:8],
        "total_real_5bit_code_bytes":sum(x["actual_5bit_bytes"]
                                       for x in diagnostics.values()),
        "physical_entire_CB59_bytes":len(blob),
        "full_checkpoint_effective_bpw":8*len(blob)/
            manifest["original_unique_parameter_count"],
    }
    return blob,manifest,diagnostics_out


@dataclass
class QuantHandle:
    name:str
    shape:tuple
    packed:np.ndarray
    row_scales:np.ndarray
    centroids:np.ndarray


class CB59:
    def __init__(self,blob,expected_keys=None):
        self.blob=blob  # one read-only compressed source archive in RAM
        if len(blob)<HEADER.size+FOOTER:
            raise ValueError("short CB59 source")
        magic,version,mlen,blen=HEADER.unpack_from(blob)
        if magic!=MAGIC or version!=1 or mlen>5_000_000 or blen>300_000_000:
            raise ValueError("bad CB59 header")
        if HEADER.size+mlen+blen+FOOTER!=len(blob):
            raise ValueError("CB59 physical length mismatch")
        if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
            raise ValueError("CB59 SHA256 error")
        if zlib.crc32(blob[:-36])!=struct.unpack_from("<I",blob,len(blob)-36)[0]:
            raise ValueError("CB59 CRC error")
        self.meta=json.loads(blob[HEADER.size:HEADER.size+mlen].decode("utf8"))
        if (self.meta["bits_per_code"]!=BITS or
            self.meta["codebook_entries"]!=K or
            self.meta["vector_length"]!=VECTOR):
            raise ValueError("incompatible direct codebook ABI")
        if expected_keys is not None and (
            set(expected_keys)!=set(self.meta["full_original_state_keys"])):
            raise ValueError("incomplete full source pretrained state")
        body=memoryview(blob)[HEADER.size+mlen:-36]
        self.handles={}
        self.raw={}
        desc=self.meta["tensor_descriptors"]
        if sorted(desc)!=self.meta["trained_unique_keys"]:
            raise ValueError("inconsistent unique source tensor names")
        for name,data in desc.items():
            parts=data["parts"]
            shape=tuple(data["shape"])
            def get(part):
                off,size=parts[part]
                if off<0 or size<0 or off+size>len(body):
                    raise ValueError("source segment offset corrupt")
                return body[off:off+size]
            if data["dtype"]=="CB59_K32V8":
                m,n=shape
                if n%VECTOR:raise ValueError("corrupt 8D source shape")
                packed=np.frombuffer(get("packed"),dtype=np.uint8)
                scales=np.frombuffer(get("row_scales"),dtype="<f2")
                centers=np.frombuffer(get("centroids"),dtype="<f2")
                if (len(packed)!=(m*n//VECTOR*BITS+7)//8 or
                    len(scales)!=m or len(centers)!=K*VECTOR or
                    not np.isfinite(scales).all() or
                    not np.isfinite(centers).all()):
                    raise ValueError("invalid physical source index/scale/codebook")
                self.handles[name]=QuantHandle(
                    name,shape,packed,scales.astype(np.float32),
                    centers.astype(np.float32).reshape(K,VECTOR))
            elif data["dtype"]=="f4":
                arr=np.frombuffer(get("raw"),dtype="<f4")
                if int(np.prod(shape,dtype=np.int64))!=len(arr):
                    raise ValueError("raw norm/bias vector length mismatch")
                self.raw[name]=arr.reshape(shape)
            else:raise ValueError("unrecognized physical source tensor type")
        aliases=self.meta["source_alias_keys"]
        for alias,canonical in aliases.items():
            if alias in self.handles or alias in self.raw:
                raise ValueError("alias duplicates real physical weights")
            if canonical in self.handles:
                self.handles[alias]=self.handles[canonical]
            elif canonical in self.raw:
                self.raw[alias]=self.raw[canonical]
            else:raise ValueError("dangling alias source")
        if (set(self.handles)|set(self.raw))!=set(
             self.meta["full_original_state_keys"]):
            raise ValueError("missing source tensor after alias restoration")
    def hash(self):return hashlib.sha256(self.blob).hexdigest()
    def physical_bytes(self):return len(self.blob)


def smoke():
    torch.manual_seed(59)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=torch.nn.Embedding(32,16)
            self.head=torch.nn.Linear(16,32,bias=False)
            self.head.weight=self.embed.weight
            self.layer=torch.nn.Linear(16,16)
    model=Toy()
    blob,meta,rate=encode(model.state_dict())
    decoded=CB59(blob,expected_keys=model.state_dict().keys())
    assert decoded.handles["head.weight"] is decoded.handles["embed.weight"]
    assert len(decoded.handles["layer.weight"].packed)>0
    assert not hasattr(decoded.handles["layer.weight"],"dense_weights")
    changed=bytearray(blob);changed[len(blob)//2]^=1
    try:CB59(bytes(changed))
    except ValueError:pass
    else:raise AssertionError("CB59 checksum didn't detect modified bytes")
    allcodes=np.arange(8,dtype=np.uint8)%32
    assert np.array_equal(unpack5(pack5(allcodes),8),allcodes)
    print("CALIBRI v59 actual physical source 5bits/8 vectors, dedup, corruption and index smoke PASS")


if __name__=="__main__":
    smoke()
