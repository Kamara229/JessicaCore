"""One physical CB59 with extra 1bit per ORIGINAL 8D group of one block.

No CB149/CB151 appended logit priors. Only the tested transformer block is
refined; all 29 others and the tied head stay identical CB59. Seven small
FP16 residual dictionaries and every extra one-bit index are on disk.
"""
from dataclasses import dataclass
import hashlib,struct,zlib
import numpy as np
from direct_codebook_v59 import CB59
from v117.protocol import MAX_SOURCE_BYTES,TOTAL_UNIQUE
from .protocol import (PARENT_CB59_SHA,EXPECTED_BLOCK_MATRICES,
                       MODEL_LAYERS)

MAGIC=b"CLB153\0\0"
HEADER=struct.Struct("<8sII")
ENTRY=struct.Struct("<HIII")
TRAILER=36
CENTERS_BYTES=32*2*8*2


@dataclass(frozen=True)
class Patch:
    name: str
    shape: tuple
    centers: np.ndarray
    bits: np.ndarray

@dataclass(frozen=True)
class Physical:
    original: bytes
    layer: int
    patches: dict
    size: int
    sha256: str


def pack(original,layer,patches):
    if hashlib.sha256(original).hexdigest()!=PARENT_CB59_SHA:
        raise ValueError("not original CB59 full trained source")
    cb=CB59(original)
    if not 0<=layer<MODEL_LAYERS or len(patches)!=EXPECTED_BLOCK_MATRICES:
        raise ValueError("fixed one-block patch must have exactly 7 2D matrices")
    records=[]
    for name,part in sorted(patches.items()):
        if not name.startswith(f"model.layers.{layer}.") or not name.endswith(".weight"):
            raise ValueError("patch from wrong transformer block")
        h=cb.handles.get(name)
        if h is None or tuple(h.shape)!=tuple(part.shape):
            raise ValueError("original codebook geometry not pinned")
        if tuple(part.centers.shape)!=(32,2,8):
            raise ValueError("bad physical residual codebook")
        centers=np.asarray(part.centers,dtype="<f2")
        if not np.isfinite(centers).all():
            raise ValueError("nonfinite physical 1bit refinements")
        raw=np.asarray(part.bits,dtype=np.uint8).tobytes()
        groups=h.shape[0]*h.shape[1]//8
        if len(raw)!=(groups+7)//8:
            raise ValueError("unpaid matrix residual plane")
        name_bytes=name.encode("utf8")
        if len(name_bytes)>512:
            raise ValueError("matrix key name too long")
        records.append(ENTRY.pack(len(name_bytes),h.shape[0],h.shape[1],len(raw))+
                       name_bytes+centers.tobytes()+raw)
    body=HEADER.pack(MAGIC,len(original),layer)+original+struct.pack("<H",len(records))+b"".join(records)
    c=body+struct.pack("<I",zlib.crc32(body))
    final=c+hashlib.sha256(c).digest()
    if len(final)>MAX_SOURCE_BYTES:
        raise ValueError("the complete improved block exceeds 0.7-bit cap")
    return final


def unpack(blob):
    raw=bytes(blob)
    if len(raw)>MAX_SOURCE_BYTES or len(raw)<HEADER.size+TRAILER:
        raise ValueError("bad full physical CB153 length")
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError("bad full physical CB153 SHA")
    if zlib.crc32(raw[:-36])!=struct.unpack_from("<I",raw,len(raw)-36)[0]:
        raise ValueError("bad complete CB153 CRC")
    magic,base_len,layer=HEADER.unpack_from(raw)
    if magic!=MAGIC or layer not in range(MODEL_LAYERS):
        raise ValueError("wrong CB153 source geometry")
    at=HEADER.size
    original=raw[at:at+base_len]
    if hashlib.sha256(original).hexdigest()!=PARENT_CB59_SHA:
        raise ValueError("original source CB59 altered")
    cb=CB59(original)
    at+=base_len
    n=struct.unpack_from("<H",raw,at)[0]
    at+=2
    if n!=EXPECTED_BLOCK_MATRICES:
        raise ValueError("missing original layer operators")
    patches={}
    for _ in range(n):
        name_len,rows,dim,nbytes=ENTRY.unpack_from(raw,at)
        at+=ENTRY.size
        if not 0<name_len<=512:
            raise ValueError("invalid matrix path length")
        name=raw[at:at+name_len].decode("utf8")
        at+=name_len
        if not name.startswith(f"model.layers.{layer}.") or name in patches:
            raise ValueError("unexpected or duplicate transformer operator")
        h=cb.handles.get(name)
        if h is None or tuple(h.shape)!=(rows,dim):
            raise ValueError("wrong original trained matrix shape")
        if nbytes!=(rows*dim//8+7)//8:
            raise ValueError("bad physically paid onebit plane byte length")
        centers=np.frombuffer(raw[at:at+CENTERS_BYTES],dtype="<f2").copy().reshape(32,2,8)
        at+=CENTERS_BYTES
        bits=np.frombuffer(raw[at:at+nbytes],dtype=np.uint8).copy()
        at+=nbytes
        if not np.isfinite(centers).all():
            raise ValueError("nonfinite original-source patch centroids")
        patches[name]=Patch(name,(rows,dim),centers.astype(np.float32),bits)
    if at!=len(raw)-TRAILER:
        raise ValueError("trailing or missing physical source data")
    return Physical(original,layer,patches,len(raw),hashlib.sha256(raw).hexdigest())


def report(blob):
    model=unpack(blob)
    return {"complete_CB153_archive_bytes":model.size,
            "all_in_bpp_unique_original":8*model.size/TOTAL_UNIQUE,
            "complete_codebook_residuals_physically_paid":True,
            "one_original_complete_CB59_nested":True,
            "refined_original_layer":model.layer,
            "paid_matrix_count":len(model.patches),
            "whole_source_SHA256":model.sha256,
            "remaining_bytes_below_cap":MAX_SOURCE_BYTES-model.size}
