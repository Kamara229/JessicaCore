"""Physical CB140: pinned complete CB59 + one bit/8 tied embedding weights.

Two FP16 V8 residual vectors per original 32 codewords, plus ONE bitplane for
the tied 49,152x576 matrix. NO v119 paid adapters included: this is a separate
<=0.7-bpp source candidate. CRC, SHA, original CB59 SHA must all verify.
"""
import hashlib
import struct
import zlib
from dataclasses import dataclass
import numpy as np
from direct_codebook_v59 import CB59
from v117.protocol import ORIGINAL_CB59_SHA,MAX_SOURCE_BYTES,TOTAL_UNIQUE
from v139.protocol import RESIDUAL_CODEBOOK_BYTES,RESIDUAL_BITPLANE_BYTES

MAGIC=b"CLB140\0\0"
HEADER=struct.Struct("<8sII")
FOOTER_BYTES=36
EXPECTED_MATRIX=(49_152,576)
EXPECTED_CENTERS=(32,2,8)


def pack_bits(bits):
    arr=np.asarray(bits,dtype=np.uint8).reshape(-1)
    if np.any(arr>1):
        raise ValueError("one-bit plane contains a non-bit")
    return np.packbits(arr,bitorder="little").tobytes()


def unpack_bits(blob,count):
    arr=np.frombuffer(blob,dtype=np.uint8)
    if len(arr)!=(count+7)//8:raise ValueError("invalid physical bitplane length")
    return np.unpackbits(arr,bitorder="little")[:count].copy()


def centers_as_physical(centers):
    x=np.asarray(centers,dtype="<f2")
    if x.shape!=EXPECTED_CENTERS or not np.isfinite(x).all():
        raise ValueError("bad K32x2xV8 FP16 refinement table")
    return x.tobytes()


def pack(base, centers, plane):
    if hashlib.sha256(base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("must refine exact original CB59, not altered source")
    physical=centers_as_physical(centers)
    bits=bytes(plane)
    if len(bits)!=RESIDUAL_BITPLANE_BYTES or len(physical)!=RESIDUAL_CODEBOOK_BYTES:
        raise ValueError("wrong paid one-bit plane dimensions")
    data=HEADER.pack(MAGIC,len(base),len(bits))+base+physical+bits
    crc=data+struct.pack("<I",zlib.crc32(data))
    blob=crc+hashlib.sha256(crc).digest()
    if len(blob)>MAX_SOURCE_BYTES:
        raise ValueError("complete original-weight source exceeds 0.7bpp")
    return blob


@dataclass(frozen=True)
class Physical:
    original: bytes
    centers: np.ndarray
    bits: np.ndarray
    size: int
    sha256: str


def unpack(blob):
    raw=bytes(blob)
    if len(raw)<HEADER.size+FOOTER_BYTES:
        raise ValueError("truncated CB140")
    magic,base_len,plane_len=HEADER.unpack_from(raw)
    if magic!=MAGIC or plane_len!=RESIDUAL_BITPLANE_BYTES:
        raise ValueError("wrong CB140 format, dimensions or version")
    if (len(raw)!=HEADER.size+base_len+RESIDUAL_CODEBOOK_BYTES+plane_len+FOOTER_BYTES
            or len(raw)>MAX_SOURCE_BYTES):
        raise ValueError("incomplete paid one-bit physical source")
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError("whole CB140 SHA corrupted")
    if zlib.crc32(raw[:-36])!=struct.unpack_from("<I",raw,len(raw)-36)[0]:
        raise ValueError("whole CB140 CRC corrupted")
    p=HEADER.size
    base=raw[p:p+base_len]
    if hashlib.sha256(base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("CB59 source changed")
    cb=CB59(base)
    if cb.handles["model.embed_tokens.weight"].shape!=EXPECTED_MATRIX:
        raise ValueError("wrong frozen embedding shape")
    if cb.handles["lm_head.weight"] is not cb.handles["model.embed_tokens.weight"]:
        raise ValueError("original tied output handle lost")
    p+=base_len
    table=np.frombuffer(raw[p:p+RESIDUAL_CODEBOOK_BYTES],dtype="<f2").copy().reshape(EXPECTED_CENTERS)
    if not np.isfinite(table).all():raise ValueError("invalid residual FP16 centers")
    p+=RESIDUAL_CODEBOOK_BYTES
    bits=np.frombuffer(raw[p:p+plane_len],dtype=np.uint8).copy()
    return Physical(base,table.astype(np.float32),bits,len(raw),hashlib.sha256(raw).hexdigest())


def report(blob):
    physical=unpack(blob)
    return {"complete_bytes":physical.size,"full_source_bpp":8*physical.size/TOTAL_UNIQUE,
            "size_cap_bytes":MAX_SOURCE_BYTES,"source_base_SHA":ORIGINAL_CB59_SHA,
            "whole_source_SHA":physical.sha256,"tied_embedding_and_head_bytes_counted_once":True,
            "one_extra_bit_per_group":True,"native_full_checkpoint_physically_paid":True}
