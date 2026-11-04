"""Fully physically paid v151: exact CB149 + 3bits/output rank2 residual.

New 49152 3bit symbol codes occupy exactly 18432 bytes, plus 576 FP16
input projection, FP16 bias, FP32 scale, 16B header, CRC32 and SHA256.
All pretrained source is nested with original CRC and SHA. No uncounted
teacher weights, Green Tunnel, dense weight matrix or dummy model.
"""
from dataclasses import dataclass
import hashlib
import struct
import math
import zlib

import numpy as np
from v149.codec import unpack as unpack_cb149
from v117.protocol import TOTAL_UNIQUE,MAX_SOURCE_BYTES
from .protocol import (PINNED_CB149_SHA,VOCAB,TRAIN_HIDDEN,
                       RANK2_PAID_INT_MIN,RANK2_PAID_INT_MAX,check_protocol)

MAGIC=b"CLB151\0\0"
HEADER=struct.Struct("<8sHHI")
PROJECTION_LEN=2*TRAIN_HIDDEN
BITPLANE_LEN=VOCAB*3//8
TAIL_LEN=2+4+BITPLANE_LEN
CHECK_LEN=4+32
EXTRA_BYTES=HEADER.size+PROJECTION_LEN+TAIL_LEN+CHECK_LEN


def pack3(q):
    values=np.asarray(q)
    if values.shape!=(VOCAB,) or not np.isfinite(values).all():
        raise ValueError("not a complete signed int3 output code")
    if np.any(values!=np.rint(values)) or np.any(values<RANK2_PAID_INT_MIN) or np.any(values>RANK2_PAID_INT_MAX):
        raise ValueError("rank2 signed int3 [-3,3] expected")
    codes=(values.astype(np.uint8)+3)
    out=bytearray(BITPLANE_LEN)
    for i in range(0,VOCAB,8):
        word=0
        for j in range(8):
            word|=int(codes[i+j])<<(3*j)
        out[i//8*3:i//8*3+3]=word.to_bytes(3,"little")
    return bytes(out)


def unpack3(blob):
    raw=bytes(blob)
    if len(raw)!=BITPLANE_LEN:raise ValueError("missing 3-bit physical plane")
    out=np.empty(VOCAB,dtype=np.int8)
    for i in range(0,VOCAB,8):
        start=(i//8)*3
        word=int.from_bytes(raw[start:start+3],"little")
        for j in range(8):
            symbol=(word>>(3*j))&7
            if symbol==7:raise ValueError("reserved int3 pattern in paid model")
            out[i+j]=symbol-3
    return out


def pack(parent_blob,projection,offset,code,step):
    check_protocol()
    parent=unpack_cb149(parent_blob)
    if parent.sha256!=PINNED_CB149_SHA:
        raise ValueError("wrong pretrained full paid CB149 base")
    u=np.asarray(projection,dtype="<f2")
    b=np.float16(offset)
    step=np.float32(step)
    if u.shape!=(TRAIN_HIDDEN,) or not np.isfinite(u).all() or not np.isfinite(b) or not np.isfinite(step) or not step>0:
        raise ValueError("invalid physically paid contextual coefficients")
    base=(HEADER.pack(MAGIC,1,3,TRAIN_HIDDEN)+bytes(parent_blob)
        +u.tobytes()+b.tobytes()+struct.pack("<f",float(step))+pack3(code))
    check=base+struct.pack("<I",zlib.crc32(base))
    final=check+hashlib.sha256(check).digest()
    if len(final)>MAX_SOURCE_BYTES:
        raise ValueError("complete 0.7 bit archive source budget breached")
    return final


@dataclass(frozen=True)
class Physical:
    base: object
    projection: np.ndarray
    offset: float
    signed_int3: np.ndarray
    scale: float
    size: int
    sha256: str

    @property
    def output_vector(self):
        return self.signed_int3.astype(np.float32)*self.scale


def unpack(source):
    raw=bytes(source)
    if len(raw)<EXTRA_BYTES+100 or len(raw)>MAX_SOURCE_BYTES:
        raise ValueError("incomplete or over-budget native rank2 source")
    if HEADER.unpack_from(raw)!=(MAGIC,1,3,TRAIN_HIDDEN):
        raise ValueError("wrong version of source or 3bit head")
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError("complete CB151 SHA integrity failure")
    if zlib.crc32(raw[:-36])!=struct.unpack_from("<I",raw,len(raw)-36)[0]:
        raise ValueError("complete CB151 CRC integrity failure")
    offset=HEADER.size
    oldlen=len(raw)-EXTRA_BYTES
    parent=unpack_cb149(raw[offset:offset+oldlen])
    if parent.sha256!=PINNED_CB149_SHA:
        raise ValueError("not original FP32-trained physically paid frozen CB149")
    offset+=oldlen
    u=np.frombuffer(raw[offset:offset+PROJECTION_LEN],dtype="<f2").copy()
    if u.shape!=(TRAIN_HIDDEN,) or not np.isfinite(u).all():
        raise ValueError("bad paid projection")
    offset+=PROJECTION_LEN
    bias=float(np.frombuffer(raw[offset:offset+2],dtype="<f2")[0])
    offset+=2
    step=float(struct.unpack_from("<f",raw,offset)[0])
    offset+=4
    q=unpack3(raw[offset:offset+BITPLANE_LEN])
    if not math.isfinite(bias) or not math.isfinite(step) or step<=0:
        raise ValueError("invalid paid model rank2 gain")
    return Physical(parent,u.astype(np.float32),bias,q,step,
                    len(raw),hashlib.sha256(raw).hexdigest())


def physical_report(source):
    m=unpack(source)
    return {"all_source_archive_bytes":m.size,
            "all_in_bpp_unique_original":8*m.size/TOTAL_UNIQUE,
            "remaining_below_0p7_bytes":MAX_SOURCE_BYTES-m.size,
            "physically_paid_second_rank_int3_vocab_symbols":VOCAB,
            "physically_paid_second_rank_FP16_projection":TRAIN_HIDDEN,
            "parent_CB149_verified":True,
            "whole_source_SHA256":m.sha256}
