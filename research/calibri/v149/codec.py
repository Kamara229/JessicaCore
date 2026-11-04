"""CB149 complete physical source: original CB147 plus contextual int4 rank1.

Total <=0.7 bits per original unique pretrained parameter, including CB59,
CB140 1-bit tied embedding residual, CB147 int8 vocab prior and this new
rank-one correction. Only a small FP16 projection and a nibble-coded output
vector are expanded in RAM, never a pretrained 2D matrix.
"""
import hashlib
import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from v147.codec import unpack as unpack_cb147
from v117.protocol import TOTAL_UNIQUE,MAX_SOURCE_BYTES
from .protocol import CB147_SHA,TRAIN_HIDDEN,VOCAB

MAGIC=b"CLB149\x00\x00"
VERSION=1
HEADER=struct.Struct("<8sHHI")
FOOTER=36
PROJECTION_BYTES=TRAIN_HIDDEN*2
VECTOR_BYTES=VOCAB//2
SCALAR_BYTES=6
EXTRA_BYTES=HEADER.size+PROJECTION_BYTES+SCALAR_BYTES+VECTOR_BYTES+FOOTER


def pack_signed_int4(codes):
    q=np.asarray(codes)
    if q.shape!=(VOCAB,) or not np.all(np.isfinite(q)) or (
            not np.all(q==q.astype(np.int8))):
        raise ValueError("expected an integer per original output token")
    if np.any(q < -7) or np.any(q > 7):
        raise ValueError("int4 palette is restricted to [-7,7], -8 reserved")
    x=q.astype(np.int8)+8
    return ((x[::2].astype(np.uint8)) |
            (x[1::2].astype(np.uint8)<<4)).tobytes()


def unpack_signed_int4(data):
    raw=np.frombuffer(data,dtype=np.uint8)
    if len(raw)!=VECTOR_BYTES:
        raise ValueError("missing one paid nibble per output token")
    out=np.empty(VOCAB,dtype=np.int8)
    out[::2]=(raw&15).astype(np.int8)-8
    out[1::2]=(raw>>4).astype(np.int8)-8
    if np.any(out < -7) or np.any(out>7):
        raise ValueError("reserved int4 code in packed source")
    return out


def pack(old_cb147,projection,offset,vocab_code,scale):
    old=unpack_cb147(old_cb147)
    if old.sha256!=CB147_SHA:
        raise ValueError("not pinned original valid complete CB147")
    u=np.asarray(projection,dtype="<f2")
    if u.shape!=(TRAIN_HIDDEN,) or not np.isfinite(u).all():
        raise ValueError("invalid FP16 contextual input projection")
    off=np.float16(offset)
    scale=np.float32(scale)
    if not np.isfinite(off) or not np.isfinite(scale) or not scale>0:
        raise ValueError("finite positive packed scalar required")
    packed=pack_signed_int4(vocab_code)
    raw=(HEADER.pack(MAGIC,VERSION,0,TRAIN_HIDDEN)
         +bytes(old_cb147)+u.tobytes()+off.tobytes()
         +struct.pack("<f",float(scale))+packed)
    crc=raw+struct.pack("<I",zlib.crc32(raw))
    complete=crc+hashlib.sha256(crc).digest()
    if len(complete)>MAX_SOURCE_BYTES:
        raise ValueError("contextual new paid operator breaks 0.7bpp")
    return complete


@dataclass(frozen=True)
class Physical:
    parent: object
    projection: np.ndarray
    offset: float
    signed_int4: np.ndarray
    scale: float
    size: int
    sha256: str

    @property
    def output_vector(self):
        return self.signed_int4.astype(np.float32)*self.scale


def unpack(blob):
    raw=bytes(blob)
    if len(raw)<EXTRA_BYTES+100:
        raise ValueError("truncated full physically paid CB149 source")
    magic,version,flags,dim=HEADER.unpack_from(raw)
    if (magic,version,flags,dim)!=(MAGIC,VERSION,0,TRAIN_HIDDEN):
        raise ValueError("unrecognized CB149 archive format")
    if len(raw)>MAX_SOURCE_BYTES:
        raise ValueError("entire source above 0.7 bpp")
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError("invalid whole-source SHA")
    if zlib.crc32(raw[:-36])!=struct.unpack_from("<I",raw,len(raw)-36)[0]:
        raise ValueError("invalid whole-source CRC")
    offset=HEADER.size
    parent_len=len(raw)-EXTRA_BYTES
    parent=unpack_cb147(raw[offset:offset+parent_len])
    if parent.sha256!=CB147_SHA:
        raise ValueError("CB147 trained source replaced")
    offset+=parent_len
    proj=np.frombuffer(raw[offset:offset+PROJECTION_BYTES],dtype="<f2").copy()
    if len(proj)!=TRAIN_HIDDEN or not np.isfinite(proj).all():
        raise ValueError("invalid paid rank-one projection")
    offset+=PROJECTION_BYTES
    bias=float(np.frombuffer(raw[offset:offset+2],dtype="<f2")[0])
    offset+=2
    scale=struct.unpack_from("<f",raw,offset)[0]
    offset+=4
    output=unpack_signed_int4(raw[offset:offset+VECTOR_BYTES])
    if not math.isfinite(bias) or not math.isfinite(scale) or scale<=0:
        raise ValueError("nonfinite physically packed contextual output")
    return Physical(parent,proj.astype(np.float32),bias,output,scale,
                    len(raw),hashlib.sha256(raw).hexdigest())


def report(blob):
    x=unpack(blob)
    return {
        "complete_all_source_bytes":x.size,
        "complete_bpp_original_unique":8*x.size/TOTAL_UNIQUE,
        "0_7bit_headroom_bytes":MAX_SOURCE_BYTES-x.size,
        "original_CB147_nested_and_checksum_verified":True,
        "extra_int4_output_nibbles_paid":VOCAB,
        "extra_FP16_hidden_projection_paid":TRAIN_HIDDEN,
        "extra_intercept_and_scale_paid_bytes":SCALAR_BYTES,
        "one_rank_contextual_adapter_physical":True,
        "sha256":x.sha256,
    }
