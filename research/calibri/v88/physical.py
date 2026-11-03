"""Physical one-file CB88 = entire original CB59 + truly paid 4D weights.

Five-bit packed original pretraining weights remain in native C GEMV. Only
separable four-axis residual and bias are physically stored as FP16.
"""
from __future__ import annotations
import hashlib
import struct
import zlib
import numpy as np
from v86.provenance import ORIGINAL_CB59_SHA
from paid_lowrank_v76 import SOURCE_BYTES
from binary_geometry_v64 import EXPECTED_UNIQUE
from .tensor import OUT1,OUT2,IN1,IN2,RANK,FACTOR_ELEMENTS,VOCAB,TRAINABLE_ELEMENTS

MAGIC=b"CLBR88\0\0"
HEADER=struct.Struct("<8sIIII")
DIGEST=36
PAYLOAD_BYTES=2*TRAINABLE_ELEMENTS
COMPLETE_BYTES=SOURCE_BYTES+HEADER.size+PAYLOAD_BYTES+DIGEST
COMPLETE_BPW=8*COMPLETE_BYTES/EXPECTED_UNIQUE
assert COMPLETE_BPW<=.700 and HEADER.size==24

def verified_source(original:bytes):
    if len(original)!=SOURCE_BYTES or hashlib.sha256(original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise AssertionError("wrong original trained whole-model physical source")

def pack(original:bytes,A,B,bias):
    verified_source(original)
    shapes=((OUT1,IN1,RANK),(RANK,OUT2,IN2),(VOCAB,))
    arrays=[]
    for data,shape in zip((A,B,bias),shapes):
        z=np.asarray(data,dtype="<f2")
        if z.shape!=shape or not np.isfinite(z).all():
            raise AssertionError("invalid physical multidimensional source tensors")
        arrays.append(z)
    body=original+HEADER.pack(MAGIC,OUT1,OUT2,IN1,IN2)+b"".join(
        t.tobytes(order="C") for t in arrays)
    raw=body+struct.pack("<I",zlib.crc32(body))
    full=raw+hashlib.sha256(raw).digest()
    if len(full)!=COMPLETE_BYTES:
        raise AssertionError("physical source length changes not accounted")
    return full

def unpack(full:bytes):
    if len(full)!=COMPLETE_BYTES:
        raise AssertionError("compressed source incomplete or wrong rate")
    original=full[:SOURCE_BYTES]
    verified_source(original)
    if HEADER.unpack_from(full,SOURCE_BYTES)!=(MAGIC,OUT1,OUT2,IN1,IN2):
        raise AssertionError("unknown CB88 tensor geometry")
    if zlib.crc32(full[:-36])!=struct.unpack("<I",full[-36:-32])[0]:
        raise AssertionError("whole physical archive CRC mismatch")
    if hashlib.sha256(full[:-32]).digest()!=full[-32:]:
        raise AssertionError("physical source SHA mismatch")
    payload=memoryview(full)[SOURCE_BYTES+HEADER.size:-36]
    pos=0
    tensors=[]
    for shape in ((OUT1,IN1,RANK),(RANK,OUT2,IN2),(VOCAB,)):
        size=2*int(np.prod(shape))
        tensors.append(np.frombuffer(payload[pos:pos+size],dtype="<f2").copy().reshape(shape))
        pos+=size
    if pos!=PAYLOAD_BYTES or not all(np.isfinite(x).all() for x in tensors):
        raise AssertionError("corrupt FP16 tensor source")
    return original,*tensors

def audit(full:bytes):
    _,A,B,b=unpack(full)
    return {"physical_complete_bytes":len(full),"original_CB59_bytes":SOURCE_BYTES,
            "four_axis_tensor_shapes":[list(A.shape),list(B.shape)],
            "bias_shape":list(b.shape),
            "all_in_bits_per_original_parameter":COMPLETE_BPW,
            "original_pretrained_unique_parameters":EXPECTED_UNIQUE,
            "genuine_FP16_tensor_parameters":TRAINABLE_ELEMENTS,
            "native_original_5bit_source_preserved":True,
            "Tensor_operator_CPU_Torch_not_native_C_fused":True,
            "complete_physical_SHA256":hashlib.sha256(full).hexdigest()}
