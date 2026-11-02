"""CALIBRI v46 physically stored all-parameter signed INT6 model checkpoint.

Only 2D float tensors are quantized to rowwise CF15 INT6; small biases,
layer-norm vectors and any other non-2D tensors are retained with exact
original dtype in the same single CB46 file. Full original Pythia model
weights are never required for a fresh from_config() restored runtime.
"""
from __future__ import annotations
import hashlib
import struct
import zlib
from pathlib import Path
import numpy as np
import torch
from fixedbit_v15 import encode as quant_encode,decode as quant_decode

MAGIC=b"CB46"
HDR=struct.Struct("<4sBI")
ENTRY=struct.Struct("<HBBI")
F64=1
F32=2
F16=3
I64=4
BOOL=5
INT6=6
DTYPES={
    F64:("<f8",torch.float64),
    F32:("<f4",torch.float32),
    F16:("<f2",torch.float16),
    I64:("<i8",torch.int64),
    BOOL:("?",torch.bool),
}
TORCH_TO_DTYPE={dtype:tag for tag,(_,dtype) in DTYPES.items()}
MAX_TENSOR_BYTES=300_000_000


def pack_state(state,preencoded=None):
    preencoded=preencoded or {}
    keys=sorted(state)
    if set(preencoded)-set(keys):
        raise ValueError("preencoded tensor not in original checkpoint")
    buf=bytearray(HDR.pack(MAGIC,1,len(keys)))
    summary={}
    for key in keys:
        w=state[key].detach().contiguous().cpu()
        if not w.isfinite().all() if w.is_floating_point() else False:
            raise ValueError("nonfinite source weights")
        if w.ndim==2 and w.is_floating_point() and min(w.shape)>0 and max(w.shape)<=65535:
            blob=preencoded.get(key)
            if blob is not None:
                q=quant_decode(blob)
                if q.shape!=tuple(w.shape) or not np.isfinite(q).all() or blob[8]!=6:
                    raise AssertionError("invalid frozen INT6 PRE-scale calibration")
            else:
                blob=quant_encode(w.numpy().astype(np.float32),6)
            codec=INT6
        else:
            if key in preencoded:
                raise ValueError("preencoded original tensor is not a 2D float matrix")
            codec=TORCH_TO_DTYPE.get(w.dtype)
            if codec is None:
                raise ValueError("unsupported raw tensor dtype "+str(w.dtype))
            arr=np.ascontiguousarray(w.numpy())
            blob=bytes([w.ndim])+b"".join(struct.pack("<I",d) for d in w.shape)+arr.tobytes()
        name=key.encode("utf-8")
        if len(name)>65535 or len(blob)>MAX_TENSOR_BYTES:
            raise ValueError("source tensor too large")
        buf.extend(ENTRY.pack(len(name),codec,w.ndim,len(blob)))
        buf.extend(name)
        buf.extend(blob)
        summary[key]={"shape":list(w.shape),"codec":codec,"bytes":len(blob),
                      "numel":w.numel()}
    content=bytes(buf)
    archive=content+struct.pack("<I",zlib.crc32(content))
    archive+=hashlib.sha256(archive).digest()
    return archive,summary


def unpack_state(data,expected_keys=None):
    """Verify CRC/SHA and every serialized item before constructing model."""
    if len(data)<HDR.size+36:
        raise ValueError("too short CB46 model")
    if hashlib.sha256(data[:-32]).digest()!=data[-32:]:
        raise ValueError("CB46 outer SHA corruption")
    if zlib.crc32(data[:-36])!=struct.unpack_from("<I",data,len(data)-36)[0]:
        raise ValueError("CB46 outer CRC corruption")
    magic,version,count=HDR.unpack_from(data)
    if magic!=MAGIC or version!=1 or count<=0 or count>5000:
        raise ValueError("wrong model manifest schema")
    off=HDR.size
    final=len(data)-36
    result={}
    for _ in range(count):
        if off+ENTRY.size>final:
            raise ValueError("truncated CB46 tensor descriptor")
        name_len,codec,ndim,payload_len=ENTRY.unpack_from(data,off)
        off+=ENTRY.size
        if not (0<name_len<=1024 and 0<payload_len<=MAX_TENSOR_BYTES and
                off+name_len+payload_len<=final and ndim<=8):
            raise ValueError("invalid CB46 tensor metadata or length")
        key=data[off:off+name_len].decode("utf-8")
        off+=name_len
        blob=data[off:off+payload_len]
        off+=payload_len
        if key in result:
            raise ValueError("duplicate CB46 tensor key")
        if codec==INT6:
            if ndim!=2:
                raise ValueError("INT6 codec must be rowwise 2D")
            w=torch.from_numpy(quant_decode(blob).copy())
        elif codec in DTYPES:
            dtype,torch_dtype=DTYPES[codec]
            expected_min=1+4*ndim
            if len(blob)<expected_min or blob[0]!=ndim:
                raise ValueError("short CB46 raw tensor header")
            dims=tuple(struct.unpack_from("<I",blob,1+4*i)[0] for i in range(ndim))
            numel=int(np.prod(dims,dtype=np.int64)) if dims else 1
            required=expected_min+numel*np.dtype(dtype).itemsize
            if numel<=0 or required!=len(blob):
                raise ValueError("incorrect CB46 raw tensor length")
            w=torch.from_numpy(np.frombuffer(blob,dtype=dtype,
                offset=expected_min,count=numel).copy().reshape(dims))
            if w.dtype!=torch_dtype:
                raise ValueError("incorrect raw tensor dtype")
        else:
            raise ValueError("unsupported CB46 tensor codec")
        if not torch.isfinite(w).all() if w.is_floating_point() else False:
            raise ValueError("nonfinite decoded tensor")
        result[key]=w
    if off!=final or len(result)!=count:
        raise ValueError("CB46 trailing bytes or missing parameters")
    if expected_keys is not None and set(result)!=set(expected_keys):
        raise ValueError("missing or extra checkpoint tensors")
    return result


def read_runtime(model,path):
    """Load all 2D INT6 tensors once into inference model's hot dense cache.

    This deliberately optimizes speed, sacrificing *runtime* RAM savings.
    Storage at rest remains genuine INT6. No teacher/original tensors
    need exist in the running process.
    """
    data=Path(path).read_bytes()
    restored=unpack_state(data,expected_keys=model.state_dict().keys())
    report=model.load_state_dict(restored,strict=True)
    if report.missing_keys or report.unexpected_keys:
        raise AssertionError("model checkpoint is incomplete")
    return {"compressed_disk_bytes":len(data),
            "sha256":hashlib.sha256(data).hexdigest(),
            "hot_decoded_weight_bytes":sum(v.numel()*v.element_size()
                                           for v in restored.values())}


def smoke():
    x=torch.arange(-64,64,dtype=torch.float32).reshape(16,8)/64
    src={"matrix":x,"vector":torch.linspace(-1,1,16),
         "indices":torch.arange(3,dtype=torch.int64)}
    buf,meta=pack_state(src)
    restored=unpack_state(buf,expected_keys=src)
    if restored["matrix"].shape!=x.shape or len(meta)!=3:
        raise AssertionError("CB46 roundtrip failed")
    for delta in (3,len(buf)//2,-2):
        corrupt=bytearray(buf)
        corrupt[delta]^=1
        try:unpack_state(bytes(corrupt))
        except ValueError:pass
        else:raise AssertionError("CB46 checksum did not reject bit flip")
    print("CB46 full-state physical INT6 binary/CRC/SHA/roundtrip smoke PASS")


if __name__=="__main__":
    smoke()
