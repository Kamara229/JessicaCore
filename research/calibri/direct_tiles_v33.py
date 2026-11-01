"""CALIBRI v33: direct 2D tiled variable-precision weights (TT-free ablation).

Each tile has an FP16 scale and a physical packed signed 3- or 4-bit
residual stream. Bit allocation is precomputed only from TRAIN activation
second moments under an exact fixed INT4 byte budget. No tensor-train cores.
This is a combination of known block quantization and fixed-budget bit
allocation, not an established novel quantization theory.
"""
from __future__ import annotations
import struct
import zlib
import numpy as np

from multidim_v32 import packed_symbols, unpacked_symbols, quant_tile

HEADER=struct.Struct("<4sHHHHII")
MAGIC=b"CT33"


def encode(w,activation_moment,budget,br=16,bc=32):
    w=np.asarray(w,np.float32)
    var=np.asarray(activation_moment,np.float32)
    if (w.ndim!=2 or min(w.shape)<1 or br<1 or bc<1
        or w.shape[0]%br or w.shape[1]%bc
        or br*bc%8 or var.shape!=(w.shape[1],)
        or not np.isfinite(w).all() or not np.isfinite(var).all()
        or (var<0).any() or budget<=HEADER.size):
        raise ValueError("unsupported trained matrix or tile geometry")
    m,n=w.shape
    all_tiles=[]
    gain=[]
    total=HEADER.size
    for row in range(0,m,br):
        for col in range(0,n,bc):
            tile=w[row:row+br,col:col+bc]
            s3,b3,d3=quant_tile(tile,3)
            s4,b4,d4=quant_tile(tile,4)
            assert len(b4)>len(b3)
            importance=var[col:col+bc][None,:]
            e3=float(np.sum(importance*(tile-d3)**2,dtype=np.float64))
            e4=float(np.sum(importance*(tile-d4)**2,dtype=np.float64))
            total+=3+len(b3)
            gain.append(e3-e4)
            all_tiles.append(((s3,b3),(s4,b4),e3,e4))
    if total>budget:
        raise ValueError("no feasible mixed-precision physical stream under INT4 budget")
    increment=br*bc//8
    slots=min(len(all_tiles),(budget-total)//increment)
    selected=set(sorted((i for i,v in enumerate(gain) if v>0),
                        key=lambda i:(-gain[i],i))[:slots])
    payload=bytearray()
    train_error=0.
    for i,(q3,q4,e3,e4) in enumerate(all_tiles):
        use4=i in selected
        scale,bits=q4 if use4 else q3
        payload.extend(struct.pack("<Be",4 if use4 else 3,float(scale)))
        payload.extend(bits)
        train_error+=e4 if use4 else e3
    raw=bytes(payload)
    blob=HEADER.pack(MAGIC,m,n,br,bc,len(raw),zlib.crc32(raw))+raw
    if len(blob)>budget or len(blob)!=HEADER.size+len(payload):
        raise AssertionError("physical budget accounting error")
    metadata={
        "tiles_total":len(all_tiles),
        "tiles_3bit":len(all_tiles)-len(selected),
        "tiles_4bit":len(selected),
        "train_weighted_MSE_proxy_sum":float(train_error),
        "train_proxy_gain_from_4bit_upgrades":float(sum(gain[i] for i in selected)),
        "physical_bytes":len(blob),
        "budget_bytes":budget,
        "block":[br,bc],
        "method":"rank tiles by TRAIN activation second moment times 3-bit to 4-bit residual error improvement",
    }
    return blob,metadata


def parse(blob):
    if len(blob)<HEADER.size:
        raise ValueError("truncated tiled weight stream")
    magic,m,n,br,bc,length,crc=HEADER.unpack_from(blob)
    if (magic!=MAGIC or min(m,n,br,bc)<1 or m%br or n%bc
        or br*bc%8 or HEADER.size+length!=len(blob)
        or zlib.crc32(blob[HEADER.size:])!=crc):
        raise ValueError("bad tiled weight header/CRC")
    tiles=[]
    idx=HEADER.size
    for row in range(0,m,br):
        for col in range(0,n,bc):
            if idx+3>len(blob):
                raise ValueError("short tiled stream metadata")
            bits,scale=struct.unpack_from("<Be",blob,idx)
            idx+=3
            if bits not in (3,4) or not np.isfinite(scale) or scale<=0:
                raise ValueError("invalid tile scale/bitwidth")
            size=br*bc*bits//8
            if idx+size>len(blob):
                raise ValueError("short tiled symbols")
            tiles.append((row,col,bits,float(scale),blob[idx:idx+size]))
            idx+=size
    if idx!=len(blob):
        raise ValueError("extra trailing tiled data")
    return m,n,br,bc,tiles


def decode(blob):
    m,n,br,bc,tiles=parse(blob)
    out=np.empty((m,n),np.float32)
    for row,col,bits,scale,raw in tiles:
        out[row:row+br,col:col+bc]=unpacked_symbols(raw,br*bc,bits).reshape(br,bc)*scale
    return out


def stream_matmul(blob,x):
    """Direct physical tiled forward: one br×bc scratch tile, no full dense W.

    Not an optimized native kernel; Python allocation/dispatch dominates.
    """
    import torch
    m,n,br,bc,tiles=parse(blob)
    xx=torch.as_tensor(x,dtype=torch.float32)
    if xx.ndim!=2 or xx.shape[1]!=n:
        raise ValueError("input must be [tokens, in_features]")
    result=torch.zeros((xx.shape[0],m),dtype=torch.float32)
    with torch.inference_mode():
        for row,col,bits,scale,raw in tiles:
            tile=unpacked_symbols(raw,br*bc,bits).reshape(br,bc)*scale
            result[:,row:row+br] += xx[:,col:col+bc] @ torch.from_numpy(tile).T
    return result


def smoke():
    from fixedbit_v15 import encode as fixed_encode
    rng=np.random.default_rng(33)
    w=rng.normal(size=(128,512)).astype(np.float32)*0.02
    moments=np.linspace(0.3,1.1,512,dtype=np.float32)
    x=rng.normal(size=(5,512)).astype(np.float32)
    cap=len(fixed_encode(w,4))
    for br,bc in ((16,32),(8,64)):
        blob,meta=encode(w,moments,cap,br,bc)
        recon=decode(blob)
        stream=stream_matmul(blob,x).detach().numpy()
        dense=x@recon.T
        assert len(blob)<=cap and meta["tiles_3bit"]>0
        if not np.allclose(stream,dense,atol=2e-5,rtol=2e-5):
            raise AssertionError("tile-backed stream inconsistent with own dense decode")
        bad=bytearray(blob)
        bad[-1]^=1
        try:
            decode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("CRC must detect corruption")
    print("v33 physical 3/4-bit variable tiles both geometries, CRC, exact byte budget and no-full-W forward PASS")


if __name__=="__main__":
    smoke()
