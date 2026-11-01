"""CALIBRI v32 experiment: physical 4D tensor-core + adaptive tiled 3/4-bit residual.

This is a new *combination/prototype* of known TT-SVD and block quantization,
NOT a demonstrated novel mathematical primitive. The contiguous byte stream
has CRC, explicit topology and tile-level precision. A separate streaming
matvec uses no full reconstructed weight; Python loops are intentionally a
slow research reference, not a production native kernel.
"""
from __future__ import annotations

import struct
import time
import zlib
import numpy as np

from geometric_v16 import tt_svd, pack as pack_tt, unpack as unpack_tt, reconstruct

HEADER = struct.Struct("<4sHHHHIII")
# magic, rows, cols, block_rows, block_cols, TT_blob_len, tiles_len, CRC32
MAGIC = b"CM32"


def packed_symbols(q, bits):
    if bits not in (3, 4):
        raise ValueError("unapproved residual precision")
    x = np.asarray(q, dtype=np.int16).reshape(-1)
    bound = (1 << (bits - 1)) - 1
    if np.any(x < -bound) or np.any(x > bound):
        raise ValueError("symbols outside signed quantization range")
    symbols = (x & ((1 << bits) - 1)).astype(np.uint8)
    bitstream = ((symbols[:, None] >> np.arange(bits - 1, -1, -1, dtype=np.uint8)) & 1).ravel()
    return np.packbits(bitstream).tobytes()


def unpacked_symbols(raw, count, bits):
    expected = (count * bits + 7) // 8
    if len(raw) != expected or bits not in (3, 4):
        raise ValueError("invalid packed block length")
    b = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:count * bits].reshape(-1, bits)
    unsigned = b.astype(np.int16) @ (2 ** np.arange(bits-1, -1, -1, dtype=np.int16))
    return np.where(unsigned >= (1 << (bits-1)), unsigned - (1 << bits), unsigned).astype(np.float32)


def quant_tile(tile, bits):
    bound = (1 << (bits - 1)) - 1
    mx = float(np.max(np.abs(tile)))
    scale = np.float16(mx / bound) if mx else np.float16(1)
    if scale == 0 or not np.isfinite(scale):
        raise ValueError("FP16 tile scale overflow/underflow")
    s = float(scale)
    q = np.clip(np.rint(tile / s), -bound, bound).astype(np.int16)
    recon = (q.astype(np.float32) * s).reshape(tile.shape)
    return scale, packed_symbols(q, bits), recon


def encode(w, input_second_moment, budget, rank=4, m2=8, n2=32, br=16, bc=32):
    w = np.asarray(w, np.float32)
    var = np.asarray(input_second_moment, np.float32)
    if (w.ndim != 2 or min(w.shape) == 0 or w.shape[0] % br or w.shape[1] % bc
            or var.shape != (w.shape[1],) or not np.isfinite(w).all()
            or not np.isfinite(var).all() or np.any(var < 0)):
        raise ValueError("invalid trained weight shape, tile size or training moments")
    m, n = w.shape
    g1, g2 = tt_svd(w, rank, m2=m2, n2=n2)
    core = pack_tt(g1, g2)
    a, b, residual = unpack_tt(core)
    if residual is not None:
        raise AssertionError("expected two TT cores only")
    residue = w - reconstruct(a, b)
    tiles = []
    scores = []
    tr, tc = m // br, n // bc
    for i in range(tr):
        for j in range(tc):
            slice_r = slice(i*br, (i+1)*br)
            slice_c = slice(j*bc, (j+1)*bc)
            part = residue[slice_r, slice_c]
            s3, d3, q3 = quant_tile(part, 3)
            s4, d4, q4 = quant_tile(part, 4)
            if len(d4)-len(d3) != br*bc//8:
                raise AssertionError("tile difference is not exactly one bit")
            importance = var[slice_c][None, :]
            err3 = float(np.sum(importance * np.square(part-q3), dtype=np.float64))
            err4 = float(np.sum(importance * np.square(part-q4), dtype=np.float64))
            # The score is a train-moment weighted proxy, not direct task PPL.
            scores.append(err3-err4)
            tiles.append(((s3,d3), (s4,d4)))
    base_bytes = HEADER.size + len(core) + sum(1+2+len(entry[0][1]) for entry in tiles)
    if base_bytes > budget:
        raise ValueError("no feasible 3-bit+TT storage under matched INT4 budget")
    upgrade = br*bc//8
    slots = min(len(tiles), (budget-base_bytes)//upgrade)
    ordered = sorted((i for i in range(len(tiles)) if scores[i] > 0),
                     key=lambda i: (-scores[i], i))
    chosen = set(ordered[:slots])
    blocks = bytearray()
    for i,options in enumerate(tiles):
        bits=4 if i in chosen else 3
        scale, stream = options[bits-3]
        blocks.extend(struct.pack("<Be", bits, scale))
        blocks.extend(stream)
    payload = core + bytes(blocks)
    blob = HEADER.pack(MAGIC,m,n,br,bc,len(core),len(blocks),zlib.crc32(payload)) + payload
    if len(blob) > budget or len(blob) != HEADER.size+len(payload):
        raise AssertionError("encoded bytes exceed requested physical storage limit")
    return blob, {
        "tiles_total":len(tiles),
        "tiles_3bit":len(tiles)-len(chosen),
        "tiles_4bit":len(chosen),
        "train_proxy_upgrade_gain":float(sum(scores[i] for i in chosen)),
        "base_3bit_plus_TT_bytes":base_bytes,
        "budget_bytes":int(budget),
        "actual_bytes":len(blob),
        "rank":rank,
        "TT_topology":[m//m2,m2,n//n2,n2],
        "residual_tile_shape":[br,bc],
        "selection":"train activation second moment times 3bit-minus-4bit per-tile squared error, ranked under physical byte cap",
    }


def parsed(blob):
    if len(blob) < HEADER.size:
        raise ValueError("truncated multidimensional format")
    magic,m,n,br,bc,clen,tlen,crc = HEADER.unpack_from(blob)
    if (magic!=MAGIC or not (m and n and br and bc and clen and tlen)
            or m%br or n%bc or HEADER.size+clen+tlen!=len(blob)
            or zlib.crc32(blob[HEADER.size:])!=crc):
        raise ValueError("bad multidimensional framing or CRC")
    core=blob[HEADER.size:HEADER.size+clen]
    a,b,res=unpack_tt(core)
    if res is not None or m!=a.shape[0]*b.shape[1] or n!=a.shape[1]*b.shape[2]:
        raise ValueError("TT core does not match block metadata")
    k=HEADER.size+clen
    blocks=[]
    for i in range(m//br):
        for j in range(n//bc):
            if len(blob)<k+3:
                raise ValueError("truncated residual header")
            bits,scale = struct.unpack_from("<Be",blob,k)
            k+=3
            if bits not in (3,4) or not np.isfinite(scale) or scale<=0:
                raise ValueError("invalid residual bitwidth or scale")
            count=br*bc
            size=(count*bits+7)//8
            if len(blob)<k+size:
                raise ValueError("truncated residual tile")
            blocks.append((i,j,bits,float(scale),blob[k:k+size]))
            k+=size
    if k != len(blob):
        raise ValueError("trailing bytes in multidimensional format")
    return a,b,m,n,br,bc,core,blocks


def decode(blob):
    a,b,m,n,br,bc,core,blocks=parsed(blob)
    w=reconstruct(a,b)
    for i,j,bits,scale,raw in blocks:
        tile=unpacked_symbols(raw,br*bc,bits).reshape(br,bc)*scale
        w[i*br:(i+1)*br,j*bc:(j+1)*bc]+=tile
    return w


def streamed_matmul(blob,x):
    """Actual byte-backed forward: TT contraction and tiled scratch, no dense W.

    EXPERIMENTAL PYTHON REFERENCE: each residual tile is decoded into an
    ephemeral br*bc FP32 scratch. Benchmarks likely much slower than native
    dense GEMM. This does not imply portable/production packed acceleration.
    """
    import torch
    from geometric_v16 import make_layer
    a,b,m,n,br,bc,core,blocks=parsed(blob)
    xx=torch.as_tensor(x,dtype=torch.float32)
    if xx.ndim!=2 or xx.shape[1]!=n:
        raise ValueError("streamed matmul input dimension")
    with torch.no_grad():
        y=make_layer(core)(xx)
        for i,j,bits,scale,raw in blocks:
            tile=unpacked_symbols(raw,br*bc,bits).reshape(br,bc)*scale
            zz=torch.from_numpy(tile)
            y[:,i*br:(i+1)*br] += xx[:,j*bc:(j+1)*bc] @ zz.T
    return y


def smoke():
    # One physical topology with real but deterministic test coefficients;
    # production experimental runs always load actual pretrained weights.
    rng=np.random.default_rng(32)
    w=rng.standard_normal((128,512)).astype(np.float32)*0.02
    v=np.linspace(0.1,1.3,512,dtype=np.float32)
    from fixedbit_v15 import encode as fixed_encode
    cap=len(fixed_encode(w,4))
    blob,stats=encode(w,v,cap,rank=4,m2=8,n2=32,br=16,bc=32)
    q=decode(blob)
    xx=rng.standard_normal((7,512)).astype(np.float32)
    y1=streamed_matmul(blob,xx).detach().numpy()
    y2=xx @ q.T
    if len(blob)>cap or np.max(np.abs(y1-y2))>3e-5:
        raise AssertionError("physical matched storage or TT + streamed tiled computation mismatch")
    damaged=bytearray(blob)
    damaged[-1]^=1
    try:
        decode(bytes(damaged))
    except ValueError:
        pass
    else:
        raise AssertionError("CRC must reject corrupted physical payload")
    print("CALIBRI v32 physical 4D TT + adaptive 3/4-bit tiles, CRC, no-full-W streaming equality PASS",flush=True)


if __name__ == "__main__":
    smoke()
