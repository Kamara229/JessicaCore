"""CALIBRI v0.11: exact two-stage QMoE-style dictionary + block/image-plane codec.

Stage 1: use the same static 65,536-entry pair-sequence dictionary construction
as IST-DASLab QMoE's sub1.py and a greedy prefix-trie row encoder. This is
QMoE-STYLE on our own ternarized model, not the original quantized QMoE ckpt.

Stage 2: keep dictionary codes unchanged; for independently seekable groups
of 16/32 rows, choose the physically shortest of raw codes, zlib bytes, or
a 2-D "image-like" low/high byte-plane reshuffle + zlib. A short mode tag and
a row-index are included. Each choice is byte-exact reversible.

Actual QMoE shared dictionary LUT (~512 KiB) must be accounted once per
model, not for every matrix. This code is CPU research, NOT direct GPU execution.
"""
from __future__ import annotations
import heapq
import math
import struct
import zlib
import numpy as np

P_ZERO=0.885
LUT_BYTES=65536*8
MAGIC=b"CL11"
HEADER="<4sIIHHI"
HEADER_BYTES=struct.calcsize(HEADER)
MODE_RAW=0
MODE_ZLIB=1
MODE_PLANES=2
MODE_TIER8=3

class StaticPairDictionary:
    def __init__(self,p0=P_ZERO,count=65536):
        if count<9 or count>65536:raise ValueError("invalid count")
        ps=[p0,(1-p0)/2,(1-p0)/2]
        pair_ps=[p*q for p in ps for q in ps]
        queue=[(-1.,())]
        sequences=[]
        while len(sequences)<count:
            negprob,seq=heapq.heappop(queue)
            if negprob!=-1.:sequences.append(seq)
            if len(seq)==14:continue
            for digit,p in enumerate(pair_ps):
                heapq.heappush(queue,(negprob*p,seq+(digit,)))
        self.sequences=sequences
        self.root=count
        self.children=np.full((count+1,9),-1,dtype=np.int32)
        lookup={():count}
        for idx,seq in enumerate(sequences):
            self.children[lookup[seq[:-1]],seq[-1]]=idx
            lookup[seq]=idx

    def encode(self,q):
        q=np.asarray(q,dtype=np.int8)
        if q.ndim!=2 or q.shape[1]%2 or q.shape[0]<1 or q.shape[1]<2:
            raise ValueError("2D even-width ternary matrix required")
        if np.any(q<-1) or np.any(q>1):raise ValueError("not ternary")
        x=np.where(q==0,0,np.where(q<0,1,2)).astype(np.uint8)
        pairs=(x[:,0::2]+3*x[:,1::2]).astype(np.uint8)
        offsets=np.zeros(len(q)+1,dtype=np.uint32)
        codes=[]
        child=self.children;root=self.root
        for r,row in enumerate(pairs):
            node=root
            for p in row:
                nxt=int(child[node,int(p)])
                if nxt<0:
                    if node==root:raise AssertionError("missing basic pair")
                    codes.append(node)
                    node=int(child[root,int(p)])
                    if node<0:raise AssertionError("missing restart")
                else:node=nxt
            if node!=root:codes.append(node)
            offsets[r+1]=len(codes)
        return np.asarray(codes,dtype='<u2'), offsets

    def decode(self,words,offsets,width):
        words=np.asarray(words,dtype='<u2')
        off=np.asarray(offsets,dtype=np.int64)
        if (off.size<2 or off[0]!=0 or off[-1]!=len(words)
            or np.any(np.diff(off)<0) or width<2 or width%2):
            raise ValueError("invalid code lengths")
        ans=np.empty((len(off)-1,width),dtype=np.int8)
        mapping=np.array([0,-1,1],dtype=np.int8)
        for r in range(len(ans)):
            col=0
            for code in words[off[r]:off[r+1]]:
                seq=self.sequences[int(code)]
                for pair in seq:
                    if col+2>width:raise ValueError("overflow decoding row")
                    ans[r,col]=mapping[pair%3]
                    ans[r,col+1]=mapping[pair//3]
                    col+=2
            if col!=width:raise ValueError("underflow decoding row")
        return ans

def original_qmoe_like_bytes(words,offsets,rows):
    return int(np.asarray(words).size*2+np.asarray(offsets).size*4+rows*4)

def _bytes_from_words(words):
    return np.asarray(words,dtype='<u2').tobytes(order="C")

def _pack_plane(words):
    b=np.frombuffer(_bytes_from_words(words),dtype=np.uint8)
    return b[0::2].tobytes()+b[1::2].tobytes()

def _unpack_plane(data,ncodes):
    if len(data)!=2*ncodes:raise ValueError("bad planar byte stream")
    b=np.frombuffer(data,dtype=np.uint8)
    w=np.empty(2*ncodes,dtype=np.uint8)
    w[0::2]=b[:ncodes]
    w[1::2]=b[ncodes:]
    return w.tobytes()

def _pack_tier8(words):
    w=np.asarray(words,dtype='<u2')
    high=w>=256
    flags=np.packbits(high,bitorder='little')
    low=(w & 255).astype(np.uint8)
    upper=(w[high]>>8).astype(np.uint8)
    return flags.tobytes()+low.tobytes()+upper.tobytes()

def _unpack_tier8(data,ncodes):
    masklen=(ncodes+7)//8
    if len(data)<masklen+ncodes:raise ValueError('short tier8 stream')
    flags=np.unpackbits(np.frombuffer(data[:masklen],dtype=np.uint8),
                        count=ncodes,bitorder='little').astype(bool)
    n_high=int(np.count_nonzero(flags))
    if len(data)!=masklen+ncodes+n_high:raise ValueError('tier8 count mismatch')
    w=np.frombuffer(data,dtype=np.uint8,count=ncodes,offset=masklen).astype(np.uint16)
    w[flags]|=(np.frombuffer(data,dtype=np.uint8,count=n_high,
                             offset=masklen+ncodes).astype(np.uint16)<<8)
    return w.tobytes()

def _payload(words,selector):
    raw=_bytes_from_words(words)
    if selector=="raw":return MODE_RAW,raw
    if selector=="zlib":return MODE_ZLIB,zlib.compress(raw,6)
    if selector=="planes":return MODE_PLANES,zlib.compress(_pack_plane(words),6)
    if selector=="tier8":return MODE_TIER8,_pack_tier8(words)
    if selector=="adaptive":
        options=[(MODE_RAW,raw),
                 (MODE_ZLIB,zlib.compress(raw,6)),
                 (MODE_PLANES,zlib.compress(_pack_plane(words),6)),
                 (MODE_TIER8,_pack_tier8(words))]
        return min(options,key=lambda v:(len(v[1]),v[0]))
    raise ValueError("unknown selector")

def encode_grouped(words,offsets,width,group=16,selector="adaptive"):
    words=np.asarray(words,dtype='<u2')
    offsets=np.asarray(offsets,dtype=np.int64)
    rows=len(offsets)-1
    if rows<1 or offsets[0]!=0 or offsets[-1]!=len(words) or np.any(np.diff(offsets)<0):
        raise ValueError("invalid row offsets")
    if width<=0 or width%2 or group<1 or group>256:raise ValueError("invalid format")
    lengths=np.diff(offsets)
    if np.any(lengths>65535) or len(words)>=2**32:raise ValueError("row code stream oversized")
    n_groups=(rows+group-1)//group
    payload=bytearray();block_offsets=[0];mode_counts={0:0,1:0,2:0,3:0}
    for g in range(n_groups):
        lo=g*group;hi=min(rows,lo+group)
        mode,b=_payload(words[offsets[lo]:offsets[hi]],selector)
        payload.append(mode);payload.extend(b)
        mode_counts[mode]+=1
        block_offsets.append(len(payload))
    meta=struct.pack(HEADER,MAGIC,rows,width,group,0,len(words))
    out=(meta + np.asarray(lengths,dtype='<u2').tobytes()
         +np.asarray(block_offsets,dtype='<u4').tobytes() + bytes(payload))
    return out,mode_counts

def _read_index(blob):
    if len(blob)<HEADER_BYTES:raise ValueError("short header")
    magic,rows,width,group,flags,nwords=struct.unpack_from(HEADER,blob)
    if magic!=MAGIC or not rows or not width or width%2 or group<1 or group>256 or flags:
        raise ValueError("bad header")
    ng=(rows+group-1)//group
    start=HEADER_BYTES+rows*2
    payload_start=start+(ng+1)*4
    if len(blob)<payload_start:raise ValueError("short metadata")
    lengths=np.frombuffer(blob,dtype='<u2',count=rows,offset=HEADER_BYTES).astype(np.uint32)
    offs=np.frombuffer(blob,dtype='<u4',count=ng+1,offset=start).astype(np.int64)
    if (offs[0]!=0 or offs[-1]!=len(blob)-payload_start
       or np.any(np.diff(offs)<=0) or int(lengths.sum())!=nwords):
        raise ValueError("corrupt offsets")
    return rows,width,group,lengths,offs,payload_start,nwords

def _decode_block(blob,group_idx,meta):
    rows,width,group,lengths,offs,payload_start,nwords=meta
    if group_idx<0 or group_idx>=len(offs)-1:raise IndexError(group_idx)
    begin=payload_start+int(offs[group_idx]);end=payload_start+int(offs[group_idx+1])
    mode=blob[begin];compressed=blob[begin+1:end]
    lo=group_idx*group;hi=min(rows,lo+group)
    expected=int(np.sum(lengths[lo:hi]));nbytes=2*expected
    if mode==MODE_RAW:
        raw=compressed
    elif mode==MODE_ZLIB:
        raw=zlib.decompress(compressed)
    elif mode==MODE_PLANES:
        raw=_unpack_plane(zlib.decompress(compressed),expected)
    elif mode==MODE_TIER8:
        raw=_unpack_tier8(compressed,expected)
    else:raise ValueError("unrecognized code mode")
    if len(raw)!=nbytes:raise ValueError("wrong restored block size")
    return np.frombuffer(raw,dtype='<u2'),lo,hi

def decode_grouped(blob):
    meta=_read_index(blob)
    rows,width,group,lengths,offs,payload_start,nwords=meta
    words=np.empty(nwords,dtype='<u2')
    starts=np.concatenate(([0],np.cumsum(lengths,dtype=np.int64))).astype(np.uint32)
    for g in range(len(offs)-1):
        w,lo,hi=_decode_block(blob,g,meta)
        if len(w)!=int(starts[hi]-starts[lo]):raise ValueError("bad group length")
        words[starts[lo]:starts[hi]]=w
    return words,starts,width

def read_one_row(blob,row):
    meta=_read_index(blob)
    rows,width,group,lengths,offs,payload_start,nwords=meta
    if row<0 or row>=rows:raise IndexError(row)
    g=row//group
    w,lo,hi=_decode_block(blob,g,meta)
    start=int(np.sum(lengths[lo:row]))
    return w[start:start+int(lengths[row])].copy()

def grouped_total_bytes(blob,rows,include_scales=True):
    return len(blob)+(rows*4 if include_scales else 0)

def stable_fp64_ffn_equivalence(wi,wo,permutation,seed=90210):
    """Exact algebraic equivalence of ReLU FFN for a *paired* hidden reorder."""
    wia=np.asarray(wi,dtype=np.float64)
    woa=np.asarray(wo,dtype=np.float64)
    p=np.asarray(permutation,dtype=np.int64)
    if not np.array_equal(np.sort(p),np.arange(wia.shape[0])):
        raise ValueError("not permutation")
    if woa.shape[1]!=wia.shape[0]:raise ValueError("not paired")
    rng=np.random.default_rng(seed)
    x=rng.normal(size=wia.shape[1])
    y0=woa @ np.maximum(wia @ x,0)
    y1=woa[:,p] @ np.maximum(wia[p] @ x,0)
    rel=float(np.linalg.norm(y0-y1)/max(np.linalg.norm(y0),1e-20))
    return rel,float(np.max(np.abs(y0-y1)))
