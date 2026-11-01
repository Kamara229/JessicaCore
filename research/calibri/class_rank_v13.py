"""CALIBRI v0.13 — CLASS-RANK: a seekable enumerative ternary codec.

Inspired by RRR succinct bitvectors, enumerative coding, and coding of
2-D binary image masks. A block of 8,16,32,64 ternary symbols is encoded:
  (1) canonical Huffman count of nonzero coefficients (popcount class),
  (2) fixed-size combinatorial support rank in C(B,k) possibilities,
  (3) one sign bit per nonzero.
Rows are independently addressable via u8/escaped-u16 lengths and 4-byte
group offsets. Every byte of side information is counted. The Huffman table
is deterministically generated from one of 4 predefined sparsity priors,
and its ID is in the header; no corpus-trained shared dictionary.

Coding operations are CPU Python reference only. No claim of a novel
mathematical primitive, GPU performance or whole-model accuracy.
"""
from __future__ import annotations

import heapq
import math
import struct

import numpy as np
from codec import Writer,Reader,rank_positions,unrank_positions

MAGIC=b'C13R'
FMT='<4sIIHBB'
HDR=struct.calcsize(FMT)
BLOCK_CHOICES=(8,16,32,64)
P_NONZERO_PRIORS=(.05,.115,.25,.50)
MAX_ROWS=4_000_000
MAX_COLS=4_000_000

def _huffman_probabilities(block,p_nonzero):
    return [(math.comb(block,k)*p_nonzero**k
             *(1.-p_nonzero)**(block-k)) for k in range(block+1)]

def _codes(block,prior):
    probabilities=_huffman_probabilities(block,P_NONZERO_PRIORS[prior])
    items=[(p,i,{i:0}) for i,p in enumerate(probabilities)]
    heapq.heapify(items)
    counter=block+1
    while len(items)>1:
        lo=heapq.heappop(items);hi=heapq.heappop(items)
        merged={key:depth+1 for sub in (lo[2],hi[2]) for key,depth in sub.items()}
        heapq.heappush(items,(lo[0]+hi[0],counter,merged))
        counter+=1
    lengths=items[0][2]
    order=sorted(lengths,key=lambda k:(lengths[k],k))
    code=0;prev=lengths[order[0]]
    forward={};reverse={}
    for k in order:
        n=lengths[k]
        code<<=n-prev
        if code>=(1<<n):raise AssertionError("invalid canonical count code")
        forward[k]=(code,n)
        reverse[(n,code)]=k
        code+=1;prev=n
    return forward,reverse,max(lengths.values())

TABLE={(n,p):_codes(n,p) for n in BLOCK_CHOICES for p in range(len(P_NONZERO_PRIORS))}
RANKBITS={(n,k):(math.comb(n,k)-1).bit_length() for n in BLOCK_CHOICES for k in range(n+1)}

def _row_encode(q,block,prior):
    q=np.asarray(q,dtype=np.int8)
    if not q.size or q.size%block or np.any(q<-1) or np.any(q>1):
        raise ValueError("input is not block-aligned ternary")
    wr=Writer();codebook=TABLE[(block,prior)][0]
    for i in range(0,len(q),block):
        current=q[i:i+block]
        where=np.flatnonzero(current)
        k=len(where)
        code,bits=codebook[k]
        wr.put(code,bits)
        wr.put(rank_positions(where.tolist()),RANKBITS[(block,k)])
        signs=sum(int(current[j]>0)<<shift for shift,j in enumerate(where))
        wr.put(signs,k)
    return wr.bytes()

def _row_decode(blob,cols,block,prior):
    if cols<block or cols%block:raise ValueError("invalid block length")
    rd=Reader(blob)
    reverse=TABLE[(block,prior)][1]
    longest=TABLE[(block,prior)][2]
    q=np.zeros(cols,dtype=np.int8)
    for start in range(0,cols,block):
        code=0
        k=None
        for bits in range(1,longest+1):
            code=code*2+rd.get(1)
            k=reverse.get((bits,code),None)
            if k is not None:break
        if k is None:raise ValueError("bad Huffman class")
        r=rd.get(RANKBITS[(block,k)])
        if r>=math.comb(block,k):raise ValueError("invalid combinatorial rank")
        p=unrank_positions(block,k,r)
        signs=rd.get(k)
        for shift,j in enumerate(p):
            q[start+j]=1 if (signs>>shift)&1 else -1
    return q

def encode(q,block=32,group=64,prior='auto',index_mode='auto'):
    q=np.asarray(q,dtype=np.int8)
    if (q.ndim!=2 or q.shape[0]<1 or q.shape[1]<1 or
        q.shape[0]>MAX_ROWS or q.shape[1]>MAX_COLS or
        block not in BLOCK_CHOICES or q.shape[1]%block or
        not 1<=group<=256 or np.any(q<-1) or np.any(q>1)):
        raise ValueError("bad ternary matrix")
    if prior=='auto':
        dens=float(np.count_nonzero(q)/q.size)
        prior=min(range(len(P_NONZERO_PRIORS)),key=lambda i:abs(P_NONZERO_PRIORS[i]-dens))
    if prior not in range(len(P_NONZERO_PRIORS)):raise ValueError("bad prior")
    if index_mode not in ('auto','u16','u8_escape'):raise ValueError("bad index mode")
    rows,cols=q.shape
    chunks=[_row_encode(row,block,prior) for row in q]
    lengths=np.asarray([len(b) for b in chunks],dtype=np.uint32)
    if np.any(lengths==0) or np.any(lengths>65535):raise ValueError("row too large")
    plain=lengths.astype('<u2').tobytes()
    esc=lengths>=255
    compact=np.minimum(lengths,255).astype(np.uint8).tobytes()+lengths[esc].astype('<u2').tobytes()
    use_compact=(index_mode=='u8_escape' or
                 (index_mode=='auto' and len(compact)<len(plain)))
    anchors=[0];payload=bytearray()
    for start in range(0,rows,group):
        for b in chunks[start:start+group]:payload.extend(b)
        anchors.append(len(payload))
    if len(payload)>=2**32:raise ValueError("payload exceeds u32")
    flag=(prior<<1)|int(use_compact)
    header=struct.pack(FMT,MAGIC,rows,cols,group,block,flag)
    return (header+(compact if use_compact else plain)
            +np.asarray(anchors,dtype='<u4').tobytes()+payload)

def _index(blob):
    if len(blob)<HDR:raise ValueError("truncated header")
    magic,rows,cols,group,block,flags=struct.unpack_from(FMT,blob)
    if (magic!=MAGIC or not 1<=rows<=MAX_ROWS or not 1<=cols<=MAX_COLS
        or not 1<=group<=256 or block not in BLOCK_CHOICES or cols%block
        or flags>=8):
        raise ValueError("bad header")
    prior=flags>>1;use_compact=bool(flags&1)
    if use_compact:
        if len(blob)<HDR+rows:raise ValueError("truncated compact lengths")
        low=np.frombuffer(blob,dtype=np.uint8,count=rows,offset=HDR)
        cnt=int(np.count_nonzero(low==255))
        end=HDR+rows+2*cnt
        if len(blob)<end:raise ValueError("truncated escapes")
        lengths=low.astype(np.uint32)
        lengths[low==255]=np.frombuffer(blob,dtype='<u2',count=cnt,offset=HDR+rows)
    else:
        end=HDR+2*rows
        if len(blob)<end:raise ValueError("truncated normal lengths")
        lengths=np.frombuffer(blob,dtype='<u2',count=rows,offset=HDR).astype(np.uint32)
    ng=(rows+group-1)//group
    datastart=end+4*(ng+1)
    if len(blob)<datastart:raise ValueError("truncated offsets")
    anchors=np.frombuffer(blob,dtype='<u4',count=ng+1,offset=end)
    if int(anchors[0])!=0 or int(anchors[-1])!=len(blob)-datastart:
        raise ValueError("invalid anchors")
    if np.any(lengths==0):raise ValueError("empty row")
    for g in range(ng):
        lo=g*group;hi=min(rows,lo+group)
        if int(anchors[g+1])-int(anchors[g])!=int(lengths[lo:hi].astype(np.int64).sum()):
            raise ValueError("group index mismatch")
    return rows,cols,group,block,prior,lengths,anchors,datastart

def _read_one_from_index(blob,row,meta):
    rows,cols,group,block,prior,lengths,anchors,datastart=meta
    if row<0 or row>=rows:raise IndexError(row)
    group_first=(row//group)*group
    pos=datastart+int(anchors[row//group])+int(lengths[group_first:row].astype(np.int64).sum())
    return _row_decode(blob[pos:pos+int(lengths[row])],cols,block,prior)

def read_row(blob,row):
    return _read_one_from_index(blob,row,_index(blob))

def decode(blob):
    meta=_index(blob)
    rows,cols=meta[:2]
    q=np.empty((rows,cols),dtype=np.int8)
    for i in range(rows):
        q[i,:]=_read_one_from_index(blob,i,meta)
    return q

def direct_matvec(blob,x,lo_hi):
    """Read each row once and dot it with x without reconstructing W globally."""
    meta=_index(blob);rows,cols=meta[:2]
    x=np.asarray(x,dtype=np.float64).ravel()
    levels=np.asarray(lo_hi,dtype=np.float64)
    if len(x)!=cols or levels.shape!=(rows,2):raise ValueError("shape mismatch")
    y=np.empty(rows,dtype=np.float64)
    for i in range(rows):
        row=_read_one_from_index(blob,i,meta)
        vals=np.where(row<0,levels[i,0],np.where(row>0,levels[i,1],0.))
        y[i]=float(vals@x)
    return y
