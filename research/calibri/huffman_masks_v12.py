"""CALIBRI v0.12-HUFF: image-mask Huffman coding of exact ternary weights.

An alternative to "dictionary of WORD IDs": code every 8-entry occupancy
mask with a fixed 94-symbol canonical Huffman code (93 masks with <=3 active
coefficients plus an escape for more dense 8-pixel masks). Immediately follow
each mask by sign bits of its positive/negative nonzero elements.

Statistical coding is established; novelty not claimed. Each matrix row is
independently addressable via compact group-index + u16 lengths.
Huffman tree is deterministic from fixed p_nonzero = 0.115.
No per-matrix entropy-model training and no external dictionary payload.
"""
from __future__ import annotations
import heapq
import math
import struct
import numpy as np
from codec import Writer,Reader

MAGIC=b'C12H'
MAGIC_COMPACT=b'C12I'
HDR='<4sIIH'
HDR_SIZE=struct.calcsize(HDR)
ESC=256

def make_codebook(p_nonzero=0.115):
    weights={}
    for mask in range(256):
        k=mask.bit_count()
        if k<=3:weights[mask]=(p_nonzero**k)*((1-p_nonzero)**(8-k))
    weights[ESC]=sum(math.comb(8,k)*(p_nonzero**k)*((1-p_nonzero)**(8-k))
                     for k in range(4,9))
    heap=[];tie=0
    for key,v in sorted(weights.items()):
        heapq.heappush(heap,(v,tie,{key:0}));tie+=1
    while len(heap)>1:
        a=heapq.heappop(heap);b=heapq.heappop(heap)
        node={k:v+1 for sub in (a[2],b[2]) for k,v in sub.items()}
        heapq.heappush(heap,(a[0]+b[0],tie,node));tie+=1
    lengths=heap[0][2]
    # canonical Huffman: shorter words first, ties by integer mask
    order=sorted(lengths,key=lambda x:(lengths[x],x))
    dictionary={};nextcode=0;prev=lengths[order[0]]
    for symbol in order:
        bits=lengths[symbol]
        nextcode<<=bits-prev
        if nextcode>=(1<<bits):raise AssertionError('invalid code lengths')
        dictionary[symbol]=(nextcode,bits)
        nextcode+=1;prev=bits
    reverse={(bits,code):symbol for symbol,(code,bits) in dictionary.items()}
    return dictionary,reverse,max(lengths.values())

CODE,REVERSE,MAX_BITS=make_codebook()

def row_encode(row):
    q=np.asarray(row,dtype=np.int8).ravel()
    if not len(q) or len(q)%8 or np.any(q<-1) or np.any(q>1):
        raise ValueError('expect 8-aligned ternary row')
    wr=Writer()
    for i in range(0,len(q),8):
        block=q[i:i+8]
        mask=sum(int(v!=0)<<j for j,v in enumerate(block))
        if mask in CODE:val,bits=CODE[mask]
        else:val,bits=CODE[ESC]
        wr.put(val,bits)
        if mask not in CODE:wr.put(mask,8)
        pos=np.flatnonzero(block)
        for j in pos:wr.put(int(block[j]>0),1)
    return wr.bytes()

def row_decode(data,width):
    if width<8 or width%8:raise ValueError('invalid width')
    rd=Reader(data)
    q=np.zeros(width,dtype=np.int8)
    for i in range(0,width,8):
        key=0
        for n in range(1,MAX_BITS+1):
            key=(key<<1)|rd.get(1)
            sym=REVERSE.get((n,key),None)
            if sym is not None:break
        else:raise ValueError('invalid Huffman prefix')
        mask=rd.get(8) if sym==ESC else sym
        for j in range(8):
            if (mask>>j)&1:q[i+j]=1 if rd.get(1) else -1
    return q

def encode(q,group=16,index_mode='auto'):
    q=np.asarray(q,dtype=np.int8)
    if q.ndim!=2 or not q.shape[0] or not q.shape[1] or q.shape[1]%8 or group<1 or group>256:
        raise ValueError('invalid dimensions')
    pieces=[row_encode(x) for x in q]
    if any(len(x)>65535 or not len(x) for x in pieces):raise ValueError('invalid length')
    rows,cols=q.shape;payload=bytearray();anchors=[0]
    for first in range(0,rows,group):
        for p in pieces[first:first+group]:payload.extend(p)
        anchors.append(len(payload))
    if len(payload)>=2**32:raise ValueError('matrix too big')
    lens=np.asarray([len(p) for p in pieces],dtype=np.uint32)
    u16=lens.astype('<u2').tobytes()
    escapes=lens>=255
    u8=np.minimum(lens,255).astype(np.uint8).tobytes()
    u8_plus_exceptions=u8+lens[escapes].astype('<u2').tobytes()
    if index_mode not in ('auto','u16','u8_escape'):
        raise ValueError('unknown index mode')
    compact=(index_mode=='u8_escape' or
             (index_mode=='auto' and len(u8_plus_exceptions)<len(u16)))
    return (struct.pack(HDR,MAGIC_COMPACT if compact else MAGIC,rows,cols,group)
            + (u8_plus_exceptions if compact else u16)
            + np.asarray(anchors,dtype='<u4').tobytes()+payload)

def _index(blob):
    if len(blob)<HDR_SIZE:raise ValueError('short header')
    magic,rows,cols,group=struct.unpack_from(HDR,blob)
    if magic not in (MAGIC,MAGIC_COMPACT) or not rows or not cols or cols%8 or not 1<=group<=256:
        raise ValueError('invalid header')
    groups=(rows+group-1)//group
    if magic==MAGIC:
        end_lens=HDR_SIZE+2*rows
        if len(blob)<end_lens:raise ValueError('short u16 indexes')
        lengths=np.frombuffer(blob,dtype='<u2',count=rows,offset=HDR_SIZE).astype(np.uint32)
    else:
        if len(blob)<HDR_SIZE+rows:raise ValueError('short u8 indexes')
        u8=np.frombuffer(blob,dtype=np.uint8,count=rows,offset=HDR_SIZE)
        escaped=u8==255
        n_extra=int(np.count_nonzero(escaped))
        end_lens=HDR_SIZE+rows+2*n_extra
        if len(blob)<end_lens:raise ValueError('short escaped indexes')
        lengths=u8.astype(np.uint32)
        lengths[escaped]=np.frombuffer(blob,dtype='<u2',
                                       count=n_extra,offset=HDR_SIZE+rows)
    payload_start=end_lens+4*(groups+1)
    if len(blob)<payload_start:raise ValueError('short anchors')
    anchor=np.frombuffer(blob,dtype='<u4',count=groups+1,offset=end_lens)
    if anchor[0]!=0 or anchor[-1]!=len(blob)-payload_start:
        raise ValueError('bad anchors')
    for g in range(groups):
        st=g*group;en=min(rows,st+group)
        if int(anchor[g+1])-int(anchor[g])!=int(lengths[st:en].astype(np.int64).sum()):
            raise ValueError('inconsistent lengths')
    return rows,cols,group,lengths,anchor,payload_start

def read_row(blob,row):
    rows,cols,group,lengths,anchor,payload_start=_index(blob)
    if row<0 or row>=rows:raise IndexError(row)
    st=(row//group)*group
    pos=payload_start+int(anchor[row//group])+int(lengths[st:row].astype(np.int64).sum())
    return row_decode(blob[pos:pos+int(lengths[row])],cols)

def decode(blob):
    rows,cols,group,lengths,anchor,payload_start=_index(blob)
    q=np.empty((rows,cols),dtype=np.int8)
    for row in range(rows):q[row,:]=read_row(blob,row)
    return q

def direct_matvec(blob,x,scales):
    """Compute row dot-products from codes, no full matrix reconstruction."""
    rows,cols,group,lengths,anchor,payload_start=_index(blob)
    x=np.asarray(x,dtype=np.float64).ravel()
    s=np.asarray(scales,dtype=np.float64)
    if x.size!=cols or s.shape!=(rows,2):raise ValueError('shape mismatch')
    out=np.empty(rows,dtype=np.float64)
    for i in range(rows):
        row=read_row(blob,i)
        out[i]=np.where(row<0,s[i,0],np.where(row>0,s[i,1],0.))@x
    return out
