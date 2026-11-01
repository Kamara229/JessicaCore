"""CALIBRI v0.12 — seekable patched bit-packed WORD IDs after QMoE dictionary.

Dictionary coding by exact QMoE-style greedy prefix code is performed by
dual_codec.StaticPairDictionary; this stage is a fully reversible re-encoding
of its 16-bit dictionary IDs. It borrows the *established* PFOR integer-codec
idea: bit-pack low bits; mark exceptions; pack high bits only for exceptions.

Each group contains independently decompressible consecutive rows. Each
subblock selects its own bit width to minimize PHYSICAL bytes. The format
counts all row indexes, group offsets, flags and per-block width metadata.
No zlib/entropy dependencies, no GPU kernel, no claims of novelty.
"""
from __future__ import annotations
import math
import struct
import numpy as np

MAGIC=b'CL12'
HDR='<4sIIHHI'
HDR_SIZE=struct.calcsize(HDR)
CANDIDATE_BITS=(0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)

def _pack(v,bits):
    v=np.asarray(v,dtype=np.uint32).ravel()
    if bits==0:
        if np.any(v):raise ValueError('cannot put nonzero with zero bits')
        return b''
    if bits>16 or np.any(v>=(1<<bits)):raise ValueError('bit width overflow')
    if bits==16:return v.astype('<u2').tobytes()
    shifts=np.arange(bits,dtype=np.uint32)
    arr=((v[:,None]>>shifts)&1).astype(np.uint8)
    return np.packbits(arr.ravel(),bitorder='little').tobytes()

def _unpack(raw,n,bits):
    expected=(n*bits+7)//8
    if len(raw)!=expected:raise ValueError('bad packed width')
    if bits==0:return np.zeros(n,dtype=np.uint16)
    if bits==16:return np.frombuffer(raw,dtype='<u2',count=n).copy()
    bitarray=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),
                           bitorder='little',count=n*bits)
    powers=(1<<np.arange(bits,dtype=np.uint32))
    return (bitarray.reshape(n,bits).astype(np.uint32)@powers).astype(np.uint16)

def _estimate(w,bits):
    n=len(w)
    if bits==16:return 1+2*n
    errors=int(np.count_nonzero(w>=(1<<bits)))
    return (1+(n*bits+7)//8+(n+7)//8+(errors*(16-bits)+7)//8)

def _block_encode(values):
    v=np.asarray(values,dtype=np.uint16).ravel()
    width=min(CANDIDATE_BITS,key=lambda b:(_estimate(v,b),-b))
    if width==16:return bytes((16,))+v.astype('<u2').tobytes(),width
    low=(v & ((1<<width)-1)).astype(np.uint16)
    high=(v>>width)
    flags=high>0
    flagbits=np.packbits(flags.astype(np.uint8),bitorder='little').tobytes()
    out=bytes((width,))+_pack(low,width)+flagbits+_pack(high[flags],16-width)
    if len(out)!=_estimate(v,width):raise AssertionError('packing length')
    return out,width

def _block_decode(data,n):
    if len(data)<1:raise ValueError('missing mode')
    bits=data[0]
    if bits>16:raise ValueError('invalid bit-width')
    if bits==16:
        m=2*n
        if len(data)<1+m:raise ValueError('short raw')
        return np.frombuffer(data[1:1+m],dtype='<u2').copy(),1+m
    lowbytes=(n*bits+7)//8
    maskbytes=(n+7)//8
    start=1
    if len(data)<start+lowbytes+maskbytes:raise ValueError('short base/mask')
    lo=_unpack(data[start:start+lowbytes],n,bits)
    mask=np.unpackbits(np.frombuffer(data[start+lowbytes:start+lowbytes+maskbytes],
                                      dtype=np.uint8),bitorder='little',count=n).astype(bool)
    n_hi=int(mask.sum());hbytes=(n_hi*(16-bits)+7)//8
    end=start+lowbytes+maskbytes+hbytes
    if len(data)<end:raise ValueError('short exception data')
    hi=_unpack(data[start+lowbytes+maskbytes:end],n_hi,16-bits)
    out=lo.astype(np.uint32)
    out[mask]|=(hi.astype(np.uint32)<<bits)
    if np.any(out>65535):raise ValueError('overflow')
    return out.astype(np.uint16),end

def encode_patched(words,row_offsets,width,group_rows=16,block_words=128):
    words=np.asarray(words,dtype=np.uint16).ravel()
    row_offsets=np.asarray(row_offsets,dtype=np.int64).ravel()
    rows=len(row_offsets)-1
    if (rows<1 or not 1<=group_rows<=256 or not 16<=block_words<=1024
        or not width or width%2 or row_offsets[0]!=0 or
        row_offsets[-1]!=len(words) or np.any(np.diff(row_offsets)<0)):
        raise ValueError('bad dimensions/index')
    lengths=np.diff(row_offsets)
    if np.any(lengths>65535) or len(words)>=2**32:
        raise ValueError('row too long/stream too large')
    body=bytearray();anchors=[0];widthhist={}
    for st in range(0,rows,group_rows):
        end=min(rows,st+group_rows)
        groupwords=words[row_offsets[st]:row_offsets[end]]
        for j in range(0,len(groupwords),block_words):
            payload,b=_block_encode(groupwords[j:j+block_words])
            body.extend(payload)
            widthhist[str(b)]=widthhist.get(str(b),0)+1
        anchors.append(len(body))
    meta=struct.pack(HDR,MAGIC,rows,width,group_rows,block_words,len(words))
    blob=(meta+np.asarray(lengths,dtype='<u2').tobytes()
          +np.asarray(anchors,dtype='<u4').tobytes()+body)
    return blob,widthhist

def _header(blob):
    if len(blob)<HDR_SIZE:raise ValueError('truncated header')
    magic,rows,width,group,block_words,nwords=struct.unpack_from(HDR,blob)
    if magic!=MAGIC or not rows or not width or width%2 or not 1<=group<=256 or not 16<=block_words<=1024:
        raise ValueError('invalid header')
    ng=(rows+group-1)//group;ptr=HDR_SIZE+2*rows
    header_end=ptr+4*(ng+1)
    if len(blob)<header_end:raise ValueError('truncated indexes')
    lengths=np.frombuffer(blob,dtype='<u2',count=rows,offset=HDR_SIZE).astype(np.int32)
    anchor=np.frombuffer(blob,dtype='<u4',count=ng+1,offset=ptr).astype(np.int64)
    if int(lengths.sum())!=nwords or anchor[0]!=0 or anchor[-1]!=len(blob)-header_end or np.any(np.diff(anchor)<=0):
        raise ValueError('bad lengths')
    return rows,width,group,block_words,nwords,lengths,anchor,header_end

def _decode_group(blob,meta,g):
    rows,width,group,block_words,nwords,lengths,anchor,start=meta
    if g<0 or g>=len(anchor)-1:raise IndexError('group out of bounds')
    st=g*group;end=min(rows,st+group)
    target=int(lengths[st:end].sum())
    pos=start+int(anchor[g]);limit=start+int(anchor[g+1])
    values=np.empty(target,dtype=np.uint16)
    done=0
    while done<target:
        n=min(block_words,target-done)
        data,consumed=_block_decode(blob[pos:limit],n)
        values[done:done+n]=data
        done+=n
        pos+=consumed
    if pos!=limit:raise ValueError('unexpected trailing group data')
    return values,st,end

def decode_patched(blob):
    meta=_header(blob)
    rows,width,group,block_words,nwords,lengths,anchor,start=meta
    words=np.empty(nwords,dtype='<u2')
    off=np.empty(rows+1,dtype=np.uint32);off[0]=0
    off[1:]=np.cumsum(lengths,dtype=np.uint32)
    for g in range(len(anchor)-1):
        w,st,end=_decode_group(blob,meta,g)
        words[off[st]:off[end]]=w
    return words,off,width

def read_row(blob,row):
    meta=_header(blob)
    rows,width,group,block_words,nwords,lengths,anchor,start=meta
    if row<0 or row>=rows:raise IndexError(row)
    ids,st,end=_decode_group(blob,meta,row//group)
    left=int(lengths[st:row].sum())
    return ids[left:left+int(lengths[row])].copy()

def codeword_matvec(blob,dictionary,x,scale_pairs):
    """Direct QMoE-style codeword matvec: no full matrix materialization.

    This is correct but CPU Python-only; runtime is NOT an ARM/GPU claim.
    """
    rows,width,group,block_words,nwords,lengths,anchor,start=_header(blob)
    x=np.asarray(x,dtype=np.float64).ravel()
    s=np.asarray(scale_pairs,dtype=np.float64)
    if len(x)!=width or s.shape!=(rows,2):
        raise ValueError('shape mismatch')
    out=np.zeros(rows,dtype=np.float64)
    for r in range(rows):
        tokens=read_row(blob,r)
        total=0.;j=0
        neg,pos=s[r]
        for word in tokens:
            for p in dictionary.sequences[int(word)]:
                if j>=width:raise ValueError('too many tokens')
                left,right=p%3,p//3
                # 0 absent, 1 negative, 2 positive
                if left:total+=(neg if left==1 else pos)*x[j]
                if right:total+=(neg if right==1 else pos)*x[j+1]
                j+=2
        if j!=width:raise ValueError('too few tokens')
        out[r]=total
    return out
