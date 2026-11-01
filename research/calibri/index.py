"""CLB8-I: grouped seekable row index with 1-byte lengths, u32 anchor/16 rows.

Falls back to CL7R if any compressed row is 255+ bytes. All metadata counted.
"""
import struct
import numpy as np
from codec import row_encode,row_decode,encode as old_encode,decode as old_decode

MAGIC=b'CL8I'
FMT='<4sIIH'
HDR=struct.calcsize(FMT)

def encode(q,group=16):
    q=np.asarray(q,dtype=np.int8)
    if q.ndim!=2 or min(q.shape)<1 or np.any(np.abs(q)>1):
        raise ValueError('ternary matrix expected')
    if not 1<=group<=256:raise ValueError('bad group size')
    pieces=[row_encode(row) for row in q]
    if any(len(p)>254 for p in pieces):return old_encode(q)
    anchors=[0];data=bytearray();rows=len(pieces)
    for st in range(0,rows,group):
        for p in pieces[st:st+group]:data.extend(p)
        anchors.append(len(data))
    if len(data)>=2**32:raise ValueError('oversized matrix')
    lengths=bytes(len(p) for p in pieces)
    return (struct.pack(FMT,MAGIC,*q.shape,group)
            +struct.pack(f'<{len(anchors)}I',*anchors)+lengths+data)

def decode(blob):
    if blob[:4]!=MAGIC:return old_decode(blob)
    if len(blob)<HDR:raise ValueError('truncated header')
    magic,rows,cols,group=struct.unpack_from(FMT,blob)
    if not rows or not cols or not 1<=group<=256:raise ValueError('bad header')
    ng=(rows+group-1)//group
    minlen=HDR+4*(ng+1)+rows
    if len(blob)<minlen:raise ValueError('truncated payload')
    anchors=struct.unpack_from(f'<{ng+1}I',blob,HDR)
    lengths=blob[HDR+4*(ng+1):minlen]
    data=memoryview(blob)[minlen:]
    if anchors[0]!=0 or anchors[-1]!=len(data):raise ValueError('invalid anchors')
    out=np.empty((rows,cols),dtype=np.int8)
    for g in range(ng):
        st=g*group;end=min(rows,st+group);offset=anchors[g]
        for i in range(st,end):
            n=lengths[i]
            if n==0 or offset+n>len(data):raise ValueError('invalid row length')
            out[i,:]=row_decode(data[offset:offset+n],cols)
            offset+=n
        if offset!=anchors[g+1]:raise ValueError('group length mismatch')
    return out
