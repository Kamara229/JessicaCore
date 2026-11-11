"""CLB row64 research format: independently seekable rows, exact ternary symbols."""
import math
import struct
import numpy as np

MAGIC=b'CL7R'
HEADER='<4sII'

class Writer:
    def __init__(self): self.out=bytearray(); self.n=0
    def put(self,v,bits):
        if v<0 or (bits and v>=(1<<bits)) or (bits==0 and v): raise ValueError('bit-field overflow')
        for j in range(bits-1,-1,-1):
            if self.n%8==0: self.out.append(0)
            self.out[-1]|=((v>>j)&1)<<(7-self.n%8)
            self.n+=1
    def bytes(self):return bytes(self.out)

class Reader:
    def __init__(self,data):self.buf=data;self.n=0
    def get(self,bits):
        if self.n+bits>len(self.buf)*8:raise ValueError('truncated')
        ans=0
        for _ in range(bits):
            ans=(ans<<1)|((self.buf[self.n//8]>>(7-self.n%8))&1)
            self.n+=1
        return ans

def rank_positions(p):return sum(math.comb(v,i+1) for i,v in enumerate(p))

def unrank_positions(n,k,rank):
    if rank>=math.comb(n,k):raise ValueError('invalid rank')
    out=[0]*k;u=n-1
    for j in range(k,0,-1):
        while u>=j and math.comb(u,j)>rank:u-=1
        out[j-1]=u;rank-=math.comb(u,j);u-=1
    return out

def row_encode(row):
    wr=Writer();blocks=[row[i:i+64] for i in range(0,len(row),64)]
    ks=[int(np.count_nonzero(b)) for b in blocks]
    for k in ks:wr.put(min(k,15),4)
    for k in ks:
        if k>=15:wr.put(k,7)
    for block,k in zip(blocks,ks):
        p=np.flatnonzero(block).tolist()
        wr.put(rank_positions(p),(math.comb(len(block),k)-1).bit_length())
        signs=sum(int(block[t]>0)<<i for i,t in enumerate(p))
        wr.put(signs,k)
    return wr.bytes()

def row_decode(data,count):
    rd=Reader(data);ns=[min(64,count-i) for i in range(0,count,64)]
    ks=[rd.get(4) for _ in ns];ks=[rd.get(7) if k==15 else k for k in ks]
    ans=np.zeros(count,dtype=np.int8);start=0
    for n,k in zip(ns,ks):
        if k>n:raise ValueError('bad cardinality')
        rank=rd.get((math.comb(n,k)-1).bit_length())
        signs=rd.get(k)
        for i,p in enumerate(unrank_positions(n,k,rank)):
            ans[start+p]=1 if (signs>>i)&1 else -1
        start+=n
    return ans

def encode(q):
    q=np.asarray(q,dtype=np.int8)
    if q.ndim!=2 or min(q.shape)<1 or np.any(np.abs(q)>1):
        raise ValueError('expected nonempty ternary matrix')
    data=bytearray();offsets=[0]
    for row in q:
        data.extend(row_encode(row));offsets.append(len(data))
    if len(data)>=2**32:raise ValueError('matrix too large')
    return struct.pack(HEADER,MAGIC,*q.shape)+struct.pack(f'<{len(offsets)}I',*offsets)+data

def decode(blob):
    h=struct.calcsize(HEADER)
    if len(blob)<h:raise ValueError('short header')
    magic,rows,cols=struct.unpack_from(HEADER,blob)
    if magic!=MAGIC or rows<1 or cols<1:raise ValueError('bad header')
    size=4*(rows+1)
    if len(blob)<h+size:raise ValueError('short offsets')
    offsets=struct.unpack_from(f'<{rows+1}I',blob,h)
    data=memoryview(blob)[h+size:]
    if offsets[0]!=0 or offsets[-1]!=len(data) or any(a>b for a,b in zip(offsets,offsets[1:])):raise ValueError('bad offsets')
    ans=np.empty((rows,cols),dtype=np.int8)
    for i in range(rows):ans[i,:]=row_decode(data[offsets[i]:offsets[i+1]],cols)
    return ans
