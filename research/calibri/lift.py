"""EXACT-LIFT: lossless paired-column transform on ternarized real weights."""
import math
import struct
import zlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from codec import encode,decode

FMT='<4sIIIIII'
HDR=struct.calcsize(FMT)

def permutation_map(q,matching=True):
    rows,cols=q.shape
    if cols%32:raise ValueError('input columns must be multiple of 32')
    out=[]
    for i in range(0,cols,32):
        if not matching:
            out.append(np.arange(16,dtype=np.uint8));continue
        a=q[:,i:i+16].astype(np.int16)
        b=q[:,i+16:i+32].astype(np.int16)
        diff=b[:,None,:]-a[:,:,None]
        cost=np.count_nonzero(diff,axis=0)+.16*np.count_nonzero(np.abs(diff)==2,axis=0)
        ii,jj=linear_sum_assignment(cost)
        if not np.array_equal(ii,np.arange(16)):raise AssertionError('matching order')
        out.append(jj.astype(np.uint8))
    return out

def fold(q,matching=True):
    rows,cols=q.shape;ps=permutation_map(q,matching)
    a=np.empty((rows,cols//2),np.int8);s=np.empty_like(a)
    d=np.empty_like(a,dtype=bool)
    for t,p in enumerate(ps):
        left=q[:,32*t:32*t+16];right=q[:,32*t+16:32*t+32][:,p]
        delta=right.astype(np.int16)-left.astype(np.int16)
        a[:,16*t:16*t+16]=left
        s[:,16*t:16*t+16]=np.sign(delta)
        d[:,16*t:16*t+16]=np.abs(delta)==2
    return a,s,d,ps

def restore(a,s,d,ps):
    rows,cols=a.shape
    out=np.empty((rows,cols*2),dtype=np.int8)
    b=a.astype(np.int16)+s.astype(np.int16)*(1+d.astype(np.int16))
    if np.any(np.abs(b)>1):raise ValueError('nonternary residual')
    for t,p in enumerate(ps):
        out[:,32*t:32*t+16]=a[:,16*t:16*t+16]
        out[:,32*t+16+p.astype(np.int32)]=b[:,16*t:16*t+16].astype(np.int8)
    return out

def rank_perm(p):
    pool=list(range(16));rank=0
    for i,x in enumerate(p):
        j=pool.index(int(x));rank+=j*math.factorial(15-i);pool.pop(j)
    return rank

def unrank_perm(rank):
    if not 0<=rank<math.factorial(16):raise ValueError('bad permutation rank')
    pool=list(range(16));out=[]
    for i in range(16):
        j,rank=divmod(rank,math.factorial(15-i))
        out.append(pool.pop(j))
    return np.array(out,dtype=np.uint8)

def encode_lift(q,matching=True):
    a,s,d,ps=fold(q,matching)
    ea=encode(a);es=encode(s)
    flags=np.packbits(d[s!=0].astype(np.uint8),bitorder='little').tobytes()
    perms=b''.join(rank_perm(p).to_bytes(6,'little') for p in ps)
    header=struct.pack(FMT,b'CL7L',*q.shape,len(ea),len(es),len(flags),int(np.count_nonzero(s)))
    blob=header+perms+ea+es+flags
    return blob+struct.pack('<I',zlib.crc32(blob)),(a,s,d,ps)

def decode_lift(blob):
    if len(blob)<HDR+4:raise ValueError('short lift')
    magic,rows,cols,alen,slen,flen,nnz=struct.unpack_from(FMT,blob)
    if magic!=b'CL7L' or rows<1 or cols%32:raise ValueError('bad lift header')
    if HDR+6*(cols//32)+alen+slen+flen+4!=len(blob):raise ValueError('bad length')
    if zlib.crc32(blob[:-4])!=struct.unpack_from('<I',blob,len(blob)-4)[0]:raise ValueError('bad crc32')
    pos=HDR;ps=[]
    for _ in range(cols//32):
        ps.append(unrank_perm(int.from_bytes(blob[pos:pos+6],'little')))
        pos+=6
    a=decode(blob[pos:pos+alen]);pos+=alen
    s=decode(blob[pos:pos+slen]);pos+=slen
    if a.shape!=(rows,cols//2) or s.shape!=a.shape or np.count_nonzero(s)!=nnz:raise ValueError('bad content')
    flag=np.unpackbits(np.frombuffer(blob[pos:pos+flen],np.uint8),bitorder='little')[:nnz]
    if len(flag)!=nnz:raise ValueError('bad flags')
    d=np.zeros_like(s,dtype=bool);d[s!=0]=flag.astype(bool)
    return restore(a,s,d,ps)

def measure(q,matching=True):
    blob,parts=encode_lift(q,matching)
    if not np.array_equal(q,decode_lift(blob)):raise AssertionError('lift roundtrip')
    a,s,d,ps=parts
    return len(blob),float(np.mean(s==0)),parts

def fused_matvec(a,s,d,ps,x,lo,hi):
    rows,half=a.shape;x=np.asarray(x,dtype=np.float64)
    out=np.zeros(rows,dtype=np.float64)
    for t,p in enumerate(ps):
        leftx=x[32*t:32*t+16];rightx=x[32*t+16:32*t+32][p]
        qa=a[:,16*t:16*t+16].astype(np.int16)
        qr=qa+s[:,16*t:16*t+16].astype(np.int16)*(1+d[:,16*t:16*t+16].astype(np.int16))
        va=np.where(qa<0,lo[:,None],np.where(qa>0,hi[:,None],0.))
        vr=np.where(qr<0,lo[:,None],np.where(qr>0,hi[:,None],0.))
        out+=(va*(leftx+rightx)+((vr-va)*rightx)).sum(axis=1)
    return out
