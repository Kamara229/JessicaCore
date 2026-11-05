"""Source-only normalized, sign-flipped Walsh-Hadamard transform.

For weight W (out,in) and orthogonal E=D H, represent W' = W E.
Then W x = W' (E.T x). Row-wise activation transformation x' = x E.
Embedding inverse is W' E.T; all transforms are implicit at runtime.
"""
import numpy as np

def signs(n):
    index=np.arange(1,n+1,dtype=np.uint64)
    v=(index*np.uint64(2654435761))&np.uint64(0xffffffff)
    v^=v>>np.uint64(16)
    return np.where((v&np.uint64(1))!=0,-1.,1.).astype(np.float32)

def fwht(x,block):
    a=np.array(x,dtype=np.float32,copy=True,order="C")
    if block not in (32,64) or a.ndim!=2 or a.shape[1]%block:
        raise ValueError("Wrong v180 Hadamard geometry")
    v=a.reshape(-1,block)
    step=1
    while step<block:
        parts=v.reshape(-1,block//(2*step),2,step)
        u=parts[:,:,0,:].copy()
        w=parts[:,:,1,:].copy()
        parts[:,:,0,:]=u+w
        parts[:,:,1,:]=u-w
        step*=2
    v*=np.float32(1/np.sqrt(block))
    return a

def apply(x,block,inverse=False):
    a=np.asarray(x,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]%block or not np.isfinite(a).all():
        raise ValueError("Nonfinite/unrepresentable orthogonal input")
    mask=signs(a.shape[1])[None,:]
    return fwht(a,block)*mask if inverse else fwht(a*mask,block)
