"""Native vector-codebook linear without reconstructing O x I weights."""
import torch
from torch.nn import functional as F

def compact_linear(x,centers,scales,codes,shape,rows=64):
    o,i=shape
    assert i%8==0 and x.shape[-1]==i
    assert centers.ndim==2 and centers.shape[1]==8
    g=i//8;k=centers.shape[0]
    assert codes.numel()==o*g and scales.shape==(o,)
    flat=x.reshape(-1,g,8)
    # One dot product for each group and codebook vector; no O x I matrix.
    table=(flat@centers.T).reshape(-1,g*k)
    offsets=torch.arange(g,device=x.device)*k
    idx=codes.reshape(o,g)
    result=[]
    for start in range(0,o,rows):
        pick=(idx[start:start+rows]+offsets).reshape(-1)
        chunk=table.index_select(1,pick).reshape(flat.shape[0],-1,g).sum(-1)
        result.append(chunk*scales[start:start+rows])
    return torch.cat(result,dim=-1).reshape(*x.shape[:-1],o)

def compact_source(x,source):
    c,s=source.fields()
    return compact_linear(x,c,s,source.codes,source.shape)

def dense_reference(x,centers,scales,codes,shape):
    return F.linear(x,centers[codes].reshape(shape)*scales[:,None])
