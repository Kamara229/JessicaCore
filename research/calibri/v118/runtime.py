"""Real CB59-native model with paid rank1 blocks and rank2 output head."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v117.protocol import roles
from direct_runtime_v59 import PackedLinear

def tensor(x):
    return torch.from_numpy(np.asarray(x,dtype=np.float32).copy())
class Block(nn.Module):
    def __init__(self,original,u,v):
        super().__init__()
        self.original=original
        self.register_buffer("u",tensor(u))
        self.register_buffer("v",tensor(v))
        self.zero=not np.any(u) or not np.any(v)
    def forward(self,x):
        base=self.original(x)
        return base if self.zero else base+F.linear(F.linear(x,self.v),self.u)
class Head(nn.Module):
    def __init__(self,original,u,v,b):
        super().__init__()
        self.original=original
        self.register_buffer("u",tensor(u))
        self.register_buffer("v",tensor(v))
        self.register_buffer("bias",tensor(b))
        self.zero=not np.any(u) and not np.any(b)
    def forward(self,x):
        base=self.original(x)
        if self.zero:return base
        return base+F.linear(F.linear(x,self.v),self.u)+self.bias

def attach(model,paid):
    old={}
    try:
        for name,r,c in roles():
            original=model.get_submodule(name)
            if (not isinstance(original,PackedLinear)
                or original.out_features!=r or original.in_features!=c):
                raise AssertionError("missing original native packed "+name)
            u,v=paid.role(name)
            p,_,leaf=name.rpartition(".")
            old[name]=original
            setattr(model.get_submodule(p),leaf,Block(original,u,v))
        old["lm_head"]=model.lm_head
        if not isinstance(model.lm_head,PackedLinear):raise AssertionError("no native head")
        model.lm_head=Head(model.lm_head,*paid.head())
        if any(p.ndim>=2 for p in model.parameters()):
            raise AssertionError("dense trained source reappeared")
        return old
    except Exception:
        restore(model,old)
        raise
def restore(model,old):
    for name,original in old.items():
        p,sep,leaf=name.rpartition(".")
        owner=model.get_submodule(p) if sep else model
        setattr(owner,leaf,original)
