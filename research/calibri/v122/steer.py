"""Apply ONE shared 6-coefficient Green Policy to 60 already-paid native roles.

This is causal context-conditioned steering, not text post-processing.
For different host sizes only the host-specific paid U/V/role manifest
changes; no coefficient dimension or neural-controller weights change.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v118.runtime import Head
from v117.protocol import roles

def fresh_tensor(a):
    return torch.from_numpy(np.asarray(a,dtype=np.float32).copy())

class SteeringBlock(nn.Module):
    def __init__(self,original,u,v):
        super().__init__()
        self.original=original
        self.register_buffer("u",fresh_tensor(u))
        self.register_buffer("v",fresh_tensor(v))
        self.gain=None
        self.zero=not np.any(u) or not np.any(v)
    def forward(self,x):
        base=self.original(x)
        if self.zero:return base
        if self.gain is None:
            raise AssertionError("controller gate not set before forward")
        return base+self.gain*F.linear(F.linear(x,self.v),self.u)

class Installed:
    def __init__(self,model,paid):
        self.model=model
        self.paid=paid
        self.originals={}
        self.gates=[]
        self.maps=[]
        try:
            keys=roles()
            levels=sorted({int(n.split(".")[2]) for n,_,_ in keys})
            mapping={layer:i for i,layer in enumerate(levels)}
            for name,rows,cols in keys:
                current=model.get_submodule(name)
                underlying=getattr(current,"original",current)
                if hasattr(underlying,"handle"):
                    underlying=current
                # Wrapped native differentiable modules expose the packed
                # source as .original; ordinary PackedLinear exposes .handle.
                out=getattr(underlying,"out_features",None)
                inp=getattr(underlying,"in_features",None)
                if out is None and hasattr(underlying,"original"):
                    out=getattr(underlying.original,"out_features",None)
                    inp=getattr(underlying.original,"in_features",None)
                if (out,inp)!=(rows,cols):
                    raise AssertionError("physical role mismatch "+name)
                u,v=paid.role(name)
                module=SteeringBlock(current,u,v)
                parent,_,leaf=name.rpartition(".")
                self.originals[name]=current
                setattr(model.get_submodule(parent),leaf,module)
                group=0 if "self_attn" in name else 1
                self.gates.append(module)
                self.maps.append((module,group,(mapping[int(name.split(".")[2])]+.5)/len(levels)))
            self.originals["lm_head"]=model.lm_head
            model.lm_head=Head(model.lm_head,*paid.head())
        except Exception:
            self.restore()
            raise
    def set_question(self,policy,question):
        coeff=policy(question)
        if tuple(coeff.shape)!=(6,):
            raise AssertionError("generic controller output must remain width 6")
        values=[]
        for module,group,relative_depth in self.maps:
            gain=policy.gates(coeff,relative_depth,group)
            module.gain=gain
            values.append(gain)
        return torch.stack(values)
    def clear_question(self):
        for gate in self.gates:
            gate.gain=None
    def restore(self):
        self.clear_question()
        for name,original in self.originals.items():
            parent,sep,leaf=name.rpartition(".")
            owner=self.model.get_submodule(parent) if sep else self.model
            setattr(owner,leaf,original)
