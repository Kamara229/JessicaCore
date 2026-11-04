"""Bounded causal multipliers on ALREADY-PAID v119 rank1 U/V corrective roles.

No new trained coefficients. This pilot is an intervention study:
can dynamic feedback change the host answer at all? Not a proof of
new semantics or trained universal controller.
"""
import torch
from torch import nn
from torch.nn import functional as F
from v117.protocol import roles
from v118.runtime import Block

class Gate(nn.Module):
    def __init__(self,block,scale):
        super().__init__()
        if not isinstance(block,Block):
            raise AssertionError("expected paid native rank1 residual")
        if not 0<=scale<=2:raise ValueError("unbounded raw corrective gate")
        self.block=block
        self.scale=scale
    def forward(self,x):
        base=self.block.original(x)
        if self.scale==0 or self.block.zero:return base
        return base+self.scale*F.linear(F.linear(x,self.block.v),self.block.u)

def attach_gates(model,paid,attn,mlp):
    from v118.runtime import attach
    originals=attach(model,paid)
    try:
        for name,_,_ in roles():
            module=model.get_submodule(name)
            if not isinstance(module,Block):
                raise AssertionError("paid parent was overwritten")
            alpha=attn if "self_attn" in name else mlp
            owner,_,leaf=name.rpartition(".")
            setattr(model.get_submodule(owner),leaf,Gate(module,alpha))
        return originals
    except Exception:
        from v118.runtime import restore
        restore(model,originals)
        raise
