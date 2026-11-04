"""Native CB59 packed-resident LLM + physically paid structural residuals.

v116 *idea* is composed onto v59; we do NOT replace source model with toy LM.
Only FP16 rank-2 factors are resident, never full trained dense 2D weights.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import roles

class PaidResidual(nn.Module):
    def __init__(self,packed,u,v):
        super().__init__()
        if tuple(u.shape)!=(packed.out_features,2) or tuple(v.shape)!=(2,packed.in_features):
            raise AssertionError("native CB59 role mismatch with paid factor geometry")
        self.packed=packed
        self.register_buffer("u",torch.from_numpy(np.array(u,dtype=np.float32,copy=True)))
        self.register_buffer("v",torch.from_numpy(np.array(v,dtype=np.float32,copy=True)))
        self.zero=not (np.any(u) and np.any(v))

    def forward(self,x):
        raw=self.packed(x)
        if self.zero:return raw
        return raw+F.linear(F.linear(x,self.v),self.u)

def attach(model,paid):
    """Replace exactly 60 PackedLinear modules, all originals remain reachable."""
    from direct_runtime_v59 import PackedLinear
    changed={}
    try:
        for name,rows,cols in roles():
            module=model.get_submodule(name)
            if not isinstance(module,PackedLinear):
                raise AssertionError("full original packed operator absent: "+name)
            if module.out_features!=rows or module.in_features!=cols:
                raise AssertionError("original source geometry incompatible: "+name)
            u,v=paid.factors(name)
            parent,_,leaf=name.rpartition(".")
            holder=model.get_submodule(parent)
            changed[name]=module
            setattr(holder,leaf,PaidResidual(module,u,v))
        if len(changed)!=60:
            raise AssertionError("missing original packed operators")
        if any(p.ndim>=2 for p in model.parameters()):
            raise AssertionError("hidden full pretrained dense weights reappeared")
        return changed
    except Exception:
        restore(model,changed)
        raise

def restore(model,changed):
    for name,module in changed.items():
        parent,_,leaf=name.rpartition(".")
        setattr(model.get_submodule(parent),leaf,module)
