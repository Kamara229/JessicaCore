"""Input-conditioned small adaptive operator: two original native C kernels.

Output = (1-g(x))*OriginalCB59(x) + g(x)*RewrittenCB59(x).
The dynamic gate belongs to the operator and reads only present activations,
not dataset labels. Exact FP16 physical source parameters are loaded via CB85.
"""
from __future__ import annotations
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from direct_runtime_v59 import PackedLinear


class AdaptivePackedLinear(nn.Module):
    def __init__(self, baseline: PackedLinear, extra_centroids: np.ndarray,
                 gate_w: np.ndarray, gate_b: float):
        super().__init__()
        if not isinstance(baseline,PackedLinear) or baseline.bias is not None:
            raise AssertionError("only original fully packed bias-free native projections")
        if extra_centroids.shape!=baseline.handle.centroids.shape:
            raise AssertionError("alternative compressed operator changes tensor geometry")
        self.base=baseline
        modified=replace(baseline.handle,centroids=np.array(
            extra_centroids,dtype=np.float32,copy=True))
        self.experimental=PackedLinear(modified,baseline.kernel)
        if len(gate_w)!=baseline.in_features:
            raise AssertionError("gating input width incompatible with packed source")
        self.register_buffer("gate_w",torch.as_tensor(
            np.asarray(gate_w,dtype=np.float32).copy()))
        self.register_buffer("gate_b",torch.tensor(float(gate_b),dtype=torch.float32))
        self.in_features=baseline.in_features
        self.out_features=baseline.out_features

    def forward(self,x):
        if x.dtype!=torch.float32 or x.device.type!="cpu":
            raise AssertionError("CB85 native CPU prototype requires FP32 activations")
        a=self.base(x)
        b=self.experimental(x)
        normalized=x/torch.sqrt(torch.mean(x.square(),dim=-1,keepdim=True)+1e-6)
        gate=torch.sigmoid((normalized*self.gate_w).sum(dim=-1,keepdim=True)
                           +self.gate_b)
        return a+(b-a)*gate


def insert_native_adapters(student,cb85,paths: tuple[str,...]):
    originals={}
    for path in paths:
        parentname,_,attr=path.rpartition(".")
        parent=student.get_submodule(parentname)
        module=getattr(parent,attr)
        if not isinstance(module,PackedLinear):
            raise AssertionError("original packed module missing for adaptive rewrite")
        if path not in cb85.extra:
            raise AssertionError("physical CB85 source lacks selected operator")
        c,w,b=cb85.extra[path]
        originals[path]=(parent,attr,module)
        setattr(parent,attr,AdaptivePackedLinear(module,c,w,b))
    return originals


def restore_native_adapters(originals):
    for parent,attr,original in originals.values():
        setattr(parent,attr,original)
