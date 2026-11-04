"""Synthesize compact U/V operators into block-local scratch for CPU execution.

The original entire trained model is ONLY the source checkpoint containing
shared U/V plus layer gains/biases and rank2 local adapters. Hot dense GEMV
blocks are DERIVED, never serialized. At 135M scale, a block-bounded LRU
cache would be required; this tiny pilot keeps all three blocks to measure
an upper bound on achievable CPU speed, with explicit additional RAM cost.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F

class FusedBlock(nn.Module):
    def __init__(self,old,shared):
        super().__init__()
        for role in ("expand","contract"):
            # Generates W = U*V + lowrank adapter, not pretrained FP32 W.
            weights=(
                shared.u[role] @ shared.v[role]
            ) * old.gain[role][:,None]
            if shared.adapter:
                weights=weights+old.low_u[role]@old.low_v[role]
            self.register_buffer(role+"_weight",weights.detach().contiguous().clone())
            self.register_buffer(role+"_bias",old.bias[role].detach().contiguous().clone())
        self.norm=copy.deepcopy(old.norm)

    def forward(self,x):
        hidden=torch.tanh(F.linear(x,self.expand_weight,self.expand_bias))
        return self.norm(x+F.linear(
            hidden,self.contract_weight,self.contract_bias))

def fuse_compact(source):
    if source.mode not in ("shared","shared_adapter"):
        raise ValueError("only compact structural models can be fused")
    with torch.no_grad():
        result=copy.deepcopy(source)
        result.layers=nn.ModuleList([
            FusedBlock(block,source.book) for block in source.layers
        ])
        result.mode="dense"
        # Discard source lowrank book from the generated inference view.
        # The independent physical original checkpoint is NEVER changed.
        del result.book
        result.eval()
    return result

def hot_generated_weight_bytes(model):
    """Exact bytes of *temporary generated* 2D operator buffers."""
    if any(not isinstance(b,FusedBlock) for b in model.layers):
        raise AssertionError("wrong generated runtime")
    return sum(
        int(buf.numel()*buf.element_size())
        for layer in model.layers
        for name,buf in layer.named_buffers()
        if name.endswith("_weight")
    )

def hot_generated_all_buffers_bytes(model):
    return sum(
        int(buf.numel()*buf.element_size())
        for layer in model.layers
        for buf in layer.buffers()
    )
