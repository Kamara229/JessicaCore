"""Inference-only generation with frozen source head component ablations."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from v88.tensor import delta,normalized
from v90.generation import generate_diagnostic

class FrozenAblationHead(nn.Module):
    def __init__(self,base,A,B,bias,mode:str):
        super().__init__()
        if mode not in ("bias_only","tensor_only","CB88"):raise ValueError(mode)
        self.base=base
        self.mode=mode
        self.register_buffer("A",torch.from_numpy(np.asarray(A,dtype=np.float32).copy()))
        self.register_buffer("B",torch.from_numpy(np.asarray(B,dtype=np.float32).copy()))
        self.register_buffer("bias",torch.from_numpy(np.asarray(bias,dtype=np.float32).copy()))
    def forward(self,x):
        base=self.base(x)
        if self.mode in ("tensor_only","CB88"):
            base=base+delta(normalized(x),self.A,self.B)
        if self.mode in ("bias_only","CB88"):base=base+self.bias
        return base

def greedy_ablations(student,tok,A,B,bias):
    original=student.lm_head
    results={}
    try:
        for mode in ("CB59","bias_only","tensor_only","CB88"):
            student.lm_head=original if mode=="CB59" else FrozenAblationHead(original,A,B,bias,mode)
            results[mode]=generate_diagnostic(student,tok)
    finally:student.lm_head=original
    return results
