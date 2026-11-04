"""Runtime attenuation of the ALREADY PAID CB140 residual.

Retains the one original shared packed CB59 code handle and the same K32x2x8
CB140 physical bitplane. Gains are NOT serialized in this experiment. Any
future compact deployed model must physically account for both gain values.
This is a source-valid diagnostic, not another 0.7-bpp physical release.
"""
import torch
from torch import nn
from direct_runtime_v59 import PackedEmbedding,PackedLinear
from v141.kernel import LUTNative,compile_lut


def check_gain(value):
    gain=float(value)
    if gain not in (0.0,0.25,0.5,0.75,1.0):
        raise ValueError("gain outside locked scalar grid")
    return gain


class Input(nn.Module):
    def __init__(self,original,kernel):
        super().__init__()
        if not isinstance(original,PackedEmbedding):
            raise TypeError("requires packed original embedding")
        self.original=original
        self.handle=original.handle
        self.kernel=kernel
        self.embedding_dim=original.embedding_dim
        self.num_embeddings=original.num_embeddings
        self.gain=0.0

    def forward(self,ids):
        basic=self.original(ids)
        if self.gain==0.0:return basic
        extra=torch.from_numpy(self.kernel.embed(ids.detach().cpu().numpy())).reshape(basic.shape)
        return basic+self.gain*extra


class Head(nn.Module):
    def __init__(self,original,kernel):
        super().__init__()
        if not isinstance(original,PackedLinear):
            raise TypeError("requires packed original head")
        self.original=original
        self.handle=original.handle
        self.kernel=kernel
        self.in_features=original.in_features
        self.out_features=original.out_features
        self.gain=0.0

    def forward(self,x):
        basic=self.original(x)
        if self.gain==0.0:return basic
        extra=torch.from_numpy(self.kernel.linear(x.detach().cpu().numpy())).reshape(basic.shape)
        return basic+self.gain*extra


class Paired:
    def __init__(self,model,physical,folder):
        original_input=model.model.embed_tokens
        original_head=model.lm_head
        if original_input.handle is not original_head.handle:
            raise ValueError("physical tied identity unexpectedly absent")
        kernel=LUTNative(compile_lut(folder/"libcalibri144_paid_lut.so"),
                         original_input.handle,physical)
        self.input=Input(original_input,kernel)
        self.head=Head(original_head,kernel)
        self.model=model
        self.initial_input=original_input
        self.initial_head=original_head
        model.model.embed_tokens=self.input
        model.lm_head=self.head
        if self.input.handle is not self.head.handle:
            raise AssertionError("lost physical tie")

    def configure(self,input_gain,head_gain):
        self.input.gain=check_gain(input_gain)
        self.head.gain=check_gain(head_gain)

    def restore(self):
        self.model.model.embed_tokens=self.initial_input
        self.model.lm_head=self.initial_head
