"""Packed physically paid contextual head. NO pretrained 2D matrix in RAM.

The only new 1-D buffers are 576 FP16 projection coefficients and a full
49152-entry int4 vocab vector unpacked as runtime FP32. This is not a
Green sidecar or a hidden large output correction tensor.
"""
import torch
from torch import nn
from v143.runtime import PackedFusedHead
from .protocol import TRAIN_HIDDEN,VOCAB


class ContextualHead(nn.Module):
    def __init__(self,old_fused_head,paid):
        super().__init__()
        if not isinstance(old_fused_head,PackedFusedHead):
            raise ValueError("requires original native physically paid CB147")
        if (paid.projection.shape!=(TRAIN_HIDDEN,) or
                paid.signed_int4.shape!=(VOCAB,)):
            raise ValueError("not a compact rank-one paid contextual head")
        self.original=old_fused_head
        self.handle=old_fused_head.handle
        self.out_features=VOCAB
        self.in_features=TRAIN_HIDDEN
        self.register_buffer("proj",torch.tensor(
            paid.projection,dtype=torch.float32))
        self.register_buffer("offset",torch.tensor(
            paid.offset,dtype=torch.float32))
        self.register_buffer("vocab_vector",torch.tensor(
            paid.output_vector,dtype=torch.float32))
        if any(p.ndim>=2 for p in self.parameters()):
            raise ValueError("dense pretrained 2D source weights materialized")

    def forward(self,hidden):
        if (hidden.device.type!="cpu" or hidden.dtype!=torch.float32
                or hidden.shape[-1]!=TRAIN_HIDDEN):
            raise ValueError("contextual calibrated output expects CPU FP32 hidden")
        old=self.original(hidden)
        scalar=(hidden*self.proj).sum(dim=-1)+self.offset
        correction=scalar.unsqueeze(-1)*self.vocab_vector
        return old+correction


def attach(model,paid):
    original=model.lm_head
    head=ContextualHead(original,paid)
    if model.model.embed_tokens.handle is not head.handle:
        raise ValueError("original source tied alias lost")
    model.lm_head=head
    return head
