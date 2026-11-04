"""Read-only small additive second rank to existing CB149 packed native head.

Two 1D vector products with paid low-bit output vectors; no dense original
pretrained weight tensors, no teacher run on CPU inference source.
"""
import torch
from torch import nn
from v149.runtime import ContextualHead
from .protocol import TRAIN_HIDDEN,VOCAB


class SuccessiveHead(nn.Module):
    def __init__(self,original_paid_head,physical):
        super().__init__()
        if not isinstance(original_paid_head,ContextualHead):
            raise ValueError("rank2 must extend actual physically paid CB149 head")
        if physical.projection.shape!=(TRAIN_HIDDEN,) or physical.signed_int3.shape!=(VOCAB,):
            raise ValueError("second rank lacks fully paid coefficients")
        self.original=original_paid_head
        self.handle=original_paid_head.handle
        self.out_features=VOCAB
        self.in_features=TRAIN_HIDDEN
        self.register_buffer("u",torch.from_numpy(physical.projection.copy()))
        self.register_buffer("bias",torch.tensor(physical.offset,dtype=torch.float32))
        self.register_buffer("output_v",torch.from_numpy(physical.output_vector.copy()))
        if any(p.ndim>=2 for p in self.parameters()):
            raise ValueError("accidentally loaded original full matrix")

    def forward(self,hidden):
        if hidden.device.type!="cpu" or hidden.dtype!=torch.float32 or hidden.shape[-1]!=TRAIN_HIDDEN:
            raise ValueError("native CPU float32 expected for successive projection")
        original=self.original(hidden)
        v=(hidden*self.u).sum(-1)+self.bias
        return original+v.unsqueeze(-1)*self.output_v


def install(model,physical):
    parent=model.lm_head
    corrected=SuccessiveHead(parent,physical)
    if model.model.embed_tokens.handle is not corrected.handle:
        raise ValueError("tied shared original prepaid codebook identity lost")
    model.lm_head=corrected
    return corrected
