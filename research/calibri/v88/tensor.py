"""Genuine 4-axis tensor-network neural output operator: 192x256x24x24.

W[o1,o2,i1,i2] = sum_{r=1..R} A[o1,i1,r] B[r,o2,i2].
Unlike ordinary matrix rank-R, this is a separation across different
paired geometric output/input axes, with direct tensor contraction.
"""
import torch
from torch import nn

OUT1,OUT2,IN1,IN2,RANK=192,256,24,24,4
VOCAB=OUT1*OUT2
HIDDEN=IN1*IN2
FACTOR_ELEMENTS=OUT1*IN1*RANK+RANK*OUT2*IN2
TRAINABLE_ELEMENTS=FACTOR_ELEMENTS+VOCAB

def delta(x,A,B):
    if (A.ndim!=3 or B.ndim!=3 or x.shape[-1]!=A.shape[1]*B.shape[2]
        or A.shape[2]!=B.shape[0]):
        raise AssertionError("invalid multidimensional contraction")
    original=x.shape[:-1]
    unit=x.reshape(-1,A.shape[1],B.shape[2])
    first=torch.einsum("nij,rbj->nrib",unit,B)
    out=torch.einsum("air,nrib->nab",A,first)
    return out.reshape(*original,A.shape[0]*B.shape[1])

def normalized(x):
    return x/torch.sqrt(x.square().mean(dim=-1,keepdim=True)+1e-6)

class TensorPackedHead(nn.Module):
    """Keeps original native-C packed source; adds paid FP16 tensor operator."""
    def __init__(self,base,A,B,bias):
        super().__init__()
        if (tuple(A.shape)!=(OUT1,IN1,RANK) or
            tuple(B.shape)!=(RANK,OUT2,IN2) or
            tuple(bias.shape)!=(VOCAB,)):
            raise AssertionError("physical FP16 tensors do not match declared format")
        self.base=base
        self.register_buffer("A",torch.as_tensor(A.astype("f4").copy()))
        self.register_buffer("B",torch.as_tensor(B.astype("f4").copy()))
        self.register_buffer("bias",torch.as_tensor(bias.astype("f4").copy()))
    def forward(self,x):
        if x.dtype!=torch.float32 or x.device.type!="cpu":
            raise AssertionError("v88 tensor head tested on original CPU FP32 activations")
        return self.base(x)+delta(normalized(x),self.A,self.B)+self.bias
