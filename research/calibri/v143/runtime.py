"""Physically paid fused packed CPU lm_head, NO source tensor rehydration."""
import ctypes
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from direct_runtime_v59 import PackedLinear
from v141.kernel import install as install_lut

PTR=ctypes.c_void_p
SIZE=ctypes.c_size_t
I64=ctypes.c_int64


def build(path):
    path=Path(path)
    src=Path(__file__).with_name("fused_output.c")
    p=subprocess.run(["gcc","-O3","-std=c11","-fPIC","-shared",
                      "-o",str(path),str(src)],
                     capture_output=True,text=True,timeout=120)
    if p.returncode:
        raise RuntimeError("fused C compile error: "+p.stderr)
    return path


class FusedKernel:
    """The original 32x8 center table and 32x2x8 residual centers stay tiny."""
    def __init__(self,so,base_handle,paid):
        self.lib=ctypes.CDLL(str(so))
        self.forward_fn=self.lib.calibri143_fused_output
        self.forward_fn.argtypes=[
            PTR,SIZE,PTR,SIZE,PTR,PTR,PTR,PTR,I64,I64,I64,PTR]
        self.forward_fn.restype=ctypes.c_int
        self.base=np.ascontiguousarray(base_handle.packed,dtype=np.uint8)
        self.bits=np.ascontiguousarray(paid.bits,dtype=np.uint8)
        self.centers=np.ascontiguousarray(base_handle.centroids,dtype=np.float32)
        self.delta=np.ascontiguousarray(paid.centers,dtype=np.float32)
        self.scales=np.ascontiguousarray(base_handle.row_scales,dtype=np.float32)
        self.rows,self.dim=base_handle.shape
        if self.rows!=49152 or self.dim!=576 or len(self.bits)!=442368:
            raise ValueError("unpinned physical 135M tied head")
        if self.centers.shape!=(32,8) or self.delta.shape!=(32,2,8):
            raise ValueError("not the 5+1bit dictionary")

    @staticmethod
    def p(x):
        return PTR(int(x.ctypes.data))

    def forward(self,tensor):
        x=np.ascontiguousarray(tensor,dtype=np.float32).reshape(-1,self.dim)
        out=np.empty((len(x),self.rows),dtype=np.float32)
        status=self.forward_fn(
            self.p(self.base),len(self.base),
            self.p(self.bits),len(self.bits),
            self.p(self.centers),self.p(self.delta),
            self.p(self.scales),self.p(x),
            len(x),self.rows,self.dim,self.p(out))
        if status:
            raise RuntimeError("failed fused packed C head: "+str(status))
        if not np.isfinite(out).all():
            raise ValueError("nonfinite fused logits")
        return out


class PackedFusedHead(nn.Module):
    def __init__(self,handle,kernel,bias=None):
        super().__init__()
        self.handle=handle
        self.kernel=kernel
        self.in_features=handle.shape[1]
        self.out_features=handle.shape[0]
        if bias is not None:
            self.register_buffer("bias",torch.from_numpy(
                np.array(bias,dtype=np.float32,copy=True)))
        else:
            self.bias=None

    def forward(self,x):
        if x.device.type!="cpu" or x.dtype!=torch.float32:
            raise ValueError("fused packed head is CPU FP32 only")
        logits=self.kernel.forward(x.detach().numpy())
        output=torch.from_numpy(logits).reshape((*x.shape[:-1],self.out_features))
        if self.bias is not None:output=output+self.bias
        return output


def install(model,paid,folder):
    """Reuse original v141 Embedding, replace only the output head operator."""
    base=model.lm_head
    if not isinstance(base,PackedLinear):
        raise ValueError("need direct-native tied CB59 source")
    input_base=model.model.embed_tokens
    if input_base.handle is not base.handle:
        raise ValueError("input and output must refer to one source codebook")
    embedding_runner=install_lut(model,paid,folder)
    fused=FusedKernel(build(Path(folder)/"calibri_fused_v143.so"),base.handle,paid)
    model.lm_head=PackedFusedHead(base.handle,fused,base.bias)
    if model.model.embed_tokens.handle is not model.lm_head.handle:
        raise ValueError("tied physical codebook identity lost")
    if any(p.ndim>=2 for p in model.parameters()):
        raise ValueError("unexpected dense pretrained source parameter")
    return embedding_runner,fused
