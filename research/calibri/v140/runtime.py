"""CPU runtime for physically prepaid 1-bit CB140 packed embedding and head.

C reads the original 5-bit group indices and the extra 1-bit plane directly,
without reconstructing an original dense 49152x576 matrix. Real trained 2D
source always remains a packed base; residual table is just 32x2x8 FP32
RUN-TIME expansion from physically stored FP16 source.
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

from direct_runtime_v59 import PackedEmbedding,PackedLinear

PTR=ctypes.c_void_p
I64=ctypes.c_int64
SIZE=ctypes.c_size_t


class OneBitNative:
    def __init__(self, library, input_handle, physical):
        self.lib=ctypes.CDLL(str(library))
        self.handle=input_handle
        self.bits=np.ascontiguousarray(physical.bits,dtype=np.uint8)
        self.centers=np.ascontiguousarray(physical.centers,dtype=np.float32)
        self.base=np.ascontiguousarray(input_handle.packed,dtype=np.uint8)
        self.scales=np.ascontiguousarray(input_handle.row_scales,dtype=np.float32)
        self.rows,self.dim=input_handle.shape
        self.embed_fn=self.lib.calibri140_embedding_delta
        self.linear_fn=self.lib.calibri140_output_delta
        args=[PTR,SIZE,PTR,SIZE,PTR,PTR,PTR,I64,I64,I64,PTR]
        self.embed_fn.argtypes=args
        self.linear_fn.argtypes=args
        self.embed_fn.restype=ctypes.c_int
        self.linear_fn.restype=ctypes.c_int
        if self.rows!=49_152 or self.dim!=576:
            raise ValueError("unexpected tied embedding dimensions")
        if len(self.bits)!=(self.rows*self.dim//8+7)//8:
            raise ValueError("bitplane missing from paid source")

    @staticmethod
    def p(array):
        return PTR(int(array.ctypes.data))

    def prefix(self):
        return (self.p(self.base),len(self.base),
                self.p(self.bits),len(self.bits),self.p(self.centers),
                self.p(self.scales))

    def embed(self,index):
        ids=np.ascontiguousarray(index,dtype=np.int64).reshape(-1)
        out=np.empty((len(ids),self.dim),dtype=np.float32)
        error=self.embed_fn(*self.prefix(),self.p(ids),len(ids),
                            self.rows,self.dim,self.p(out))
        if error:raise RuntimeError(f"CB140 native embedding residual failed: {error}")
        return out

    def linear(self,x):
        data=np.ascontiguousarray(x,dtype=np.float32).reshape(-1,self.dim)
        out=np.empty((len(data),self.rows),dtype=np.float32)
        error=self.linear_fn(*self.prefix(),self.p(data),len(data),
                             self.rows,self.dim,self.p(out))
        if error:raise RuntimeError(f"CB140 native output residual failed: {error}")
        return out


def compile_onebit(out):
    command=["gcc","-std=c11","-O3","-fPIC","-shared","-o",str(out),
             str(Path(__file__).with_name("onebit_kernel.c"))]
    res=subprocess.run(command,text=True,capture_output=True,timeout=120)
    if res.returncode:
        raise RuntimeError("failed to compile native residual kernel: "+res.stderr)
    return out


class Embedding(nn.Module):
    def __init__(self,base,kernel):
        super().__init__()
        if not isinstance(base,PackedEmbedding):raise TypeError("not packed embedding")
        self.original=base
        self.kernel=kernel
        self.handle=base.handle
        self.num_embeddings=base.num_embeddings
        self.embedding_dim=base.embedding_dim

    def forward(self,ids):
        if ids.device.type!="cpu":
            raise ValueError("CB140 CPU kernel only")
        base=self.original(ids)
        additional=torch.from_numpy(self.kernel.embed(ids.detach().cpu().numpy())).reshape(base.shape)
        return base+additional


class OutputHead(nn.Module):
    def __init__(self,base,kernel):
        super().__init__()
        if not isinstance(base,PackedLinear):raise TypeError("not packed output")
        self.original=base
        self.kernel=kernel
        self.handle=base.handle
        self.out_features=base.out_features
        self.in_features=base.in_features

    def forward(self,x):
        if x.device.type!="cpu" or x.dtype!=torch.float32:
            raise ValueError("CB140 CPU output only")
        base=self.original(x)
        additional=torch.from_numpy(self.kernel.linear(x.detach().cpu().numpy())).reshape(base.shape)
        return base+additional


def install(model,physical,folder):
    old_input=model.model.embed_tokens
    old_output=model.lm_head
    if not isinstance(old_input,PackedEmbedding) or not isinstance(old_output,PackedLinear):
        raise ValueError("base model not direct-native packed CB59")
    if old_input.handle is not old_output.handle:
        raise ValueError("original tied weights must share exact packed code handle")
    lib=compile_onebit(Path(folder)/"libcalibri_onebit_v140.so")
    runner=OneBitNative(lib,old_input.handle,physical)
    model.model.embed_tokens=Embedding(old_input,runner)
    model.lm_head=OutputHead(old_output,runner)
    if model.model.embed_tokens.handle is not model.lm_head.handle:
        raise AssertionError("tied physical codebook identity lost")
    if any(p.ndim>=2 for p in model.parameters()):
        raise AssertionError("dense pretrained 2D source materialized")
    return runner
