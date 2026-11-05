"""Facade: build native model from one physically complete CB178 file."""
import ctypes
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_runtime_v59 import compile_kernel, bind_into_llama
from .codec import CB178

PTR=ctypes.c_void_p
I64=ctypes.c_int64


class ColumnNative:
    def __init__(self,library):
        self.lib=ctypes.CDLL(str(library))
        fn=self.lib.cb178_linear
        fn.argtypes=[PTR,ctypes.c_size_t,PTR,PTR,PTR,PTR,I64,I64,I64,PTR]
        fn.restype=ctypes.c_int
        fn=self.lib.cb178_embedding
        fn.argtypes=[PTR,ctypes.c_size_t,PTR,PTR,PTR,PTR,I64,I64,I64,PTR]
        fn.restype=ctypes.c_int

    @staticmethod
    def ptr(a):
        return PTR(int(a.ctypes.data))

    def linear(self,handle,x):
        if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=handle.shape[1]:
            raise ValueError("Column-scaled C backend takes contiguous FP32 CPU input")
        x=np.ascontiguousarray(x)
        out=np.empty((x.shape[0],handle.shape[0]),dtype=np.float32)
        rc=self.lib.cb178_linear(self.ptr(handle.packed),len(handle.packed),
                self.ptr(handle.row_scales),self.ptr(handle.centroids),
                self.ptr(handle.col_scales),self.ptr(x),x.shape[0],
                handle.shape[0],handle.shape[1],self.ptr(out))
        if rc or not np.isfinite(out).all():
            raise ValueError("Native column-quantized matvec failed: "+str(rc))
        return out

    def embedding(self,handle,ids):
        ids=np.ascontiguousarray(ids,dtype=np.int64).reshape(-1)
        out=np.empty((len(ids),handle.shape[1]),dtype=np.float32)
        rc=self.lib.cb178_embedding(self.ptr(handle.packed),len(handle.packed),
                self.ptr(handle.row_scales),self.ptr(handle.centroids),
                self.ptr(handle.col_scales),self.ptr(ids),len(ids),
                handle.shape[0],handle.shape[1],self.ptr(out))
        if rc or not np.isfinite(out).all():
            raise ValueError("Native column-quantized embedding failed: "+str(rc))
        return out


def build(blob,folder):
    archive=CB178(blob)
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    kernel=ColumnNative(compile_kernel(Path(__file__).with_name("kernel.c"),
                                       folder/"cb178_native.so"))
    evidence=bind_into_llama(model,archive,kernel)
    if any(p.ndim>=2 for p in model.parameters()):
        raise AssertionError("Dense trained source leaked into native inference")
    if model.model.embed_tokens.handle is not model.lm_head.handle:
        raise AssertionError("Tied original embedding/output source broke")
    return model,archive,evidence
