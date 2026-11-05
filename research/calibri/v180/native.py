"""v180 facade: Hadamard-weight source => no-dense native CPU model."""
import ctypes
from pathlib import Path
import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_runtime_v59 import compile_kernel, bind_into_llama
from .codec import read
from .protocol import BLOCKS

PTR=ctypes.c_void_p
I64=ctypes.c_int64

class RotatedNative:
    def __init__(self, library, block):
        if block not in (32,64):
            raise ValueError("Unregistered source block")
        self.block=block
        self.lib=ctypes.CDLL(str(library))
        for name in ("cb180_linear","cb180_embedding"):
            fn=getattr(self.lib,name)
            fn.argtypes=[PTR,ctypes.c_size_t,PTR,PTR,PTR,
                         I64,I64,I64,I64,PTR]
            fn.restype=ctypes.c_int
    @staticmethod
    def ptr(a):
        return PTR(int(a.ctypes.data))
    def linear(self,h,x):
        if x.ndim!=2 or x.dtype!=np.float32 or x.shape[1]!=h.shape[1]:
            raise ValueError("Hadamard native CPU GEMV requires FP32 2D input")
        x=np.ascontiguousarray(x)
        out=np.empty((x.shape[0],h.shape[0]),dtype=np.float32)
        rc=self.lib.cb180_linear(self.ptr(h.packed),len(h.packed),
              self.ptr(h.row_scales),self.ptr(h.centroids),self.ptr(x),
              x.shape[0],h.shape[0],h.shape[1],self.block,self.ptr(out))
        if rc or not np.isfinite(out).all():
            raise ValueError("Physical source Hadamard C GEMV failed "+str(rc))
        return out
    def embedding(self,h,ids):
        ids=np.ascontiguousarray(ids,dtype=np.int64).reshape(-1)
        out=np.empty((len(ids),h.shape[1]),dtype=np.float32)
        rc=self.lib.cb180_embedding(self.ptr(h.packed),len(h.packed),
              self.ptr(h.row_scales),self.ptr(h.centroids),self.ptr(ids),
              len(ids),h.shape[0],h.shape[1],self.block,self.ptr(out))
        if rc or not np.isfinite(out).all():
            raise ValueError("Physical source Hadamard C embedding failed "+str(rc))
        return out

def build(blob,arm,folder):
    archive=read(blob,arm)
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    kernel=RotatedNative(compile_kernel(Path(__file__).with_name("kernel.c"),
                         folder/"cb180_hadamard.so"),BLOCKS[arm])
    audit=bind_into_llama(model,archive,kernel)
    if any(p.ndim>=2 for p in model.parameters()):
        raise AssertionError("Dense weight persisted into compressed runtime")
    if model.model.embed_tokens.handle is not model.lm_head.handle:
        raise AssertionError("Tied embedding-head no longer physically shared")
    return model,archive,audit
