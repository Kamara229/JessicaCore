"""Source-paid CB59+rank2 direct C CPU runtime; never decode model 2D."""
import ctypes
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_runtime_v59 import NativeKernel,compile_kernel,bind_into_llama
from .codec import Paid184
from . import protocol as p

ADDRESS=ctypes.c_void_p
INT=ctypes.c_int64

class Joint184Kernel(NativeKernel):
    def __init__(self,path,paid):
        super().__init__(path)
        self.paid=paid
        lin=self.lib.cb184_linear
        lin.argtypes=[ADDRESS,ctypes.c_size_t,ADDRESS,ADDRESS,ADDRESS,
                      INT,INT,INT,ADDRESS,ADDRESS,ctypes.c_float,ADDRESS,INT]
        lin.restype=ctypes.c_int
        emb=self.lib.cb184_embedding
        emb.argtypes=[ADDRESS,ctypes.c_size_t,ADDRESS,ADDRESS,ADDRESS,
                      INT,INT,INT,ADDRESS,ADDRESS,ctypes.c_float,ADDRESS,INT]
        emb.restype=ctypes.c_int

    def _get_factors(self,h):
        alpha,scale,beta=self.paid.adapter_for(h)
        if (alpha.shape!=(h.shape[0],p.RANK) or
            beta.shape!=(p.RANK,h.shape[1])):
            raise ValueError("Rank2 native source shape mismatch")
        return alpha,scale,np.ascontiguousarray(beta,dtype=np.float32)

    def linear(self,h,x):
        if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=h.shape[1]:
            raise ValueError("Original source CPU FP32 GEMV required")
        x=np.ascontiguousarray(x)
        a,scale,b=self._get_factors(h)
        out=np.empty((x.shape[0],h.shape[0]),dtype=np.float32)
        result=self.lib.cb184_linear(
            self.pointer(h.packed),len(h.packed),
            self.pointer(h.row_scales),self.pointer(h.centroids),
            self.pointer(x),x.shape[0],h.shape[0],h.shape[1],self.pointer(out),
            self.pointer(a),float(scale),self.pointer(b),p.RANK)
        if result or not np.isfinite(out).all():
            raise RuntimeError("Paid physical joint source GEMV failed: "+str(result))
        return out

    def embedding(self,h,ids):
        ids=np.ascontiguousarray(ids,dtype=np.int64).reshape(-1)
        a,scale,b=self._get_factors(h)
        out=np.empty((len(ids),h.shape[1]),dtype=np.float32)
        result=self.lib.cb184_embedding(
            self.pointer(h.packed),len(h.packed),
            self.pointer(h.row_scales),self.pointer(h.centroids),
            self.pointer(ids),len(ids),h.shape[0],h.shape[1],self.pointer(out),
            self.pointer(a),float(scale),self.pointer(b),p.RANK)
        if result or not np.isfinite(out).all():
            raise RuntimeError("Paid rank2 source embedding failed: "+str(result))
        return out

def build(blob,folder):
    paid=Paid184(blob)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    kernel=Joint184Kernel(compile_kernel(Path(__file__).with_name("kernel.c"),
                         folder/"calibri184_native.so"),paid)
    audit=bind_into_llama(model,paid.base,kernel)
    if any(p.ndim>=2 for p in model.parameters()):
        raise AssertionError("Original dense 2D source materialized in native runtime")
    if model.get_input_embeddings().handle is not model.get_output_embeddings().handle:
        raise AssertionError("Shared head source lost")
    if paid.hash()!=__import__("hashlib").sha256(blob).hexdigest():
        raise AssertionError("Paid physical complete source hash mismatch")
    return model,paid,audit
