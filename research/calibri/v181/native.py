"""v181 facade: literal V1/V2 native physical decoder, no pretrained dense tensors."""
import ctypes
from pathlib import Path
import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_runtime_v59 import compile_kernel,bind_into_llama
from .codec import CB181
from .protocol import GROUPS

PTR=ctypes.c_void_p
I64=ctypes.c_int64

class Direct181:
    def __init__(self,path,group):
        self.group=group
        self.lib=ctypes.CDLL(str(path))
        for name in ("cb181_linear","cb181_embedding"):
            fn=getattr(self.lib,name)
            fn.argtypes=[PTR,ctypes.c_size_t,PTR,PTR,PTR,I64,I64,I64,I64,PTR]
            fn.restype=ctypes.c_int
    @staticmethod
    def ptr(a):return PTR(int(a.ctypes.data))
    def linear(self,h,x):
        if x.ndim!=2 or x.dtype!=np.float32 or x.shape[1]!=h.shape[1]:
            raise ValueError("v181 CPU expects 2D float32")
        x=np.ascontiguousarray(x)
        result=np.empty((x.shape[0],h.shape[0]),dtype=np.float32)
        rc=self.lib.cb181_linear(self.ptr(h.packed),len(h.packed),
                self.ptr(h.row_scales),self.ptr(h.centroids),self.ptr(x),
                x.shape[0],h.shape[0],h.shape[1],h.group,self.ptr(result))
        if rc or not np.isfinite(result).all():raise ValueError("Native C181 linear "+str(rc))
        return result
    def embedding(self,h,ids):
        ids=np.ascontiguousarray(ids,dtype=np.int64).reshape(-1)
        result=np.empty((len(ids),h.shape[1]),dtype=np.float32)
        rc=self.lib.cb181_embedding(self.ptr(h.packed),len(h.packed),
                self.ptr(h.row_scales),self.ptr(h.centroids),self.ptr(ids),
                len(ids),h.shape[0],h.shape[1],h.group,self.ptr(result))
        if rc or not np.isfinite(result).all():raise ValueError("Native C181 embed "+str(rc))
        return result

def build(blob,arm,folder):
    reader=CB181(blob)
    if arm not in GROUPS or reader.meta["vector_length"]!=GROUPS[arm]:
        raise ValueError("Oracle source group mismatch")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    native=Direct181(compile_kernel(Path(__file__).with_name("kernel.c"),
                                    folder/"lib181.so"),GROUPS[arm])
    report=bind_into_llama(model,reader,native)
    if any(t.ndim>=2 for t in model.parameters()):
        raise AssertionError("Dense source in diagnostic physical CPU model")
    return model,reader,report
