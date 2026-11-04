"""CB104 original tied head, 8-bank source-paid direct native C execution.

Native 5-bit indices and centroid dictionaries stay compressed in RAM;
one small C LUT is built per GEMV call. Non-head transformer matrices
stay CB59 original packed modules. No external FP32 teacher at inference.
"""
from __future__ import annotations
import ctypes
from pathlib import Path
import numpy as np
import torch
from torch import nn
from direct_runtime_v59 import compile_kernel,ADDRESS,INT
from .spec import BANKS,VOCAB,DIM,K,V

class BankedKernel:
    def __init__(self,source_path,output_path):
        compiled=compile_kernel(source_path,output_path)
        self.library=ctypes.CDLL(str(compiled))
        self.head=self.library.clb104_head
        self.embedding=self.library.clb104_embedding
        self.head.argtypes=[ADDRESS,ctypes.c_size_t,ADDRESS,ADDRESS,ADDRESS,
                            INT,INT,INT,ADDRESS]
        self.head.restype=ctypes.c_int
        self.embedding.argtypes=[ADDRESS,ctypes.c_size_t,ADDRESS,ADDRESS,ADDRESS,
                                 INT,INT,INT,ADDRESS]
        self.embedding.restype=ctypes.c_int
    @staticmethod
    def ptr(x):return ADDRESS(int(x.ctypes.data))
    def matvec(self,h,banks,features):
        x=np.ascontiguousarray(features,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=DIM:
            raise AssertionError("C banked whole-head GEMV requires 2D FP32 input")
        y=np.empty((x.shape[0],VOCAB),dtype=np.float32)
        rc=self.head(self.ptr(h.packed),len(h.packed),self.ptr(h.row_scales),
                     self.ptr(banks),self.ptr(x),len(x),VOCAB,DIM,self.ptr(y))
        if rc or not np.isfinite(y).all():
            raise RuntimeError("CB104 direct native head read failed: "+str(rc))
        return y
    def lookup(self,h,banks,tokens):
        ids=np.ascontiguousarray(tokens,dtype=np.int64).reshape(-1)
        y=np.empty((len(ids),DIM),dtype=np.float32)
        rc=self.embedding(self.ptr(h.packed),len(h.packed),
                    self.ptr(h.row_scales),self.ptr(banks),self.ptr(ids),
                    len(ids),VOCAB,DIM,self.ptr(y))
        if rc or not np.isfinite(y).all():
            raise RuntimeError("CB104 direct native embedding decode failed: "+str(rc))
        return y

class BankedPackedLinear(nn.Module):
    def __init__(self,h,banks,kernel):
        super().__init__()
        self.handle=h
        self.banks=banks
        self.kernel=kernel
        self.in_features=DIM
        self.out_features=VOCAB
    def forward(self,x):
        if x.device.type!="cpu" or x.dtype!=torch.float32 or x.shape[-1]!=DIM:
            raise AssertionError("CB104 full source C projection expects CPU FP32 hidden")
        shape=tuple(x.shape[:-1])+(VOCAB,)
        v=np.ascontiguousarray(x.detach().reshape(-1,DIM).numpy())
        return torch.from_numpy(self.kernel.matvec(self.handle,self.banks,v)).reshape(shape)

class BankedPackedEmbedding(nn.Module):
    def __init__(self,h,banks,kernel):
        super().__init__()
        self.handle=h
        self.banks=banks
        self.kernel=kernel
        self.num_embeddings=VOCAB
        self.embedding_dim=DIM
    def forward(self,ids):
        if ids.device.type!="cpu":raise AssertionError("CB104 CPU-only direct lookup")
        z=np.ascontiguousarray(ids.detach().numpy(),dtype=np.int64)
        return torch.from_numpy(
            self.kernel.lookup(self.handle,self.banks,z)).reshape(tuple(ids.shape)+(DIM,))

def bind_native_banks(model,banks,source_path:Path,library_path:Path):
    head=model.get_output_embeddings()
    emb=model.get_input_embeddings()
    if not (hasattr(head,"handle") and hasattr(emb,"handle") and
            head.handle is emb.handle and
            tuple(head.handle.shape)==(VOCAB,DIM)):
        raise AssertionError("real tied trained native CB59 whole head is required")
    a=np.asarray(banks,dtype=np.float32)
    if a.shape!=(BANKS,K,V) or not np.isfinite(a).all():
        raise AssertionError("wrong physically saved FP16 bank codebook")
    codebooks=np.ascontiguousarray(a)
    kernel=BankedKernel(source_path,library_path)
    model.model.embed_tokens=BankedPackedEmbedding(head.handle,codebooks,kernel)
    model.lm_head=BankedPackedLinear(head.handle,codebooks,kernel)
    if (model.get_input_embeddings().handle is not model.get_output_embeddings().handle
         or model.get_input_embeddings().banks is not model.get_output_embeddings().banks):
        raise AssertionError("runtime output/input bank handles do not alias")
    if any(p.ndim>=2 for p in model.parameters()):
        raise AssertionError("source trained 2D dense model weights left in runtime")
    return model,{"real_physical_CB104_direct_CPU_C":True,
                  "eight_banks_source_paid_and_buffered_in_runtime":True,
                  "one_tied_original_output_input_bank_codebook":True,
                  "no_dense_pretrained_2D_weights":True}
