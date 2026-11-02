#!/usr/bin/env python3
"""CALIBRI v59 packed-resident transformer runtime on CPU.

Replaces ALL trained 2D original Embedding and Linear tensors with a
native C LUT-GEMV (five-bit / eight weights) or on-demand row decoder.
No full pretrained 2D weight is materialized while executing inference.
The model retains tiny 1D norms/biases in FP32.

This is an intentionally elementary CPU reference for DIRECT execution,
NOT a speed-optimized CUDA/QMoE reproduction. Batch/sequence prefill
will likely be slower than standard dense FP32 BLAS at this stage.
"""
from __future__ import annotations
import ctypes,gc,os,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from direct_codebook_v59 import CB59,QuantHandle,unpack5

ADDRESS=ctypes.c_void_p
INT=ctypes.c_int64


def compile_kernel(source,output=None):
    source=Path(source)
    if output is None:
        output=Path(tempfile.gettempdir())/"libcalibri_direct_v59.so"
    result=subprocess.run(["gcc","-O3","-std=c11","-fPIC","-shared",
                           "-o",str(output),str(source)],
                          text=True,capture_output=True,timeout=120)
    if result.returncode:
        raise RuntimeError("CB59 native source compile failed: "+result.stderr)
    return output


class NativeKernel:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(path))
        lin=self.lib.cb59_linear
        lin.argtypes=[ADDRESS,ctypes.c_size_t,ADDRESS,ADDRESS,ADDRESS,
                      INT,INT,INT,ADDRESS]
        lin.restype=ctypes.c_int
        emb=self.lib.cb59_embedding
        emb.argtypes=[ADDRESS,ctypes.c_size_t,ADDRESS,ADDRESS,ADDRESS,
                      INT,INT,INT,ADDRESS]
        emb.restype=ctypes.c_int

    @staticmethod
    def pointer(a):
        return ADDRESS(int(a.ctypes.data))

    def linear(self,h,x):
        if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=h.shape[1]:
            raise ValueError("direct packed matvec requires contiguous CPU FP32 activations")
        x=np.ascontiguousarray(x)
        out=np.empty((x.shape[0],h.shape[0]),dtype=np.float32)
        rc=self.lib.cb59_linear(
            self.pointer(h.packed),len(h.packed),
            self.pointer(h.row_scales),self.pointer(h.centroids),
            self.pointer(x),x.shape[0],h.shape[0],h.shape[1],
            self.pointer(out))
        if rc:
            raise RuntimeError(f"packed 5-bit GEMV native code error {rc}")
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("sub0.7 packed linear produced NaN/Inf")
        return out

    def embedding(self,h,ids):
        ids=np.ascontiguousarray(ids,dtype=np.int64).reshape(-1)
        out=np.empty((len(ids),h.shape[1]),dtype=np.float32)
        rc=self.lib.cb59_embedding(
            self.pointer(h.packed),len(h.packed),
            self.pointer(h.row_scales),self.pointer(h.centroids),
            self.pointer(ids),len(ids),h.shape[0],h.shape[1],
            self.pointer(out))
        if rc:raise RuntimeError(f"packed 5-bit embedding native code {rc}")
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("sub0.7 packed embedding produced NaN/Inf")
        return out


class PackedLinear(nn.Module):
    def __init__(self,h:QuantHandle,kernel:NativeKernel,bias=None):
        super().__init__()
        self.handle=h
        self.kernel=kernel
        self.in_features=h.shape[1]
        self.out_features=h.shape[0]
        if bias is not None:
            self.register_buffer("bias",torch.from_numpy(np.array(bias,
                                       dtype=np.float32,copy=True)))
        else:
            self.bias=None
    def forward(self,x):
        if x.device.type!="cpu" or x.dtype!=torch.float32:
            raise ValueError("CB59 direct CPU kernel requires FP32 activations")
        shape=tuple(x.shape[:-1])+(self.out_features,)
        a=np.ascontiguousarray(x.detach().reshape(-1,self.in_features).numpy())
        result=self.kernel.linear(self.handle,a)
        y=torch.from_numpy(result).reshape(shape)
        if self.bias is not None:
            y=y+self.bias
        return y


class PackedEmbedding(nn.Module):
    def __init__(self,h:QuantHandle,kernel:NativeKernel,padding_idx=None):
        super().__init__()
        self.handle=h
        self.kernel=kernel
        self.num_embeddings=h.shape[0]
        self.embedding_dim=h.shape[1]
        self.padding_idx=padding_idx
    def forward(self,ids):
        if ids.device.type!="cpu":
            raise ValueError("CB59 embedding prototype CPU-only")
        array=np.ascontiguousarray(ids.detach().numpy(),dtype=np.int64)
        out=self.kernel.embedding(self.handle,array)
        return torch.from_numpy(out).reshape(tuple(ids.shape)+(self.embedding_dim,))


def bind_into_llama(model,archive:CB59,kernel:NativeKernel):
    """Replace ALL trained dense 2D model modules, including tied head."""
    required=set(archive.handles)
    actual=set()
    seen_types={}
    # Snapshot names: modifying module registry while iterating is unsafe.
    named=list(model.named_modules(remove_duplicate=False))
    for modname,mod in named:
        if isinstance(mod,(nn.Linear,nn.Embedding)):
            key=modname+".weight"
            if key not in required:
                raise AssertionError("missing original trained compressed matrix: "+key)
            h=archive.handles[key]
            if tuple(mod.weight.shape)!=h.shape:
                raise AssertionError("trained tensor geometry mismatch "+key)
            parent_name,_,attr=modname.rpartition(".")
            parent=model.get_submodule(parent_name) if parent_name else model
            if isinstance(mod,nn.Embedding):
                compressed=PackedEmbedding(h,kernel,padding_idx=mod.padding_idx)
            else:
                bkey=modname+".bias"
                compressed=PackedLinear(
                    h,kernel,archive.raw.get(bkey)
                    if mod.bias is not None else None)
            setattr(parent,attr,compressed)
            actual.add(key)
            seen_types[modname]=type(compressed).__name__
    if actual!=required:
        raise AssertionError("not all source 2D weights have direct packed module")
    # Raw small 1D norms/biases remain native FP32; inspect model actual
    # parameter names after removing original dense matmuls.
    with torch.no_grad():
        cur=dict(model.named_parameters())
        for name,t in archive.raw.items():
            if name not in cur:
                if name.endswith(".bias"):
                    continue  # stored as a buffer in PackedLinear
                raise AssertionError("raw trained tensor isn't a live runtime parameter: "+name)
            if tuple(cur[name].shape)!=tuple(t.shape):
                raise AssertionError("source 1D shape mismatch "+name)
            cur[name].copy_(torch.from_numpy(np.array(t,copy=True)))
    leftovers={n:tuple(p.shape) for n,p in model.named_parameters()
               if p.ndim>=2}
    if leftovers:
        raise AssertionError("DENSE source weights persisted in packed runtime: "+str(leftovers))
    if model.get_input_embeddings().handle is not model.get_output_embeddings().handle:
        raise AssertionError("tied embedding/out-head must reference ONE physical codebook")
    gc.collect()
    return {
        "compressed_real_2D_module_count":len(actual),
        "runtime_dense_multi_dim_source_parameters":len(leftovers),
        "tied_embed_and_output_codebook_SHARED_IN_RAM":True,
        "first_loaded_modules":dict(list(seen_types.items())[:8]),
    }


def smoke():
    from direct_codebook_v59 import encode
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=nn.Embedding(32,16)
            self.linear=nn.Linear(16,16)
            self.head=nn.Linear(16,32,bias=False)
            self.head.weight=self.embed.weight
        def get_input_embeddings(self):return self.embed
        def get_output_embeddings(self):return self.head
    torch.manual_seed(59)
    toy=Toy()
    data,_,_=encode(toy.state_dict())
    a=CB59(data,expected_keys=toy.state_dict().keys())
    so=compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"))
    kernel=NativeKernel(so)
    h=a.handles["linear.weight"]
    # Audit only: explicit dense reconstruction used here solely as test
    # oracle. Production PackedLinear/Embedding never reconstruct W.
    ix=unpack5(h.packed.tobytes(),h.shape[0]*(h.shape[1]//8))
    dense=(h.centroids[ix.reshape(h.shape[0],-1)].reshape(h.shape)*
           h.row_scales[:,None])
    x=np.arange(16,dtype=np.float32)[None,:]/16
    direct=kernel.linear(h,x)
    oracle=x@dense.T
    if np.max(abs(direct-oracle))>1e-4:
        raise AssertionError("native 5-bit direct compressed GEMV != naive decoded matrix")
    ids=np.array([0,4,20],dtype=np.int64)
    emb=a.handles["embed.weight"]
    result=kernel.embedding(emb,ids)
    selected=unpack5(emb.packed.tobytes(),emb.shape[0]*(emb.shape[1]//8))
    ref=emb.centroids[selected.reshape(emb.shape[0],-1)].reshape(emb.shape)
    ref*=emb.row_scales[:,None]
    if np.max(abs(result-ref[ids]))>1e-6:
        raise AssertionError("embedding direct-on-demand rows do not match 5bit weight")
    report=bind_into_llama(toy,a,kernel)
    out=toy.head(toy.linear(toy.embed(torch.tensor([0,4]))))
    if out.shape!=(2,32):
        raise AssertionError("truly packed model real forward invalid")
    print("CALIBRI v59 native C direct 5bit GEMV+embedding vs oracle, full packed Toy forward PASS")


if __name__=="__main__":
    smoke()
