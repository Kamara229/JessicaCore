"""NRA2 3.0 S2: native C CB59_K32V8 CPU execution for EVERY 2D area weight.

All 6.1M learned 2D parameters stay in original Calibri packed 5-bit
indices/FP16 book/FP16 scale source. Only small centroid+row_scale metadata is
converted to float32 for the existing direct_kernel_v59.c. No dense matrices
are materialized or retained in the inference area. No optimizer, original
Calibri 43M checkpoint, Green Tunnel weights, or evaluation data is read.

CPU-only and strict gate=0: decoder outputs remain UNCHANGED in shadow mode.
This is a reference native CPU model, not optimized CB59/GT speed certification.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import (
    MORPH, SEMANTIC_ROLES, UPOS, NRA3Config, NRA3Features,
)


def expected_shapes(c: NRA3Config) -> dict[str, tuple[int, ...]]:
    d, r = c.hidden, c.relation_width
    f = d * c.ffn_multiplier
    shapes: dict[str, tuple[int, ...]] = {}
    def norm(name, size):
        shapes[name+".weight"] = (size,)
        shapes[name+".bias"] = (size,)
    def linear(name, out, ins, bias=True):
        if ins%8:
            raise ValueError("CB59 requires matrix input features divisible by 8")
        shapes[name+".weight"] = (out, ins)
        if bias:
            shapes[name+".bias"] = (out,)
    norm("input_projection.0",c.io_width)
    linear("input_projection.1",d,c.io_width)
    for family, count in (("grammar",c.grammar_blocks),
                          ("semantic",c.semantic_blocks)):
        for k in range(count):
            p=f"{family}.{k}"
            norm(p+".norm_attn",d)
            shapes[p+".attn.in_proj_weight"]=(3*d,d)
            shapes[p+".attn.in_proj_bias"]=(3*d,)
            linear(p+".attn.out_proj",d,d)
            norm(p+".norm_ffn",d)
            linear(p+".ffn.0",f,d)
            linear(p+".ffn.2",d,f)
    linear("upos_head",len(UPOS),d)
    linear("morph_head",len(MORPH),d)
    linear("dep_q",d,d,False)
    linear("dep_k",d,d,False)
    linear("role_head",len(SEMANTIC_ROLES),d)
    linear("core_in",r,d)
    for name in ("rel_q","rel_k","rel_value"):
        linear(name,r,r,False)
    linear("ante_q",d,d,False)
    linear("ante_k",d,d,False)
    shapes["rel_update.weight_ih"]=(3*r,r)
    shapes["rel_update.weight_hh"]=(3*r,r)
    shapes["rel_update.bias_ih"]=(3*r,)
    shapes["rel_update.bias_hh"]=(3*r,)
    linear("readout",c.io_width,r,False)
    return shapes


class NativeCB59Kernel:
    """Loads EXISTING CALIBRI v59 C source as a precompiled CPU shared library."""
    def __init__(self, path: str | Path):
        path=Path(path)
        if not path.is_file():
            raise FileNotFoundError("NRA3 requires compiled trusted Calibri CB59 C kernel")
        self._lib=ctypes.CDLL(str(path.resolve()))
        self._fn=self._lib.cb59_linear
        u8=ctypes.POINTER(ctypes.c_uint8)
        fl=ctypes.POINTER(ctypes.c_float)
        self._fn.argtypes=[u8,ctypes.c_size_t,fl,fl,fl,
                           ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,fl]
        self._fn.restype=ctypes.c_int

    def matmul(self, handle, x: torch.Tensor) -> torch.Tensor:
        if (x.device.type!="cpu" or x.dtype!=torch.float32
                or x.ndim<2 or x.shape[-1]!=handle.shape[1]):
            raise ValueError("NRA3 direct K32V8 GEMV expects float32 CPU features")
        xnp=np.ascontiguousarray(x.detach().numpy(),dtype=np.float32)
        size=int(np.prod(x.shape[:-1]))
        output=np.empty((size,handle.shape[0]),dtype=np.float32)
        ptr_u8=ctypes.POINTER(ctypes.c_uint8)
        ptr_fl=ctypes.POINTER(ctypes.c_float)
        result=self._fn(handle.packed.ctypes.data_as(ptr_u8),
                        handle.packed.nbytes,
                        handle.row_scales.ctypes.data_as(ptr_fl),
                        handle.centroids.ctypes.data_as(ptr_fl),
                        xnp.ctypes.data_as(ptr_fl),
                        size,handle.shape[0],handle.shape[1],
                        output.ctypes.data_as(ptr_fl))
        if result!=0:
            raise RuntimeError(f"NRA3 CB59 native kernel failed status={result}")
        return torch.from_numpy(output.reshape(*x.shape[:-1],handle.shape[0]))


class NRA3NativePackedCPU(nn.Module):
    """All matrix operators compiled C CB59, never dense pretrained weights.

    CPU-only, no autograd. Accepts physically authenticated *raw* CB59 only.
    If NRA weights are stored in S61 lossless envelope, decode once before
    instantiating this class; native 5-bit matmul still runs on actual CB59.
    """
    NRA3_NATIVE_CPU_PACKED=True

    def __init__(self, physical_cb59: bytes,
                 config: NRA3Config, library_path: str | Path):
        super().__init__()
        if not isinstance(physical_cb59,bytes):
            raise ValueError("Native NRA3 needs verified complete physical CB59 bytes")
        self.config=config
        self.kernel=NativeCB59Kernel(library_path)
        self.source=CB59(physical_cb59)
        shapes=expected_shapes(config)
        desc=self.source.meta["tensor_descriptors"]
        if (set(desc)!=set(shapes) or set(self.source.handles)|
                set(self.source.raw)!=set(shapes)):
            raise ValueError("NRA3 native ABI source operator names incomplete")
        for name, shape in shapes.items():
            if tuple(desc[name]["shape"])!=shape:
                raise ValueError("NRA3 native source shape mismatch: "+name)
            if len(shape)==2:
                if name not in self.source.handles or desc[name]["dtype"]!="CB59_K32V8":
                    raise ValueError("NRA3 forbidden dense matrix: "+name)
            elif name not in self.source.raw or desc[name]["dtype"]!="f4":
                raise ValueError("NRA3 nonphysical 1D source: "+name)
        self.source_sha256=hashlib.sha256(physical_cb59).hexdigest()
        self.unique_parameters=self.source.meta["original_unique_parameter_count"]
        if self.unique_parameters!=sum(math.prod(s) for s in shapes.values()):
            raise ValueError("NRA3 native parameter count drift")
        # NumPy view objects refer only to archived 1D source vectors,
        # or compact coded 2D payloads: no dense 2D matrix is resident.
        self._bias={n:torch.from_numpy(np.array(self.source.raw[n],copy=True))
                    for n,shape in shapes.items() if len(shape)==1}
        self.output_gate=0
        self.candidate_accepted=False
        self.native_cpu_weights_dense_resident=False

    def linear(self,name,x):
        h=self.source.handles.get(name+".weight")
        if h is None:
            raise ValueError("Native NRA3 unknown operator "+name)
        y=self.kernel.matmul(h,x)
        bias=self._bias.get(name+".bias")
        return y if bias is None else y+bias

    def _layernorm(self,name,x):
        return F.layer_norm(x,(x.shape[-1],),
                            self._bias[name+".weight"],
                            self._bias[name+".bias"],eps=1e-5)

    def _block(self,x,name,allowed,valid):
        c=self.config
        b,t,d=x.shape
        z=self._layernorm(name+".norm_attn",x)
        # Original torch.nn.MultiheadAttention W_qkv and W_out are
        # directly executed in 5-bit C; no FP32 weight matrix.
        qkv=self.kernel.matmul(self.source.handles[name+".attn.in_proj_weight"],z)
        qkv=qkv+self._bias[name+".attn.in_proj_bias"]
        q,k,v=qkv.chunk(3,dim=-1)
        def to_heads(y):
            return y.reshape(b,t,c.heads,d//c.heads).transpose(1,2)
        q,k,v=map(to_heads,(q,k,v))
        attn=F.scaled_dot_product_attention(
            q,k,v,attn_mask=allowed[:,None,:,:],dropout_p=0.0)
        flat=attn.transpose(1,2).contiguous().reshape(b,t,d)
        x=x+self.linear(name+".attn.out_proj",flat)
        y=self._layernorm(name+".norm_ffn",x)
        y=self.linear(name+".ffn.2",F.gelu(self.linear(name+".ffn.0",y)))
        x=x+y
        return x.masked_fill(~valid.unsqueeze(-1),0.)

    def _gru_cell(self,inp,h):
        gi=self.kernel.matmul(
            self.source.handles["rel_update.weight_ih"],inp)+self._bias["rel_update.bias_ih"]
        gh=self.kernel.matmul(
            self.source.handles["rel_update.weight_hh"],h)+self._bias["rel_update.bias_hh"]
        ir,iz,inn=gi.chunk(3,dim=-1)
        hr,hz,hn=gh.chunk(3,dim=-1)
        reset=torch.sigmoid(ir+hr)
        update=torch.sigmoid(iz+hz)
        new=torch.tanh(inn+reset*hn)
        return new+update*(h-new)

    def forward(self,h:torch.Tensor,valid:torch.Tensor|None=None)->NRA3Features:
        if (not isinstance(h,torch.Tensor) or h.ndim!=3
                or h.shape[-1]!=320 or h.dtype!=torch.float32
                or h.device.type!="cpu" or not bool(torch.isfinite(h).all())):
            raise ValueError("NRA3 packed source expects finite CPU float32 [B,T,320]")
        b,t,_=h.shape
        if b<1 or t<1:
            raise ValueError("NRA3 empty native input")
        if valid is None:
            valid=torch.ones((b,t),dtype=torch.bool)
        if (valid.shape!=(b,t) or valid.dtype!=torch.bool or valid.device.type!="cpu"
                or not bool(valid[:,0].all())
                or not bool((valid.long().cummin(dim=1).values==valid.long()).all())):
            raise ValueError("NRA3 packed only contiguous right-padding mask")
        if self.output_gate!=0 or self.candidate_accepted:
            raise RuntimeError("NRA3 packed READ-ONLY: no candidate activation")
        allowed=torch.tril(torch.ones(t,t,dtype=torch.bool))[None,:,:] & valid[:,None,:]
        x=self.linear("input_projection.1",
                      self._layernorm("input_projection.0",h))
        x=x.masked_fill(~valid.unsqueeze(-1),0.)
        for i in range(self.config.grammar_blocks):
            x=self._block(x,f"grammar.{i}",allowed,valid)
        grammar=x
        upos=self.linear("upos_head",x)
        morph=self.linear("morph_head",x)
        dep=self.linear("dep_q",x)@self.linear("dep_k",x).transpose(1,2)
        dep=dep/math.sqrt(self.config.hidden)
        dep=dep.masked_fill(~allowed,-1e4)
        for i in range(self.config.semantic_blocks):
            x=self._block(x,f"semantic.{i}",allowed,valid)
        semantics=x
        role=self.linear("role_head",x)
        ante=self.linear("ante_q",x)@self.linear("ante_k",x).transpose(1,2)
        ante=ante/math.sqrt(self.config.hidden)
        ante=ante.masked_fill(~allowed,-1e4)
        relation=self.linear("core_in",x)
        for _ in range(self.config.relation_steps):
            q=self.linear("rel_q",relation)
            k=self.linear("rel_k",relation)
            score=(q@k.transpose(1,2))/math.sqrt(self.config.relation_width)
            score=score.masked_fill(~allowed,-1e4)
            v=self.linear("rel_value",relation)
            message=score.softmax(dim=-1)@v
            relation=self._gru_cell(message,relation)
            relation=relation.masked_fill(~valid.unsqueeze(-1),0.)
        read=self.linear("readout",relation)
        if any(not bool(torch.isfinite(v).all()) for v in
               (grammar,semantics,relation,upos,morph,role,read)):
            raise ValueError("NRA3 native generated nonfinite features")
        return NRA3Features(grammar,semantics,relation,upos,morph,dep,role,ante,read)

    def packed_matrix_count(self):
        return len(self.source.handles)

    def packed_resident_source_bytes(self):
        """Physical packed payload only (excluding C executable code)."""
        return self.source.physical_bytes()

    def forward_train(self,*args,**kwargs):
        raise RuntimeError("Native NRA3 is inference-only; use packed codebook QAT elsewhere")
