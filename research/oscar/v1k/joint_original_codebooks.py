"""Oscar v1K — differentiable, physically PAID existing ORIGINAL CB59 codebooks.

Exactly four original pretrained TinyStories eight-layer K32/V8 matrices
are modified via their pre-existing 32x8 FP16 centroids and FP16 per-row
scales. All original packed 5-bit codes and all other pretrained tensors
remain bitwise identical. The connected v1F/v1G Green Tunnels are separately
real FP16 sidecars and are jointly trainable in the same optimizer.

The original pretrained teacher's FP32 weights are NEVER stored in student.
"""
from __future__ import annotations

import hashlib
import struct
import types
import zlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from direct_codebook_v59 import CB59,HEADER,unpack5


TARGETS=(
    "layers.7.feed_forward.w2.weight",
    "layers.1.feed_forward.w2.weight",
    "layers.6.attention.wq.weight",
    "layers.1.feed_forward.w3.weight",
)
K=32
VECTOR=8


class ExactOriginalK32V8Matrix(nn.Module):
    def __init__(self,handle):
        super().__init__()
        if len(handle.shape)!=2 or handle.shape[1]%VECTOR:
            raise AssertionError("Oscar original 42M source tensor not 2D K32/V8")
        m,n=handle.shape
        codes=unpack5(handle.packed.tobytes(),m*n//VECTOR)
        self.register_buffer("fixed_original_5bit_indices",
                             torch.from_numpy(codes.astype(np.int64).reshape(m,n//VECTOR).copy()))
        self.register_buffer("initial_original_FP16_centroids",
                             torch.from_numpy(handle.centroids.copy()).float())
        self.register_buffer("initial_original_FP16_row_scales",
                             torch.from_numpy(handle.row_scales.copy()).float())
        self.center=nn.Parameter(self.initial_original_FP16_centroids.clone())
        self.log_scale=nn.Parameter(torch.zeros(m,dtype=torch.float32))
        self.m=m
        self.n=n

    def weight(self):
        choices=self.center[self.fixed_original_5bit_indices.reshape(-1)]
        reconstructed=choices.reshape(self.m,self.n)
        row=self.row_scales()
        result=reconstructed*row[:,None]
        return result

    def row_scales(self):
        return torch.clamp(self.initial_original_FP16_row_scales*
                           torch.exp(self.log_scale),min=2**-14,max=60000.)

    def penalty(self):
        return (0.5*self.log_scale.square().mean()+
                0.025*(self.center-self.initial_original_FP16_centroids).square().mean())

    def changed_fp16(self):
        return {
          "centroid_fp16_changed":int(torch.count_nonzero(
                self.center.detach().half()!=self.initial_original_FP16_centroids.half())),
          "row_scale_fp16_changed":int(torch.count_nonzero(
                self.row_scales().detach().half()!=self.initial_original_FP16_row_scales.half())),
          "5bit_codeword_indices_changed":0,
        }


class JointOriginalSource(nn.Module):
    def __init__(self,original_blob:bytes):
        super().__init__()
        self.original_blob=original_blob
        orig=CB59(original_blob)
        if (len(orig.meta["tensor_descriptors"])!=len(orig.meta["trained_unique_keys"])
            or orig.meta["original_unique_parameter_count"]!=41689600):
            raise AssertionError("Oscar v1K expected complete authentic 41,689,600 original pretrained weights")
        for name in TARGETS:
            if name not in orig.handles:
                raise AssertionError("Oscar v1K nominated original FP32 causal sensitive matrix missing: "+name)
        self.names=TARGETS
        self.layers=nn.ModuleList([ExactOriginalK32V8Matrix(orig.handles[n])
                                   for n in TARGETS])
        self._replaced=[]
        self._installed=False

    def install(self,model):
        if self._installed:
            raise AssertionError("Oscar v1K source 5bit differentiable replacement already installed")
        for name,source in zip(self.names,self.layers):
            target=model.get_submodule(name.removesuffix(".weight"))
            if not isinstance(target,nn.Linear) or target.bias is not None:
                raise AssertionError("Oscar true pretrained source module is not original unbiased nn.Linear: "+name)
            if tuple(target.weight.shape)!=(source.m,source.n):
                raise AssertionError("Oscar original pretrained matrix unverified shape: "+name)
            saved=target.forward
            def forward(_self,x,*,_source=source):
                return F.linear(x,_source.weight())
            target.forward=types.MethodType(forward,target)
            self._replaced.append((target,saved))
        self._installed=True
        return self

    def uninstall(self):
        for target,forward in reversed(self._replaced):
            target.forward=forward
        self._replaced=[]
        self._installed=False

    def original_anchoring_loss(self):
        return sum(m.penalty() for m in self.layers)/len(self.layers)

    def measure_changed_fp16(self):
        return {name:mod.changed_fp16() for name,mod in zip(self.names,self.layers)}

    def rewrite_physical_CB59(self)->bytes:
        if self._installed:
            raise AssertionError("Oscar v1K must uninstall transient differentiable FP32 diagnostic before writing actual K32V8 packed source")
        source=CB59(self.original_blob)
        raw=bytearray(self.original_blob)
        _,_,mlen,_=HEADER.unpack_from(raw)
        body=HEADER.size+mlen
        expected_allowed=[]
        for name,layer in zip(self.names,self.layers):
            desc=source.meta["tensor_descriptors"][name]
            if desc["dtype"]!="CB59_K32V8":
                raise AssertionError("Oscar original 5bit pretrained matrix ABI type changed")
            for part,value in (
                ("centroids",layer.center.detach().cpu().numpy().astype("<f2").tobytes()),
                ("row_scales",layer.row_scales().detach().cpu().numpy().astype("<f2").tobytes())
            ):
                start,length=desc["parts"][part]
                if start<0 or len(value)!=length:
                    raise AssertionError("Oscar physical original pretrained codebook segment length corrupted")
                begin=body+start
                raw[begin:begin+length]=value
                expected_allowed.append((begin,begin+length,name,part))
        if len(expected_allowed)!=len(self.names)*2:
            raise AssertionError("Missing physically paid original FP16 CB59 source segment")
        raw[-36:-32]=struct.pack("<I",zlib.crc32(raw[:-36]))
        raw[-32:]=hashlib.sha256(raw[:-32]).digest()
        physically_new=bytes(raw)
        assert len(physically_new)==len(self.original_blob)
        # STRONG bytewise integrity gate: ONLY the eight authorized source
        # segments and the original standard trailing checksums can differ.
        compare=bytearray(physically_new)
        for start,end,_,_ in expected_allowed:
            compare[start:end]=self.original_blob[start:end]
        compare[-36:]=self.original_blob[-36:]
        if bytes(compare)!=self.original_blob:
            raise AssertionError("Oscar v1K modified unauthorized original pretrained 5bit codes, other matrix parts or header!")
        decoded=CB59(physically_new)
        if decoded.meta!=source.meta:
            raise AssertionError("Oscar v1K altered immutable original source metadata")
        for key in source.handles:
            a=source.handles[key]
            b=decoded.handles[key]
            if a.packed.tobytes()!=b.packed.tobytes():
                raise AssertionError("Oscar v1K accidentally rewrote EXISTING 5-bit original pretrained packed indices: "+key)
            if key not in TARGETS:
                if (not np.array_equal(a.centroids,b.centroids) or
                    not np.array_equal(a.row_scales,b.row_scales)):
                    raise AssertionError("Oscar v1K accidentally rewrote OTHER pretrained matrix: "+key)
        for key in source.raw:
            if not np.array_equal(source.raw[key],decoded.raw[key]):
                raise AssertionError("Oscar v1K accidentally rewrote raw original norm/bias: "+key)
        if (decoded.handles["output.weight"] is not decoded.handles["tok_embeddings.weight"]):
            raise AssertionError("Oscar v1K damaged original trained 32k tied output head/input alias")
        return physically_new
