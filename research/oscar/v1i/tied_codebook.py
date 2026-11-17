"""Oscar v1I differentiable ORIGINAL COMPRESSED model tied embedding codebook.

Train existing 32×8 K32/V8 FP16 centers + 32k FP16 row scales
using independent true pretrained FP32 teacher activation and token losses.
Keep every existing packed 5-bit code, ALL original other 2D matrices,
original source manifest, original tied embedding ↔ output, GT weights.

Rewrites literally these two parts inside original authenticated CB59
archive, then updates physical CRC32 and SHA256. No unpaid inference model
parameter and no new codebook tensors at runtime.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import struct
import types
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from direct_codebook_v59 import CB59, HEADER, unpack5

NAME="tok_embeddings.weight"
ALIAS="output.weight"
VOCAB=32000
WIDTH=512
K=32
VECTOR=8
SCALARS=VOCAB


class LearnableOriginalTiedEmbedding(nn.Module):
    """Exact existing K32V8 source with trainable physically paid scalars.

    No new source codes/indices are inserted. One independent original
    teacher-supervised row-scale multiplier per vocabulary token and
    one original 32×8 codebook. Teacher never present during inference.
    """
    def __init__(self,original_CB59:bytes):
        super().__init__()
        original=CB59(original_CB59)
        if (NAME not in original.handles or
            ALIAS not in original.handles or
            original.handles[NAME] is not original.handles[ALIAS]):
            raise AssertionError("Oscar v1I source authentic original tied head/embedding missing")
        h=original.handles[NAME]
        if (tuple(h.shape)!=(VOCAB,WIDTH) or
            h.centroids.shape!=(K,VECTOR) or
            h.row_scales.shape!=(VOCAB,)):
            raise AssertionError("Oscar independent original FP32 42M K32/V8 vocabulary shape changed")
        code=unpack5(h.packed.tobytes(),VOCAB*WIDTH//VECTOR).astype(np.int64)
        if (code.shape!=(VOCAB*WIDTH//VECTOR,) or
            np.max(code)>=K):
            raise AssertionError("Oscar original source 5-bit code index not authentic")
        self.register_buffer("original_code5",torch.from_numpy(
            code.reshape(VOCAB,WIDTH//VECTOR).copy()))
        self.register_buffer("original_row_scale",torch.from_numpy(
            h.row_scales.copy().astype(np.float32)))
        self.register_buffer("original_center",torch.from_numpy(
            h.centroids.copy().astype(np.float32)))
        self.log_gain=nn.Parameter(torch.zeros(VOCAB,dtype=torch.float32))
        self.center=nn.Parameter(self.original_center.clone())
        self.original_CB59=original_CB59
        self._forward_hooks_installed=False
        self._model=None
        self._original_embed_forward=None
        self._original_output_forward=None
        self._preforward_handle=None
        self._cached_quant=None
        self.last_real_quant=None

    def row_scale(self):
        return torch.clamp(
            self.original_row_scale*torch.exp(self.log_gain),
            min=2.**-14,max=60000.)

    def weight(self):
        if self._cached_quant is None:
            embedded=self.center[
                self.original_code5.reshape(-1)].reshape(VOCAB,WIDTH)
            quantized=embedded*self.row_scale()[:,None]
            if not bool(torch.isfinite(quantized).all()):
                raise AssertionError("Oscar pretrained source tuned 32k embedding nonfinite")
            self._cached_quant=quantized
            self.last_real_quant=quantized
        return self._cached_quant

    def clear_cache(self):
        self._cached_quant=None

    def install(self,model):
        if self._forward_hooks_installed:
            raise AssertionError("Tied embedding codebook already installed")
        if not hasattr(model,"tok_embeddings") or not hasattr(model,"output"):
            raise AssertionError("Not original pretrained eight-layer LLaMA model")
        if (model.output.weight is not model.tok_embeddings.weight or
            model.output.weight.shape!=(VOCAB,WIDTH)):
            raise AssertionError("Oscar rewrite must retain original physical tied output")
        self._model=model
        self._original_embed_forward=model.tok_embeddings.forward
        self._original_output_forward=model.output.forward
        context=self

        def embed_forward(_mod,ids):
            if ids.dtype!=torch.long:
                raise AssertionError("Real pretrained model embedding token dtype changed")
            return F.embedding(ids,context.weight())

        def output_forward(_mod,x):
            return F.linear(x,context.weight())

        def before_model(_model,_args):
            context.clear_cache()

        model.tok_embeddings.forward=types.MethodType(
            embed_forward,model.tok_embeddings)
        model.output.forward=types.MethodType(
            output_forward,model.output)
        self._preforward_handle=model.register_forward_pre_hook(before_model)
        self._forward_hooks_installed=True
        return self

    def uninstall(self):
        if not self._forward_hooks_installed:
            return
        self._model.tok_embeddings.forward=self._original_embed_forward
        self._model.output.forward=self._original_output_forward
        self._preforward_handle.remove()
        self._preforward_handle=None
        self._forward_hooks_installed=False
        self.clear_cache()
        self._model=None

    def original_anchor_loss(self):
        # Penalize destabilizing normal-codebook geometry. This is
        # optimization regularization, not an extra inference tensor.
        return (0.7*self.log_gain.square().mean()+
                .07*(self.center-self.original_center).square().mean())

    def physically_changed_FP16_count(self):
        with torch.no_grad():
            row=self.row_scale().half()
            cent=self.center.half()
            return {
              "original_embedded_fp16_row_scales_changed":int(torch.count_nonzero(
                  row!=self.original_row_scale.half())),
              "original_embedded_fp16_codebook_centroids_changed":int(
                  torch.count_nonzero(cent!=self.original_center.half())),
              "original_packed_5bit_indices_changed":0,
            }

    def rewrite_actual_CB59_source(self)->bytes:
        if self._forward_hooks_installed:
            # Saving while installed isn't forbidden mathematically but
            # makes it easier to confuse transient GPU/CPU weights with
            # actually reloaded CB59; require explicit release.
            raise AssertionError("Uninstall Oscar differentiable source before physical re-encode")
        before=CB59(self.original_CB59)
        desc=before.meta["tensor_descriptors"][NAME]
        if desc["dtype"]!="CB59_K32V8":
            raise AssertionError("Original pretrained tied embedding descriptor no longer K32/V8")
        raw=bytearray(self.original_CB59)
        _,_,mlen,_=HEADER.unpack_from(raw)
        payload=HEADER.size+mlen
        center=self.center.detach().cpu().numpy().astype("<f2")
        scales=self.row_scale().detach().cpu().numpy().astype("<f2")
        if (center.shape!=(K,VECTOR) or scales.shape!=(VOCAB,)
            or not np.isfinite(center).all() or
            not np.isfinite(scales).all() or np.any(scales<=0)):
            raise AssertionError("Original independent FP32 model new codebook not FP16 safe")
        expected={ "centroids":center.tobytes(),
                   "row_scales":scales.tobytes() }
        accounted=[]
        for part,bytes_ in expected.items():
            offset,length=desc["parts"][part]
            if len(bytes_)!=length:
                raise AssertionError("Oscar original 5bit codec had unexpected per-tensor scalar physical size")
            raw[payload+offset:payload+offset+length]=bytes_
            accounted.append((payload+offset,payload+offset+length))
        if any(a>=b for a,b in accounted):
            raise AssertionError("Oscar original tied source codebook offsets empty")
        # This test is important: *NO* non-embedding K32/V8 matrices, 5bit
        # indices or source tensor alias data are allowed to change.
        prev=memoryview(self.original_CB59)
        for idx,(start,end) in enumerate(sorted(accounted)):
            left=HEADER.size if idx==0 else max(e for _,e in sorted(accounted)[:idx])
            if left<start and raw[left:start]!=prev[left:start]:
                raise AssertionError("Unexpected original CB59 source bytes modified")
        raw[-36:-32]=struct.pack("<I",zlib.crc32(raw[:-36]))
        raw[-32:]=hashlib.sha256(raw[:-32]).digest()
        fresh=bytes(raw)
        if len(fresh)!=len(self.original_CB59):
            raise AssertionError("Oscar physical 5bit original pretrained binary size changed")
        checked=CB59(fresh)
        if (checked.meta!=before.meta or
            checked.handles[NAME] is not checked.handles[ALIAS] or
            checked.handles[NAME].packed.tobytes()!=before.handles[NAME].packed.tobytes()):
            raise AssertionError("Oscar changed original pretrained K32V8 bits or head alias")
        for k in before.handles:
            if k in (NAME,ALIAS):
                continue
            a,b=before.handles[k],checked.handles[k]
            if (a.packed.tobytes()!=b.packed.tobytes() or
                not np.array_equal(a.centroids,b.centroids) or
                not np.array_equal(a.row_scales,b.row_scales)):
                raise AssertionError("Oscar silently changed OTHER original pretrained FP32 matrix "+k)
        for k in before.raw:
            if not np.array_equal(before.raw[k],checked.raw[k]):
                raise AssertionError("Oscar silently changed original norm/bias "+k)
        return fresh
