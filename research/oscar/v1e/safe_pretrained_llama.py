"""Locally audited, remote-code-FREE LLaMA-2.c 42M FP32 reference.

Minimal model to inspect original independently pretrained Karpathy TinyStories
FP32 weights with the original 8-layer 512-wide architecture. No external
Python is imported/executed. The 8-layer teacher is *NOT* CALIBRI 12-layer.
Weights are tied token_embedding/output, plain FP32 reference and decoded
CB59 mirrors are ONLY diagnostics, NEVER weight files in a model runtime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Shape:
    vocab:int
    dim:int
    layers:int
    heads:int
    hidden:int
    max_ctx:int=1024


class RMS(nn.Module):
    def __init__(self,n:int,eps:float=1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(n))
        self.eps=eps
    def forward(self,x):
        return (x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+
                              self.eps)).to(x.dtype)*self.weight


class CausalAttention(nn.Module):
    def __init__(self,dim:int,heads:int):
        super().__init__()
        self.dim=dim
        self.heads=heads
        self.wq=nn.Linear(dim,dim,bias=False)
        self.wk=nn.Linear(dim,dim,bias=False)
        self.wv=nn.Linear(dim,dim,bias=False)
        self.wo=nn.Linear(dim,dim,bias=False)
    def forward(self,x,cos,sin):
        b,t,c=x.shape
        d=c//self.heads
        q=self.wq(x).reshape(b,t,self.heads,d)
        k=self.wk(x).reshape(b,t,self.heads,d)
        v=self.wv(x).reshape(b,t,self.heads,d)
        # Exactly reference llama2.c pair-interleaved RoPE, not half-split.
        def rope(u):
            r,im=u.float().reshape(b,t,self.heads,d//2,2).unbind(-1)
            c0=cos[None,:,None,:]
            s0=sin[None,:,None,:]
            out=torch.stack((r*c0-im*s0,r*s0+im*c0),-1)
            return out.flatten(-2).to(u.dtype).transpose(1,2)
        q,k=rope(q),rope(k)
        v=v.transpose(1,2)
        context=F.scaled_dot_product_attention(
            q,k,v,attn_mask=None,dropout_p=0.,is_causal=True)
        return self.wo(context.transpose(1,2).contiguous().view(b,t,c))


class SwiGLUFFN(nn.Module):
    def __init__(self,d:int,h:int):
        super().__init__()
        self.w1=nn.Linear(d,h,bias=False)
        self.w2=nn.Linear(h,d,bias=False)
        self.w3=nn.Linear(d,h,bias=False)
    def forward(self,x):
        return self.w2(F.silu(self.w1(x))*self.w3(x))


class LlamaBlock(nn.Module):
    def __init__(self,d:int,h:int,heads:int):
        super().__init__()
        self.attention=CausalAttention(d,heads)
        self.feed_forward=SwiGLUFFN(d,h)
        self.attention_norm=RMS(d)
        self.ffn_norm=RMS(d)
    def forward(self,x,cos,sin):
        h=x+self.attention(self.attention_norm(x),cos,sin)
        return h+self.feed_forward(self.ffn_norm(h))


class LocalOriginalPretrained42M(nn.Module):
    def __init__(self,p:Shape):
        super().__init__()
        self.p=p
        self.tok_embeddings=nn.Embedding(p.vocab,p.dim)
        self.layers=nn.ModuleList(
            [LlamaBlock(p.dim,p.hidden,p.heads) for _ in range(p.layers)])
        self.norm=RMS(p.dim)
        self.output=nn.Linear(p.dim,p.vocab,bias=False)
        self.output.weight=self.tok_embeddings.weight
        half=p.dim//p.heads//2
        freq=1./(10000.**(torch.arange(half,dtype=torch.float32)/half/2))
        at=torch.arange(p.max_ctx,dtype=torch.float32)[:,None]*freq[None,:]
        self.register_buffer("cos",at.cos(),persistent=False)
        self.register_buffer("sin",at.sin(),persistent=False)

    def forward(self,ids,*,hidden=False):
        if ids.ndim!=2 or not 0 < ids.shape[1]<=self.p.max_ctx:
            raise ValueError("Oscar independent pretrained 42M prompt too long")
        x=self.tok_embeddings(ids)
        hs=[x.float().detach()] if hidden else []
        cos=self.cos[:x.shape[1]]
        sin=self.sin[:x.shape[1]]
        for layer in self.layers:
            x=layer(x,cos,sin)
            if hidden:hs.append(x.float().detach())
        x=self.norm(x)
        logits=self.output(x)
        if hidden:hs.append(x.float().detach())
        return logits,hs


def normalize_checkpoint(obj:dict[str,torch.Tensor])->dict[str,torch.Tensor]:
    if not isinstance(obj,dict) or not obj:
        raise ValueError("Original independent pretrained 42M numeric state absent")
    keys=list(obj)
    if all(k.startswith("_orig_mod.") for k in keys):
        obj={k.removeprefix("_orig_mod."):v for k,v in obj.items()}
    if not all(isinstance(v,torch.Tensor) and v.dtype==torch.float32
               for v in obj.values()):
        raise ValueError("Oscar safe pretrained source must be all FP32 tensors")
    return obj


def shape_from_state(s):
    names=list(s)
    if "tok_embeddings.weight" not in s or "layers.0.feed_forward.w1.weight" not in s:
        raise ValueError("Not local Llama2.c 42M raw original teacher state")
    vocab,d=s["tok_embeddings.weight"].shape
    h=s["layers.0.feed_forward.w1.weight"].shape[0]
    indices={int(k.split(".")[1]) for k in names if k.startswith("layers.")}
    nl=max(indices)+1
    if sorted(indices)!=list(range(nl)):
        raise ValueError("Noncontiguous original 42M pretrained decoder layers")
    if (d!=512 or nl!=8 or vocab<10000 or vocab>64000 or h%8 or
        s["norm.weight"].shape!=(d,)):
        raise ValueError("Original pretrained 42M source wrong 8x512 geometry")
    return Shape(vocab=vocab,dim=d,layers=nl,heads=8,hidden=h)


def from_checkpoint(s):
    s=normalize_checkpoint(s)
    p=shape_from_state(s)
    model=LocalOriginalPretrained42M(p)
    keys=model.state_dict()
    missing=set(keys)-set(s)
    unexpected=set(s)-set(keys)
    if missing or unexpected:
        raise ValueError("Original 42M local full source mismatched keys: "+str(
            {"missing":sorted(missing),"unexpected":sorted(unexpected)}))
    with torch.no_grad():
        model.load_state_dict(s,strict=True)
    model.eval().requires_grad_(False)
    if model.output.weight is not model.tok_embeddings.weight:
        raise AssertionError("Original Llama2.c source tied token embedding and head broken")
    return model


def materialize_decoded_cb59(model,packed):
    """Diagnostic ONLY. This FP32 model is NEVER shipped in compressed package."""
    from direct_codebook_v59 import CB59,unpack5
    import numpy as np
    cb=CB59(packed,expected_keys=model.state_dict().keys())
    if cb.meta["original_unique_parameter_count"]!=sum(
            p.numel() for p in model.parameters()):
        raise AssertionError("Oscar independent pretrained 42M source count changed")
    local=LocalOriginalPretrained42M(model.p)
    state=local.state_dict()
    with torch.no_grad():
        for name,t in state.items():
            if name in cb.handles:
                h=cb.handles[name]
                m,n=h.shape
                code=unpack5(h.packed.tobytes(),m*(n//8))
                arr=h.centroids[code].reshape(m,n)*h.row_scales[:,None]
            else:
                arr=cb.raw[name]
            if arr.shape!=tuple(t.shape):
                raise AssertionError("Source original tensor decode shape mismatch: "+name)
            t.copy_(torch.from_numpy(np.array(arr,dtype=np.float32,copy=True)))
    local.eval().requires_grad_(False)
    return local
