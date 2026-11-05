"""Jointly trainable *shared* two-dimensional residual coordinates.

All original CB59 five-bit codebooks/indices and row scales are frozen.
For every trained original operator W (m,n):
  W_eff x = CB59(W) x + (A[m,2] @ B_shared[2,n]) x.
A is physically serialized as INT8 per row/channel and B is shared
across ALL layers with the same input width (576 or 1536), stored FP16.
The same tied A and B correct both input embedding and output head.
Only A and B train; quantization STE is active *during* fitting.
"""
import torch
from torch import nn
from torch.nn import functional as F
from v176.operators import build as build_frozen_source, clear
from . import protocol as p

def ste_half(t):
    rounded=t.to(torch.float16).to(torch.float32)
    return t+(rounded-t).detach()

def ste_int8(t):
    full=t.detach().abs().amax().clamp_min(1.e-8)
    step=full/p.INT8_MAX
    scaled=(t/step).round().clamp(-p.INT8_MAX,p.INT8_MAX)*step
    return t+(scaled-t).detach()

class AdditiveLinear(nn.Module):
    def __init__(self,source,alpha,beta):
        super().__init__()
        self.source=source
        self.alpha=alpha
        self.beta=beta
        self.in_features=source.shape[1]
        self.out_features=source.shape[0]
        self.bias=None
    def forward(self,x):
        base=F.linear(x,self.source())
        low=F.linear(F.linear(x,ste_half(self.beta)),ste_int8(self.alpha))
        return base+low

class AdditiveEmbedding(nn.Module):
    def __init__(self,source,alpha,beta):
        super().__init__()
        self.source=source
        self.alpha=alpha
        self.beta=beta
        self.num_embeddings=source.shape[0]
        self.embedding_dim=source.shape[1]
    def forward(self,ids):
        return F.embedding(ids,self.source()) + F.linear(
            F.embedding(ids,ste_int8(self.alpha)),
            ste_half(self.beta).T)

def build(archive):
    torch.manual_seed(p.SEED)
    model,sources=build_frozen_source(archive)
    # Freeze every original 1D norm as well as the 211 source codebooks.
    # Only the freshly installed compact rank-two factors are trainable.
    model.requires_grad_(False)
    if len(sources)!=211:raise AssertionError("211 original source operators required")
    for src in sources.values():src.requires_grad_(False)
    shapes={source.shape[1] for source in sources.values()}
    if shapes!={576,1536}:
        raise AssertionError("Unexpected trained transformer source widths: "+str(shapes))
    betas={n:nn.Parameter(
        torch.randn((p.RANK,n),dtype=torch.float32)*(1/(n**.5)))
        for n in sorted(shapes)}
    alphas={name:nn.Parameter(torch.zeros((s.shape[0],p.RANK),dtype=torch.float32))
            for name,s in sources.items()}
    aliases=archive.meta["source_alias_keys"]
    for full_path in archive.handles:
        canon=aliases.get(full_path,full_path)
        source=sources[canon]
        if canon not in alphas:raise AssertionError("Unknown tied pretrained tensor")
        path=full_path.removesuffix(".weight")
        parent,_,attr=path.rpartition(".")
        owner=model.get_submodule(parent) if parent else model
        existing=getattr(owner,attr)
        if path=="model.embed_tokens":
            module=AdditiveEmbedding(source,alphas[canon],betas[source.shape[1]])
        else:
            module=AdditiveLinear(source,alphas[canon],betas[source.shape[1]])
        if not hasattr(existing,"source") or existing.source is not source:
            raise AssertionError("Unexpected frozen pretrained operator "+full_path)
        setattr(owner,attr,module)
    if model.model.embed_tokens.alpha is not model.lm_head.alpha:
        raise AssertionError("Tied source rank-2 A must be one stored object")
    if model.model.embed_tokens.beta is not model.lm_head.beta:
        raise AssertionError("Tied source rank-2 B must be one stored object")
    intended={id(a) for a in alphas.values()}|{id(b) for b in betas.values()}
    actual={id(x) for x in model.parameters() if x.requires_grad}
    if actual!=intended:
        raise AssertionError("Unexpected trained dense source param or missing shared factor")
    if sum(t.numel() for t in alphas.values())!=409344:
        raise AssertionError("Unexpected original 211 matrix row count")
    if sum(t.numel() for t in betas.values())!=4224:
        raise AssertionError("Unexpected shared B byte count")
    return model,sources,alphas,betas

def clear_frozen(sources):
    clear(sources)
