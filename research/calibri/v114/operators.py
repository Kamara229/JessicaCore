"""Trained FROM SCRATCH compact functional operators, not distilled weights.

The shared block actually executes U @ (V @ x) and per-layer gain/bias.
An optional rank-2 correction acts on the same layer. No full dense W
is ever assembled at inference in shared modes.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .protocol import TOKENS,D,DEPTH,RANK,ADAPTER_RANK

ROLES=("expand","contract")

class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.ModuleDict({r:nn.Linear(D,D) for r in ROLES})
        self.norm=nn.LayerNorm(D)

    def forward(self,x):
        intermediate=torch.tanh(self.proj["expand"](x))
        return self.norm(x+self.proj["contract"](intermediate))

class SharedBook(nn.Module):
    def __init__(self,adapter):
        super().__init__()
        self.adapter=adapter
        self.u=nn.ParameterDict({
            r:nn.Parameter(torch.randn(D,RANK)*.12) for r in ROLES})
        self.v=nn.ParameterDict({
            r:nn.Parameter(torch.randn(RANK,D)*.12) for r in ROLES})

    def role(self,name,x,block):
        result=F.linear(F.linear(x,self.v[name]),self.u[name])
        result=result*block.gain[name]+block.bias[name]
        if self.adapter:
            result+=F.linear(
                F.linear(x,block.low_v[name]),block.low_u[name])
        return result

class SharedBlock(nn.Module):
    def __init__(self,adapter):
        super().__init__()
        self.gain=nn.ParameterDict({
            r:nn.Parameter(torch.ones(D)) for r in ROLES})
        self.bias=nn.ParameterDict({
            r:nn.Parameter(torch.zeros(D)) for r in ROLES})
        self.norm=nn.LayerNorm(D)
        if adapter:
            self.low_u=nn.ParameterDict({
                r:nn.Parameter(torch.randn(D,ADAPTER_RANK)*.015)
                for r in ROLES})
            self.low_v=nn.ParameterDict({
                r:nn.Parameter(torch.randn(ADAPTER_RANK,D)*.015)
                for r in ROLES})

    def forward(self,x,book):
        hidden=torch.tanh(book.role("expand",x,self))
        return self.norm(x+book.role("contract",hidden,self))

class TinyLanguageModel(nn.Module):
    """Token-by-token true autoregressive decoder, trained on original answers."""
    def __init__(self,mode):
        super().__init__()
        if mode not in ("dense","shared","shared_adapter"):
            raise ValueError(mode)
        self.mode=mode
        vocab=len(TOKENS)
        self.enc=nn.Embedding(vocab,D)
        self.dec=nn.Embedding(vocab,D)
        self.pos=nn.Embedding(6,D)
        if mode=="dense":
            self.layers=nn.ModuleList(DenseBlock() for _ in range(DEPTH))
        else:
            self.book=SharedBook(mode=="shared_adapter")
            self.layers=nn.ModuleList(
                SharedBlock(mode=="shared_adapter") for _ in range(DEPTH))
        self.output=nn.Linear(D,vocab)

    def forward(self,context,previous):
        h=self.enc(context).mean(dim=1,keepdim=True)+self.dec(previous)
        positions=torch.arange(previous.shape[1],device=previous.device)
        h=h+self.pos(positions)[None,:,:]
        for layer in self.layers:
            h=layer(h) if self.mode=="dense" else layer(h,self.book)
        return self.output(h)

    @torch.inference_mode()
    def generate(self,context,max_tokens=6):
        previously=[2]
        result=[]
        for _ in range(max_tokens):
            x=torch.tensor([previously],device=context.device)
            scores=self(context[None,:],x)[0,-1]
            token=int(scores.argmax())
            result.append(token)
            if token==3:
                break
            previously.append(token)
        return result
