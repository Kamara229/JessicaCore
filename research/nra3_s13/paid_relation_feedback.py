"""NRA3 S13: REAL causal relation -> grammar/semantic/dependency readouts.

S12c showed S11's trained selector is downstream of all linguistic heads:
on 16 NEW Russian heldout, UPOS/roles/UD were UNCHANGED by construction.
This stage therefore adds a small, EXPLICITLY PHYSICALLY CHARGED,
trainable, rank-8 relation-conditioned refinement to these *readouts*.

The original 43 full native-C CB59 5-bit operators, original S11 paid
39-FP16 selector, Calibri 43M and Green Tunnel are all frozen.
This isn't backprop to grammar/semantic layers or recurrent feedback
inside the six-cycle relation core: it is a new readout from that core.
Never call it an end-to-end learning/understanding breakthrough.

Strictly causal: at token i, only relation[i] and relation[j<=i]
influence the correction; no full-prefix summary is added to every
earlier token, no gold labels/UD heads/QA in inference. TRAIN-only use
of S9 72 labeled rows is handled by another module.

Physical FP16 contains all 1,266 parameters (no unbilled bias/scale):
128x8 projection (1,024), 8x26 output (208), 26 bias, and 8
dependency-pair coefficient scalars. Version + full SHA256 is counted.
Original unmodified CB59 source remains real C for all 43 matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.nra3_s0.neural_area import UPOS,SEMANTIC_ROLES

MAGIC=b"N3S13R8\x00"
VERSION=1
WIDTH=128
RANK=8
UPOS_DIM=len(UPOS)   # 17
ROLE_DIM=len(SEMANTIC_ROLES)   # 9
OUT=UPOS_DIM+ROLE_DIM   # 26
PARAMETERS=RANK*WIDTH+OUT*RANK+OUT+RANK  # 1266
HEADER=struct.Struct("<8sBHHHH") # Magic + version + in/rank/UPOS/ROLE
PHYSICAL_BYTES=HEADER.size+PARAMETERS*2+32 # 2581

def require(check:bool,why:str):
    if not bool(check):
        raise ValueError("NRA3 S13 causal physically-paid relation readout "+why)


@dataclass(frozen=True)
class CausalRefinedHeads:
    upos_logits:torch.Tensor       # [B,T,17]
    semantic_role_logits:torch.Tensor # [B,T,9]
    dependency_logits:torch.Tensor # [B,T,T]
    relation_rank:torch.Tensor     # [B,T,8]
    relation_delta:torch.Tensor    # [B,T,26]


class PaidRelationFeedback(nn.Module):
    """1,266 tiny FP16 parameters physically billed separately from CB59.

    During training FP16 rounding is straight-through; the exported
    inference module is reconstructed exclusively from physical FP16
    fields and has no trainable parameter. `output_gate=0` forbids
    influencing original Calibri response tokens.
    """
    def __init__(self):
        super().__init__()
        rng=torch.Generator(device="cpu").manual_seed(313000013)
        self.input_weight=nn.Parameter(torch.randn((RANK,WIDTH),generator=rng)*.035)
        self.output_weight=nn.Parameter(torch.zeros((OUT,RANK),dtype=torch.float32))
        self.output_bias=nn.Parameter(torch.zeros((OUT,),dtype=torch.float32))
        self.dependency_diagonal=nn.Parameter(torch.zeros((RANK,),dtype=torch.float32))
        self.output_gate=0.
        self.candidate_accepted=False

    def _snapped(self,p:torch.Tensor)->torch.Tensor:
        # Actual FP16 is the physical source. No unbilled train-time
        # state is permitted to participate in the native reload.
        return p+(p.half().float()-p).detach()

    def forward(self,relation:torch.Tensor,
                upos_logits:torch.Tensor,
                role_logits:torch.Tensor,
                dependency_logits:torch.Tensor,
                valid:torch.Tensor)->CausalRefinedHeads:
        if self.output_gate!=0 or self.candidate_accepted:
            raise RuntimeError("S13 feedback cannot activate Calibri generation")
        require(isinstance(relation,torch.Tensor) and relation.ndim==3
                and relation.dtype==torch.float32 and relation.device.type=="cpu",
                "relation CPU FP32 [B,T,128] required")
        b,t,w=relation.shape
        require(t>=1 and w==WIDTH
                and upos_logits.shape==(b,t,UPOS_DIM)
                and role_logits.shape==(b,t,ROLE_DIM)
                and dependency_logits.shape==(b,t,t)
                and valid.shape==(b,t) and valid.dtype==torch.bool
                and valid.device.type=="cpu" and bool(valid[:,0].all())
                and bool((valid.long().cummin(dim=1).values==valid.long()).all()),
                "full causal grammar+semantics 3way source ABI mismatch")
        require(all(bool(torch.isfinite(v).all())
                    for v in (relation,upos_logits,role_logits,dependency_logits)),
                "nonfinite original native C source/head logits")
        x=F.layer_norm(relation,(WIDTH,))
        z=torch.tanh(F.linear(x,self._snapped(self.input_weight)))
        z=z.masked_fill(~valid.unsqueeze(-1),0.)
        delta=F.linear(z,self._snapped(self.output_weight),
                       self._snapped(self.output_bias))
        delta=delta.masked_fill(~valid.unsqueeze(-1),0.)
        pos=upos_logits+delta[:,:,:UPOS_DIM]
        sem=role_logits+delta[:,:,UPOS_DIM:]
        d=self._snapped(self.dependency_diagonal)
        pair=(z*d)@z.transpose(1,2)/math.sqrt(RANK)
        # No future arc/padded key enters a dependency or score:
        # preserve original causal -1e4 cells EXACTLY.
        allow=torch.tril(torch.ones((t,t),dtype=torch.bool))[None]&valid[:,None,:]
        pair=pair.masked_fill(~allow,0.)
        dependency=dependency_logits+pair
        require(all(bool(torch.isfinite(v).all())
                    for v in (pos,sem,dependency,z,delta)),
                "nonfinite physically paid causal refinement")
        return CausalRefinedHeads(pos,sem,dependency,z,delta)

    def physical_bytes(self)->bytes:
        if self.output_gate!=0 or self.candidate_accepted:
            raise RuntimeError("S13 cannot export active model")
        fields=(self.input_weight.flatten(),self.output_weight.flatten(),
                self.output_bias.flatten(),self.dependency_diagonal.flatten())
        with torch.no_grad():
            full=torch.cat([f.detach().cpu() for f in fields]).contiguous()
            require(full.numel()==PARAMETERS and torch.isfinite(full).all(),
                    "missing/nonfinite paid parameter field")
            values=full.half().numpy().astype("<f2",copy=False).tobytes()
        raw=HEADER.pack(MAGIC,VERSION,WIDTH,RANK,UPOS_DIM,ROLE_DIM)+values
        out=raw+hashlib.sha256(raw).digest()
        require(len(out)==PHYSICAL_BYTES,
                "all FP16 parameters, version and physical SHA not charged")
        return out

    @classmethod
    def from_physical(cls,data:bytes)->"PaidRelationFeedback":
        require(type(data) is bytes and len(data)==PHYSICAL_BYTES,
                "complete exact physically paid source required")
        require(HEADER.unpack_from(data,0)==
                (MAGIC,VERSION,WIDTH,RANK,UPOS_DIM,ROLE_DIM),
                "source header architecture/version mismatch")
        require(hashlib.sha256(data[:-32]).digest()==data[-32:],
                "S13 physical digest invalid")
        vec=np.frombuffer(data[HEADER.size:-32],dtype="<f2").astype("float32")
        require(vec.size==PARAMETERS and np.isfinite(vec).all(),
                "source physical FP16 fields invalid")
        model=cls().eval()
        idx=0
        with torch.no_grad():
            for p in (model.input_weight,model.output_weight,
                      model.output_bias,model.dependency_diagonal):
                n=p.numel()
                p.copy_(torch.from_numpy(vec[idx:idx+n].reshape(tuple(p.shape)).copy()))
                idx+=n
        require(idx==PARAMETERS,
                "S13 physical source didn't load all fields")
        for p in model.parameters():p.requires_grad_(False)
        require(model.physical_bytes()==data,
                "exact physical source failed round-trip")
        return model
