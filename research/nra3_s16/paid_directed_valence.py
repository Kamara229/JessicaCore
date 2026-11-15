"""NRA2 3.0 S16: physically PAID directional predicate-argument readout.

Research-only, NO training/scoring on consumed EVAL24/S12c EVAL16.
The earlier S14 rank8 changed semantic ROLE scores on internal TRAIN
but caused ZERO net improvement on genuine S15c one-time EVAL24.
S16 therefore predicts an explicit directed ROLE-bearing link between
predicate and argument, not merely each token's isolated semantic class.

Two causal orientations:
- predicate appears NOW, argument appeared EARLIER:
  pred_now[i,j,r] = score(predicate=i, argument=j, role=r);
- argument appears NOW, predicate appeared EARLIER:
  arg_now[i,j,r] = score(predicate=j, argument=i, role=r).
At i no tokens >i can influence any link. The semantic relation in
the output is always directed PREDICATE -> ARGUMENT, despite word order.
Class 0 is NONE, others AGENT, PATIENT, RECIPIENT, SOURCE. An
explicit negation/event-truth or coreference decision is NOT implemented.

The original 43 packed native-C CB59 5-bit matrices, S11 39-FP16
router, real frozen Calibri 43M, Green Tunnel are never modified.
New trainable fields are separately physically billed:
predicate 128x8 + argument 128x8 = 2048 FP16; four 8x8
role bilinear matrices = 256 FP16; five biases = 5 FP16;
2309 parameters + header and SHA256 -> 4665 physical bytes.
No unbilled 2D weights are hidden in inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROLES=("NONE","AGENT","PATIENT","RECIPIENT","SOURCE")
MAGIC=b"N3S16D8\x00"
VERSION=1
WIDTH=128
RANK=8
PARAMETERS=2*WIDTH*RANK+(len(ROLES)-1)*RANK*RANK+len(ROLES)
HEADER=struct.Struct("<8sBHHH")
PHYSICAL_BYTES=HEADER.size+2*PARAMETERS+32


def guard(ok,why):
    if not bool(ok):
        raise ValueError("NRA3 S16 directed predicate→argument source "+why)


@dataclass(frozen=True)
class DirectedCausalLinks:
    predicate_now:torch.Tensor # [B,T,T,5], current token predicate, prior argument
    argument_now:torch.Tensor  # [B,T,T,5], current token argument, prior predicate
    allowed:torch.Tensor       # [B,T,T], strict j<i and no padded keys
    predicate8:torch.Tensor    # [B,T,8]
    argument8:torch.Tensor     # [B,T,8]


class PaidDirectedValence(nn.Module):
    """Explicitly billed rank8 predicate→argument relation scores, shadow ONLY.

    Parameter source for scoring is FP16 via straight-through rounding.
    Exact physical FP16 bytes are reconstructed on load; weights can be
    optimized on existing S9 TRAIN72 only in a separate S16b stage.
    """
    def __init__(self):
        super().__init__()
        rng=torch.Generator(device="cpu").manual_seed(316001)
        self.predicate_weight=nn.Parameter(torch.randn((RANK,WIDTH),generator=rng)*.035)
        self.argument_weight=nn.Parameter(torch.randn((RANK,WIDTH),generator=rng)*.035)
        self.role_bilinear=nn.Parameter(torch.zeros(len(ROLES)-1,RANK,RANK))
        self.bias=nn.Parameter(torch.tensor([.25,0.,0.,0.,0.]))
        self.output_gate=0
        self.candidate_accepted=False

    @staticmethod
    def _paid(t):
        return t+(t.half().float()-t).detach()

    def forward(self,relation:torch.Tensor,valid:torch.Tensor)->DirectedCausalLinks:
        if self.output_gate!=0 or self.candidate_accepted:
            raise RuntimeError("S16 read-only: never connect to live decoder")
        guard(isinstance(relation,torch.Tensor) and relation.ndim==3 and
              relation.device.type=="cpu" and relation.dtype==torch.float32
              and bool(torch.isfinite(relation).all()),
              "actual original native C relation CPU FP32 [B,T,128] required")
        b,t,w=relation.shape
        guard(w==WIDTH and t>=2 and valid.shape==(b,t) and
              valid.device.type=="cpu" and valid.dtype==torch.bool
              and bool(valid[:,0].all()) and
              bool((valid.long().cummin(dim=1).values==valid.long()).all()),
              "right-padded source cannot reveal future or padded token")
        pred=F.linear(F.layer_norm(relation,(WIDTH,)),
                      self._paid(self.predicate_weight))
        arg=F.linear(F.layer_norm(relation,(WIDTH,)),
                     self._paid(self.argument_weight))
        pred=pred.masked_fill(~valid.unsqueeze(-1),0.)
        arg=arg.masked_fill(~valid.unsqueeze(-1),0.)
        w4=self._paid(self.role_bilinear)
        # All hypothesis matrices/biases physically present. Scores are
        # oriented PREDICATE -> ARGUMENT regardless of surface word order.
        now_pred=torch.einsum("bti,rij,bsj->btsr",pred,w4,arg)
        now_arg=torch.einsum("bsi,rij,btj->btsr",pred,w4,arg)
        bs=self._paid(self.bias)
        none=bs[0].expand(b,t,t,1)
        logits_pred=torch.cat((none,now_pred+bs[1:]),dim=-1)
        logits_arg=torch.cat((none,now_arg+bs[1:]),dim=-1)
        allow=torch.tril(torch.ones((t,t),dtype=torch.bool),diagonal=-1)[None]
        allow=allow&valid[:,:,None]&valid[:,None,:]
        logits_pred=logits_pred.masked_fill(~allow.unsqueeze(-1),-1e4)
        logits_arg=logits_arg.masked_fill(~allow.unsqueeze(-1),-1e4)
        guard(logits_pred.shape==logits_arg.shape==(b,t,t,len(ROLES))
              and bool(torch.isfinite(logits_pred).all())
              and bool(torch.isfinite(logits_arg).all()),
              "invalid paid direction/role link score")
        return DirectedCausalLinks(logits_pred,logits_arg,allow,pred,arg)

    def physical_bytes(self)->bytes:
        if self.output_gate!=0 or self.candidate_accepted:
            raise RuntimeError("S16 unaccepted live output gate must be zero")
        with torch.no_grad():
            vec=torch.cat([x.detach().cpu().flatten() for x in (
                self.predicate_weight,self.argument_weight,
                self.role_bilinear,self.bias)])
            guard(vec.numel()==PARAMETERS and bool(torch.isfinite(vec).all()),
                  "unpaid or nonfinite source parameter")
            physical=vec.half().numpy().astype("<f2",copy=False).tobytes()
        header=HEADER.pack(MAGIC,VERSION,WIDTH,RANK,len(ROLES))
        blob=header+physical
        output=blob+hashlib.sha256(blob).digest()
        guard(len(output)==PHYSICAL_BYTES,
              "missing charged FP16 weights/header/full SHA256")
        return output

    @classmethod
    def from_physical(cls,data:bytes)->"PaidDirectedValence":
        guard(type(data) is bytes and len(data)==PHYSICAL_BYTES,
              "complete raw physically paid rank8 source required")
        guard(HEADER.unpack_from(data,0)==
              (MAGIC,VERSION,WIDTH,RANK,len(ROLES)),
              "wrong paid architecture header/version")
        guard(hashlib.sha256(data[:-32]).digest()==data[-32:],
              "original SHA256 full paid source corrupted")
        val=np.frombuffer(data[HEADER.size:-32],dtype="<f2").astype("float32")
        guard(len(val)==PARAMETERS and np.isfinite(val).all(),
              "invalid paid source physical FP16 values")
        model=cls().eval()
        i=0
        with torch.no_grad():
            for parameter in (model.predicate_weight,model.argument_weight,
                              model.role_bilinear,model.bias):
                n=parameter.numel()
                parameter.copy_(torch.from_numpy(
                    val[i:i+n].reshape(parameter.shape).copy()))
                i+=n
        guard(i==PARAMETERS and model.physical_bytes()==data,
              "physical tensor roundtrip lost a paid parameter")
        for p in model.parameters():
            p.requires_grad_(False)
        return model
