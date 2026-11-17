"""S19g research-only fully learned, gold-blind, typed event-graph head.

Inputs are genuine original frozen Calibri43M+Green Tunnel hidden320 + raw18;
S19d frozen learned grammar and contextual scope are physically paid as
part of the model. No human annotation, event anchor, gold POS, predicted
gold edge or hand-written parse enters the neural forward function.

Five distinct predicted outputs: event factuality on every word; directed
semantic role on every ordered word pair; participant polarity/rejection
on every ordered pair; coreference mention->antecedent; event-event link.
All outputs are disabled (output_gate=0); no actual NRA2/decoder integration.
Only TRAIN9 supervised optimization is supported here. NO independent EVAL.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19f_real43m_train9_graph_targets import (
    STATUS, ROLE, EVENT_RELATION,
)

PAID_PARENT_FP16 = 9332
NODE_WIDTH = 24
NODE_INPUT = 16+16+18
ARG_STATUS = ("NONE","POSITIVE","REJECTED")
COREF_STATUS = ("NONE","LINK")
MAGIC = b"N3S19G00"
HEADER = struct.Struct("<8sBIII")
VERSION = 1


def require(ok, reason):
    if not bool(ok):
        raise ValueError("S19g learned typed event graph shadow: "+reason)


@dataclass(frozen=True)
class GraphLogits:
    event_status: torch.Tensor     # [B,W,6] incl NONE
    role: torch.Tensor             # [B,W,W,9] incl NONE
    argument_polarity: torch.Tensor # [B,W,W,3] incl NONE
    coreference: torch.Tensor      # [B,W,W,2] incl NONE
    event_relation: torch.Tensor   # [B,W,W,5] incl NONE


class PaidSemanticEventGraph(nn.Module):
    def __init__(self, frozen_parent: PaidContextualEventScope):
        super().__init__()
        require(isinstance(frozen_parent,PaidContextualEventScope),
                "genuine prior contextual-grammar shadow parent required")
        require(frozen_parent.output_gate == 0
                and not frozen_parent.candidate_accepted
                and not frozen_parent.activation_allowed,
                "parent model cannot be active")
        self.parent=frozen_parent.eval()
        self.parent.requires_grad_(False)
        self.node=nn.Linear(NODE_INPUT,NODE_WIDTH)
        self.event=nn.Linear(NODE_WIDTH,len(STATUS))
        self.query=nn.Linear(NODE_WIDTH,NODE_WIDTH)
        self.key=nn.Linear(NODE_WIDTH,NODE_WIDTH)
        self.role=nn.Linear(NODE_WIDTH,len(ROLE))
        self.argument_polarity=nn.Linear(NODE_WIDTH,len(ARG_STATUS))
        self.coreference=nn.Linear(NODE_WIDTH,len(COREF_STATUS))
        self.event_relation=nn.Linear(NODE_WIDTH,len(EVENT_RELATION))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False

    def semantic_params(self):
        return (p for name,p in self.named_parameters()
                if not name.startswith("parent."))

    def forward(self,x:torch.Tensor,valid:torch.Tensor)->GraphLogits:
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and self.parent.output_gate==0
                and not self.parent.candidate_accepted
                and not self.parent.activation_allowed,
                "semantic output gate must remain disabled")
        # Parent validates original CPU 338-D gold-blind model input, including
        # original raw-only 18 binary surface cues, right-padding, finiteness.
        with torch.no_grad():
            check=self.parent(x,valid)
            require(check.shape==(*valid.shape,4),
                    "source parent event logit ABI")
            g=F.gelu(self.parent.grammar.normalization(
                self.parent.grammar.projection(x)))
            g=g.masked_fill(~valid.unsqueeze(-1),0.)
            raw=x[...,-18:].masked_fill(~valid.unsqueeze(-1),0.)
            context=F.gelu(self.parent.context(
                torch.cat((g,raw),dim=-1).transpose(1,2)).transpose(1,2))
            context=context.masked_fill(~valid.unsqueeze(-1),0.)
        node=F.gelu(self.node(torch.cat((g,context,raw),dim=-1)))
        node=node.masked_fill(~valid.unsqueeze(-1),0.)
        event=self.event(node)
        # Predicate and argument are NOT supplied by a human oracle:
        # every eligible ordered (i,j) pair gets every head's prediction.
        pair=torch.tanh(self.query(node).unsqueeze(2)
                        +self.key(node).unsqueeze(1))
        pair_mask=(valid.unsqueeze(2)&valid.unsqueeze(1))
        n=valid.shape[1]
        pair_mask=pair_mask & ~torch.eye(n,dtype=torch.bool).unsqueeze(0)
        def pairs(layer):
            return layer(pair).masked_fill(~pair_mask.unsqueeze(-1),-1e4)
        return GraphLogits(
            event.masked_fill(~valid.unsqueeze(-1),-1e4),
            pairs(self.role),pairs(self.argument_polarity),
            pairs(self.coreference),pairs(self.event_relation),
        )

    def head_physical(self)->bytes:
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed,"semantic model cannot activate")
        with torch.no_grad():
            tensors=[p.detach().cpu().contiguous()
                     for p in self.semantic_params()]
            count=sum(p.numel() for p in tensors)
            require(count==semantic_parameter_count(self)
                    and all(bool(torch.isfinite(t).all()) for t in tensors),
                    "unpaid/nonfinite typed head weights")
            arr=torch.cat([v.flatten() for v in tensors]).half().numpy()
            return arr.astype("<f2",copy=False).tobytes()

    def physical_bytes(self)->bytes:
        parent=self.parent.physical_bytes()
        new=self.head_physical()
        n=semantic_parameter_count(self)
        require(len(new)==2*n,"complete paid graph parameter serialization")
        body=HEADER.pack(MAGIC,VERSION,PAID_PARENT_FP16,n,len(parent))+parent+new
        return body+hashlib.sha256(body).digest()

    @classmethod
    def from_physical(cls,blob:bytes)->"PaidSemanticEventGraph":
        require(type(blob) is bytes and len(blob)>=HEADER.size+32,
                "typed graph physical blob missing")
        magic,ver,pcount,ncount,parent_len=HEADER.unpack_from(blob)
        require((magic,ver,pcount)==(MAGIC,VERSION,PAID_PARENT_FP16)
                and parent_len>=100 and ncount>0
                and len(blob)==HEADER.size+parent_len+2*ncount+32
                and hashlib.sha256(blob[:-32]).digest()==blob[-32:],
                "typed graph payload physical header/SHA invalid")
        parent_blob=blob[HEADER.size:HEADER.size+parent_len]
        parent=PaidContextualEventScope.from_physical(parent_blob)
        obj=cls(parent).eval()
        require(semantic_parameter_count(obj)==ncount,"typed graph ABI parameter count drift")
        payload=blob[HEADER.size+parent_len:-32]
        arr=np.frombuffer(payload,dtype="<f2").astype("float32")
        require(arr.size==ncount and np.isfinite(arr).all(),
                "typed graph paid semantic weights corrupt")
        with torch.no_grad():
            offset=0
            for t in obj.semantic_params():
                count=t.numel()
                t.copy_(torch.from_numpy(arr[offset:offset+count]
                                         .reshape(t.shape).copy()))
                offset+=count
            require(offset==ncount,"incomplete physical semantic weights")
        obj.requires_grad_(False)
        require(obj.head_physical()==payload,
                "typed graph FP16 physical semantic values fail exact roundtrip")
        return obj


def semantic_parameter_count(model:PaidSemanticEventGraph)->int:
    return sum(t.numel() for t in model.semantic_params())


def full_parameter_count(model:PaidSemanticEventGraph)->int:
    return sum(t.numel() for t in model.parameters())
