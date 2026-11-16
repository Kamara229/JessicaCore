"""NRA3 S18c4 research-only physically PAID raw-grammar rank8 role scorer.

Extend S16's genuine 128-dim frozen-native43 relation vectors with
18 raw-text morphological surface indicators at EACH endpoint.
The 18 input bits are deterministic indicators (not linguistic truth),
computed from raw text without any oracle POS, dependency or semantic role.
New physically charged projections: two 8x18 matrices = 288 FP16.
Original 2309 FP16 S16 rank8 fields remain accounted for separately:
TOTAL=2597 FP16 params, 5286 physical bytes inc both ABI headers/SHA.
No gate can connect this to LIVE; no claim of strict word-time causality,
as frozen S95 original activations use whole-sentence context.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.nra3_s16.paid_directed_valence import (
    PaidDirectedValence, PARAMETERS as BASE_PARAMETERS,
    PHYSICAL_BYTES as BASE_BYTES, RANK, WIDTH, ROLES,
)
from research.nra3_s18.s18c3_raw_russian_grammar_cues import DIM

MAGIC=b"N3S18C4\x00"
VERSION=1
HEADER=struct.Struct("<8sBHH")
EXTRA_PARAMETERS=2*RANK*DIM
PARAMETERS=BASE_PARAMETERS+EXTRA_PARAMETERS
PHYSICAL_BYTES=HEADER.size+BASE_BYTES+2*EXTRA_PARAMETERS+32


def require(ok,reason):
    if not bool(ok):
        raise ValueError("S18c4 paid raw-only Russian rank8 source: "+reason)


class PaidRawGrammarDirectedValence(nn.Module):
    """No oracle features or learned tokenizer, only raw bits + native relation."""
    def __init__(self):
        super().__init__()
        self.base=PaidDirectedValence().eval()
        generator=torch.Generator(device="cpu").manual_seed(318004)
        self.predicate_grammar=nn.Parameter(
            torch.randn((RANK,DIM),generator=generator)*0.035)
        self.argument_grammar=nn.Parameter(
            torch.randn((RANK,DIM),generator=generator)*0.035)
        self.output_gate=0
        self.candidate_accepted=False

    @staticmethod
    def _paid(value):
        return value+(value.half().float()-value).detach()

    def score_pairs(self,predicate,argument,
                    predicate_positions,argument_positions,current_positions,
                    raw_predicate,raw_argument):
        require(self.output_gate==0 and self.candidate_accepted is False
                and self.base.output_gate==0 and self.base.candidate_accepted is False,
                "gated decoder or candidate activation prohibited")
        require(predicate.ndim==argument.ndim==2
                and predicate.shape==argument.shape
                and predicate.shape[1]==WIDTH
                and predicate.dtype==argument.dtype==torch.float32
                and predicate.device.type==argument.device.type=="cpu"
                and torch.isfinite(predicate).all() and torch.isfinite(argument).all(),
                "only actual CPU native-C 128-dim vectors expected")
        n=predicate.shape[0]
        require(n>=1
                and raw_predicate.shape==raw_argument.shape==(n,DIM)
                and raw_predicate.dtype==raw_argument.dtype==torch.float32
                and raw_predicate.device.type==raw_argument.device.type=="cpu"
                and bool(((raw_predicate==0.)|(raw_predicate==1.)).all())
                and bool(((raw_argument==0.)|(raw_argument==1.)).all()),
                "only immutable RAW TEXT-derived 18-bit word features accepted")
        require(all(k.shape==(n,) and k.dtype==torch.long
                    and k.device.type=="cpu" for k in (
                        predicate_positions,argument_positions,current_positions))
                and bool((predicate_positions>=0).all())
                and bool((argument_positions>=0).all())
                and bool((predicate_positions!=argument_positions).all())
                and bool((torch.maximum(predicate_positions,argument_positions)
                           ==current_positions).all()),
                "pred-arg links only after both real wordpieces are present")
        base=self.base
        pred=F.linear(F.layer_norm(predicate,(WIDTH,)),
                      base._paid(base.predicate_weight))
        arg=F.linear(F.layer_norm(argument,(WIDTH,)),
                     base._paid(base.argument_weight))
        pred=pred+F.linear(raw_predicate,self._paid(self.predicate_grammar))
        arg=arg+F.linear(raw_argument,self._paid(self.argument_grammar))
        role_scores=torch.einsum(
            "ni,rij,nj->nr",pred,base._paid(base.role_bilinear),arg)
        bias=base._paid(base.bias)
        logits=torch.cat((bias[0].expand(n,1),role_scores+bias[1:]),dim=-1)
        require(logits.shape==(n,len(ROLES)) and bool(torch.isfinite(logits).all()),
                "new paid rank8 raw-input relational source nonfinite")
        return logits

    def physical_bytes(self)->bytes:
        require(self.output_gate==0 and self.candidate_accepted is False,
                "cannot serialize active output gate")
        base=self.base.physical_bytes()
        with torch.no_grad():
            vectors=torch.cat((self.predicate_grammar.detach().flatten(),
                               self.argument_grammar.detach().flatten()))
            require(vectors.numel()==EXTRA_PARAMETERS and
                    bool(torch.isfinite(vectors).all()),
                    "288 extra physically charged FP16 projection fields missing")
            data=vectors.cpu().half().numpy().astype("<f2",copy=False).tobytes()
        header=HEADER.pack(MAGIC,VERSION,DIM,EXTRA_PARAMETERS)
        raw=header+base+data
        blob=raw+hashlib.sha256(raw).digest()
        require(len(blob)==PHYSICAL_BYTES==5286,
                "physical FP16 source missing ABI/full payload or SHA")
        return blob

    @classmethod
    def from_physical(cls,blob:bytes)->"PaidRawGrammarDirectedValence":
        require(type(blob) is bytes and len(blob)==PHYSICAL_BYTES,
                "complete real 2597 paid FP16 parameter source required")
        require(HEADER.unpack_from(blob,0)==(MAGIC,VERSION,DIM,EXTRA_PARAMETERS),
                "architecture/raw grammar dimension/version drift")
        require(hashlib.sha256(blob[:-32]).digest()==blob[-32:],
                "full grammar FP16 model physical SHA checksum failed")
        start=HEADER.size
        original=blob[start:start+BASE_BYTES]
        new_bytes=blob[start+BASE_BYTES:-32]
        base=PaidDirectedValence.from_physical(original)
        require(len(new_bytes)==EXTRA_PARAMETERS*2,
                "new 288 FP16 params absent/unpaid")
        values=np.frombuffer(new_bytes,dtype="<f2").astype("float32")
        require(len(values)==EXTRA_PARAMETERS and np.isfinite(values).all(),
                "invalid grammar physical extra values")
        out=cls().eval()
        out.base=base
        with torch.no_grad():
            i=0
            for tensor in (out.predicate_grammar,out.argument_grammar):
                count=tensor.numel()
                tensor.copy_(torch.from_numpy(values[i:i+count].reshape(
                    tensor.shape).copy()))
                i+=count
        require(i==EXTRA_PARAMETERS and out.physical_bytes()==blob,
                "inexact physical reload of every extra FP16 field")
        for p in out.parameters():
            p.requires_grad_(False)
        return out
