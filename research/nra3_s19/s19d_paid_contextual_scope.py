"""S19d: RESEARCH-ONLY contextual event factuality via frozen S19b grammar.

Original S19c was a LOCAL per-word event classifier and failed to tag any
NEGATED event even in lexical TRAIN-internal CV. This candidate learns over
five neighbouring words in a full visible sentence, including RAW negation
and future surface cues. Never consumes gold UPOS, gold predicates, event
truth, dependency arcs, coreference or source text annotations as inputs.

Original 43M+Green Tunnel S95 hidden states cover the entire prompt and
the convolution looks both directions. Strict streaming causality is NOT
claimed, tested or assumed. Not a decoder integration or LIVE module.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load as safe_load

from research.nra3_s19.paid_grammar_parser import (
    CompactWordGrammar, RANK, INPUT_DIM, RAW_DIM,
    load_paid_fp16_bytes, paid_fp16_bytes, trainable_parameter_count,
)
from research.nra3_s19.s19c_paid_event_factuality import STATUS

KERNEL_WIDTH = 5
CONTEXT_WIDTH = RANK
CONTEXT_INPUT = RANK + RAW_DIM
EXTRA_PARAMETERS = (
    CONTEXT_WIDTH * CONTEXT_INPUT * KERNEL_WIDTH + CONTEXT_WIDTH +
    len(STATUS) * CONTEXT_WIDTH + len(STATUS)
)
MAGIC = b"N3S19D00"
VERSION = 1
HEADER = struct.Struct("<8sBIII")


def require(ok, reason):
    if not bool(ok):
        raise ValueError("S19d contextual shadow adapter: " + reason)


class PaidContextualEventScope(nn.Module):
    """Frozen grammar + learnable RAW/context scope; every FP16 weight billed."""

    def __init__(self, frozen_grammar: CompactWordGrammar):
        super().__init__()
        require(isinstance(frozen_grammar, CompactWordGrammar),
                "frozen learned-grammar ABI required")
        self.grammar = frozen_grammar.eval()
        self.grammar.requires_grad_(False)
        self.context = nn.Conv1d(
            CONTEXT_INPUT, CONTEXT_WIDTH, kernel_size=KERNEL_WIDTH,
            padding=KERNEL_WIDTH//2, bias=True)
        self.event_head = nn.Linear(CONTEXT_WIDTH, len(STATUS))
        self.output_gate = 0
        self.candidate_accepted = False
        self.activation_allowed = False

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        require(self.output_gate == 0 and not self.candidate_accepted
                and not self.activation_allowed,
                "forbidden decoder activation")
        require(x.ndim == 3 and x.shape[-1] == INPUT_DIM
                and x.dtype == torch.float32 and x.device.type == "cpu"
                and valid.shape == x.shape[:2] and valid.dtype == torch.bool
                and valid.device.type == "cpu" and bool(valid[:, 0].all())
                and bool((valid.long().cummin(1).values == valid.long()).all())
                and bool(torch.isfinite(x).all()),
                "real original43M hidden320 + plain-text raw18 word features only")
        # The frozen parser is used as a learned feature extractor;
        # NO gold labels / true predicate positions are ever read here.
        with torch.no_grad():
            gram = F.gelu(self.grammar.normalization(
                self.grammar.projection(x)))
        gram = gram.masked_fill(~valid.unsqueeze(-1), 0.)
        raw18 = x[..., -RAW_DIM:]
        require(bool(((raw18 == 0.) | (raw18 == 1.)).all()),
                "scope input raw18 must be original deterministic binary cues")
        raw18 = raw18.masked_fill(~valid.unsqueeze(-1), 0.)
        ctx = torch.cat((gram, raw18), dim=-1)
        contextual = F.gelu(self.context(ctx.transpose(1, 2)).transpose(1, 2))
        out = self.event_head(contextual)
        return out.masked_fill(~valid.unsqueeze(-1), -1e4)

    def head_physical(self) -> bytes:
        require(self.output_gate == 0 and not self.candidate_accepted
                and not self.activation_allowed,
                "forbidden output activation")
        values = (self.context.weight, self.context.bias,
                  self.event_head.weight, self.event_head.bias)
        with torch.no_grad():
            flat = torch.cat(tuple(z.detach().cpu().flatten() for z in values))
            require(flat.numel() == EXTRA_PARAMETERS
                    and bool(torch.isfinite(flat).all()),
                    "missing or nonfinite FP16 scope weights")
            return flat.half().numpy().astype("<f2", copy=False).tobytes()

    def physical_bytes(self) -> bytes:
        parent = paid_fp16_bytes(self.grammar)
        extra = self.head_physical()
        require(len(extra) == 2*EXTRA_PARAMETERS,
                "all contextual FP16 weights must be physically paid")
        header = HEADER.pack(
            MAGIC, VERSION, trainable_parameter_count(self.grammar),
            EXTRA_PARAMETERS, len(parent))
        body = header + parent + extra
        return body + hashlib.sha256(body).digest()

    @classmethod
    def from_physical(cls, blob: bytes) -> "PaidContextualEventScope":
        require(type(blob) is bytes and len(blob) >= HEADER.size+32,
                "physical grammar/context payload missing")
        magic,ver,grammar_params,extra_params,parent_len = HEADER.unpack_from(blob)
        require((magic,ver,grammar_params,extra_params) ==
                (MAGIC,VERSION,6528,EXTRA_PARAMETERS)
                and len(blob) == HEADER.size+parent_len+2*EXTRA_PARAMETERS+32
                and hashlib.sha256(blob[:-32]).digest()==blob[-32:],
                "physical ABI/header SHA drift")
        parent_blob=blob[HEADER.size:HEADER.size+parent_len]
        obj=cls(load_paid_fp16_bytes(parent_blob)).eval()
        tensors=safe_load(paid_fp16_bytes(obj.grammar))
        originals=safe_load(parent_blob)
        require(set(tensors)==set(originals) and
                all(torch.equal(tensors[k],originals[k]) for k in originals),
                "frozen parent grammar FP16 values drifted")
        raw=np.frombuffer(blob[HEADER.size+parent_len:-32],
                          dtype="<f2").astype("float32")
        require(raw.size==EXTRA_PARAMETERS and np.isfinite(raw).all(),
                "physical context classifier invalid")
        with torch.no_grad():
            position=0
            for t in (obj.context.weight,obj.context.bias,
                      obj.event_head.weight,obj.event_head.bias):
                count=t.numel()
                t.copy_(torch.from_numpy(raw[position:position+count]
                                          .reshape(t.shape).copy()))
                position+=count
        require(position==EXTRA_PARAMETERS and
                obj.head_physical()==blob[HEADER.size+parent_len:-32],
                "not all physically paid event/context values restored")
        obj.requires_grad_(False)
        return obj
