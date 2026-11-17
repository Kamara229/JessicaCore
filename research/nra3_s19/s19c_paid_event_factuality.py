"""S19c: trained factuality head on frozen S19b grammar embeddings.

This is a RESEARCH-ONLY semantic adapter SHADOW; not standalone language
understanding or decoder output. Input is ONLY the physically obtained
original Calibri43M+Green Tunnel hidden320 and raw18 surface indicators.
S19b grammar features are PREDICTED, never gold POS/UD/roles. Label NONE
covers all non-predicate words; the event classes are ASSERTED, NEGATED and
FUTURE_UNREALIZED on the original 96 TRAIN sentence source.

No conditioned truth is available in TRAIN96; CONDITION is deliberately NOT
a trained output label until a separate TRAIN-only source is captured.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.nra3_s19.paid_grammar_parser import (
    CompactWordGrammar, RANK, INPUT_DIM, load_paid_fp16_bytes,
    paid_fp16_bytes, trainable_parameter_count,
)

STATUS = ("NONE", "ASSERTED", "NEGATED", "FUTURE_UNREALIZED")
HEAD_PARAMETERS = RANK * len(STATUS) + len(STATUS)
MAGIC = b"N3S19C00"
HEADER = struct.Struct("<8sBIII")
VERSION = 1


def require(ok, reason):
    if not bool(ok):
        raise ValueError("S19c event SHADOW: " + reason)


class PaidEventFactuality(nn.Module):
    """Gold-blind event tags for all words, explicitly paid in FP16."""

    def __init__(self, grammar: CompactWordGrammar):
        super().__init__()
        require(isinstance(grammar, CompactWordGrammar), "parent grammar ABI")
        self.grammar = grammar.eval()
        self.grammar.requires_grad_(False)
        self.event_head = nn.Linear(RANK, len(STATUS))
        self.output_gate = 0
        self.candidate_accepted = False
        self.activation_allowed = False

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        require(self.output_gate == 0 and not self.candidate_accepted
                and not self.activation_allowed, "LIVE decoder hook prohibited")
        require(x.ndim == 3 and x.shape[-1] == INPUT_DIM
                and x.dtype == torch.float32 and x.device.type == "cpu"
                and valid.shape == x.shape[:2] and valid.dtype == torch.bool
                and valid.device.type == "cpu" and bool(valid[:,0].all())
                and bool((valid.long().cummin(1).values == valid.long()).all())
                and bool(torch.isfinite(x).all()),
                "only original43M hidden320 + raw18 valid CPU word features")
        with torch.no_grad():
            # Identical physical FP16-rounded frozen grammar computation.
            z = F.gelu(self.grammar.normalization(self.grammar.projection(x)))
            z = z.masked_fill(~valid.unsqueeze(-1), 0.)
        logits = self.event_head(z)
        return logits.masked_fill(~valid.unsqueeze(-1), -1e4)

    def head_physical(self):
        require(self.output_gate == 0 and not self.candidate_accepted
                and not self.activation_allowed, "activation cannot serialize")
        with torch.no_grad():
            flat = torch.cat((self.event_head.weight.detach().cpu().flatten(),
                              self.event_head.bias.detach().cpu().flatten()))
            require(flat.numel() == HEAD_PARAMETERS
                    and bool(torch.isfinite(flat).all()), "unpaid or nonfinite head")
            return flat.half().numpy().astype("<f2", copy=False).tobytes()

    def physical_bytes(self):
        """Joint immutable grammar (6528 FP16) + event (68 FP16), SHA sealed."""
        grammar_payload = paid_fp16_bytes(self.grammar)
        new_payload = self.head_physical()
        require(len(new_payload) == 2 * HEAD_PARAMETERS,
                "event head must contain every physically paid parameter")
        header = HEADER.pack(MAGIC, VERSION,
                             trainable_parameter_count(self.grammar),
                             HEAD_PARAMETERS, len(grammar_payload))
        body = header + grammar_payload + new_payload
        return body + hashlib.sha256(body).digest()

    @classmethod
    def from_physical(cls, raw: bytes):
        require(isinstance(raw, bytes) and len(raw) >= HEADER.size + 32,
                "joint physical source missing")
        magic, version, grammar_p, event_p, grammar_bytes = HEADER.unpack_from(raw)
        require((magic,version,grammar_p,event_p) ==
                (MAGIC,VERSION,6528,HEAD_PARAMETERS)
                and len(raw) == HEADER.size + grammar_bytes + 2*event_p+32
                and hashlib.sha256(raw[:-32]).digest() == raw[-32:],
                "physical joint payload header or SHA invalid")
        start = HEADER.size
        source_grammar = raw[start:start+grammar_bytes]
        loaded = load_paid_fp16_bytes(source_grammar)
        obj = cls(loaded).eval()
        values = np.frombuffer(raw[start+grammar_bytes:-32],dtype="<f2").astype("float32")
        require(values.size == HEAD_PARAMETERS and np.isfinite(values).all(),
                "corrupt paid event weights")
        with torch.no_grad():
            w = obj.event_head.weight
            obj.event_head.weight.copy_(torch.from_numpy(values[:w.numel()].reshape(w.shape).copy()))
            obj.event_head.bias.copy_(torch.from_numpy(values[w.numel():].copy()))
        obj.requires_grad_(False)
        # The FP16 tensors and SHA must be identical, not just model predictions.
        # safetensors header byte ordering is stable for a fixed grammar source.
        from safetensors.torch import load as safe_load
        current = safe_load(paid_fp16_bytes(obj.grammar))
        before = safe_load(source_grammar)
        require(set(current) == set(before) and
                all(torch.equal(current[k],before[k]) for k in before) and
                obj.head_physical() == raw[start+grammar_bytes:-32],
                "physically paid event + grammar tensor round-trip changed")
        return obj


def extra_fp16_weight_count():
    return HEAD_PARAMETERS
