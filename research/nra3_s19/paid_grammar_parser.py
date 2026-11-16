"""S19b compact multi-head Russian grammar learner: research/CPU/TRAIN ONLY.

Inputs: genuine frozen Calibri43M+GT 320-dim CPU hidden vectors at raw-word
anchors + 18 surface features computed exclusively from RAW text. Targets:
human-annotated S18 TRAIN96 UPOS, UD dependency relation, word's head,
and predicate-VERB flag. No UPOS/UD/roles/event gold ever enters inference.
This head does NOT assert strict word-incremental causality: original S95
frozen real 43M features were computed with full-sentence context.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load, save

from research.nra3_s0.neural_area import UPOS
from research.nra3_s18.s18c3_raw_russian_grammar_cues import DIM as RAW_DIM

WIDTH = 320
RANK = 16
INPUT_DIM = WIDTH + RAW_DIM
DEP_LABELS = ("root", "nsubj", "iobj", "obj", "punct", "nsubj:pass",
              "aux", "obl:agent", "obl", "case", "advmod", "amod", "mark",
              "advcl")
PHYSICAL_FORMAT = "NRA3_S19B_GOLD_BLIND_RAW43M_CPU_WORD_GRAMMAR_FP16_V1"


@dataclass(frozen=True)
class GrammarLogits:
    upos: torch.Tensor           # [batch, words, 17]
    dependency: torch.Tensor     # [batch, words, 14]
    predicate: torch.Tensor      # [batch, words], VERB anchors only
    heads: torch.Tensor          # [batch, words, words+1], 0=ROOT


class CompactWordGrammar(nn.Module):
    """All physically billed weights belong to this model, not to LIVE."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(INPUT_DIM, RANK)
        self.normalization = nn.LayerNorm(RANK)
        self.upos_head = nn.Linear(RANK, len(UPOS))
        self.deprel_head = nn.Linear(RANK, len(DEP_LABELS))
        self.predicate_head = nn.Linear(RANK, 1)
        self.arc_query = nn.Linear(RANK, RANK, bias=False)
        self.arc_key = nn.Linear(RANK, RANK, bias=False)
        self.root_key = nn.Parameter(torch.zeros((RANK,)))
        self.output_gate = 0
        self.candidate_accepted = False

    def forward(self, input_338: torch.Tensor,
                valid_words: torch.Tensor) -> GrammarLogits:
        if self.output_gate != 0 or self.candidate_accepted:
            raise ValueError("S19b learned grammar candidate may not activate")
        if (input_338.ndim != 3 or input_338.shape[-1] != INPUT_DIM or
                input_338.dtype != torch.float32 or
                input_338.device.type != "cpu" or
                valid_words.shape != input_338.shape[:2] or
                valid_words.dtype != torch.bool or
                valid_words.device.type != "cpu" or
                not bool(valid_words[:, 0].all()) or
                not bool((valid_words.long().cummin(1).values ==
                          valid_words.long()).all()) or
                not bool(torch.isfinite(input_338).all())):
            raise ValueError("S19b invalid gold-blind CPU raw hidden+word tensor")
        b, n, _ = input_338.shape
        z = F.gelu(self.normalization(self.projection(input_338)))
        z = z.masked_fill(~valid_words.unsqueeze(-1), 0.)
        upos = self.upos_head(z)
        deps = self.deprel_head(z)
        pred = self.predicate_head(z).squeeze(-1)
        query = self.arc_query(z)
        key = self.arc_key(z)
        candidates = torch.cat((self.root_key.expand(b, 1, -1), key), dim=1)
        arc = torch.matmul(query, candidates.transpose(1, 2)) / math.sqrt(RANK)
        keep = torch.cat((torch.ones(b, 1, dtype=torch.bool), valid_words),
                         dim=1)
        arc = arc.masked_fill(~keep.unsqueeze(1), -1e4)
        self_arcs = torch.eye(n, dtype=torch.bool).unsqueeze(0)
        arc = torch.cat((arc[:, :, :1],
                         arc[:, :, 1:].masked_fill(self_arcs, -1e4)), dim=-1)
        return GrammarLogits(upos, deps, pred, arc)


def trainable_parameter_count(model: CompactWordGrammar) -> int:
    return sum(p.numel() for p in model.parameters())


def paid_fp16_bytes(model: CompactWordGrammar) -> bytes:
    """Serialize EVERY trainable weight in physically charged FP16."""
    if model.output_gate != 0 or model.candidate_accepted:
        raise ValueError("S19b cannot export active weights")
    state = {name: tensor.detach().cpu().to(torch.float16).contiguous()
             for name, tensor in sorted(model.state_dict().items())}
    if sum(v.numel() for v in state.values()) != trainable_parameter_count(model):
        raise ValueError("S19b physical FP16 dropped a trainable parameter")
    if any(not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("S19b physical FP16 nonfinite weights")
    return save(state, metadata={"format": PHYSICAL_FORMAT,
                                 "activation_allowed": "false",
                                 "source": "TRAIN96_REAL43M_CPU_ONLY"})


def load_paid_fp16_bytes(raw: bytes) -> CompactWordGrammar:
    """Round-trip from physical, never evaluate unrounded FP32 train master."""
    weights = load(raw)
    model = CompactWordGrammar().eval()
    expected = set(model.state_dict())
    if (set(weights) != expected or
            any(t.dtype != torch.float16 for t in weights.values()) or
            sum(v.numel() for v in weights.values())
                != trainable_parameter_count(model)):
        raise ValueError("S19b physical weight ABI changed or incomplete")
    model.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
    model.requires_grad_(False)
    rebuilt = load(paid_fp16_bytes(model))
    if set(rebuilt) != set(weights) or any(
            not torch.equal(rebuilt[k], weights[k]) for k in weights):
        raise ValueError("S19b physical FP16 tensors do not round-trip")
    return model


def physical_sha256(model: CompactWordGrammar) -> str:
    return hashlib.sha256(paid_fp16_bytes(model)).hexdigest()
