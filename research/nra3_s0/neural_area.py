"""NRA2 3.0: unified causal grammar, semantics and relational neural area.

RESEARCH ONLY. No language competence is implied before TRAIN and independent
verification. The area consumes Calibri hidden320, never gold graph edges, answers
or held-out EVAL metadata. Output readout is zero-initialized and the matching
Calibri hook is strictly read-only. Efficient decoding cache is NOT implemented.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn

UPOS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")
MORPH = ("Case=Nom", "Case=Gen", "Case=Dat", "Case=Acc", "Case=Ins",
         "Case=Loc", "Number=Sing", "Number=Plur", "Gender=Masc",
         "Gender=Fem", "Gender=Neut", "Tense=Past", "Tense=Pres",
         "Tense=Fut", "Voice=Act", "Voice=Pass")
SEMANTIC_ROLES = ("NONE", "AGENT", "PATIENT", "RECIPIENT", "SOURCE",
                  "DESTINATION", "EXPERIENCER", "INSTRUMENT", "LOCATION")


@dataclass(frozen=True)
class NRA3Config:
    io_width: int = 320
    hidden: int = 256
    grammar_blocks: int = 4
    semantic_blocks: int = 3
    heads: int = 8
    ffn_multiplier: int = 4
    relation_width: int = 128
    relation_steps: int = 6
    parallel_hypotheses: int = 1  # S7 opt-in 3, original single remains default

    def __post_init__(self):
        if not (self.io_width == 320 and self.hidden >= 16
                and self.hidden % self.heads == 0 and self.heads >= 1
                and self.grammar_blocks >= 1 and self.semantic_blocks >= 1
                and self.ffn_multiplier >= 2 and self.relation_width >= 8
                and self.relation_steps >= 1
                and self.parallel_hypotheses in (1, 3)):
            raise ValueError("NRA3 invalid dimension/causal architecture contract")


class CausalBlock(nn.Module):
    def __init__(self, c: NRA3Config):
        super().__init__()
        self.norm_attn = nn.LayerNorm(c.hidden)
        self.attn = nn.MultiheadAttention(c.hidden, c.heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(c.hidden)
        self.ffn = nn.Sequential(nn.Linear(c.hidden, c.hidden * c.ffn_multiplier),
                                 nn.GELU(),
                                 nn.Linear(c.hidden * c.ffn_multiplier, c.hidden))

    def forward(self, x, valid, future_mask):
        q = self.norm_attn(x)
        value, _ = self.attn(q, q, q, attn_mask=future_mask,
                             key_padding_mask=~valid, need_weights=False)
        x = x + value
        x = x + self.ffn(self.norm_ffn(x))
        return x.masked_fill(~valid.unsqueeze(-1), 0.)


@dataclass
class NRA3Features:
    grammar: torch.Tensor
    semantics: torch.Tensor
    relations: torch.Tensor
    upos_logits: torch.Tensor
    morph_logits: torch.Tensor
    dependency_logits: torch.Tensor
    semantic_role_logits: torch.Tensor
    antecedent_logits: torch.Tensor
    readout: torch.Tensor
    parallel_hypothesis_gates: torch.Tensor | None = None


class NRA3NeuralArea(nn.Module):
    """One three-level area. Heads are untrained; they cannot parse Russian yet.

    All predictions at time t use tokens <= t only. Future UD heads or entities
    cannot be predicted for earlier prefixes until additional text arrives.
    This does not maintain an incremental semantic cache across decoder calls.
    """
    def __init__(self, config: NRA3Config = NRA3Config()):
        super().__init__()
        self.config = config
        d = config.hidden
        r = config.relation_width
        self.input_projection = nn.Sequential(nn.LayerNorm(config.io_width),
                                              nn.Linear(config.io_width, d))
        self.grammar = nn.ModuleList([CausalBlock(config)
                                      for _ in range(config.grammar_blocks)])
        self.upos_head = nn.Linear(d, len(UPOS))
        self.morph_head = nn.Linear(d, len(MORPH))
        self.dep_q = nn.Linear(d, d, bias=False)
        self.dep_k = nn.Linear(d, d, bias=False)
        self.semantic = nn.ModuleList([CausalBlock(config)
                                       for _ in range(config.semantic_blocks)])
        self.role_head = nn.Linear(d, len(SEMANTIC_ROLES))
        self.core_in = nn.Linear(d, r)
        self.rel_q = nn.Linear(r, r, bias=False)
        self.rel_k = nn.Linear(r, r, bias=False)
        self.rel_value = nn.Linear(r, r, bias=False)
        self.rel_update = nn.GRUCell(r, r)
        self.ante_q = nn.Linear(d, d, bias=False)
        self.ante_k = nn.Linear(d, d, bias=False)
        self.readout = nn.Linear(r, config.io_width, bias=False)
        # Keep source nondegenerate for exact CB59 K32V8 vector coding.
        # The Calibri bridge, not a zero matrix, guarantees zero output influence.
        nn.init.normal_(self.readout.weight, mean=0.0, std=0.01)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _prefix_mask(valid: torch.Tensor) -> torch.Tensor:
        n = valid.shape[1]
        causal = torch.tril(torch.ones(n, n, device=valid.device, dtype=torch.bool))
        return causal.unsqueeze(0) & valid[:, None, :]

    def forward(self, hidden320: torch.Tensor,
                valid: torch.Tensor | None = None) -> NRA3Features:
        if not isinstance(hidden320, torch.Tensor) or hidden320.ndim != 3:
            raise ValueError("NRA3 expects [batch,time,320] actual hidden states")
        b, t, d = hidden320.shape
        if d != self.config.io_width or b < 1 or t < 1 or not hidden320.is_floating_point():
            raise ValueError("NRA3 invalid input hidden ABI")
        if hidden320.device != self.readout.weight.device:
            raise ValueError("NRA3 hidden/device mismatch")
        if not torch.isfinite(hidden320).all():
            raise ValueError("NRA3 nonfinite hidden states")
        if valid is None:
            valid = torch.ones((b, t), device=hidden320.device, dtype=torch.bool)
        if (valid.shape != (b, t) or valid.dtype != torch.bool
                or valid.device != hidden320.device or not valid[:, 0].all()
                or not (valid.long().cummin(dim=1).values == valid.long()).all()):
            raise ValueError("NRA3 requires right padding, first token valid")
        allowed = self._prefix_mask(valid)
        future_mask = ~torch.tril(torch.ones(t, t, device=valid.device, dtype=torch.bool))
        x = self.input_projection(hidden320.float()).masked_fill(~valid.unsqueeze(-1), 0.)
        for block in self.grammar:
            x = block(x, valid, future_mask)
        grammar = x
        upos = self.upos_head(grammar)
        morph = self.morph_head(grammar)
        dep = (self.dep_q(grammar) @ self.dep_k(grammar).transpose(1, 2)) / math.sqrt(self.config.hidden)
        dep = dep.masked_fill(~allowed, -1e4)
        for block in self.semantic:
            x = block(x, valid, future_mask)
        semantics = x
        roles = self.role_head(semantics)
        antecedent = (self.ante_q(semantics) @ self.ante_k(semantics).transpose(1, 2)) / math.sqrt(self.config.hidden)
        antecedent = antecedent.masked_fill(~allowed, -1e4)
        relation = self.core_in(semantics)
        parallel_gates = None
        for _ in range(self.config.relation_steps):
            scores = (self.rel_q(relation) @ self.rel_k(relation).transpose(1, 2)) / math.sqrt(self.config.relation_width)
            scores = scores.masked_fill(~allowed, -1e4)
            if self.config.parallel_hypotheses == 3:
                from research.nra3_s7.parallel_hypotheses import parallel_three_relations
                # Three competing relation maps, but one GRU update and no
                # additional paid parameters. No gold text labels enter.
                evidence=parallel_three_relations(
                    query=self.rel_q(relation),value=self.rel_value(relation),
                    scores=scores,allowed=allowed,valid=valid,
                    semantic_role_logits=roles,
                    local_window=min(4,t))
                message=evidence.message
                parallel_gates=evidence.hypothesis_gates
            else:
                weights = scores.softmax(dim=-1)
                message = weights @ self.rel_value(relation)
            relation = self.rel_update(message.reshape(-1, self.config.relation_width),
                                       relation.reshape(-1, self.config.relation_width)).reshape(b, t, -1)
            relation = relation.masked_fill(~valid.unsqueeze(-1), 0.)
        readout = self.readout(relation)
        if not all(torch.isfinite(z).all() for z in (readout, roles, upos, morph, relation)):
            raise AssertionError("NRA3 nonfinite stage features")
        return NRA3Features(grammar, semantics, relation, upos, morph, dep,
                            roles, antecedent, readout, parallel_gates)

    def fp16_lower_bound_bpp(self, parent_parameters: int,
                             parent_bpp: float) -> float:
        """Optimistic FP16 payload bound, EXCLUDES metadata and code.

        This is NOT a physical compressed-model budget certificate.
        """
        if (parent_parameters <= 0 or not math.isfinite(parent_bpp)
                or parent_bpp < 0):
            raise ValueError("NRA3 invalid parent physical accounting")
        n = self.parameter_count()
        return (parent_parameters * parent_bpp + 16 * n) / (parent_parameters + n)
