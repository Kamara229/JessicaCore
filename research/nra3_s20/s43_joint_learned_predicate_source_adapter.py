"""S43 TRAIN-only learned predicate-source and directed role consistency head.

Unlike S41/S42, no output-side gold edge completion, greedy matching or
one-to-one pair cap is applied. The event SOURCE head itself is trainable,
and a separate learned relation head attends to its current predictions.
The baseline fixed S40 structural surface prior is inherited (already
informed by inspected synthetic templates); no claim of new general
Russian grammar or integration with genuine Calibri43M+GT/NRA2.

Forward arguments: ONLY feature tensors parsed from plain Unicode text.
TRAIN-only event/quote/rejected annotations enter loss, NEVER forward.
"""
from __future__ import annotations

import torch
from torch import nn
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import GraphPrediction
from research.nra3_s20.s40_compositional_predicate_negation_graph import (
    CompositionalNRA2S40,STRUCTURE_DIM,
)

S43_PRED_INPUT_DIM=20
S43_LINK_INPUT_DIM=STRUCTURE_DIM+3
S43_TRAIN_EVENT_SOURCE_COEFF=2.0
S43_TRAIN_LINK_SOURCE_CONSISTENCY_COEFF=0.65
S43_TRAIN_HARD_NEGATIVE_COEFF=1.0
S43_BASELINE_S42_RUN=44246124526

class JointLearnedPredicateSourceS43(nn.Module):
    """Separate experimental 3-head grammar graph, not active NRA2."""
    def __init__(self):
        super().__init__()
        self.base=CompositionalNRA2S40()
        self.source_head=nn.Sequential(
            nn.Linear(S43_PRED_INPUT_DIM,32),
            nn.Tanh(),
            nn.Linear(32,1))
        self.edge_head=nn.Sequential(
            nn.Linear(S43_LINK_INPUT_DIM,24),
            nn.Tanh(),
            nn.Linear(24,1))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False

    def forward(self,predicate,quotation,pair,structure):
        single=predicate.ndim==2
        if single:
            predicate=predicate.unsqueeze(0)
            quotation=quotation.unsqueeze(0)
            pair=pair.unsqueeze(0)
            structure=structure.unsqueeze(0)
        if not (predicate.ndim==3 and quotation.ndim==3
                and pair.ndim==4 and structure.ndim==4
                and predicate.shape[:2]==quotation.shape[:2]
                and pair.shape[:3]==structure.shape[:3]
                and pair.shape[:2]==predicate.shape[:2]
                and predicate.shape[-1]==16 and quotation.shape[-1]==5
                and pair.shape[-1]==32 and structure.shape[-1]==STRUCTURE_DIM):
            raise ValueError("S43 only plain raw text feature tensors allowed")
        b,w=predicate.shape[:2]
        raw=self.base(predicate,quotation,pair,structure)
        # Only textual morphology and candidate-structural signals.
        # NEVER human-labelled source indexes or TRAIN row groups.
        source_possible=structure[...,0].amax(dim=2)
        outgoing_eligible=structure[...,11].amax(dim=2)
        passive_source=predicate[...,2]
        infinitive_source=predicate[...,3]
        source_input=torch.cat((
            predicate,
            source_possible.unsqueeze(-1),
            outgoing_eligible.unsqueeze(-1),
            passive_source.unsqueeze(-1),
            infinitive_source.unsqueeze(-1),
        ),dim=-1)
        source_logits=raw.event_logits+self.source_head(
            source_input).squeeze(-1)
        rel_in=torch.cat((
            structure,
            source_logits.unsqueeze(2).unsqueeze(-1).expand(b,w,w,1),
            raw.quote_logits.unsqueeze(1).unsqueeze(-1).expand(b,w,w,1),
            raw.rejected_logits.unsqueeze(-1),
        ),dim=-1)
        relation_logits=raw.rejected_logits+self.edge_head(
            rel_in).squeeze(-1)
        if single:
            return GraphPrediction(source_logits[0],raw.quote_logits[0],
                                   relation_logits[0])
        return GraphPrediction(source_logits,raw.quote_logits,relation_logits)
