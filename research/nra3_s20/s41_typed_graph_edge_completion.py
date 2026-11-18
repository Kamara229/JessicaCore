"""S41: type-agnostic compositional directed-edge completion after S40.

S40's frozen synthetic TRAIN80 18-fold report (#44217309516) confirms
the manual grammar eligibility bit is 1 on all 56 original positive
directed pairs, yet the learned rejection logits missed all 16 held
short-passive / infinitive examples. This S41 test imposes a SINGLE
predeclared output-side lower confidence bound on EVERY structurally
eligible pair, not on named constructions or gold classes.

The lower bound is an explicit rule-based inductive bias, not evidence
of a learned universal predicate grammar, generalization or true43M
integration. It may increase incorrect structurally eligible links.
It never changes other event, quote or role logits, S40 learned weights,
original NRA2/Calibri43M/Green Tunnel/616J7, or LIVE outputs.
"""
from __future__ import annotations

import torch
from torch import nn
from research.nra3_s20.s40_compositional_predicate_negation_graph import (
    CompositionalNRA2S40, STRUCTURE_DIM,
)
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import GraphPrediction

S41_ELIGIBLE_LOWER_LOGIT=0.25
S41_RULE="generic_directed_grammatical_eligible_edge_floor_for_all_predicate_types"
S41_ORIGINAL_S40_RUN=44217309516

def structurally_complete_REJECTED(output:GraphPrediction,structure:torch.Tensor)->GraphPrediction:
    """No gold, fold/pattern IDs, labels, text or lexicon in decoder."""
    single=(output.rejected_logits.ndim==2)
    legal_shape=structure.shape[:-1]==output.rejected_logits.shape
    if structure.ndim not in (3,4) or structure.shape[-1]!=STRUCTURE_DIM or not legal_shape:
        raise ValueError("S41 compositional 12D graph and learned pair logit geometry invalid")
    eligible=structure[...,11]>0.5
    # Only eligible edge logits may be increased. Noneligible edges are
    # kept EXACTLY as learned by S40; this rule does NOT tune thresholds
    # on the held synthetic TRAIN fold.
    new_rejected=torch.where(
        eligible,
        torch.maximum(output.rejected_logits,
                      torch.full_like(output.rejected_logits,S41_ELIGIBLE_LOWER_LOGIT)),
        output.rejected_logits,
    )
    return GraphPrediction(output.event_logits,output.quote_logits,new_rejected)

class CompositionalNRA2S41(nn.Module):
    """TRAIN-only S40 graph+learned heads with transparent S41 edge floor."""
    def __init__(self):
        super().__init__()
        self.base=CompositionalNRA2S40()
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False

    def forward(self,pred,quote,pair,structure):
        original=self.base(pred,quote,pair,structure)
        return structurally_complete_REJECTED(original,structure)
