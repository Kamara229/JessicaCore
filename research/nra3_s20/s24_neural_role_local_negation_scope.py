"""S24 research-only three-parameter negation-scope residual on S22d graph.

A hypothesis test for S23's complete failure on the held-fronted-recipient
REJECTED pattern (0 TP / 8). Use ONLY original genuinely frozen 43M+GT
raw338 word features and *predicted* role probabilities. Never inspect gold,
pre-declared held group IDs, lexicon, or source graphs in forward. The
residual is invariant to relative predicate/argument order and has ZERO
initial effect on the full S22d architecture.

No independent EVAL, no LIVE, output gate=0. Learning and reporting run
exclusively on the original S22 TRAIN256 with the same 40 frozen splits.
"""
from __future__ import annotations

import torch
from torch import nn

from research.nra3_s18.s18c3_raw_russian_grammar_cues import FEATURES
from research.nra3_s19.s19f_real43m_train9_graph_targets import ROLE
from research.nra3_s19.s19g_paid_semantic_graph import (
    PaidSemanticEventGraph,GraphLogits,semantic_parameter_count,
)

PREV_NEG = FEATURES.index("previous_word_negative_particle")
LIKELY_DATIVE = FEATURES.index("ending_dative_possible")
RECIPIENT = ROLE.index("RECIPIENT")
EXTRA_SCOPE_FP16=3
OLD_GRAPH_FP16=3049


class ScopedRejectionGraph(PaidSemanticEventGraph):
    """Order-invariant role-local contrast cue, gold-blind at inference.

    Base architecture computes all 5 graph heads exactly as S22d did.
    This residual adds to REJECTED (class index 2) logits ONLY at
    candidate directed predicate->recipient pairs. Its features are
    raw-text preceding 'не', possible dative ending and the model's
    **own predicted** probability of RECIPIENT; no human oracle roles.
    """

    def __init__(self,parent):
        super().__init__(parent)
        self.local_scope_weights=nn.Parameter(torch.zeros(EXTRA_SCOPE_FP16))
        assert self.output_gate==0 and not self.candidate_accepted and not self.activation_allowed
        assert semantic_parameter_count(self)==OLD_GRAPH_FP16+EXTRA_SCOPE_FP16

    def forward(self,x:torch.Tensor,valid:torch.Tensor)->GraphLogits:
        base=super().forward(x,valid)
        # x is the sealed original43M post-GT hidden320 + text-only
        # raw18 surface features, never oracle morphological annotations.
        prev_negative=x[...,320+PREV_NEG]
        dative=x[...,320+LIKELY_DATIVE]
        role_probability=torch.softmax(base.role,dim=-1)[...,RECIPIENT]
        # Candidate word j is the argument, i is the predicted predicate.
        # No predicate-to-argument order test: same local rule before/after.
        local_negative=prev_negative.unsqueeze(1) * role_probability
        strength=(self.local_scope_weights[0]
                  +self.local_scope_weights[1]*dative.unsqueeze(1)
                  +self.local_scope_weights[2]*role_probability)
        delta=local_negative*strength
        legal=(valid.unsqueeze(2)&valid.unsqueeze(1))
        legal=legal & (~torch.eye(valid.shape[1],dtype=torch.bool,device=valid.device)
                       .unsqueeze(0))
        delta=delta.masked_fill(~legal,0.)
        zeros=torch.zeros_like(delta)
        polar=base.argument_polarity+torch.stack((zeros,zeros,delta),dim=-1)
        return GraphLogits(
            base.event_status,base.role,polar,base.coreference,
            base.event_relation,
        )
