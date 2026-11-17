"""S26: order-invariant event-grounded recipient/REJECTED residual, gold-blind.

S24 real 40-fold matched data showed 0/8 preposed REJECTED and no gain.
S25 independently verified that in *physical original43M+GT RAW338* all
8 preposed rejected recipient words have correct negative-adjacent/dative
text-only features. Therefore, test a structural change: no dependency
on the *possibly wrong* directed RECIPIENT role probability for the
fundamental rescue path. Use the network's own predicted event source
probability as a symmetric predicate/recipient connection gate.
No human role/event gold, templates, relative-order or fold ID in forward.
Exactly three newly trained FP16 parameters and zero-initialized parity.
"""
from __future__ import annotations

import torch
from torch import nn

from research.nra3_s18.s18c3_raw_russian_grammar_cues import FEATURES
from research.nra3_s19.s19f_real43m_train9_graph_targets import ROLE
from research.nra3_s19.s19g_paid_semantic_graph import (
    PaidSemanticEventGraph, GraphLogits,semantic_parameter_count,
)

PREV_NEG=FEATURES.index("previous_word_negative_particle")
LIKELY_DATIVE=FEATURES.index("ending_dative_possible")
ROLE_RECIPIENT=ROLE.index("RECIPIENT")
EXTRA_FP16=3
ORIGINAL_HEAD_FP16=3049


class EventGroundedInversionGraph(PaidSemanticEventGraph):
    """Three trainable scalar biases only on local predicted event pairs.

    Strictly NOT gold: event probability is the model's own output, and
    all syntactic cues are computed from plain surface text via S18c3.
    The same rule works for recipient before or after its predicate.
    """

    def __init__(self,parent):
        super().__init__(parent)
        self.event_scoped_recipient_weights=nn.Parameter(torch.zeros(EXTRA_FP16))
        assert semantic_parameter_count(self)==ORIGINAL_HEAD_FP16+EXTRA_FP16
        assert self.output_gate==0 and not self.candidate_accepted and not self.activation_allowed

    def forward(self,x:torch.Tensor,valid:torch.Tensor)->GraphLogits:
        base=super().forward(x,valid)
        # i is a potential source event, j a candidate recipient.
        # No known token order, predetermined anchors or reference labels.
        event_prob=1.-torch.softmax(base.event_status,dim=-1)[...,0]
        local_negative_dative=(x[...,320+PREV_NEG]*x[...,320+LIKELY_DATIVE])
        pair_cue=event_prob.unsqueeze(2)*local_negative_dative.unsqueeze(1)
        legal=valid.unsqueeze(2)&valid.unsqueeze(1)
        legal=legal & ~torch.eye(valid.shape[1],dtype=torch.bool,device=valid.device).unsqueeze(0)
        pair_cue=pair_cue.masked_fill(~legal,0.)

        w=self.event_scoped_recipient_weights
        # Explicitly rescue a recipient role even if the OLD directed
        # role head predicts NONE for an inverted pre-verbal recipient.
        # The same learned cue supplies an independent REJECTED polarity
        # potential. Third weight learns pair-local calibration from
        # the model's *predicted* recipient class, never an oracle.
        recipient_prob=torch.softmax(base.role,dim=-1)[...,ROLE_RECIPIENT]
        delta_role=w[0]*pair_cue
        delta_rejected=pair_cue*(w[1]+w[2]*recipient_prob)
        role_extra=torch.zeros_like(base.role)
        role_extra[...,ROLE_RECIPIENT]=delta_role
        polar_extra=torch.stack((torch.zeros_like(delta_rejected),
                                 torch.zeros_like(delta_rejected),
                                 delta_rejected),dim=-1)
        return GraphLogits(base.event_status,base.role+role_extra,
                           base.argument_polarity+polar_extra,
                           base.coreference,base.event_relation)
