"""NRA3 S100 TRAIN-only ALL VERB multirelation neural arc loss.

S99 morphology pretrained in-Calibri weights but role graph overfit:
264/307 ~86% teacher-anchored TRAIN vs 18/108 ~16.7% Taiga
TRAIN-b internal developer. That internal developer is SPENT;
DO NOT tune S100 on it.

This introduces a new OFFLINE label objective, not a sidecar. It
applies directly to the SAME in-Calibri neural child of S97, with
ALL original frozen 12 Calibri+GreenTunnel layer representations.
Full UD graph contains multiple verb predicate positions, multiple
role-specific argument words per predicate (when present), and
NOT a single oracle first verb per sentence. Unknown role slots
are NOT supervised negatives and unknown event mentions remain
unknown. A multi-positive masked marginal log-likelihood sums over
positive arguments for an event-role slot.

Only after a new TRAIN-d original Calibri source-only native-BPE
frozen SHA is sealed will S100 teacher labels be mapped to BPE.
No UD runtime parser and no production/new 616 J7 activation.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F

from research.calibri_nra3.s91_neural_multilayer_event_graph import N_ROLES
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyGraph,
)

STAGE="CALIBRI_NRA3_S100_ALL_UD_VERB_MULTI_POSITIVE_DEPENDENCY_ARC_TRAIN_ONLY"


def require(v,message):
    if not bool(v):
        raise ValueError("NRA3 S100 source-only multi-predicate in-Calibri TRAIN: "+message)


@dataclass(frozen=True)
class OriginalBPEAllVerbsEdges:
    """Labels NEVER passed to model.forward.

    argument_gold_positive [B,R,T,T] is a multi-hot label for every
    source BPE event token to each observed argument BPE token.
    Only events and role types with >=1 known link are supervised.
    No distant task-specific event/assertion claims inferred.
    """
    native_BPE_valid:torch.Tensor
    argument_gold_positive:torch.Tensor
    event_role_supervised:torch.Tensor
    event_verb_positive_only:torch.Tensor


def offline_all_verb_multi_positive_relation_loss(
        graph:NRA3MorphologyGraph,target:OriginalBPEAllVerbsEdges):
    valid=graph.native_valid
    b,t=valid.shape
    positive=target.argument_gold_positive
    supervision=target.event_role_supervised
    mention=target.event_verb_positive_only
    require(valid.shape==target.native_BPE_valid.shape
            and torch.equal(valid,target.native_BPE_valid)
            and valid.dtype==torch.bool
            and positive.shape==(b,N_ROLES,t,t)
            and supervision.shape==(b,N_ROLES,t)
            and mention.shape==(b,t)
            and positive.dtype==torch.bool
            and supervision.dtype==torch.bool
            and mention.dtype==torch.bool
            and not positive.requires_grad
            and not supervision.requires_grad
            and graph.role_arc_logits.shape==positive.shape
            and bool(torch.isfinite(graph.role_arc_logits).all()),
            "strict native BPE, in-Calibri neural logits and teacher multi-edge geometry")
    pairs=valid[:,None,:,None] & valid[:,None,None,:]
    require(not bool((positive&~pairs).any())
            and not bool((supervision&~valid[:,None,:]).any())
            and not bool((mention&~valid).any())
            and bool(supervision.any())
            and bool(positive.any())
            and torch.equal(positive.any(-1),supervision)
            and not bool((supervision&~mention[:,None,:]).any()),
            "multi-link role must have one or more explicit gold arguments anchored to known source VERB")
    # [N,T] for supervised (event,role) slots. Non-target events
    # and unannotated roles receive ZERO contribution to loss.
    score=graph.role_arc_logits[supervision].float()
    gold=positive[supervision]
    require(score.ndim==2 and score.shape[1]==t
            and bool(gold.any(-1).all()),
            "each supervised role has >=1 explicit valid UD arguments")
    total_logmass=torch.logsumexp(score,dim=-1)
    correct_logmass=torch.logsumexp(
        score.masked_fill(~gold,-1.e7),dim=-1)
    edge_loss=(total_logmass-correct_logmass).mean()
    require(bool(torch.isfinite(edge_loss))
            and float(edge_loss.detach())>=-1.e-5,
            "finite positive training-only multi-argument marginal")
    return edge_loss,{
        "n_source_BPE_verbs_explicit":int(mention.sum()),
        "n_supervised_event_role_slots":int(supervision.sum()),
        "n_supervised_event_to_argument_edges":int(positive.sum()),
        "multiple_distinct_role_args_preserved":True,
        "source_native_Calibri_BPE_only":True,
        "unknown_event_role_slots_NEVER_negative":True,
        "labels_only_OFFLINE_TRAIN_not_neural_forward":True,
        "not_independent_factuality_or_grammar_accuracy":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
