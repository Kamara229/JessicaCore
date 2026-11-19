"""NRA3 S93 research TRAIN-only grammatical dependency loss.

IMPORTANT: S91/S92 NRA3 is a neural child INSIDE Calibri weights.
This separate module defines *offline supervision losses*, NOT an
external runtime parser, model, grammar rules, inference service or
neural features. Any Russian UD/Stanza/UDPipe tool is a possible
OFFLINE teacher only. Its labels must be calibrated, independently
checked, and mapped onto original Calibri native BPE boundaries;
the source-only model.forward never imports/accepts this module.

Every loss target has a 'not annotated' sentinel. Never label
unannotated mentions/roles as NOT EVENTS. Role targets describe
predicate token i -> argument token j for named role r. All annotated
indices must be original native BPE offsets, not raw-word indices.
No S79/S86 permanently SPENT firstread samples, no S66 held gold,
no sealed EVAL32/EVAL128, nor prior S83 diag for retuning.

This is a TRAIN-LOSS ABI with synthetic unit tests, NOT actual
grammatical accuracy or knowledge retention evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3NeuralGraph,N_ROLES,N_STATES,N_MICRO,
)

STAGE="CALIBRI_NRA3_S93_OFFLINE_TEACHER_EVENT_DEPENDENCY_TRAIN_ONLY"
IGNORE=-100
UNKNOWN=-1
ROLE_WEIGHT=2.0
MENTION_WEIGHT=1.0
STATE_WEIGHT=1.5
MINIAREA_WEIGHT=0.6
MEMORY_CONSOLIDATION_WEIGHT=0.1


def require(value,why):
    if not bool(value):
        raise ValueError("NRA3 S93 strictly supervised TRAIN-only graph loss: "+why)


@dataclass(frozen=True)
class NRA3TrainOnlyTargets:
    """OFFLINE supervision; NEVER feed into NRA3 forward.

    mention: [B,T] {0,1,IGNORE}; only 0 if all mentions adjudicated.
    role_argument_token: [B,8,T] {-100 or valid native BPE token index};
      repeated actions may have multiple roles, but each role slot has
      at most ONE supervised argument per predicate for this first ABI.
      Multi-argument roles need a future multi-target supervision.
    state: [B,T,6] {-1,0,1}; UNKNOWN is not a negative.
    mini_area: [B,T,8] {-1,0,1}, retaining NRA2 individual heads.
    original_native_valid: [B,T], must be same original BPE mask.
    evidence: human annotation or offline parser; never accepted based
      only on synthetic/generative pseudo-labels.
    """
    mention:torch.Tensor
    role_argument_token:torch.Tensor
    state:torch.Tensor
    mini_area:torch.Tensor
    original_native_valid:torch.Tensor
    evidence:str="train_only_constructed_not_independently_adjudicated"


def graph_train_only_loss(graph:NRA3NeuralGraph,
                          target:NRA3TrainOnlyTargets,
                          *,
                          old_memory_snapshot:Optional[torch.Tensor]=None,
                          new_neural_model=None):
    """Differentiable supervision of neural in-Calibri event-graph heads.

    All four data-dependent losses are masked to explicitly annotated
    native BPE positions; no assumptions from unlabelled tokens.
    """
    valid=graph.native_valid
    mention=target.mention
    role=target.role_argument_token
    state=target.state
    areas=target.mini_area
    b,t=valid.shape
    require(
        valid.dtype==torch.bool and b>=1 and t>=2
        and torch.equal(valid,target.original_native_valid)
        and mention.shape==(b,t)
        and role.shape==(b,N_ROLES,t)
        and state.shape==(b,t,N_STATES)
        and areas.shape==(b,t,N_MICRO)
        and not mention.requires_grad
        and not role.requires_grad
        and not state.requires_grad
        and not areas.requires_grad
        and mention.dtype in (torch.long,torch.int64,torch.int32)
        and role.dtype==torch.long
        and state.dtype in (torch.float32,torch.float64)
        and areas.dtype in (torch.float32,torch.float64)
        and bool(((mention==IGNORE)|(mention==0)|(mention==1)).all())
        and bool(((state==UNKNOWN)|(state==0)|(state==1)).all())
        and bool(((areas==UNKNOWN)|(areas==0)|(areas==1)).all()),
        "explicit native BPE supervised labels with unknown sentinel required",
    )
    mm=mention!=IGNORE
    sm=state!=UNKNOWN
    am=areas!=UNKNOWN
    rm=role!=IGNORE
    require(not bool((mm&~valid).any())
            and not bool((sm&~valid[...,None]).any())
            and not bool((am&~valid[...,None]).any())
            and not bool((rm&~valid[:,None,:]).any())
            and bool(mm.any()) and bool(rm.any())
            and bool(sm.any()) and bool(am.any()),
            "explicitly annotated TRAIN source positions only; unknown not negative")
    require(bool(((role==IGNORE)|((role>=0)&(role<t))).all()),
            "role argument indexes must be native BPE or ignored")
    if bool(rm.any()):
        b_idx,r_idx,event_idx=torch.where(rm)
        arguments=role[rm]
        require(bool(valid[b_idx,arguments].all()),
                "all supervised argument heads need source-native BPE valid")
        require(bool((mention[b_idx,event_idx]==1).all()),
                "supervised event→role dependency must be an annotated EVENT")
    scores=graph.role_arc_logits
    require(scores.shape==(b,N_ROLES,t,t)
            and graph.event_mention_logits.shape==(b,t)
            and graph.event_state_logits.shape==(b,t,N_STATES)
            and graph.mini_area_logits.shape==(b,t,N_MICRO)
            and all(bool(torch.isfinite(z).all()) for z in (
                scores,graph.event_mention_logits,graph.event_state_logits,
                graph.mini_area_logits)),
            "neural graph produced from original 12-layer source only")
    l_mention=F.binary_cross_entropy_with_logits(
        graph.event_mention_logits[mm],mention[mm].float())
    l_role=F.cross_entropy(scores[rm],role[rm])
    l_state=F.binary_cross_entropy_with_logits(
        graph.event_state_logits[sm],state[sm].float())
    l_areas=F.binary_cross_entropy_with_logits(
        graph.mini_area_logits[am],areas[am].float())
    total=(MENTION_WEIGHT*l_mention+ROLE_WEIGHT*l_role
           +STATE_WEIGHT*l_state+MINIAREA_WEIGHT*l_areas)
    mem=None
    if old_memory_snapshot is not None:
        require(new_neural_model is not None
                and hasattr(new_neural_model,"knowledge_consolidation_penalty"),
                "knowledge consolidation must compare SAME in-Calibri neural child")
        mem=new_neural_model.knowledge_consolidation_penalty(old_memory_snapshot)
        total=total+MEMORY_CONSOLIDATION_WEIGHT*mem
    require(bool(torch.isfinite(total)),
            "NRA3 strictly supervised relation loss must remain finite")
    return total,{
        "mention":l_mention.detach(),
        "event_to_argument_role":l_role.detach(),
        "event_state":l_state.detach(),
        "eight_NRA2_mini_areas":l_areas.detach(),
        "optional_memory_prototype_distillation":None if mem is None else mem.detach(),
        "annotated_event_positions":int((mention==1).sum()),
        "annotated_role_edges":int(rm.sum()),
        "annotated_state_bits":int(sm.sum()),
        "annotated_mini_area_bits":int(am.sum()),
        "unannotated_ignored_by_loss":True,
        "labels_only_offline_training_not_in_model_forward":True,
        "synthetic_test_not_real_relation_F1":True,
        "activation_allowed":False,
    }
