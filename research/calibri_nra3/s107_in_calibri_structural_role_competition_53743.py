"""S107: rank-competitive neural predicate->argument graph INSIDE Calibri.

Physical S106 frozen real-original12 TRAIN-only analysis (17/442 top1
flips from S105 learned 8,200 affinity weights) found small morphological
residual logit changes relative to the old lexical native-BPE role margins.
S105 internal diagnostic is SPENT; never tune on it. This is a NEW
architecture, with NO inference teacher, parser, label or ground truth.

Preserves every previous 37,087 in-Calibri neural weight including all
8 original NRA2 mini-areas, 12-layer learned Green Tunnel routing,
16 relation memory slots, S97 morphology and S103 morph affinities.
Adds trainable predicate/argument structural Q/K, eight bilinear role
operators, directional native-BPE distance biases and a zero-init gate.
Unlike additive S105 corrections, learnable per-role *normalized*
competition can displace existing lexical ranking at full gate.

Every original Calibri/CB59/GT weight remains frozen. Gate zero means
bit-identical S105 original child output on all native BPE positions.
No LIVE, main, Render, Supabase, Jessica, APK, merge/deploy; not active.
Quality or retention NOT established; future TRAIN-only validation is
required BEFORE independent first-read. Old diagnostics all SPENT and
entire official Taiga TRAIN-c prospective split UNOPENED.
"""
from __future__ import annotations

from dataclasses import replace
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    N_ROLES,N_STREAMS,PROJ,HIDDEN,N_LAYERS,require,
)
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,
)
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,MORPH_FEATURES,
)

STAGE="CALIBRI_NRA3_S107_TWELVE_LAYER_STRUCTURAL_PREDICATE_ARGUMENT_COMPETITION_GATE0"
STREAM_FEATURES=N_STREAMS*PROJ
PAIR_INPUT=STREAM_FEATURES+MORPH_FEATURES
STRUCTURAL_DIM=32
DISTANCE_BINS=33
NEW_NEURAL_PARAMETERS=(2*PAIR_INPUT*STRUCTURAL_DIM+
                       N_ROLES*STRUCTURAL_DIM*STRUCTURAL_DIM+
                       DISTANCE_BINS*N_ROLES+N_ROLES)
OLD_NEURAL_PARAMETERS=37087
TOTAL_NEURAL_PARAMETERS=OLD_NEURAL_PARAMETERS+NEW_NEURAL_PARAMETERS
NEW_KEYS=frozenset((
    "structural_predicate_query.weight",
    "structural_argument_key.weight",
    "structural_role_bilinear",
    "structural_native_BPE_distance.weight",
    "structural_competition_gate",
))


class NRA3StructuralCompetitionInCalibriWeights(
        NRA3RelationalMorphologyInCalibriWeights):
    """In-model learned 12-layer predicate/argument, normalized across ARG positions."""

    def __init__(self):
        super().__init__()
        self.structural_predicate_query=nn.Linear(
            PAIR_INPUT,STRUCTURAL_DIM,bias=False)
        self.structural_argument_key=nn.Linear(
            PAIR_INPUT,STRUCTURAL_DIM,bias=False)
        self.structural_role_bilinear=nn.Parameter(
            torch.empty(N_ROLES,STRUCTURAL_DIM,STRUCTURAL_DIM))
        nn.init.xavier_uniform_(self.structural_role_bilinear)
        self.structural_native_BPE_distance=nn.Embedding(
            DISTANCE_BINS,N_ROLES)
        nn.init.zeros_(self.structural_native_BPE_distance.weight)
        self.structural_competition_gate=nn.Parameter(torch.zeros(N_ROLES))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.is_trained=False

    def budget(self):
        # Bypass S103's exact-37087 budget() assertion: inherited params
        # are strictly retained and child size is now 53743.
        v=NRA3MorphologyInCalibriWeights.budget(self)
        require(v["new_neural_parameters"]==TOTAL_NEURAL_PARAMETERS
                and NEW_NEURAL_PARAMETERS==16656
                and TOTAL_NEURAL_PARAMETERS==53743,
                "only 16,656 appended neural weights, 37,087 inherited unchanged")
        v.update({
            "stage":STAGE,
            "old_in_calibri_NRA3_S105_params_preserved":OLD_NEURAL_PARAMETERS,
            "new_structural_neural_params":NEW_NEURAL_PARAMETERS,
            "in_calibri_child_NOT_external_model":True,
            "NRA2_eight_original_mini_areas_preserved":True,
            "in_weight_memory_preserved":True,
            "original12_Calibri_GreenTunnel_learned_router":True,
            "native_BPE_source_only_predicate_argument_competition":True,
            "new_structural_UD_gold_or_external_parser_required_at_inference":False,
            "S105_internal_dev_SPENT_no_reuse":True,
            "prospective_Taiga_TRAIN_c_reserved_UNOPENED":True,
            "new_quality_not_yet_established":True,
        })
        return v

    def source_only_pair_features(self,old_graph,original_all12_hidden):
        """Full12 source neural route + predictions, never teacher tags."""
        h=original_all12_hidden
        valid=old_graph.native_valid
        require(h.ndim==4 and h.shape[1]==N_LAYERS
                and h.shape[-1]==HIDDEN
                and h.shape[0]==valid.shape[0]
                and h.shape[2]==valid.shape[1]
                and (not h.requires_grad or
                     getattr(self, '_unified_core_grad_only_research', False))
                and bool(torch.isfinite(h).all()),
                "strict frozen original twelve Calibri+GT source only")
        routed=torch.einsum(
            "sl,blth->bsth",old_graph.layer_router_weights,h)
        projected=self.shared_hidden_projector(routed)
        b,_,t,_=projected.shape
        streams=projected.permute(0,2,1,3).reshape(
            b,t,STREAM_FEATURES)
        morph=self.source_only_morph_features(old_graph)
        v=torch.cat((streams,morph),dim=-1)
        v=v.masked_fill(~valid[...,None],0.)
        require(v.shape==(b,t,PAIR_INPUT)
                and bool(torch.isfinite(v).all()),
                "source-only 12-layer+32 predicted morphology neural pair features")
        return v

    def structural_role_scores(self,base,original_all12_hidden):
        features=self.source_only_pair_features(
            base,original_all12_hidden)
        q=self.structural_predicate_query(features)
        k=self.structural_argument_key(features)
        structural=torch.einsum(
            "bti,rij,bsj->brts",q,self.structural_role_bilinear,k
        )/sqrt(STRUCTURAL_DIM)
        t=features.shape[1]
        ids=torch.arange(t,device=features.device)
        relative=(ids[None,:]-ids[:,None]).clamp(-16,16)+16
        bias=self.structural_native_BPE_distance(
            relative.long()).permute(2,0,1)
        structural=structural+bias[None]
        valid=base.native_valid
        pair=valid[:,None,:,None]&valid[:,None,None,:]
        require(structural.shape==base.role_arc_logits.shape
                and torch.isfinite(structural).all(),
                "role-specific structural native-BPE predicate argument pair scores")
        return structural.masked_fill(~pair,-1.e4)

    def combine_role_competition(self,base_scores,structural_scores,native_valid):
        """Rank competition, NOT small additive bias.

        New = old + alpha * (logP_struct - logP_old), alpha=tanh(gate).
        At exactly zero alpha, logits are bit-for-bit identical to old.
        As alpha approaches +1, ranking follows structural P instead of
        old lexical rank; no forced scalar old-logit margin obstacle.
        Alpha is learned and can be negative, never label-set by hand.
        """
        require(base_scores.shape==structural_scores.shape
                and base_scores.ndim==4
                and base_scores.shape[1]==N_ROLES
                and native_valid.shape==(base_scores.shape[0],base_scores.shape[-1])
                and native_valid.dtype==torch.bool
                and torch.isfinite(base_scores).all()
                and torch.isfinite(structural_scores).all(),
                "source-only real predicate role arc shape + finite scores")
        pair=(native_valid[:,None,:,None]&native_valid[:,None,None,:])
        old=base_scores.float().masked_fill(~pair,-1.e4)
        new=structural_scores.float().masked_fill(~pair,-1.e4)
        old_log=F.log_softmax(old,dim=-1)
        new_log=F.log_softmax(new,dim=-1)
        gate=torch.tanh(self.structural_competition_gate)[
            None,:,None,None]
        result=old+gate*(new_log-old_log)
        result=result.masked_fill(~pair,-1.e4)
        require(result.shape==base_scores.shape
                and torch.isfinite(result).all(),
                "learnable calibrated role-wise structural arg distribution")
        return result

    def forward(self,original_layer_hidden,source_native_token_valid):
        base=super().forward(
            original_layer_hidden,source_native_token_valid)
        new=self.structural_role_scores(base,original_layer_hidden)
        arcs=self.combine_role_competition(
            base.role_arc_logits,new,base.native_valid)
        return replace(base,role_arc_logits=arcs)
