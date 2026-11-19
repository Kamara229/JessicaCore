"""NRA3 S103: learnable STRUCTURAL morphology-to-argument affinity, IN CALIBRI.

Rationale: physical S102 on 251 real original12 Calibri+GT TRAIN-d
source had teacher-anchored role top1 416/432 TRAIN but 39/160
same TRAIN-d developer, including patient 3/52. This diagnosis is
SPENT. S103 introduces a genuine *neural* inductive bias: argument
roles are scored from predicted POS, Case, Number, Voice distributions
of predicate and candidate argument rather than their lexical 320D
hidden vectors alone. No linguistics parser or labels at runtime.

This child subclasses the old 28,887-param morphology NRA3 IN the
Calibri module state_dict, preserving all previous eight distinct
NRA2 mini-area heads, original neural arcs, 12-layer routing, and
in-weight relational memories. Add exactly 8×32×32 trainable affinity
weights and eight neural residual gates = 8,200 additional weights,
total 37,087. At initialization, residual gate zeros guarantee
exactly identical S102 output (including old role decisions).
No hardcoded grammatical case→semantic role lookup.

CPU all-in bpp, online causal, independent J7 and naturalistic
factuality remain unverified. Research output gate=0, not active.
"""
from __future__ import annotations

from dataclasses import replace
from math import sqrt

import torch
from torch import nn

from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,UD_UPOS,UD_CASE,UD_NUMBER,UD_VOICE,
)
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    N_ROLES,require,
)

STAGE="CALIBRI_NRA3_S103_MORPHO_SYNTACTIC_ROLE_AFFINITY_IN_CALIBRI_RESEARCH"
MORPH_CLASSES=(len(UD_UPOS),len(UD_CASE),len(UD_NUMBER),len(UD_VOICE))
MORPH_FEATURES=sum(MORPH_CLASSES)
NEW_AFFINITY_PARAMETERS=N_ROLES*MORPH_FEATURES*MORPH_FEATURES+N_ROLES
NEW_TOTAL_NRA3_PARAMS=28887+NEW_AFFINITY_PARAMETERS


class NRA3RelationalMorphologyInCalibriWeights(NRA3MorphologyInCalibriWeights):
    """Trainable prediction-driven predicate→argument compatibility.

    All grammar probabilities are inferred only from original Calibri
    all12 input; labels are used solely by an OFFLINE trainer.
    """

    def __init__(self):
        super().__init__()
        self.role_morph_affinity=nn.Parameter(torch.empty(
            N_ROLES,MORPH_FEATURES,MORPH_FEATURES))
        nn.init.xavier_uniform_(self.role_morph_affinity)
        self.role_morph_affinity_gate=nn.Parameter(torch.zeros(N_ROLES))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.is_trained=False

    def budget(self):
        old=super().budget()
        require(old["new_neural_parameters"]==NEW_TOTAL_NRA3_PARAMS,
                "all old original S97/S91 8 NRA2 miniheads plus precisely 8200 neural affinity weights")
        old.update({
            "stage":STAGE,
            "structural_morphology_feature_dimension":MORPH_FEATURES,
            "new_morphosyntactic_predicted_role_pair_parameters":
                NEW_AFFINITY_PARAMETERS,
            "all_predicted_UPOS_Case_Number_Voice_used_for_role_arc":True,
            "old_neural_NRA2_eight_mini_areas_preserved":True,
            "old_in_weight_memory_and_all_12_router_preserved":True,
            "no_grammar_gold_or_external_parser_at_neural_inference":True,
            "no_patient_role_transfer_claim":True,
        })
        return old

    @staticmethod
    def source_only_morph_features(graph):
        """No gold: neural 17+9+3+3 predictive distributions."""
        heads=(graph.upos_logits,graph.case_logits,
               graph.number_logits,graph.voice_logits)
        valid=graph.native_valid
        require(valid.dtype==torch.bool and valid.ndim==2
                and all(x.shape[:2]==valid.shape for x in heads),
                "same original source-native BPE token geometry for morphology")
        predictions=[x.float().softmax(dim=-1) for x in heads]
        features=torch.cat(predictions,dim=-1)
        features=features.masked_fill(~valid[...,None],0.)
        require(features.shape==(valid.shape[0],valid.shape[1],MORPH_FEATURES)
                and torch.isfinite(features).all(),
                "learned neural morphosyntax feature distribution valid")
        return features

    def morph_role_correction(self,graph):
        """Explicit pair compatibility is equivariant to *word permutation*
        when frozen source-only tag predictions are permuted as well.
        This structural score does NOT use absolute lexical IDs.
        """
        z=self.source_only_morph_features(graph)
        corrected=torch.einsum(
            "bti,rij,bsj->brts",z,self.role_morph_affinity,z
        )/sqrt(MORPH_FEATURES)
        valid=graph.native_valid
        corrected=corrected.masked_fill(
            ~(valid[:,None,:,None]&valid[:,None,None,:]),0.)
        require(corrected.shape==graph.role_arc_logits.shape
                and torch.isfinite(corrected).all(),
                "neural argument morphology all-roles pair score valid")
        return corrected

    def forward(self,original_layer_hidden,source_native_token_valid):
        base=super().forward(original_layer_hidden,source_native_token_valid)
        residual=self.morph_role_correction(base)
        scales=torch.tanh(self.role_morph_affinity_gate)[None,:,None,None]
        scores=base.role_arc_logits+scales*residual
        valid=base.native_valid
        scores=scores.masked_fill(
            ~(valid[:,None,:,None]&valid[:,None,None,:]),-1.e4)
        require(torch.isfinite(scores).all(),
                "all new neural morphology-conditioned role scores finite")
        return replace(base,role_arc_logits=scores)
