"""NRA3 S112: bounded source-only structural competition INSIDE Calibri weights.

S111 original12 BPE source-only (NO GOLD) found S110 role-logit correction
magnitude 6.4–11.2 vs old margin 1.7–4.6, so structural prediction flips
55–87% of unlabeled positions. This is NEURAL LOGIT OVERSHOOT.

Keep all 53,743 learned S110 in-Calibri params bit-identical incl. eight
NRA2 mini-areas, 12 learned layer route, 16 learned memory slots, morph
and all S107 predicate/query/key/bilinear/distance weights. Add EXACTLY
eight new in-weight trainable role trust parameters, initialized to 0.
Bound any correction in native-BPE logit units by |delta|<=1 per candidate,
so with any trust value an old top1-vs-top2 margin >2 cannot flip.
At zero trust, exact old 37,087 Calibri NRA3 role logits; no teacher,
parser or UD GOLD needed at inference. Freeze output gate0 ALWAYS.

This is a structurally safe neural mechanism, NOT an accuracy result.
No retraining against spent GSD internal/S96/S99/S102/S105 dev,
no burned TaigaC, no main/LIVE/Render/Supabase/deploy.
"""
from __future__ import annotations
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from research.calibri_nra3.s91_neural_multilayer_event_graph import N_ROLES,require
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import NRA3MorphologyInCalibriWeights
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import NRA3RelationalMorphologyInCalibriWeights
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import NRA3StructuralCompetitionInCalibriWeights

OLD=53743
NEW=8
TOTAL=OLD+NEW
MAX_LOGIT_CORRECTION=1.0
STAGE="CALIBRI_NRA3_S112_BOUNDED_PREDICATE_ARGUMENT_NEURAL_TRUST_INMODEL_GATE0"

class NRA3BoundedStructuralTrustInCalibriWeights(NRA3StructuralCompetitionInCalibriWeights):
    def __init__(self):
        super().__init__()
        self.structural_trust_gate=nn.Parameter(torch.zeros(N_ROLES))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.is_trained=False

    def budget(self):
        v=NRA3MorphologyInCalibriWeights.budget(self)
        require(v["new_neural_parameters"]==TOTAL and OLD==53743 and NEW==8,
                "exact 53,751 in-Calibri weights; 53,743 old FP16 unchanged")
        v.update({
            "stage":STAGE,
            "old_neural_53743_exactly_retained":True,
            "only_8_new_learned_role_trust_parameters":True,
            "structural_correction_strictly_bounded_one_native_BPE_logit":True,
            "zero_trust_exact_prior_in_Calibri_role_logits":True,
            "NRA2_original8_mini_areas_12layer_router_memory_morphology_kept":True,
            "no_external_runtime_dependency_or_teacher_or_GOLD":True,
            "internal_GSD_dev_SPENT_never_reused":True,
            "TaigaC_firstread_BURNED_never_reused":True,
            "new_quality_not_proven":True,
        })
        return v

    def bounded_role_competition(self,old_logits,raw_structural_logits,native_valid):
        """Learned in-model trust * capped distribution contrast.

        old_top1-old_top2 > 2 means mathematically impossible to flip by
        this operation for any trust setting (absolute candidate change
        <=1, pair preference change <=2). No label enters the forward.
        """
        require(old_logits.ndim==4 and old_logits.shape==raw_structural_logits.shape
                and old_logits.shape[1]==N_ROLES
                and native_valid.dtype==torch.bool
                and native_valid.shape==(old_logits.shape[0],old_logits.shape[-1])
                and bool(torch.isfinite(old_logits).all())
                and bool(torch.isfinite(raw_structural_logits).all()),
                "actual masked original12 role distributions, no teacher labels")
        pair=native_valid[:,None,:,None]&native_valid[:,None,None,:]
        prior=old_logits.float().masked_fill(~pair,-1.e4)
        structural=raw_structural_logits.float().masked_fill(~pair,-1.e4)
        old_log=F.log_softmax(prior,dim=-1)
        new_log=F.log_softmax(structural,dim=-1)
        delta=MAX_LOGIT_CORRECTION*torch.tanh(
            (new_log-old_log)/MAX_LOGIT_CORRECTION)
        trust=torch.tanh(self.structural_trust_gate)[None,:,None,None]
        combined=prior+trust*delta
        combined=combined.masked_fill(~pair,-1.e4)
        require(combined.shape==prior.shape and bool(torch.isfinite(combined).all()),
                "bounded corrected role logits remain finite with masked BPE")
        # All native pairwise logit corrections (valid candidates) <=1.
        require(not bool((((combined-prior).abs()>
                          MAX_LOGIT_CORRECTION+1.e-5)&pair).any()),
                "strict bounded S112 residual cannot override large old margins")
        return combined

    def forward(self,original_layer_hidden,source_native_token_valid):
        # Jump over S107 unbounded rank override; reuse its LEARNED
        # predicate/argument Q/K, all role bilinear + relative offsets.
        base=NRA3RelationalMorphologyInCalibriWeights.forward(
            self,original_layer_hidden,source_native_token_valid)
        raw=self.structural_role_scores(base,original_layer_hidden)
        corrected=self.bounded_role_competition(
            base.role_arc_logits,raw,base.native_valid)
        return replace(base,role_arc_logits=corrected)
