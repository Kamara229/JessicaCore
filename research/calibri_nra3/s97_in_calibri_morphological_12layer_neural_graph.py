"""NRA3 S97: IN-Calibri neural morphology + relational arcs, not a sidecar.

S96 true original Calibri+GreenTunnel all12 yielded 75.08% teacher-
anchored UD role accuracy on 317 TRAIN edges but just 15.45% on 110
same-TRAIN-a internal-development edges (not independent first-read).
It demonstrated template overfitting, not reliable syntax. NO tuning
using the already-spent S96 internal diagnostic or earlier S79/S86.

S97 keeps the old eight independent NRA2 mini-areas, original NRA3
neural event-role graph, all-12 learned layer streams, and in-weight
event memory prototypes. It adds trainable token-level UPOS, Case,
Number and Voice heads INSIDE the same registered Calibri child, plus
morphology-conditioned role-arc and mini-area corrections. Neither
NRA3.forward nor Calibri.forward imports the UD parser or uses gold.
External UD/CoNLL-U is OFFLINE TRAIN LABELS ONLY (separate module).

Original CB59/GreenTunnel/S59/S65 completely frozen; gate zero.
No actual syntax transfer, protected integrated616, bpp or CPU
certification by constructing this architecture alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3InCalibriNeuralWeights,
    NRA3NeuralGraph,
    N_LAYERS,N_ROLES,N_MICRO,PROJ,N_STREAMS,
    require,
)

STAGE="CALIBRI_NRA3_S97_MORPHOLOGY_RELATIONAL_IN_CALIBRI_NEURAL_WEIGHTS_GATE0"

UD_UPOS=("NOUN","PROPN","PRON","VERB","AUX","ADJ","ADV","ADP",
         "DET","NUM","PART","CCONJ","SCONJ","INTJ","PUNCT","SYM","X")
UD_CASE=("Nom","Gen","Dat","Acc","Ins","Loc","Par","Voc","Other")
UD_NUMBER=("Sing","Plur","Other")
UD_VOICE=("Act","Pass","Mid")
MORPH_D=32
ARC_D=12
DISTANCE_BINS=33


@dataclass(frozen=True)
class NRA3MorphologyGraph(NRA3NeuralGraph):
    upos_logits:torch.Tensor       # [B,T,17]
    case_logits:torch.Tensor       # [B,T,9]
    number_logits:torch.Tensor     # [B,T,3]
    voice_logits:torch.Tensor      # [B,T,3]
    # All old NRA2 miniheads and event-memory attention remain base fields.


class NRA3MorphologyInCalibriWeights(NRA3InCalibriNeuralWeights):
    """Same actual Calibri model.model.nra3_neural_weights child, V2.

    The inherited 22,311 trainable S91 weights and all eight distinct
    NRA2 microheads are retained by EXACT state_dict names. New morph
    weights and role refinements are appended, not a replacement model.
    """

    def __init__(self):
        super().__init__()
        self.morphology_fuse=nn.Linear(N_STREAMS*PROJ,MORPH_D,bias=False)
        self.morph_upos=nn.Linear(MORPH_D,len(UD_UPOS))
        self.morph_case=nn.Linear(MORPH_D,len(UD_CASE))
        self.morph_number=nn.Linear(MORPH_D,len(UD_NUMBER))
        self.morph_voice=nn.Linear(MORPH_D,len(UD_VOICE))
        self.morph_role_q=nn.Linear(MORPH_D,ARC_D,bias=False)
        self.morph_role_k=nn.Linear(MORPH_D,ARC_D,bias=False)
        self.morph_role_bilinear=nn.Parameter(
            torch.empty(N_ROLES,ARC_D,ARC_D))
        nn.init.xavier_uniform_(self.morph_role_bilinear)
        self.source_relative_distance=nn.Embedding(DISTANCE_BINS,N_ROLES)
        nn.init.zeros_(self.source_relative_distance.weight)
        self.morph_role_scale=nn.Parameter(torch.zeros(N_ROLES))
        self.morph_to_nra2_miniareas=nn.Linear(
            MORPH_D,N_MICRO,bias=False)
        nn.init.zeros_(self.morph_to_nra2_miniareas.weight)
        # Every initial new contribution is exactly zero. The S91
        # existing eight miniareas/role arc values remain unchanged.
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.is_trained=False

    def budget(self):
        result=super().budget()
        result.update({
            "stage":STAGE,
            "in_calibri_child_old_nra3_s91_weights_retained":True,
            "all_eight_NRA2_neural_miniareas_retained":True,
            "new_morphological_supervision":(
                "UPOS","Case","Number","Voice"),
            "source_relative_role_bias_neural_not_hardcoded_relation":True,
            "external_inference_UD_parser_required":False,
            "no_grammar_generalization_claim":True,
        })
        return result

    def forward(self,original_layer_hidden,source_native_token_valid):
        base=super().forward(original_layer_hidden,source_native_token_valid)
        h=original_layer_hidden
        valid=source_native_token_valid
        # Learned four source-only router distributions use all 12
        # genuine original Calibri+GT frozen hidden states.
        routed=torch.einsum(
            "sl,blth->bsth",base.layer_router_weights,h)
        projected=self.shared_hidden_projector(routed)
        batch,_,width,_=projected.shape
        flattened=projected.permute(0,2,1,3).reshape(
            batch,width,N_STREAMS*PROJ)
        morph=F.gelu(self.morphology_fuse(flattened))
        outputs=(
            self.morph_upos(morph),
            self.morph_case(morph),
            self.morph_number(morph),
            self.morph_voice(morph),
        )
        outputs=tuple(z.masked_fill(~valid[:,:,None],-1.e4)
                      for z in outputs)
        q=self.morph_role_q(morph)
        k=self.morph_role_k(morph)
        new_arcs=torch.einsum("bti,rij,bsj->brts",q,
                             self.morph_role_bilinear,k)/sqrt(ARC_D)
        # Relative native-BPE event->argument distance is a trainable
        # neural bias, not a hard-coded subject/object decision.
        idx=torch.arange(width,device=h.device)
        distance=(idx[None,:]-idx[:,None]).clamp(-16,16)+16
        relative=self.source_relative_distance(
            distance.long()).permute(2,0,1)
        new_arcs=new_arcs+relative[None]
        scales=torch.tanh(self.morph_role_scale)[None,:,None,None]
        arcs=base.role_arc_logits+scales*new_arcs
        pairs=valid[:,None,:,None]&valid[:,None,None,:]
        arcs=arcs.masked_fill(~pairs,-1.e4)
        minis=base.mini_area_logits+self.morph_to_nra2_miniareas(morph)
        minis=minis.masked_fill(~valid[:,:,None],-1.e4)
        require(arcs.shape==base.role_arc_logits.shape
                and minis.shape==base.mini_area_logits.shape
                and all(torch.isfinite(z).all() for z in outputs)
                and torch.isfinite(arcs).all()
                and torch.isfinite(minis).all(),
                "morphology in-Calibri output shape/finite ABI")
        return NRA3MorphologyGraph(
            event_mention_logits=base.event_mention_logits,
            role_arc_logits=arcs,
            event_state_logits=base.event_state_logits,
            mini_area_logits=minis,
            layer_router_weights=base.layer_router_weights,
            memory_slot_attention=base.memory_slot_attention,
            native_valid=valid,
            upos_logits=outputs[0],
            case_logits=outputs[1],
            number_logits=outputs[2],
            voice_logits=outputs[3],
        )
