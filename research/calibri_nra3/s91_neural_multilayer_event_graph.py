"""NRA3 S91: trainable event/grammar/knowledge neural weights INSIDE Calibri.

A Calibri model can register this nn.Module as a child of model.model,
not as an external language-processing service. The child reads
the ORIGINAL hidden states of all 12 layers through opt-in read-only
research hooks. Each old layer output is returned unchanged (None).
There are eight separate trainable mini-area heads retained from NRA2,
plus neural grammatical event->argument arcs, event state heads,
four learned streams routing all 12 layers, and learnable event memory.

THIS IS AN UNTRAINED RESEARCH ARCHITECTURE. Graph and prototype
outputs are not certified grammatical relations or persistent facts.
All original Calibri/GT/paid weights and original616 J7 are protected.
No S79 or S86 SPENT first-read data, GOLD, external runtime parser,
LIVE, main, Render, Supabase, Jessica, APK, merge/deploy.
No activation until independent checks, output_gate=0.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s20.s82_predicate_centered_situation_fields import AREAS

STAGE = "CALIBRI_NRA3_S91_NEURAL_12LAYER_RELATIONAL_EVENT_GRAPH_RESEARCH_ONLY"
N_LAYERS, HIDDEN, N_STREAMS, PROJ, CORE, ARC = 12, 320, 4, 24, 32, 16
STREAM_NAMES = ("morphology","syntax","event_semantics","knowledge")
ROLE_NAMES = ("actor","patient","recipient","negation_scope",
              "speech_source","time_anchor","coreference","event_relation")
STATE_NAMES = ("assertion","negated","reported","hypothetical",
               "uncertain","passive")
N_ROLES, N_STATES, N_MICRO, MEMORY_SLOTS = len(ROLE_NAMES),len(STATE_NAMES),len(AREAS),16


def require(value, reason):
    if not bool(value):
        raise ValueError("NRA3 S91 in-Calibri neural graph: "+reason)


@dataclass(frozen=True)
class NRA3NeuralGraph:
    event_mention_logits: torch.Tensor  # [B,T]
    role_arc_logits: torch.Tensor       # [B,8,T,T] event->argument
    event_state_logits: torch.Tensor    # [B,T,6]
    mini_area_logits: torch.Tensor      # [B,T,8]
    layer_router_weights: torch.Tensor  # [4,12]
    memory_slot_attention: torch.Tensor # [B,T,16]
    native_valid: torch.Tensor          # [B,T] original source BPE


class NRA3InCalibriNeuralWeights(nn.Module):
    """Persistent native Calibri neural child, never an external service."""

    def __init__(self):
        super().__init__()
        self.layer_router_logits = nn.Parameter(torch.zeros(N_STREAMS,N_LAYERS))
        self.shared_hidden_projector = nn.Linear(HIDDEN,PROJ,bias=False)
        self.combine_streams = nn.Linear(N_STREAMS*PROJ,CORE,bias=False)
        self.mention_head = nn.Linear(CORE,1)
        self.role_query = nn.Linear(CORE,ARC,bias=False)
        self.role_key = nn.Linear(CORE,ARC,bias=False)
        self.role_bilinear = nn.Parameter(torch.empty(N_ROLES,ARC,ARC))
        nn.init.xavier_uniform_(self.role_bilinear)
        self.role_attention_mix = nn.Parameter(torch.zeros(N_ROLES))
        self.event_memory_slots = nn.Parameter(torch.empty(MEMORY_SLOTS,CORE))
        nn.init.normal_(self.event_memory_slots,std=.02)
        self.memory_query = nn.Linear(CORE,CORE,bias=False)
        self.event_state_head = nn.Linear(CORE*3,N_STATES)
        self.mini_areas = nn.ModuleDict({
            name:nn.Sequential(nn.Linear(CORE*3,8),nn.GELU(),nn.Linear(8,1))
            for name in AREAS})
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.online_causal_certified=False
        self.protected_616_candidate_certified=False
        self.is_trained=False

    def budget(self):
        n=sum(int(p.numel()) for p in self.parameters())
        return {
            "new_neural_parameters":n,
            "nominal_FP16_weight_bytes":2*n,
            "all_12_original_calibri_layers":N_LAYERS,
            "learned_layer_streams":N_STREAMS,
            "grammatical_event_argument_role_heads":N_ROLES,
            "NRA2_independent_mini_areas_preserved":N_MICRO,
            "event_state_heads":N_STATES,
            "trainable_in_model_memory_slots":MEMORY_SLOTS,
            "external_runtime_parser_required":False,
            "original_calibri_GT_weights_modified":False,
            "new_616_J7_not_certified":True,
            "all_in_bpp_and_CPU_not_certified":True,
            "output_gate":self.output_gate,
            "candidate_accepted":self.candidate_accepted,
            "activation_allowed":self.activation_allowed,
        }

    def forward(self,original_layer_hidden,source_native_token_valid):
        """No oracle event/role labels. Multi-layer graph from source only.

        Original frozen hidden: [B,12,T,320]. Native BPE valid: [B,T].
        Event arc position i -> candidate argument j for each neural role.
        """
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed
                and not self.online_causal_certified,
                "cannot activate experimental neural graph")
        h=original_layer_hidden;valid=source_native_token_valid
        require(h.ndim==4 and h.shape[1]==N_LAYERS
                and h.shape[-1]==HIDDEN and h.shape[0]>=1
                and h.shape[2]>=2 and not h.requires_grad
                and valid.shape==(h.shape[0],h.shape[2])
                and valid.dtype==torch.bool
                and bool(valid.any(-1).all())
                and bool(torch.isfinite(h).all()),
                "genuine original twelve layer detached source tensor")
        layer_mix=F.softmax(self.layer_router_logits,dim=-1)
        streams=torch.einsum("sl,blth->bsth",layer_mix,h)
        projected=self.shared_hidden_projector(streams)
        b,_,t,_=projected.shape
        base=self.combine_streams(
            projected.permute(0,2,1,3).reshape(b,t,-1))
        mentions=self.mention_head(base).squeeze(-1).masked_fill(~valid,-1.e4)
        queries=self.role_query(base);keys=self.role_key(base)
        arcs=torch.einsum("bti,rij,bsj->brts",queries,
                          self.role_bilinear,keys)/math.sqrt(ARC)
        pair_valid=valid[:,None,:,None]&valid[:,None,None,:]
        arcs=arcs.masked_fill(~pair_valid,-1.e4)
        relations=torch.einsum("brts,bsc->brtc",
                               F.softmax(arcs.float(),dim=-1).to(base.dtype),base)
        weighted_roles=F.softmax(self.role_attention_mix,dim=-1)
        role_context=torch.einsum("r,brtc->btc",weighted_roles,relations)
        memory_logits=torch.einsum(
            "btc,mc->btm",self.memory_query(base),
            self.event_memory_slots)/math.sqrt(CORE)
        memory_attention=F.softmax(memory_logits,dim=-1)
        memory_context=torch.einsum(
            "btm,mc->btc",memory_attention,self.event_memory_slots)
        joint=torch.cat((base,role_context,memory_context),dim=-1)
        states=self.event_state_head(joint).masked_fill(~valid[...,None],-1.e4)
        minis=torch.stack([head(joint).squeeze(-1)
                           for head in self.mini_areas.values()],dim=-1)
        minis=minis.masked_fill(~valid[...,None],-1.e4)
        require(arcs.shape==(b,N_ROLES,t,t)
                and states.shape==(b,t,N_STATES)
                and minis.shape==(b,t,N_MICRO)
                and memory_attention.shape==(b,t,MEMORY_SLOTS)
                and torch.isfinite(arcs).all()
                and torch.isfinite(states).all()
                and torch.isfinite(minis).all(),
                "invalid neural event graph or area ABI")
        return NRA3NeuralGraph(mentions,arcs,states,minis,
                              layer_mix,memory_attention,valid)

    def preserve_original_output(self,original_outputs,
                                 _graph:Optional[NRA3NeuralGraph]=None):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed,
                "research block cannot influence original decoder")
        return original_outputs  # exact object identity, not numeric approximation

    def knowledge_consolidation_penalty(self,older_frozen_slots):
        """TRAIN-only distillation candidate; never proof of no forgetting."""
        require(older_frozen_slots.shape==self.event_memory_slots.shape
                and not older_frozen_slots.requires_grad
                and torch.isfinite(older_frozen_slots).all(),
                "prior learned slots must be immutable non-GOLD snapshot")
        return F.mse_loss(self.event_memory_slots,
                          older_frozen_slots.detach())

    def attach_to_RESEARCH_calibri(self,model):
        """Register params as direct Calibri child: model.model.nra3_neural_weights."""
        backbone=getattr(model,"model",None)
        layers=getattr(backbone,"layers",None)
        require(isinstance(backbone,nn.Module)
                and layers is not None and len(layers)==N_LAYERS
                and all(isinstance(x,nn.Module) for x in layers)
                and not hasattr(backbone,"nra3_neural_weights")
                and self.output_gate==0 and not self.candidate_accepted,
                "only fresh 12-layer research Calibri accepts new weights")
        backbone.add_module("nra3_neural_weights",self)
        self.eval()
        require(backbone.nra3_neural_weights is self,
                "NRA3 MUST be part of actual Calibri module weights")
        return self

    @contextmanager
    def read_only_all12_source_scope(self,model):
        """Opt-in original CPU prefill, NOT installed during protected616 J7."""
        backbone=getattr(model,"model",None)
        require(isinstance(backbone,nn.Module)
                and getattr(backbone,"nra3_neural_weights",None) is self
                and len(backbone.layers)==N_LAYERS
                and not hasattr(self,"_active_s91")
                and self.output_gate==0 and not self.candidate_accepted,
                "research model must own inactive NRA3 neural weights")
        self._active_s91=True
        captured={}
        handles=[]
        self.latest_source_graph=None
        self.latest_source_hidden=None
        self.capture_count=0
        def make_hook(index):
            def observe(_layer,_args,out):
                require(index not in captured,
                        "each original Calibri layer observed exactly once")
                hidden=out[0] if isinstance(out,tuple) else out
                require(isinstance(hidden,torch.Tensor)
                        and hidden.ndim==3 and hidden.shape[-1]==HIDDEN
                        and hidden.shape[0]>=1 and hidden.shape[1]>=2,
                        "original Calibri layer result must be [B,T,320]")
                captured[index]=hidden.detach()
                if index==N_LAYERS-1:
                    require(len(captured)==N_LAYERS,
                            "all original twelve layers must run before graph")
                    all_hidden=torch.stack([captured[i] for i in range(N_LAYERS)],1)
                    mask=torch.ones((all_hidden.shape[0],all_hidden.shape[2]),
                                    device=all_hidden.device,dtype=torch.bool)
                    self.latest_source_hidden=all_hidden.detach().to(torch.float16).contiguous().clone()
                    # Original 43M/GT often runs FP16; neural child CPU
                    # research weights are FP32 until final export.
                    self.latest_source_graph=self.forward(all_hidden.detach().float(),mask)
                    self.capture_count+=1
                return None  # preserves actual original residual object
            return observe
        try:
            handles=[layer.register_forward_hook(make_hook(i))
                     for i,layer in enumerate(backbone.layers)]
            yield self
            require(self.capture_count==1 and len(captured)==N_LAYERS
                    and self.latest_source_graph is not None,
                    "S91 one and only one read-only prefill per scope")
        finally:
            for handle in handles:handle.remove()
            del self._active_s91
