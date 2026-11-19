"""NRA2 S72 situation micro-areas: compositional, frozen-parent, shadow-only.

Motivation: S66 first-read 32 recipient F1 75.47% and multiple false
recipient edges in quotation/report-only or not-performed transfer.
A single global S65 attribution/NEGATION classifier is not an adequate
representation of simultaneous quote + negation + passive + time + roles.

Eight TRAINABLE independent micro-areas (names = supervision contracts):
  quoted_or_reported, negation_scope, passive_role_assignment,
  temporal_revision, hypothetical_or_dream, reference_resolution,
  competing_event_clauses, uncertainty_abstain.

NOT mutually exclusive sentence classes: each predicate may engage up to
four areas. Exact real original 43M native Layer8 320D word vectors and
S65 source-only event features are inputs, along with word positions.
No handwritten GOLD enters forward. The old S65/S63/S62 logits and all
original 43M/GT/expansion weights are FROZEN INPUTS. Shadow deltas are
not connected to an accepted/active output. No training, historical held
data, EVAL32/EVAL128, or production code in this module.

This stage only establishes a trainable architecture, valid shape/masking,
independent multi-label routing, and zero-effect initialization. It does
NOT claim increased accuracy, genuinely causal streaming, retention, or
physical bpp/CPU certification. Subsequent fitting must use OLD TRAIN
only, source-group-disjoint splits and prospective FIRST-READ evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

AREAS=(
    "quoted_or_reported",
    "negation_scope",
    "passive_role_assignment",
    "temporal_revision",
    "hypothetical_or_dream",
    "reference_resolution",
    "competing_event_clauses",
    "uncertainty_abstain",
)
EVENT_CUES=36  # exact S65 source discourse features
WORD_D=320     # original frozen original43M Layer8 post-GT
WORD_RANK=12
EVENT_RANK=12
ROUTER_INPUT=EVENT_CUES+WORD_RANK
PAIR_GEOMETRY=4
MAX_ACTIVE=4
NEG_INF=-1.e9


def require(condition:bool,message:str)->None:
    if not bool(condition):
        raise ValueError("NRA2 S72 fail-closed shadow micro-areas: "+message)


@dataclass(frozen=True)
class SituationShadow:
    """Research-only candidate tensor, never a production output."""
    target_or_none:Tensor     # B,K,W+1
    rejected_edge:Tensor      # B,K,W
    event_factuality:Tensor   # B,K,3
    router_logits:Tensor      # B,K,8, auxiliary OLD TRAIN only
    active_area_weights:Tensor # B,K,8, up to four concurrent contexts
    named_areas:tuple[str,...]=AREAS
    output_gate:int=0
    candidate_accepted:bool=False
    activation_allowed:bool=False
    online_causal_certified:bool=False


class MiniArea(nn.Module):
    """Independent small trainable domain: event and target subheads."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(ROUTER_INPUT,EVENT_RANK),
            nn.GELU(),
        )
        self.event=nn.Linear(EVENT_RANK,5)  # link, NONE, 3 factuality
        self.pair=nn.Linear(EVENT_RANK+WORD_RANK+PAIR_GEOMETRY,2)
        # No pretrained weight or old prediction is altered by installation.
        # An untrained new micro-area has EXACTLY zero effect.
        nn.init.zeros_(self.event.weight)
        nn.init.zeros_(self.event.bias)
        nn.init.zeros_(self.pair.weight)
        nn.init.zeros_(self.pair.bias)


class SituationMicroAreas(nn.Module):
    """Compositional factor experts on isolated S65 shadow logits only."""
    def __init__(self):
        super().__init__()
        self.word_projection=nn.Linear(WORD_D,WORD_RANK,bias=False)
        self.router=nn.Linear(ROUTER_INPUT,len(AREAS))
        self.areas=nn.ModuleList(MiniArea() for _ in AREAS)
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.online_causal_certified=False

    @staticmethod
    def _check(f:Tensor,words:Tensor,sources:Tensor,valid_targets:Tensor,
               valid_events:Tensor,old_joint:Tensor,old_rejected:Tensor,
               old_factuality:Tensor)->tuple[int,int,int]:
        require(f.ndim==3 and f.shape[-1]==EVENT_CUES
                and words.ndim==3 and words.shape[-1]==WORD_D,
                "source-only S65 event36 and genuine original H8 word320")
        b,k,_=f.shape;bw,w,_=words.shape
        require(b>0 and k>0 and w>2 and b==bw
                and sources.shape==(b,k)
                and sources.dtype==torch.long
                and valid_targets.shape==(b,k,w)
                and valid_events.shape==(b,k)
                and valid_targets.dtype==torch.bool
                and valid_events.dtype==torch.bool
                and old_joint.shape==(b,k,w+1)
                and old_rejected.shape==(b,k,w)
                and old_factuality.shape==(b,k,3),
                "event×word original S65/GT input ABI")
        require(all(torch.is_floating_point(x) and
                    torch.isfinite(x).all().item() and
                    x.device==f.device and not x.requires_grad
                    for x in (f,words,old_joint,old_rejected,old_factuality)),
                "all parents frozen, finite and same device")
        require(sources.device==f.device
                and valid_targets.device==f.device
                and valid_events.device==f.device
                and bool(((sources>=0)&(sources<w)).all()),
                "event positions native word-index valid")
        return b,k,w

    def forward(self,source_event36:Tensor,word320:Tensor,event_indices:Tensor,
                target_safe:Tensor,event_allowed:Tensor,
                old_target_or_none:Tensor,old_rejected:Tensor,
                old_factuality:Tensor)->SituationShadow:
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and not self.online_causal_certified
                and len(self.areas)==len(AREAS), "research-only never activate")
        b,k,w=self._check(source_event36,word320,event_indices,
                          target_safe,event_allowed,old_target_or_none,
                          old_rejected,old_factuality)
        # 43M H8 inputs may contain full-sentence context: this interface
        # MUST NOT be described as streaming until separately verified.
        projected=self.word_projection(word320.detach().float())
        source_vectors=projected.gather(
            1,event_indices[:,:,None].expand(b,k,WORD_RANK))
        input_event=torch.cat((source_event36.detach().float(),
                               source_vectors),dim=-1)
        route_logits=self.router(input_event)
        route_prob=route_logits.sigmoid()
        active_indices=route_prob.topk(MAX_ACTIVE,dim=-1).indices
        top_mask=torch.zeros_like(route_prob).scatter_(
            -1,active_indices,1.)
        gated=route_prob*top_mask
        gated=gated/gated.sum(-1,keepdim=True).clamp_min(1.e-8)
        gated=gated*event_allowed[:,:,None]
        event_deltas=[]
        pair_deltas=[]
        src=event_indices.to(dtype=torch.float32)[:,:,None]
        dst=torch.arange(w,device=word320.device,dtype=torch.float32)[None,None,:]
        diff=(dst-src)/max(1,w-1)
        geometry=torch.stack((
            diff.expand(b,k,w),
            diff.abs().expand(b,k,w),
            (dst/max(1,w-1)).expand(b,k,w),
            (dst<src).float().expand(b,k,w)),dim=-1)
        target_lat=projected[:,None,:,:].expand(b,k,w,WORD_RANK)
        for area in self.areas:
            z=area.encoder(input_event)
            event_deltas.append(area.event(z))
            pair_in=torch.cat((
                z[:,:,None,:].expand(b,k,w,EVENT_RANK),
                target_lat,geometry),dim=-1)
            pair_deltas.append(area.pair(pair_in))
        ev=torch.stack(event_deltas,dim=2)   # B,K,8,5
        pr=torch.stack(pair_deltas,dim=3)   # B,K,W,8,2
        correction_event=(ev*gated[:,:,:,None]).sum(2)
        correction_pair=(pr*gated[:,:,None,:,None]).sum(3)
        corrected_edges=(old_target_or_none[:,:,:w] +
                         correction_event[:,:,0,None] +
                         correction_pair[:,:,:,0])
        corrected_edges=corrected_edges.masked_fill(
            ~target_safe | ~event_allowed[:,:,None],NEG_INF)
        corrected_none=(old_target_or_none[:,:,w] +
                        correction_event[:,:,1]).masked_fill(
                            ~event_allowed,NEG_INF)
        corrected_joint=torch.cat((
            corrected_edges,corrected_none[:,:,None]),dim=-1)
        corrected_polar=(old_rejected+correction_pair[:,:,:,1])
        corrected_cls=(old_factuality+correction_event[:,:,2:5])
        require(all(torch.isfinite(x).all().item() for x in (
            corrected_joint,corrected_polar,corrected_cls,
            route_logits,gated)), "S72 finite shadow-only weighted scores")
        return SituationShadow(
            corrected_joint,corrected_polar,corrected_cls,
            route_logits,gated)

    @staticmethod
    def old_train_only_area_supervision(router_logits:Tensor,
                                        TRAIN_area_labels:Tensor,
                                        TRAIN_valid_events:Tensor)->Tensor:
        """Auxiliary loss ONLY for explicitly labelled OLD TRAIN examples.

        These labels are never accepted by forward(). Multiple areas may
        simultaneously be true, e.g. quote + passive + negation.
        Labels not provided by the caller cannot be inferred from gold
        target/rejected edge or EVAL data.
        """
        require(router_logits.ndim==3 and
                router_logits.shape[-1]==len(AREAS)
                and TRAIN_area_labels.shape==router_logits.shape
                and TRAIN_valid_events.shape==router_logits.shape[:2]
                and TRAIN_valid_events.dtype==torch.bool
                and bool(TRAIN_valid_events.any())
                and TRAIN_area_labels.device==router_logits.device
                and TRAIN_valid_events.device==router_logits.device
                and bool(torch.isfinite(router_logits).all())
                and bool(torch.isfinite(TRAIN_area_labels).all())
                and bool(((TRAIN_area_labels==0) |
                          (TRAIN_area_labels==1)).all()),
                "S72 source-matched binary situation labels OLD TRAIN only")
        per=F.binary_cross_entropy_with_logits(
            router_logits,TRAIN_area_labels.float(),reduction="none")
        return (per*TRAIN_valid_events[:,:,None]).sum() / (
            TRAIN_valid_events.sum()*len(AREAS))

    def parameter_budget(self)->dict:
        n=sum(int(p.numel()) for p in self.parameters())
        return {
            "new_architecture_parameters":n,
            "new_FP16_weight_bytes_if_fully_serialized":n*2,
            "original43M_or_GT_weights_touched":False,
            "new_trainable_cells":len(AREAS),
            "max_simultaneously_active_areas":MAX_ACTIVE,
            "output_gate":0,
            "candidate_accepted":False,
            "physical_all_in_bpp_and_CPU_speed_NOT_certified":True,
        }
