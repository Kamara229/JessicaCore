"""NRA2 3.0 S7: 3 simultaneous relation hypotheses inside ONE neural area.

Inspired by Calibri's S159 source-bound parallel slots and S177 5/10
consensus, but fundamentally different: these are THREE structurally distinct,
causal **relation attention hypotheses**, NOT independently sampled answers,
not 3 copies of the 6M model, and never a popularity-based truth oracle.

All hypotheses share *exactly the same* original 43 CB59 matrices:
  0: global past attention (long-range entity relation);
  1: local past attention  (recent syntactic dependency);
  2: role-informed past   (soft semantic role complementarity).
One shared GRU cell processes the dynamically weighted mixture ONCE.
Extra trainable parameters and physically paid weight bytes: ZERO.

The source of semantic roles is NRA3's own inferred `role_head` logits.
NO ground-truth labels/UD arcs/QA answers are used during inference.
This code runs in original dense TRAIN and native packed C CPU paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F

from research.nra3_s0.neural_area import SEMANTIC_ROLES


# Explicit, fixed prior for role *complementarity*; not a gold sentence table.
# Pairs are symmetric because this function only defines attention, not
# source-vs-destination direction. Directed relations are learned downstream.
_ROLE_EDGES=(
    ("AGENT","PATIENT"),
    ("AGENT","RECIPIENT"),
    ("SOURCE","RECIPIENT"),
    ("SOURCE","DESTINATION"),
    ("EXPERIENCER","LOCATION"),
    ("AGENT","INSTRUMENT"),
)


@dataclass
class ParallelRelationEvidence:
    message:torch.Tensor             # [B,T,R] shared GRU input
    hypothesis_gates:torch.Tensor    # [B,T,3] selected soft mixture
    hypothesis_attentions:torch.Tensor  # [B,T,3,T] causal support
    hypothesis_messages:torch.Tensor   # [B,T,3,R]


def parallel_three_relations(*,
                             query:torch.Tensor,
                             value:torch.Tensor,
                             scores:torch.Tensor,
                             allowed:torch.Tensor,
                             valid:torch.Tensor,
                             semantic_role_logits:torch.Tensor,
                             local_window:int=4,
                             selector_version:int=0)->ParallelRelationEvidence:
    """Three differentiable alternatives from ONE frozen neural forward.

    No random sampling and no hidden state modifying Calibri. Aggregation
    never combines roles from future tokens. One shared recurrence update.
    """
    if not isinstance(query,torch.Tensor) or query.ndim!=3:
        raise ValueError("NRA3 S7 query must be [B,T,R]")
    b,t,r=query.shape
    if (query.dtype!=torch.float32 or value.shape!=(b,t,r) or
        value.dtype!=torch.float32 or scores.shape!=(b,t,t) or
        scores.dtype!=torch.float32 or
        semantic_role_logits.shape!=(b,t,len(SEMANTIC_ROLES)) or
        semantic_role_logits.dtype!=torch.float32 or
        allowed.shape!=(b,t,t) or allowed.dtype!=torch.bool or
        valid.shape!=(b,t) or valid.dtype!=torch.bool or
        not 1<=local_window<=t+1 or
        selector_version not in (0,1,2) or
        query.device!=value.device or query.device!=scores.device or
        query.device!=semantic_role_logits.device or
        query.device!=allowed.device or query.device!=valid.device or
        not bool(torch.isfinite(query).all()) or
        not bool(torch.isfinite(value).all()) or
        not bool(torch.isfinite(scores).all()) or
        not bool(torch.isfinite(semantic_role_logits).all()) or
        not bool(valid[:,0].all())):
        raise ValueError("NRA3 S7 parallel relation CPU/full-sequence ABI drift")
    expected=torch.tril(torch.ones(t,t,dtype=torch.bool,
                                   device=valid.device))[None] & valid[:,None,:]
    if not torch.equal(allowed,expected):
        raise ValueError("NRA3 S7 future-key or valid-mask mismatch")
    i=torch.arange(t,device=query.device)
    distance=(i[:,None]-i[None,:])
    local=(distance>=0)&(distance<local_window)
    local_allowed=allowed&local[None,:,:]
    # A PAD query with no local key gets key0 fallback, then its gate/output
    # is explicitly zeroed; valid query rows are unaffected.
    pad_fallback=(~valid)[:,:,None]&(i==0)[None,None,:]
    general_allowed=allowed|pad_fallback
    local_allowed=local_allowed|pad_fallback
    role_prob=F.softmax(semantic_role_logits.float(),dim=-1)
    prior=torch.zeros((len(SEMANTIC_ROLES),len(SEMANTIC_ROLES)),
                      device=query.device,dtype=torch.float32)
    for left,right in _ROLE_EDGES:
        a,c=SEMANTIC_ROLES.index(left),SEMANTIC_ROLES.index(right)
        prior[a,c]=1.
        prior[c,a]=1.
    affinity=(role_prob@prior)@role_prob.transpose(1,2)
    # Affinity is soft predictions from the current visible prefix ONLY.
    # Three probability maps are computed in parallel over one score tensor.
    masked=torch.stack((
        scores.masked_fill(~general_allowed,-1e4),
        scores.masked_fill(~local_allowed,-1e4),
        (scores+3.0*affinity).masked_fill(~general_allowed,-1e4),
    ),dim=2) # [B,T,H=3,T]
    attn=F.softmax(masked,dim=-1)
    candidates=attn@value[:,None,:,:]  # [B,T,3,R], same shared values
    # Compare each hypothesis against the SAME projected relation query.
    # No extra embedding/scorer matrices: branch scores are learned via the
    # existing paid rel_q/k/v and role_head upstream projections.
    agreement=(candidates*query.unsqueeze(2)).sum(dim=-1)/math.sqrt(r)
    if selector_version==0:
        # Preserves S7 original numerical path exactly, including its
        # near-uniform branch weights. No hidden change to old model.
        gate_logits=agreement
    else:
        # S8: amplify *measurable evidence*, not stochastic voting:
        # - normalized alignment with existing paid relation query;
        # - strong attention outside the local window for the global route;
        # - concentration of local attention for the local route;
        # - predicted complementary-role affinity for the semantic route.
        # No new parameters. These scores are causal and differentiable
        # through rel_q/k/v, role_head and all earlier NRA3 blocks.
        normalized_query=F.normalize(query,dim=-1,eps=1e-5)
        normalized_candidate=F.normalize(candidates,dim=-1,eps=1e-5)
        cosine=(normalized_candidate*normalized_query.unsqueeze(2)).sum(-1)
        far_valid=general_allowed & ~local_allowed
        long_range_mass=(attn[:,:,0,:]*far_valid.float()).sum(-1)
        local_peakedness=attn[:,:,1,:].square().sum(-1)
        role_support=(attn[:,:,2,:]*affinity).sum(-1)
        structural=torch.stack((2.0*long_range_mass,
                                1.15*local_peakedness,
                                3.0*role_support),dim=-1)
        gate_logits=3.5*cosine+structural
        # This is a small *bounded*, static temperature, not an untracked
        # extra weight matrix or sample-specific learned calibration oracle.
        gate_logits=gate_logits.clamp(-16.,16.)
        if selector_version==2:
            # S8b fail-closed debiasing: remove persistent global-vs-local
            # calibration preference using ONLY the running *causal*
            # history of this same already-visible prefix. No training
            # labels, no additional learned parameters and no future leak.
            # Avoids the S8-v1 tendency to pick GLOBAL for 63/66 words.
            position=torch.arange(1,t+1,device=gate_logits.device,
                                  dtype=torch.float32)[None,:,None]
            causal_mean=gate_logits.cumsum(dim=1)/position
            gate_logits=(2.5*(gate_logits-causal_mean)).clamp(-16.,16.)
    gates=F.softmax(gate_logits,dim=-1)
    combined=(gates.unsqueeze(-1)*candidates).sum(dim=2)
    combined=combined.masked_fill(~valid.unsqueeze(-1),0.)
    gates=gates.masked_fill(~valid.unsqueeze(-1),0.)
    attn=attn.masked_fill(~valid[:,:,None,None],0.)
    candidates=candidates.masked_fill(~valid[:,:,None,None],0.)
    if (not bool(torch.isfinite(combined).all()) or
        not bool(torch.isfinite(gates).all()) or
        not bool(torch.isfinite(attn).all())):
        raise AssertionError("NRA3 S7 nonfinite parallel relation hypothesis")
    return ParallelRelationEvidence(combined,gates,attn,candidates)
