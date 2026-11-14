"""NRA2 S3BM: meaning-oriented *joint* source/destination adapter, research only.

Original S3O role head individually scores each query mention with short local
windows. That representation fails to generalize across unseen grammar, as the
one-time S3BH and S3BL holdouts established. This HEAD ONLY introduces shared
clause-level cross-attention and an antisymmetric pair comparison, rather than
hard-coding first/last mentions, individual keywords, or any spent holdout text.

The original 43M token states, Green Tunnel, S3O graph, normalizer and standalone
NRA remain frozen. No LIVE, main, Render, Supabase, QA/TEST or Jessica chat calls.
Inputs: authentic [B,T,320] frozen CPU tokens and literal tokenizer offsets ONLY.
No labelled source/target, reachability, graph oracle or answer in inference.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from research.nra2_bridge.stage3o_neural_query_roles import (
    LearnedQueryRoleHead, QueryTextMask,
)
from research.nra2_bridge.stage3k_neural_entity_bridge import (
    NodeSpanBoundGraph, EntityBinding, NODES, WIDTH, H,
)


class JointSemanticPairRoleHead(nn.Module):
    """Two-entity semantic roles from *joint* clause context and pair evidence.

    Core proof: swapping only the arbitrary two candidate indices (and matching
    literal masks/offsets) does not change semantic source/target logits. The
    optional frozen S3BA base initializes the head identically to S3BA; a zero
    final residual makes new architecture research-safe before TRAIN-only fit.
    No linguistic keyword lists, regex classification or answer label shortcuts.
    """

    def __init__(self):
        super().__init__()
        self.base=LearnedQueryRoleHead()
        self.base.requires_grad_(False)
        # Reuse S3BA's frozen 320->48 context features, then give BOTH mention
        # candidates access to the whole interrogative clause via cross-attention.
        self.anchor_query=nn.Sequential(
            nn.LayerNorm(3*H),nn.Linear(3*H,H),nn.GELU(),
        )
        self.context_key=nn.Linear(H,H,bias=False)
        self.context_value=nn.Linear(H,H,bias=False)
        self.side=nn.Sequential(
            nn.LayerNorm(4*H+1),nn.Linear(4*H+1,2*H),
            nn.GELU(),nn.Linear(2*H,H),nn.GELU(),
        )
        # True pair comparison: swap candidates => change sign of odd
        # difference inputs. Shared score evaluated in both orientations.
        self.pair_score=nn.Sequential(
            nn.LayerNorm(3*H+1),
            nn.Linear(3*H+1,H),
            nn.GELU(),
            nn.Linear(H,1),
        )
        # Exact original S3BA predictions at initialization. The first
        # gradient step trains the residual output and later all new layers.
        nn.init.zeros_(self.pair_score[-1].weight)
        nn.init.zeros_(self.pair_score[-1].bias)

    def forward(self,tokens:torch.Tensor,token_mask:torch.Tensor,q:QueryTextMask):
        base_src,base_dst=self.base(tokens,token_mask,q)
        b,t,width=tokens.shape
        if (width!=WIDTH or q.node_order.shape!=(b,2)
            or q.relative_positions.shape!=(b,2)
            or q.anchor_mask.shape!=(b,2,t)
            or q.left_context_mask.shape!=(b,2,t)
            or q.right_context_mask.shape!=(b,2,t)
            or q.query_mask.shape!=(b,t)
            or bool((q.query_mask & ~token_mask).any())):
            raise AssertionError("S3BM literal no-gold text/mask ABI mismatch")
        # Freeze baseline features. New trainable layers cannot modify S3BA.
        with torch.no_grad():
            hidden=self.base.encoder(tokens.float())
        anchor=self.base.pool(hidden,q.anchor_mask)
        before=self.base.pool(hidden,q.left_context_mask)
        after=self.base.pool(hidden,q.right_context_mask)
        local=torch.cat([anchor,before,after],dim=-1)     # [B,2,144]
        query=self.anchor_query(local)
        keys=self.context_key(hidden)
        values=self.context_value(hidden)
        # Query clause is the same for both mentions, while mention-conditioned
        # attention weights select different directional cues across that clause.
        attention=torch.einsum("bkh,bth->bkt",query,keys)/math.sqrt(H)
        attention=attention.masked_fill(
            ~q.query_mask[:,None,:],torch.finfo(attention.dtype).min)
        attended=torch.einsum("bkt,bth->bkh",attention.softmax(dim=-1),values)
        side=self.side(torch.cat(
            [anchor,before,after,attended,q.relative_positions[:,:,None]],dim=-1))
        clause=self.base.pool(hidden,q.query_mask[:,None,:]).squeeze(1)
        common=side[:,0]*side[:,1]
        rel=q.relative_positions[:,0:1]-q.relative_positions[:,1:2]
        fwd=torch.cat([side[:,0]-side[:,1],clause,common,rel],dim=-1)
        rev=torch.cat([side[:,1]-side[:,0],clause,common,-rel],dim=-1)
        correction=(self.pair_score(fwd)-self.pair_score(rev)).squeeze(-1)
        # The two opposite endpoints are constrained in a single score; no
        # independent source and target classifiers can disagree.
        plusminus=torch.stack((correction,-correction),dim=-1)
        src=base_src.scatter(
            1,q.node_order,base_src.gather(1,q.node_order)+plusminus)
        dst=base_dst.scatter(
            1,q.node_order,base_dst.gather(1,q.node_order)-plusminus)
        if (src.shape!=(b,NODES) or dst.shape!=(b,NODES)
            or not bool(torch.isfinite(src).all() and torch.isfinite(dst).all())):
            raise AssertionError("S3BM nonfinite semantic pair role logits")
        return src,dst


class S3BMGraphWithJointRoles(nn.Module):
    """Original frozen graph head; only joint query role head is new."""
    def __init__(self):
        super().__init__()
        self.graph=NodeSpanBoundGraph()
        self.query=JointSemanticPairRoleHead()

    def forward(self,tokens,token_mask,binding:EntityBinding,query:QueryTextMask):
        edges,_unused_src,_unused_dst=self.graph(tokens,token_mask,binding)
        src,dst=self.query(tokens,token_mask,query)
        return edges,src,dst
