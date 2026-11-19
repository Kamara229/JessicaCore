"""NRA2 S89: independent source-only MENTION candidate TOP1 selector.

S88 (one-time S86 new48 synthetic first-read) showed the original
source-only S59 MENTION top8 contained constructed event 48/48, but
TOP1 was only 30/48, and top1+all situation fields only 10/48.
It does NOT establish a paired benchmark vs another model.

S89 introduces a SMALL stand-alone learned main-event selector using
only original frozen 43M/GT H8 320D, local same-clause H8, 32 source
S82 grammatical indicators, frozen parent S59 event scores, source
word position and original source-only rank. A quoted mention may
be selected as MENTION but this output NEVER becomes asserted truth,
recipient edge, or decoder instruction.

Model.forward NEVER takes labels or oracle event-index. Trainable loss
on ORIGINAL S83 56 TRAIN-only author-constructed focus anchors is
a *ranking choice* among real mention candidates. Negatives mean
'not focus for this one question', NOT 'not actual events'. The old
S86 48 and S79 48 first-read sets are SPENT and cannot be imported,
used to train, optimize, rescore or select the selector.

Architecture/contract only in S89: not trained or physical J7-tested;
all-in bpp<=0.7/CPU>=90%, streaming, transfer correctness and
false-positive event precision remain unmeasured. Gate stays zero.
"""
from __future__ import annotations

from typing import Sequence
import torch
from torch import nn

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s82_predicate_centered_situation_fields import (
    _raw_only_clauses_and_quotes,source_only_event_fields,
)

STAGE="NRA2_S89_SOURCE_ONLY_MENTION_TOP1_EVENT_SELECTOR_SHADOW"
N_FIELDS=32
NUM_HEAD_FEATURES=27
PAID=5725


def require(condition,reason):
    if not bool(condition):
        raise ValueError("NRA2 S89 source-only event selector: "+reason)


class S89MainEventMentionTop1(nn.Module):
    """Independent research-only ranking; no factuality/role output."""
    def __init__(self):
        super().__init__()
        self.event_proj=nn.Linear(320,8,bias=False)
        self.left_proj=nn.Linear(320,4,bias=False)
        self.right_proj=nn.Linear(320,4,bias=False)
        self.field_proj=nn.Linear(N_FIELDS,8,bias=False)
        self.score_head=nn.Sequential(
            nn.Linear(NUM_HEAD_FEATURES,12),nn.GELU(),nn.Linear(12,1))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.online_causal_certified=False
        self.is_trained=False
        require(sum(x.numel() for x in self.parameters())==PAID,
                "exact 5725 newly trainable selector params, old 43M unchanged")

    def forward(self,original_h8_320:torch.Tensor,
                word_valid:torch.Tensor,original_bpe_unsafe:torch.Tensor,
                mention_candidate_indices:torch.Tensor,
                mention_candidate_allowed:torch.Tensor,
                original_S59_source_scores:torch.Tensor,
                source_texts:Sequence[str]):
        """Return source-only [B,K] MENTION focus logits, never assertions."""
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and
                not self.online_causal_certified,
                "selector cannot activate prior to independent acceptance")
        h=original_h8_320
        require(h.ndim==3 and h.shape[-1]==320 and h.dtype==torch.float32
                and not h.requires_grad and torch.isfinite(h).all()
                and word_valid.shape==original_bpe_unsafe.shape==h.shape[:2]
                and original_S59_source_scores.shape==h.shape[:2]
                and not original_S59_source_scores.requires_grad
                and word_valid.dtype==original_bpe_unsafe.dtype==torch.bool
                and mention_candidate_indices.ndim==2
                and mention_candidate_indices.dtype==torch.long
                and mention_candidate_allowed.shape==mention_candidate_indices.shape
                and mention_candidate_allowed.dtype==torch.bool
                and len(source_texts)==h.shape[0]
                and all(isinstance(x,str) for x in source_texts),
                "original frozen H8 and OLD S59 only raw source tensors")
        b,w,_=h.shape;k=mention_candidate_indices.shape[1]
        require(1<=k<=16
                and bool(((mention_candidate_indices>=0)
                          &(mention_candidate_indices<w)).all())
                and bool(mention_candidate_allowed.any(-1).all())
                and not bool((original_bpe_unsafe&~word_valid).any())
                and not bool((mention_candidate_allowed&
                              ~word_valid.gather(1,mention_candidate_indices)).any())
                and not bool((mention_candidate_allowed&
                              original_bpe_unsafe.gather(1,mention_candidate_indices)).any()),
                "source-only topK candidates need valid safe real BPE positions")
        # Coreference/quote/negation indicators source-derived, NEVER gold.
        fields=source_only_event_fields(
            source_texts,word_valid,original_bpe_unsafe,
            mention_candidate_indices,mention_candidate_allowed).to(h.device)
        clause=[]
        for i,t in enumerate(source_texts):
            raw=raw_surface_grammar(t)
            n=int(word_valid[i].sum().item())
            require(len(raw)==n and n>=1 and n<=w
                    and bool(word_valid[i,:n].all())
                    and not bool(word_valid[i,n:].any()),
                    "same genuine original native BPE raw word ordering")
            cid,_=_raw_only_clauses_and_quotes(raw)
            clause.append(cid+[-100]*(w-n))
        clause=torch.tensor(clause,dtype=torch.long,device=h.device)
        ci=mention_candidate_indices
        event_clause=clause.gather(1,ci)
        safe=word_valid & ~original_bpe_unsafe
        event=h.gather(1,ci[:,:,None].expand(b,k,320))
        left=torch.zeros_like(event)
        right=torch.zeros_like(event)
        for dist in (1,2,3,4):
            lid=(ci-dist).clamp_min(0)
            rid=(ci+dist).clamp_max(w-1)
            lmask=((ci>=dist)&safe.gather(1,lid)&
                   (clause.gather(1,lid)==event_clause))
            rmask=((ci+dist<w)&safe.gather(1,rid)&
                   (clause.gather(1,rid)==event_clause))
            left+=h.gather(1,lid[:,:,None].expand(b,k,320))*lmask[:,:,None]
            right+=h.gather(1,rid[:,:,None].expand(b,k,320))*rmask[:,:,None]
        left=left/4.
        right=right/4.
        # Frozen old S59 scores, normalized per candidate set. This
        # is a SOURCE input, never a target, ground truth or re-ranker.
        old=original_S59_source_scores.gather(1,ci).detach()
        require(bool(torch.isfinite(old[mention_candidate_allowed]).all()),
                "frozen old source candidate scores must be finite")
        clean=torch.where(mention_candidate_allowed,old,torch.zeros_like(old))
        denom=mention_candidate_allowed.sum(1,keepdim=True).clamp_min(1)
        mean=clean.sum(1,keepdim=True)/denom
        centered=torch.where(mention_candidate_allowed,clean-mean,0.)
        relative_score=(centered/
                        centered.abs().amax(1,keepdim=True).clamp_min(1.)).unsqueeze(-1)
        rank=(torch.arange(k,device=h.device,dtype=h.dtype)
              .view(1,k,1).expand(b,k,1)/max(k-1,1))
        relative_word=(ci.float()/
                       word_valid.sum(1,keepdim=True).clamp_min(1)).unsqueeze(-1)
        features=torch.cat((
            self.event_proj(event),self.left_proj(left),self.right_proj(right),
            self.field_proj(fields),relative_score,rank,relative_word
        ),dim=-1)
        require(features.shape==(b,k,NUM_HEAD_FEATURES),
                "8+4+4 H8 +8 clause +3 source rank dimensions")
        logits=self.score_head(features).squeeze(-1)
        logits=logits.masked_fill(~mention_candidate_allowed,-1.e5)
        require(logits.shape==(b,k) and bool(torch.isfinite(logits).all()),
                "new small independent top1 MENTION logits without factuality")
        return logits

    def budget(self):
        return {
            "new_trainable_parameters":PAID,
            "new_FP16_weight_bytes":PAID*2,
            "same_clause_source_relative_features":True,
            "original_S59_parent_and_asserted_S65_unchanged":True,
            "not_event_factuality_or_recipient_detection":True,
            "no_real_43m_J7_candidate_test_or_CPU_bpp_certificate":True,
            "output_gate":0,
            "candidate_accepted":False,
            "activation_allowed":False,
        }
