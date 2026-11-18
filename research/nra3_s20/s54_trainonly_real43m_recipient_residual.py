"""S54 research-only learnable residual over REAL 43M+GT native word338+S26 graph.

The existing genuine Calibri43M+GT post-layer8 320D features and frozen
S26 paid five-head graph are INPUTS, NEVER updated. A 2-head trainable
tiny logistic readout learns directed verb->dative participant links
and rejection polarity ONLY from already scored S53 TRAIN-development24.

Strict: fixed features use ONLY frozen native43M+GT 320D (through frozen
S26 graph predictions), S18 raw18 binary source, S48 quote text bit,
S51 shared-BPE unsafe mask, and word-relative position. No gold/event
anchor, row IDs, dataset family, template hash, human predicate or
handwritten recipient enters model forward. The OLD S26 12,384 FP16
and original 43M/GT remain frozen; new weights serialized as FP16.
S54 new HELD24 labels may enter an evaluator AFTER all weights fit and
HELD predictions have been frozen; no held hyperparameter selection.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s18.s18c3_raw_russian_grammar_cues import FEATURES
from research.nra3_s19.s19f_real43m_train9_graph_targets import ROLE,STATUS
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    SafeGraphReadOnly,
)

FEATURE_NAMES=(
    "intercept",
    "source_S26_P_event_nonNONE",
    "source_S26_P_negated",
    "old_S26_P_recipient_directed",
    "old_S26_P_agent_directed",
    "old_S26_P_pair_rejected",
    "source_raw_past_verb_shape",
    "source_raw_immediately_preceded_ne",
    "target_raw_dative_possible",
    "target_raw_uppercase",
    "target_raw_accusative_possible",
    "target_raw_preposition_other",
    "target_follows_source",
    "target_precedes_source",
    "signed_pair_distance_div12",
    "absolute_pair_distance_div12",
    "source_S26_event_times_target_dative",
    "source_S26_neg_times_target_dative",
)
PAID_READOUT=2*len(FEATURE_NAMES)
ORIGINAL_GATE=0


def require(ok,why):
    if not bool(ok):raise ValueError("S54 real43M GT TRAIN-only recipient residual: "+why)


def pair_features_gold_blind(x339:torch.Tensor,obs:SafeGraphReadOnly):
    """Frozen full-text contextual/native features, not causal streaming decoder."""
    valid=obs.native_word_mask
    b,w=valid.shape
    require(x339.shape==(b,w,339) and x339.dtype==torch.float32
            and x339.device.type=="cpu" and not x339.requires_grad
            and valid.dtype==torch.bool
            and all(getattr(obs.original_paid_graph_logits,key).requires_grad is False
                    for key in ("event_status","role","argument_polarity",
                                "coreference","event_relation")),
            "S54 requires frozen original 43M/S26 precomputed native source, no training")
    logits=obs.original_paid_graph_logits
    status_prob=torch.softmax(logits.event_status.detach(),dim=-1)
    role_prob=torch.softmax(logits.role.detach(),dim=-1)
    pol_prob=torch.softmax(logits.argument_polarity.detach(),dim=-1)
    pair=obs.eligible_directed_pair_mask
    require(pair.shape==(b,w,w) and not bool((pair&~valid.unsqueeze(1)).any())
            and not bool((pair&~valid.unsqueeze(2)).any()),
            "unsafe quote/BPE or invalid pair leaked into S54 native graph")
    raw=x339[...,320:338]
    pos=torch.arange(w,dtype=torch.float32)
    d=(pos[None,:]-pos[:,None]).expand(b,w,w)
    def at_source(v):return v.unsqueeze(2).expand(b,w,w)
    def at_target(v):return v.unsqueeze(1).expand(b,w,w)
    event=1-status_prob[...,STATUS.index("NONE")]
    neg=status_prob[...,STATUS.index("NEGATED")]
    dative=at_target(raw[...,FEATURES.index("ending_dative_possible")])
    components=(
        torch.ones((b,w,w),dtype=torch.float32),
        at_source(event),
        at_source(neg),
        role_prob[...,ROLE.index("RECIPIENT")],
        role_prob[...,ROLE.index("AGENT")],
        pol_prob[...,2],
        at_source(raw[...,FEATURES.index("past_verb_shape_possible")]),
        at_source(raw[...,FEATURES.index("previous_word_negative_particle")]),
        dative,
        at_target(raw[...,FEATURES.index("initial_uppercase")]),
        at_target(raw[...,FEATURES.index("ending_feminine_accusative_possible")]),
        at_target(raw[...,FEATURES.index("preposition_other")]),
        (d>0).float(),
        (d<0).float(),
        torch.clamp(d/12.,-1.,1.),
        torch.clamp(d.abs()/12.,0.,1.),
        at_source(event)*dative,
        at_source(neg)*dative,
    )
    z=torch.stack(components,dim=-1).detach()
    require(z.shape==(b,w,w,len(FEATURE_NAMES))
            and bool(torch.isfinite(z).all())
            and not z.requires_grad,
            "S54 learned features need exact frozen source 43M/S26 ABI")
    return z


class S54DirectedRecipientResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.link=nn.Parameter(torch.zeros(len(FEATURE_NAMES)))
        self.polarity=nn.Parameter(torch.zeros(len(FEATURE_NAMES)))
        self.output_gate=ORIGINAL_GATE
        self.candidate_accepted=False
        self.activation_allowed=False
        self.learning_scope="S53_TRAIN_ONLY_24_PREVIOUSLY_SCORED"
        self.no_original43m_or_GT_training=True

    def forward(self,features,eligible_pair):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed,
                "S54 candidate activation forbidden")
        require(features.ndim==4 and features.shape[-1]==len(FEATURE_NAMES)
                and features.dtype==torch.float32
                and eligible_pair.shape==features.shape[:3]
                and eligible_pair.dtype==torch.bool
                and bool(torch.isfinite(features).all())
                and self.link.shape==self.polarity.shape==(len(FEATURE_NAMES),),
                "S54 paid readout input ABI or FP16 parameter drift")
        link_logit=features.matmul(self.link)
        neg_logit=features.matmul(self.polarity)
        # Do NOT apply masked_fill to logits before gradients. Return
        # explicit separate pair mask; no source oracle or edge truth.
        return link_logit,neg_logit

    def predict(self,features,eligible_pair):
        with torch.inference_mode():
            link,neg=self(features,eligible_pair)
            recipient=(link>0)&eligible_pair
            rejected=recipient&(neg>0)
        return recipient,rejected

    def physical_fp16(self):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed,"S54 readout cannot activate")
        with torch.no_grad():
            params=torch.stack((self.link.detach(),self.polarity.detach())).cpu()
            require(params.shape==(2,len(FEATURE_NAMES))
                    and bool(torch.isfinite(params).all()),
                    "S54 paid 36-param readout unavailable")
            return params.to(torch.float16).contiguous()

    @classmethod
    def from_physical_fp16(cls,weights):
        require(isinstance(weights,torch.Tensor)
                and weights.dtype==torch.float16
                and weights.shape==(2,len(FEATURE_NAMES))
                and not weights.requires_grad
                and bool(torch.isfinite(weights).all()),
                "exact S54 paid FP16 physical candidate missing")
        obj=cls().eval()
        with torch.no_grad():
            obj.link.copy_(weights[0].float())
            obj.polarity.copy_(weights[1].float())
        obj.requires_grad_(False)
        require(torch.equal(obj.physical_fp16(),weights),
                "S54 FP16 paid readout roundtrip failed")
        return obj


def train_on_previously_scored_S53_only(
    features,eligible_pair,truth_recipient,truth_rejected,
):
    """Fixed optimizer predetermined BEFORE S54 HELD24 first-read.

    S53 labels have already been inspected so results on that source
    are training fit ONLY, never independent evaluation.
    """
    require(features.ndim==4 and features.shape[-1]==len(FEATURE_NAMES)
            and eligible_pair.shape==truth_recipient.shape==truth_rejected.shape
            and eligible_pair.dtype==truth_recipient.dtype==truth_rejected.dtype==torch.bool
            and bool((truth_recipient&~eligible_pair).sum()==0)
            and bool((truth_rejected&~truth_recipient).sum()==0)
            and int(truth_recipient.sum())==18
            and int(truth_rejected.sum())==12,
            "S54 exactly S53 previously scored 18 recipient/12 negated gold TRAIN required")
    torch.manual_seed(541118)
    learner=S54DirectedRecipientResidual()
    optim=torch.optim.Adam(learner.parameters(),lr=0.075,weight_decay=0.)
    pair=eligible_pair
    n_pos=int((truth_recipient&pair).sum())
    n_neg=int((~truth_recipient&pair).sum())
    require(n_pos==18 and n_neg>n_pos,
            "S54 TRAIN loss requires both positive links and all other eligible negatives")
    pos_weight=torch.tensor(min(n_neg/n_pos,24.),dtype=torch.float32)
    # Separate positive gold pair polarity; negatives for REJECTED include
    # the six positive links. Never use manually provided predicate
    # index as input to neural readout; only to loss here in TRAIN.
    for step in range(300):
        optim.zero_grad(set_to_none=True)
        link_logit,neg_logit=learner(features,pair)
        link_loss=F.binary_cross_entropy_with_logits(
            link_logit[pair],truth_recipient[pair].float(),
            pos_weight=pos_weight)
        neg_loss=F.binary_cross_entropy_with_logits(
            neg_logit[truth_recipient],truth_rejected[truth_recipient].float(),
            pos_weight=torch.tensor(.5))
        l2=.0004*(learner.link.square().sum()+learner.polarity.square().sum())
        loss=link_loss+neg_loss+l2
        require(bool(torch.isfinite(loss)), "nonfinite S54 TRAIN loss")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in learner.parameters()),"S54 TRAIN gradient invalid")
        optim.step()
    learner.eval().requires_grad_(False)
    require(all(not p.requires_grad for p in learner.parameters()),"S54 TRAIN readout unfrozen")
    physical=learner.physical_fp16()
    restored=S54DirectedRecipientResidual.from_physical_fp16(physical)
    train_rec,train_rej=restored.predict(features,pair)
    return restored,physical,{
       "fixed_train_steps":300,
       "fixed_LR":.075,
       "train_RECIPIENT_gold":n_pos,
       "train_REJECTED_gold":int(truth_rejected.sum()),
       "train_eligible_negative_pairs":n_neg,
       "train_positive_BCE_weight":float(pos_weight),
       "resubstitution_pred_RECIPIENT":int(train_rec.sum()),
       "resubstitution_pred_REJECTED":int(train_rej.sum()),
       "resubstitution_not_quality_certification":True,
       "only_S53_previously_scored_TRAIN24_in_gradient":True,
    }
