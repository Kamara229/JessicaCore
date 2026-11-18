"""S52: frozen S26 paid event graph on actual original43M+GT S51 native words.

This is an *OFFLINE RESEARCH OBSERVER*, not a production decoder.
Native features are from S51 genuine Calibri43M 12-layer CPU post-GT
layer8 capture (protected 616 J7 replay at original capture time).
S26 graph is a *previously trained* 12,384-FP16 research head on
S22 original43M TRAIN256, not newly trained/selected here.

Restrict structural candidate eligibility by: real native word support,
no shared original BPE piece / template crossing, raw quote exclusion,
and raw non-punctuation. Actual S26 original logits are returned by
reference, unchanged, and mask is SEPARATE. No gold or annotation in
forward. This merely produces safe *ABSTAIN* masks, not evidence that
S26 semantics are correct. Neither S26 bidirectional CNN context nor
original full-prefill GT causal safety has been certified; therefore
these masks do NOT certify fully online autoregressive causality.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import torch

from research.nra3_s18.s18c3_raw_russian_grammar_cues import FEATURES
from research.nra3_s19.s19g_paid_semantic_graph import GraphLogits
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph

PUNCT_FEATURE=FEATURES.index("is_punctuation")
REAL_NATIVE_WORD_DIM=338
WITH_QUOTE_BIT=339
S52_NATIVE_MODEL_GATE=0


def require(ok,why):
    if not bool(ok):
        raise ValueError("S52 genuine43M+GT S26 graph zero-gate: "+why)


@dataclass(frozen=True)
class SafeGraphReadOnly:
    """Native candidate mask separate from frozen graph logits."""
    original_paid_graph_logits:GraphLogits
    native_word_mask:torch.Tensor
    quote_word_mask:torch.Tensor
    bpe_shared_unsafe_word_mask:torch.Tensor
    punctuation_mask:torch.Tensor
    eligible_word_mask:torch.Tensor
    eligible_directed_pair_mask:torch.Tensor
    # No gold, model improvement, online runtime, or quality claims.


class S52QuoteAwareFrozenPaidGraphBridge:
    def __init__(self,frozen_model:EventGroundedInversionGraph):
        require(isinstance(frozen_model,EventGroundedInversionGraph)
                and frozen_model.output_gate==0
                and not frozen_model.candidate_accepted
                and not frozen_model.activation_allowed
                and frozen_model.parent.output_gate==0
                and not frozen_model.parent.candidate_accepted
                and not frozen_model.parent.activation_allowed,
                "S26 original paid 12384-FP16 graph must be frozen and zero-gated")
        self.model=frozen_model.eval().requires_grad_(False)
        self.output_gate=S52_NATIVE_MODEL_GATE
        self.candidate_accepted=False
        self.activation_allowed=False
        self.training_allowed=False

    def _inputs(self,real_native_word339,valid,unsafe):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and not self.training_allowed,
                "cannot activate the research S26/Russian graph read port")
        require(isinstance(real_native_word339,torch.Tensor)
                and real_native_word339.ndim==3
                and real_native_word339.shape[-1]==WITH_QUOTE_BIT
                and real_native_word339.dtype==torch.float32
                and real_native_word339.device.type=="cpu"
                and not real_native_word339.requires_grad
                and bool(torch.isfinite(real_native_word339).all()),
                "genuine original CPU 43M post-GT native339 required")
        require(isinstance(valid,torch.Tensor) and valid.dtype==torch.bool
                and valid.device.type=="cpu"
                and isinstance(unsafe,torch.Tensor) and unsafe.dtype==torch.bool
                and unsafe.device.type=="cpu"
                and valid.shape==unsafe.shape==real_native_word339.shape[:2]
                and bool(valid[:,0].all())
                and bool((valid.long().cummin(1).values==valid.long()).all())
                and not bool((unsafe&~valid).any())
                and bool((real_native_word339[~valid]==0).all()),
                "native original word validity/BPE-unsafe contract drift")
        raw18=real_native_word339[...,320:338]
        q=real_native_word339[...,338]
        require(bool(((raw18==0)|(raw18==1)).all())
                and bool(((q==0)|(q==1)).all()),
                "raw Unicode grammar and quote flags must be input-only binary")
        return raw18,q

    def observe(self,real_native_word339,valid,unsafe):
        raw18,quote_bit=self._inputs(real_native_word339,valid,unsafe)
        # Exclude punctuation and quoted content from event/role candidate
        # eligibility to avoid incorrectly treating a reported quote as a
        # direct event assertion. This is a conservative policy, not an
        # optimized semantic NLP predictor.
        punctuation=raw18[...,PUNCT_FEATURE]==1
        quote=quote_bit==1
        safe=valid&~unsafe&~punctuation&~quote
        pair=safe.unsqueeze(1)&safe.unsqueeze(2)
        identity=torch.eye(valid.shape[1],dtype=torch.bool)
        pair=pair&~identity.unsqueeze(0)
        require(not bool((safe&~valid).any())
                and not bool((safe&unsafe).any())
                and not bool((safe&quote).any())
                and not bool((safe&punctuation).any())
                and not bool((pair&(~safe.unsqueeze(1))).any())
                and not bool((pair&(~safe.unsqueeze(2))).any())
                and not bool((pair&identity.unsqueeze(0)).any()),
                "S52 unsafe/quote protected pair mask leaked")
        # Original S26 paid model inputs are EXACT first338 features;
        # appended S48 quote flag is never passed to incompatible 338D
        # original S26 parent/heads.
        x=real_native_word339[...,:REAL_NATIVE_WORD_DIM].detach()
        require(x.shape[-1]==338,"original S26 native word338 ABI modified")
        with torch.inference_mode():
            old=self.model(x,valid)
        for kind,head,geometry in (
            ("event",old.event_status,(*valid.shape,6)),
            ("role",old.role,(*valid.shape,valid.shape[1],9)),
            ("argument_polarity",old.argument_polarity,(*valid.shape,valid.shape[1],3)),
            ("coreference",old.coreference,(*valid.shape,valid.shape[1],2)),
            ("event_relation",old.event_relation,(*valid.shape,valid.shape[1],5)),
        ):
            require(head.shape==geometry and bool(torch.isfinite(head).all()),
                    "original physically paid S26 "+kind+" graph head ABI drift")
        require(self.model.output_gate==0 and self.model.parent.output_gate==0
                and self.output_gate==0,"S52 physical graph activated unexpectedly")
        return SafeGraphReadOnly(old,valid,quote,unsafe,punctuation,safe,pair)


def tensor_sha(*tensors):
    h=hashlib.sha256()
    for t in tensors:
        require(t.device.type=="cpu", "native source hashing only CPU")
        h.update(str(t.shape).encode())
        h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def prediction_totals(observation):
    """Descriptive counts ONLY; never accuracy or validation."""
    mask=observation.eligible_directed_pair_mask
    full=observation.native_word_mask
    offdiag=full.unsqueeze(1)&full.unsqueeze(2)
    offdiag=offdiag&~torch.eye(full.shape[1],dtype=torch.bool).unsqueeze(0)
    heads=observation.original_paid_graph_logits
    event=heads.event_status.argmax(-1)!=0
    role=heads.role.argmax(-1)!=0
    polarity=heads.argument_polarity.argmax(-1)==2
    coref=heads.coreference.argmax(-1)!=0
    rel=heads.event_relation.argmax(-1)!=0
    predictions={
        "event":(event,full,observation.eligible_word_mask),
        "role":(role,offdiag,mask),
        "argument_REJECTED":(polarity,offdiag,mask),
        "coreference":(coref,offdiag,mask),
        "event_relation":(rel,offdiag,mask),
    }
    counts={}
    for name,(pred,all_mask,safe_mask) in predictions.items():
        require(pred.shape==all_mask.shape==safe_mask.shape,
                "S52 descriptive frozen graph mask geometry differs")
        raw=int((pred&all_mask).sum())
        accepted=int((pred&safe_mask).sum())
        require(accepted<=raw,"mask produced additional graph predictions")
        counts[name]={
            "original_frozen_S26_argmax_nonNONE_unvalidated":raw,
            "structurally_eligible_unvalidated":accepted,
            "abstained_due_to_quote_BPE_or_punctuation":raw-accepted,
        }
    return counts
