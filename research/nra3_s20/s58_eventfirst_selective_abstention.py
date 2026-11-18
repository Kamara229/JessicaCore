"""S58 conservative RESEARCH-ONLY selector above frozen S57 event-first logits.

This changes no Calibri43M / Green Tunnel / S26 / S54 / S56 / S57 weights.
S57 first-read HELD24 is SPENT, not an independent evaluation for S58.
A small source-only structural mask may reduce false links but can also
suppress true recipients: this is NOT a demonstrated language improvement.
Only offline full-context inference; no online causality/J7 integration claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class GuardInput:
    event: torch.Tensor                    # B,W frozen S57 logits
    transfer: torch.Tensor                 # B,W frozen S57 logits
    recipient: torch.Tensor                # B,W,W frozen S57 logits
    rejected: torch.Tensor                 # B,W,W frozen S57 logits
    event_safe: torch.Tensor               # B,W boolean
    pair_safe: torch.Tensor                # B,W,W boolean
    surface_verb: torch.Tensor             # B,W boolean, source S18 only
    same_clause: torch.Tensor              # B,W,W boolean, source S56 only
    source_left_negative: torch.Tensor     # B,W boolean, source S56 only


def require(ok, why):
    if not bool(ok):
        raise ValueError("NRA3 S58 research-only abstain: " + why)


def selective_abstain(x: GuardInput):
    """Freeze old logits; conservatively choose <=1 safe pair per sentence."""
    e, t, r, n = x.event, x.transfer, x.recipient, x.rejected
    require(e.ndim == 2, "event dims")
    b, w = e.shape
    require(b > 0 and w >= 2 and t.shape == (b,w) and
            r.shape == n.shape == (b,w,w), "score ABI mismatch")
    for z in (x.event_safe, x.surface_verb, x.source_left_negative):
        require(z.shape == (b,w) and z.dtype == torch.bool, "source mask ABI")
    for z in (x.pair_safe, x.same_clause):
        require(z.shape == (b,w,w) and z.dtype == torch.bool, "pair mask ABI")
    require(all(z.device == e.device for z in
                (t,r,n,x.event_safe,x.pair_safe,x.surface_verb,
                 x.same_clause,x.source_left_negative)), "mixed devices")
    require(all(bool(torch.isfinite(z).all()) for z in (e,t,r,n)),
            "nonfinite frozen logits")
    require(not any(z.requires_grad for z in (e,t,r,n)),
            "trainable logits forbidden")
    chosen=torch.full((b,), -1, dtype=torch.long, device=e.device)
    links=torch.zeros((b,w,w), dtype=torch.bool, device=e.device)
    reject=torch.zeros_like(links)
    reasons={k:0 for k in (
        "no_verb_like_source", "not_transfer", "no_same_clause_safe_target",
        "recipient_emitted", "unlocalized_rejection_suppressed")}
    with torch.inference_mode():
        events=x.event_safe & x.surface_verb
        for row in range(b):
            if not bool(events[row].any()):
                reasons["no_verb_like_source"]+=1
                continue
            src=int(e[row].masked_fill(~events[row], float("-inf")).argmax())
            chosen[row]=src
            if float(t[row,src])<=0:
                reasons["not_transfer"]+=1
                continue
            eligible=x.pair_safe[row,src] & x.same_clause[row,src]
            if not bool(eligible.any()):
                reasons["no_same_clause_safe_target"]+=1
                continue
            dst=int(r[row,src].masked_fill(~eligible, float("-inf")).argmax())
            links[row,src,dst]=True
            reasons["recipient_emitted"]+=1
            if float(n[row,src,dst])>0:
                if bool(x.source_left_negative[row,src]):
                    reject[row,src,dst]=True
                else:
                    reasons["unlocalized_rejection_suppressed"]+=1
    require(int(links.sum())<=b and not bool((reject&~links).any()) and
            not bool((links&~x.pair_safe).any()) and
            not bool((links&~x.same_clause).any()), "unsafe directed link")
    require(sum(reasons[k] for k in (
        "no_verb_like_source", "not_transfer", "no_same_clause_safe_target",
        "recipient_emitted"))==b, "incomplete abstention accounting")
    return {
        "chosen_event_minus1_on_abstain":chosen,
        "directed_recipient":links,
        "directed_rejected":reject,
        "reasons":reasons,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
        "research_only":True,"online_causality_certified":False,
    }


def bridge_frozen_s57(model, data):
    """Gold-blind S57/S18/S56 features only; never imports any HELD gold."""
    from research.nra3_s18.s18c3_raw_russian_grammar_cues import FEATURES
    from research.nra3_s20.s56_grammar_clause_scope_semantic_readout import (
        FEATURE_NAMES, VERB_BITS,
    )
    from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
        EVENT_DIM,PAIR_DIM,
    )
    require(model.output_gate==0 and model.candidate_accepted is False
            and model.activation_allowed is False
            and model.streaming_causality_certified is False,
            "old S57 gate cannot be enabled")
    require(not any(p.requires_grad for p in model.parameters()),
            "old S57 must be frozen")
    require(len(FEATURES)==18 and EVENT_DIM==41 and PAIR_DIM==54
            and data.event.shape[-1]==EVENT_DIM
            and data.pair.shape[-1]==PAIR_DIM,
            "frozen original feature ABI drift")
    with torch.inference_mode():
        e,t,r,n=model(data)
        verb=(data.event[...,list(VERB_BITS)]>0).any(-1)
        same=data.pair[...,FEATURE_NAMES.index("same_clause")]>0.5
        neg=(data.pair[...,FEATURE_NAMES.index(
            "negative_within_left_four_same_clause")]>0.5)
        # Reported negation from another clause MUST NOT itself assert
        # a directed REJECTED pair; use source-local evidence only.
        neg_source=(neg & same & data.valid_pair).any(-1)
        return GuardInput(e.detach(),t.detach(),r.detach(),n.detach(),
                          data.candidate_event.detach(),
                          data.valid_pair.detach(),verb.detach(),
                          same.detach(),neg_source.detach())
