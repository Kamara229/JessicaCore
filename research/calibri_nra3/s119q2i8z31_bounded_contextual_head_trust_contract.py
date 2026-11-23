"""Z31: reversible true-native49152 contextual lowrank head trust region.

Strictly prevent repeating Z29 collapse: fixed global output_bias
copied from Z10, optimizer touches ONLY copied 320x4 down and
4x49152 up (197888 parameters). The learned per-vocab correction
is bounded by ±2 PRE native-head-gain. Reference Z10 lowrank
correction is evaluated from the SAME observed original S142
paid h320; no second original43M forward required per position.

native_lm_logits_new = original_S142_lm_logits +
  0.25 * (old_Z10_delta + 2*tanh((new_delta-old_delta)/2))
native_lm_logits_reference = original_S142_lm_logits + 0.25*old_Z10_delta
reference = (new_model_logits - 0.25*clipped_diff +
             0.25*(old_delta-old_delta))  [detached]
All native49152 candidate token probabilities intact, no EOS masking.
TRAIn only: KL(reference softmax || new) weight 1 + L2 weight .02,
plus position genuine CE and anti-echo and premature EOS from Z28.
INFERENCE also uses same bounded head hook; never DEV gold.
Do not create or edit original or prior fitted model weights.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z31_84legaltrain_new14dev_bounded_context_source import (
    NATIVE_REFERENCE_KL,CONTEXT_PREGAIN_RESIDUAL_BOUND,ORIGINAL_WEIGHT_L2,
    TRAINABLE_DOWN_UP,FIXED_VOCAB_OUTPUT_BIAS,NATIVE_HEAD_GAIN,
)
VOCAB=49152
HIDDEN=320
GAIN=.25
BOUND=2.0

def check(x,m):
    if not x:raise ValueError("Z31 true original native S142 rank4 bounded causal correction: "+m)

def frozen_new_head_params(learned,original):
    check(learned is not original
          and learned.output_gate==original.output_gate==0
          and learned.candidate_accepted is original.candidate_accepted is False
          and learned.activation_allowed is original.activation_allowed is False,
          "both copied and original Z10 native readout are research-only")
    nd=list(learned.down.parameters())+list(learned.up.parameters())
    check(sum(p.numel() for p in nd)==TRAINABLE_DOWN_UP==197888
          and learned.output_bias.numel()==FIXED_VOCAB_OUTPUT_BIAS==49152
          and sum(p.numel() for p in learned.parameters())==247040
          and set(learned.state_dict())==set(original.state_dict())==
             {"down.weight","up.weight","output_bias"},
          "only exact 320x4x49152 lowrank head trainable, original bias fixed")
    learned.output_bias.requires_grad_(False)
    for p in nd:p.requires_grad_(True)
    for p in original.parameters():p.requires_grad_(False)
    check(torch.equal(learned.output_bias.detach(),original.output_bias.detach())
          and not learned.output_bias.requires_grad
          and all(p.requires_grad for p in nd)
          and NATIVE_REFERENCE_KL==1.0 and ORIGINAL_WEIGHT_L2==.02
          and CONTEXT_PREGAIN_RESIDUAL_BOUND==2.0 and NATIVE_HEAD_GAIN==.25,
          "frozen baseline global output bias, train only contextual lowrank trust")
    return nd

@contextmanager
def bounded_frozen_original_native_delta(learner_head,original_head):
    """Register before gain hook. One SAME paid native hidden state only."""
    check(learner_head is not original_head and
          learner_head.output_gate==original_head.output_gate==0,
          "bounded native original head vs newly copied head")
    before=set(learner_head._forward_hooks)
    captures=[]
    def bound(_module,inputs,unbounded):
        check(len(inputs)==1 and inputs[0].ndim==3
              and inputs[0].shape[-1]==HIDDEN
              and unbounded.shape[-1]==VOCAB
              and bool(torch.isfinite(unbounded).all()),
              "actual same paid original S142 native head h320")
        with torch.no_grad():
            old=original_head(inputs[0].detach()).detach()
        check(old.shape==unbounded.shape and bool(torch.isfinite(old).all()),
              "old frozen Z10 rank4 readout same native hidden, no extra S142 decoder")
        difference=unbounded-old
        clipped=BOUND*torch.tanh(difference/BOUND)
        check(bool(torch.isfinite(clipped).all())
              and bool((clipped.abs()<=BOUND+1e-6).all()),
              "all 49152 contextual corrections bounded")
        captures.append({"bounded_contextual_difference":clipped,
                         "reference_old_delta":old,
                         "same_original_S142_last_hidden":inputs[0].detach(),
                         "original43M_extra_decoder_calls":0})
        return old+clipped
    handle=learner_head.register_forward_hook(bound)
    try:
        yield captures
    finally:
        handle.remove()
        check(set(learner_head._forward_hooks)==before,
              "reversible native bounded head hook, no LIVE changes")

def trust_kl_after_same_paid_forward(new_native_logits,cap,old_head,new_head):
    check(isinstance(new_native_logits,torch.Tensor)
          and new_native_logits.shape==(1,VOCAB)
          and bool(torch.isfinite(new_native_logits).all())
          and new_native_logits.requires_grad
          and isinstance(cap,dict)
          and cap["bounded_contextual_difference"].shape[-1]==VOCAB
          and cap["original43M_extra_decoder_calls"]==0
          and GAIN==NATIVE_HEAD_GAIN==.25,
          "actual model full49152 new logit, same-pass reference from old Z10 head")
    bounded=cap["bounded_contextual_difference"]
    # The native S142 logits and normal learned Z10 head are in same
    # physical original lm_head hook. Remove ONLY the learned bounded
    # deviation to recover exactly old Z10 logits as reference.
    reference=(new_native_logits.detach() - GAIN*bounded.detach())
    check(reference.shape==(1,VOCAB)
          and torch.isfinite(reference).all() and not reference.requires_grad,
          "reference full native logits reconstructed without ANY second original S142 pass")
    kl=F.kl_div(F.log_softmax(new_native_logits.float(),dim=-1),
                F.softmax(reference.float(),dim=-1),reduction="sum")
    old_d=old_head.down.weight.detach()
    old_u=old_head.up.weight.detach()
    l2=(new_head.down.weight-old_d).square().mean()+(
        new_head.up.weight-old_u).square().mean()
    loss=NATIVE_REFERENCE_KL*kl+ORIGINAL_WEIGHT_L2*l2
    check(torch.isfinite(loss) and loss.requires_grad and torch.isfinite(kl)
          and bool((bounded.detach().abs()<=BOUND+1e-6).all()),
          "native KL-to-original output plus contextual L2 tether")
    return loss,{
        "full49152_exact_reference_KL":float(kl.detach()),
        "original_contextual_weight_L2":float(l2.detach()),
        "max_abs_bounded_native_delta_before_gain":
            float(bounded.detach().abs().max()),
        "all_correct_native_token_candidates_unmasked":True,
        "new_original_S142_model_forwards_for_reference":0,
    }
