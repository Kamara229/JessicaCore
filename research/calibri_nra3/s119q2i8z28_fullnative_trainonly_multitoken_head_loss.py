"""Z28 exactly preregistered REAL full-native human TRAIN BPE1-4+EOS objective.

The task is no longer to add token scoring on top of degenerate
teacher logits. Train ONLY a new *COPY* of the Z10 fitted native rank4
language-head weights via REAL original S142 43M 49152-way teacher
forced CE on five fixed human TRAIN target positions per example.
Add explicit differentiable margin against echoing preceding native
BPE when next gold differs, and against premature EOS before real EOS.
Full native 49152 logits, no token list masks, no special output words.
No gold or teacher-forcing at DEV/inference. Original pretrained
S142/GT12/Core/NRA3/NPA1 and existing Z10 A/B FP16 never optimized.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z28_precommit_84train14fresh_native_head_multistep import (
    TRAIN_SUPERVISED_POSITIONS,TRAIN_WEIGHTS,
    TRAIN_REPETITION_MARGIN,TRAIN_REPETITION_LOSS,
    TRAIN_PREMATURE_EOS_LOSS,HEAD_GAIN,HEAD_LR,
    HEAD_WEIGHT_DECAY,TRAIN_EPOCHS,EVAL_NATIVE_FIRST_K,
)
VOCAB=49152
EOS=2
HEAD_TOTAL_PARAMS=247040

def check(ok,msg):
    if not ok:raise ValueError("Z28 real native full-vocab multi-position TRAIN contract: "+msg)

def select_train_positions(gold_native_with_eos):
    check(isinstance(gold_native_with_eos,(list,tuple))
          and len(gold_native_with_eos)>=5
          and type(gold_native_with_eos[-1]) is int
          and gold_native_with_eos[-1]==EOS
          and all(type(x) is int and 0<=x<VOCAB
                  for x in gold_native_with_eos)
          and TRAIN_SUPERVISED_POSITIONS==(0,1,2,3,-1)
          and TRAIN_WEIGHTS==(2.,1.,1.,1.,2.),
          "legal Z16/Z18/Z21 TRAIN native token sequence with true EOS")
    indexes=[0,1,2,3,len(gold_native_with_eos)-1]
    check(len(set(indexes))==5 and all(0<=i<len(gold_native_with_eos)
                                      for i in indexes),
          "first4 full human BPE and EOS distinct, no future leaks")
    return list(zip(indexes,TRAIN_WEIGHTS))

def full_native_trainonly_pos_loss(logits,gold,position,weight,*,split):
    check(split=="TRAIN" and isinstance(logits,torch.Tensor)
          and logits.shape==(1,VOCAB)
          and logits.requires_grad
          and bool(torch.isfinite(logits).all())
          and isinstance(gold,(list,tuple))
          and len(gold)>=5 and gold[-1]==EOS
          and type(position) is int and 0<=position<len(gold)
          and (position<4 or position==len(gold)-1)
          and float(weight) in (1.,2.)
          and (position==len(gold)-1)==(gold[position]==EOS)
          and TRAIN_REPETITION_MARGIN==.25
          and TRAIN_REPETITION_LOSS==.25
          and TRAIN_PREMATURE_EOS_LOSS==.10,
          "ONLY human TRAIN teacher-forced actual original49152 next token")
    target=int(gold[position])
    ce=F.cross_entropy(logits,torch.tensor([target],dtype=torch.long))
    repeat=torch.zeros_like(ce)
    if position>0 and int(gold[position-1])!=target and gold[position-1]!=EOS:
        repeat=F.softplus(
            logits[0,int(gold[position-1])]-logits[0,target]+TRAIN_REPETITION_MARGIN)
    premature=torch.zeros_like(ce)
    if target!=EOS:
        premature=F.softplus(logits[0,EOS]-logits[0,target]+TRAIN_REPETITION_MARGIN)
    loss=(weight*ce+TRAIN_REPETITION_LOSS*repeat+
          TRAIN_PREMATURE_EOS_LOSS*premature)/sum(TRAIN_WEIGHTS)
    check(bool(torch.isfinite(loss).all()) and loss.requires_grad,
          "finite TRUE full-vocab CE and anti-stutter/early EOS gradients")
    return loss,{
        "human_gold_native_index":position,
        "is_REAL_native_EOS_target":target==EOS,
        "TRAIN_correct_next_BPE_target_only":True,
        "base_native_full49152_CE":float(ce.detach()),
        "echo_margin":float(repeat.detach()),
        "premature_EOS_margin":float(premature.detach()),
        "weighted_summed_loss":float(loss.detach()),
    }

def verify_shadow_head_params(head,original_old_head=None):
    check(hasattr(head,"down") and hasattr(head,"up")
          and hasattr(head,"output_bias")
          and sum(p.numel() for p in head.parameters())==HEAD_TOTAL_PARAMS
          and head.output_gate==0 and not head.candidate_accepted
          and not head.activation_allowed
          and (original_old_head is None or
               head is not original_old_head)
          and all(bool(torch.isfinite(p).all()) for p in head.parameters())
          and HEAD_GAIN==.25 and HEAD_LR==.006
          and HEAD_WEIGHT_DECAY==.02 and TRAIN_EPOCHS==2
          and EVAL_NATIVE_FIRST_K==6,
          "exact COPY of original 247040param learned rank4 native head, not live")
    return True
