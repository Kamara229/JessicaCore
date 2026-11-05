"""Model-agnostic Green Tunnel full-output-vocabulary repair signal.

This is an OPTIMIZER objective. The user-facing quality gates are
whole generated answers (and native CPU), NEVER these loss magnitudes.
At every selected TRAIN token, all 49,152 vocabulary entries contribute
to the teacher-guided gradient, not only 'Yes', 'No', 'red', '5'.
"""
import torch
from torch.nn import functional as F

from v186.fit import paired_loss
from . import protocol as P
from .policy import POLICY


def language_objective(student,reference):
    ids=reference["ids"]
    positions=reference["positions"]
    targets=reference["gold_actual_NEXT_tokens"]
    teacher_probs=reference["full_49152_token_teacher_probs_fp16"].float()
    if teacher_probs.ndim!=2 or teacher_probs.shape!=(len(positions),P.VOCAB_SIZE):
        raise AssertionError("Missing universal full teacher vocabulary")
    logits=student(input_ids=ids,use_cache=False).logits[0,positions]
    if logits.shape!=teacher_probs.shape or not torch.isfinite(logits).all():
        raise AssertionError("All-vocabulary restore training invalid model output")
    softlog=F.log_softmax(logits/POLICY.teacher_soft_temperature,dim=-1)
    # Full-vocabulary teacher CE, not select-gold or top-k distillation.
    full_vocab_guidance=-(teacher_probs*softlog).sum(dim=-1).mean()
    actual_word_sequence=F.cross_entropy(logits,targets)
    supervised=reference["kind"]=="supervised_chat"
    # Preserve original style and *complete* teacher distribution
    # while prioritizing the actual answer tokens for chat records.
    teacher_weight=0.55 if supervised else 0.85
    loss=teacher_weight*full_vocab_guidance+(
         1-teacher_weight)*actual_word_sequence
    if not torch.isfinite(loss):
        raise ValueError("Green Tunnel nonfinite full-vocabulary objective")
    return loss,{"full_vocabulary_teacher_CE_TRAIN_ONLY":float(full_vocab_guidance.detach()),
      "actual_next_word_TRAIN_only_CE":float(actual_word_sequence.detach()),
      "full_vocab_entries_count":P.VOCAB_SIZE,
      "actual_target_token_positions":len(positions),
      "supervised_full_answer_tokens":supervised}


def balanced_pair_objective(student,pair):
    loss,record=paired_loss(student,None,pair)
    if not torch.isfinite(loss):
        raise ValueError("Green Tunnel paired behavior objective nonfinite")
    return loss,{"paired_TRAIN_contrast":record,
        "paired_model_response_changes_can_be_tested_on_HELD_OUT_valid":True}


def choose_objective(student,step,refs,pairs):
    if (step+1)%POLICY.pair_every_n_steps==0:
        index=((step+1)//POLICY.pair_every_n_steps-1)%len(pairs)
        return balanced_pair_objective(student,pairs[index])+( "paired",index )
    index=(step-(step//POLICY.pair_every_n_steps))%len(refs)
    return language_objective(student,refs[index])+("whole_vocabulary",index)
