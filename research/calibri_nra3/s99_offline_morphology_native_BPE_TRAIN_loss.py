"""NRA3 S99: STRICT OFFLINE UD-Taiga morphology TRAIN loss.

This function is ONLY called by the training driver AFTER actual
Calibri+GT S98 physical 12-layer source checksum verification and
after separate UD teacher labels are opened. Never import this loss
inside the NRA3 neural forward, the decoder, or a live process.

UNKNOWN=-100 is not a negative morphological class. POS, Case,
Number, Voice are learned independently at source-native Calibri BPE
last-piece positions, with no guessed labels for missing UD features.
No old S96 internal dev, S79/S86 spent firstreads or upstream UD
dev/test. CC BY-SA 4.0 training-only pending legal review.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyGraph,UD_UPOS,UD_CASE,UD_NUMBER,UD_VOICE,
)

UNKNOWN=-100
STAGE="CALIBRI_NRA3_S99_OFFLINE_NATIVE_BPE_MORPHOLOGY_TRAIN_LOSS_ONLY"


def require(ok,reason):
    if not bool(ok):
        raise ValueError("NRA3 S99 training-only morphology labels: "+reason)


def offline_native_morphology_supervision_loss(
        graph:NRA3MorphologyGraph,upos,case,number,voice,source_valid):
    """Gradients only to registered Calibri child, no teacher inference."""
    valid=graph.native_valid
    require(source_valid.dtype==torch.bool
            and torch.equal(valid,source_valid)
            and valid.ndim==2 and bool(valid.any(-1).all()),
            "source-native BPE original tokenizer mask immutable")
    mapping={
        "upos":(graph.upos_logits,upos,len(UD_UPOS)),
        "case":(graph.case_logits,case,len(UD_CASE)),
        "number":(graph.number_logits,number,len(UD_NUMBER)),
        "voice":(graph.voice_logits,voice,len(UD_VOICE)),
    }
    losses={}
    support={}
    for name,(scores,truth,num_classes) in mapping.items():
        require(scores.shape==valid.shape+(num_classes,)
                and truth.shape==valid.shape
                and truth.dtype==torch.long
                and not truth.requires_grad
                and bool(((truth==UNKNOWN)|
                          ((truth>=0)&(truth<num_classes))).all())
                and not bool(((truth!=UNKNOWN)&~valid).any())
                and bool(torch.isfinite(scores).all()),
                "explicit morph annotation only on valid original native BPE")
        has=truth!=UNKNOWN
        support[name]=int(has.sum())
        if bool(has.any()):
            losses[name]=F.cross_entropy(scores[has],truth[has])
        else:
            # No fabricated negatives. Still a differentiable zero
            # when an entire small TRAIN microbatch has no Voice labels.
            losses[name]=scores.sum()*0.
    require(support["upos"]>0 and sum(support.values())>0,
            "at least one explicitly annotated POS for this batch")
    total=(losses["upos"]+losses["case"]+losses["number"]
           +losses["voice"])
    require(torch.isfinite(total),"finite morphological TRAIN-only loss")
    return total,{
        "per_head_train_loss_detached":{k:float(x.detach()) for k,x in losses.items()},
        "n_explicitly_annotated_original_BPE_by_head":support,
        "unannotated_morphology_not_assumed_negative":True,
        "original_Calibri_hidden_detached_not_updated":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
