"""S11 strictly TRAIN-only causal weak 3way labels on S9's 72 new UD rows.

These are labels used in a *TRAIN loss only*; never enter the original
Calibri 43M, the packed-C NRA, the paid router inference, or any EVAL.
They are NOT semantic accuracy, only an imperfect observable heuristic.
"""
from __future__ import annotations
import torch

from research.nra3_s9.new_ru_train72 import (
    NewRussianTrain,S9_TRAIN_SPLIT,
    build_s9_train72,
)
from research.nra3_s4.complete_prefix_training_contract import CausalAlignment
from research.nra3_s7.parallel_hypotheses import _ROLE_EDGES

BRANCHES=("GLOBAL","LOCAL","PREDICTED_ROLE")
_COMPLEMENT=frozenset(tuple(sorted((a,b))) for a,b in _ROLE_EDGES)


def weak_train_router_labels(row:NewRussianTrain,align:CausalAlignment,
                             *,local_window:int=16)->torch.Tensor:
    old={r.sha256():r for r in build_s9_train72()}
    if (not isinstance(row,NewRussianTrain) or
        row.split!=S9_TRAIN_SPLIT or row!=old.get(row.sha256()) or
        not isinstance(align,CausalAlignment) or
        align.independent_eval_labels_loaded or
        align.original_prompt_sha256!=row.sha256() or
        len(align.word_alignment)!=len(row.words) or
        align.complete_prefix_position!=align.tokenizer_token_count-1 or
        not 1<=local_window<=16):
        raise ValueError("S11 cannot optimize unknown/unfrozen/EVAL Russian TRAIN")
    positions=[a.last_token for a in align.word_alignment]
    if (not positions or not all(isinstance(p,int) and 0<=p<align.tokenizer_token_count
                                 for p in positions) or
        not all(positions[i]<positions[i+1] for i in range(len(positions)-1))):
        raise ValueError("S11 cannot supervise future/overlapping wordpieces")
    labels=[]
    for i,(word,semantic) in enumerate(zip(row.words,row.roles)):
        complementary=semantic!="NONE" and any(
            tuple(sorted((semantic,other))) in _COMPLEMENT
            for other in row.roles[:i])
        head=word.head
        head_observed=(0<head<=len(row.words) and
                       positions[head-1]<=positions[i])
        nearby=(head_observed and
                positions[i]-positions[head-1]<local_window)
        labels.append(2 if complementary else (1 if nearby else 0))
    return torch.tensor(labels,dtype=torch.long)
