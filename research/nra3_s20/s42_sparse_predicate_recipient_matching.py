"""S42 research-only sparse directed predicate→recipient matching decoder.

S41's output-only rule added 33 false REJECTED edges in the construction
TRAIN-held development folds by *simultaneously accepting all* eligible
pairs. The S40 original learned edge scores may provide a rank signal
for a sparse graph.

One fixed, label-blind greedy bipartite hypothesis:
  - candidates = S40 structurally eligible directed edges only
  - sort candidate edges by original S40 learned logit DESC
  - at most one outgoing REJECTED per source and incoming per recipient
  - selected scores floored to +0.25, all others blocked
  - identical original S40 event and quote logits; no weight updates

This is specifically a *one-to-one* simplifying prior and WILL fail some
legitimate Russian multi-recipient / reentrant predicates. S39 TRAIN80
is already-inspected synthetic development data with exactly one gold
REJECTED edge per positive example. No claim of general grammar, real
Calibri43M+Green Tunnel, original NRA2, or independent EVAL.
"""
from __future__ import annotations
import torch
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import GraphPrediction
from research.nra3_s20.s40_compositional_predicate_negation_graph import STRUCTURE_DIM

S42_LINK_FLOOR_LOGIT=0.25
S42_RULE="one_source_one_target_directed_structural_greedy_matching"
S42_BASELINE_S41_RUN=44244725606

def matching_links_from_s40(original:GraphPrediction,
                            structure:torch.Tensor)->tuple[GraphPrediction,torch.Tensor]:
    """Plain scores and geometry only: labels and held-group never enter.

    Greedy max weight sparse matching uses S40 model ranking among
    structurally eligible edges. Supports [W,W] or [B,W,W] logits.
    """
    singleton=(original.rejected_logits.ndim==2)
    original_scores=original.rejected_logits
    if singleton: original_scores=original_scores.unsqueeze(0)
    if structure.ndim==3: structure=structure.unsqueeze(0)
    if not (structure.ndim==4 and structure.shape[-1]==STRUCTURE_DIM
            and structure.shape[:3]==original_scores.shape
            and original_scores.shape[1]==original_scores.shape[2]):
        raise ValueError("S42 requires original trained directed logits and raw12 graph")
    if not bool(torch.isfinite(original_scores).all()):
        raise ValueError("S42 source ranking must be finite")
    b,w,_=original_scores.shape
    eligible=structure[...,11]>0.5
    matched=torch.zeros_like(eligible)
    for index in range(b):
        # Use only original S40 score and eligibility, NOT human word,
        # construction type, gold or item identity.
        ranked=[
            (float(original_scores[index,i,j]),i,j)
            for i in range(w) for j in range(w)
            if i!=j and bool(eligible[index,i,j])
        ]
        ranked.sort(key=lambda t:(-t[0],t[1],t[2]))
        used_source=set()
        used_target=set()
        for _,i,j in ranked:
            if i in used_source or j in used_target: continue
            matched[index,i,j]=True
            used_source.add(i)
            used_target.add(j)
    scores=torch.full_like(original_scores,-1e6)
    scores[matched]=torch.maximum(
        original_scores[matched],
        torch.full_like(original_scores[matched],S42_LINK_FLOOR_LOGIT))
    if singleton:
        scores=scores[0]
        matched=matched[0]
    return GraphPrediction(original.event_logits,original.quote_logits,scores),matched


def verify_readonly_matching(original,matched_output,matched_mask,structure):
    """Protected heads and all selected S40 numeric logits provenance."""
    if not torch.equal(original.event_logits,matched_output.event_logits):
        raise ValueError("S42 changed frozen event classifier")
    if not torch.equal(original.quote_logits,matched_output.quote_logits):
        raise ValueError("S42 changed frozen quotation classifier")
    if not bool((matched_output.rejected_logits[~matched_mask]==-1e6).all()):
        raise ValueError("S42 unselected REJECTED edge not blocked")
    if not bool((matched_output.rejected_logits[matched_mask]>=S42_LINK_FLOOR_LOGIT).all()):
        raise ValueError("S42 selected edge lacked fixed positive floor")
    if not bool(((structure[...,11]>0.5)|(~matched_mask)).all()):
        raise ValueError("S42 selected structurally ineligible edge")
    bmask=matched_mask.unsqueeze(0) if matched_mask.ndim==2 else matched_mask
    if not bool((bmask.sum(-1)<=1).all() and (bmask.sum(-2)<=1).all()):
        raise ValueError("S42 matching violates directed source/target uniqueness")
    return True
