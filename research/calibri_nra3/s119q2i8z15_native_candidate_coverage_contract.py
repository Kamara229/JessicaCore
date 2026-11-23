"""S119q2i8z15 target-blind ORIGINAL native49152 BPE2 topK coverage contract.

Candidate top-k and mass are computed WITHOUT gold. Only then posthoc
separately score the correct human BPE2, solely when FIRST token was
actually generated correctly by the real model. Does not improve/suggest
model logits, generation, learning, hardware throughput or any deployment.
"""
from __future__ import annotations
import torch
from research.calibri_nra3.s119q2i8z14_fresh14_bpe2_coverage_source import BUDGETS

VOCAB=49152
EOS_ID=2

def check(ok,msg):
    if not ok:
        raise ValueError("Z15 no-oracle original49k candidate contract: "+msg)

@torch.no_grad()
def candidate_distribution(native_logits:torch.Tensor)->dict:
    check(isinstance(native_logits,torch.Tensor)
          and native_logits.shape==(1,VOCAB)
          and bool(torch.isfinite(native_logits).all()),
          "actual full49152 finite native next-token logits")
    v=native_logits.detach().to(torch.float32)[0]
    lp=torch.log_softmax(v,dim=-1)
    vals, ids=torch.topk(lp, BUDGETS[-1], sorted=True)
    choices=[int(i) for i in ids.tolist()]
    check(len(choices)==512 and len(set(choices))==512,
          "ordered top512 all unique native BPE candidates")
    mass={str(k):float(torch.exp(torch.logsumexp(vals[:k],dim=0)))
          for k in BUDGETS}
    check(all(0<mass[str(k)]<=1+1e-5 for k in BUDGETS),
          "nonzero finite native full-vocab probability mass within budget")
    eos_rank=1+int((v>v[EOS_ID]).sum().item())
    return {
        "top512_candidate_native_ids":choices,
        "native_greedy_second_id":choices[0],
        "topK_probability_mass":mass,
        "rank_of_EOS_native49152":eos_rank,
        "EOS_in_top8":EOS_ID in choices[:8],
        "EOS_in_top512":EOS_ID in choices,
        "full_vocab_log_probability_EOS":float(lp[EOS_ID]),
        "entropy_native49152_nats":float(-torch.exp(lp).mul(lp).sum()),
        "target_gold_never_used_for_candidate_selection":True,
    }

def posthoc_gold_second(a:dict,gold_second:int,*,own_first_correct:bool)->dict:
    check(type(gold_second) is int and 0<=gold_second<VOCAB
          and type(own_first_correct) is bool
          and a["target_gold_never_used_for_candidate_selection"] is True,
          "human gold never passed to search, only conditional posthoc")
    candidates=a["top512_candidate_native_ids"]
    check(len(candidates)==512 and len(set(candidates))==512,
          "target-blind candidate list and explicit budget")
    coverage={str(k):bool(own_first_correct and gold_second in candidates[:k])
              for k in BUDGETS}
    return {
        "own_first_token_correct":own_first_correct,
        "posthoc_second_gold_ID_not_used_for_selection":gold_second,
        "greedy_SECOND_correct":bool(own_first_correct and
                                     a["native_greedy_second_id"]==gold_second),
        "conditional_correct_gold_BPE2_in_topK":coverage,
        "native_gold_rank_if_in_top512":(
            candidates.index(gold_second)+1 if own_first_correct and
            gold_second in candidates else None),
        "target_scores_do_not_affect_topK_ranking":True,
    }
