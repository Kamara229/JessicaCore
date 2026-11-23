"""Z12 TRAIN-FREE native49k multi-candidate lookahead contract.

Hypothesis: after the model produces its OWN first token, branch at the
SECOND native BPE (up to top8), look ahead to up to top2 THIRD native
BPE for each branch, and select max average joint model log probability.
No gold or target may be passed to search. NO EOS mask, no model weight
updates, no new trainable decoder, no inference logit boost. Greedy and
lookahead use IDENTICAL original native49152 model distributions.

CPU prototype evaluates branches sequentially, not batched parallel GPU
hardware. This tests alternative future contexts, NOT CPU acceleration.
At current weights, high-rank missing true BPE2 cannot be rescued by
top8 search; gold top8 coverage is POSTHOC diagnostic ONLY.
"""
from __future__ import annotations
from typing import Callable
import math
import torch

NATIVE_VOCAB=49152
EOS=2
WIDTH=8
THIRD_WIDTH=2
HORIZON=2
MAX_PREFIX=160

def check(v, message):
    if not v:
        raise ValueError("Z12 parallel native branch strict contract: "+message)

@torch.no_grad()
def native_lookahead_second_third(
    forward_native:Callable[[tuple[int,...]],torch.Tensor],
    prefix_with_own_first:list[int] | tuple[int,...],
)->dict:
    """Return only own-logit-conditional search, NEVER consume target/gold."""
    check(isinstance(prefix_with_own_first,(list,tuple))
          and 1<len(prefix_with_own_first)<MAX_PREFIX-2
          and all(type(v) is int and 0<=v<NATIVE_VOCAB
                  for v in prefix_with_own_first),
          "an independently generated, valid native prefix incl first OWN BPE")
    prefix=tuple(prefix_with_own_first)
    calls=0
    def fetch(tokens:tuple[int,...]):
        nonlocal calls
        check(len(tokens)<=MAX_PREFIX,"all branches must fit original decoder limit")
        x=forward_native(tokens)
        calls+=1
        check(isinstance(x,torch.Tensor)
              and tuple(x.shape)==(1,NATIVE_VOCAB)
              and bool(torch.isfinite(x).all())
              and not bool(x.requires_grad),
              "TRUE no-gradient native49152 logits for each exact branch")
        return torch.log_softmax(x.detach().to(torch.float32),dim=-1)[0]
    lp2=fetch(prefix)
    t2values,t2ids=torch.topk(lp2,WIDTH,sorted=True)
    t2=[(int(id.item()),float(val.item())) for id,val in zip(t2ids,t2values)]
    check(len(t2)==WIDTH and len(set(t for t,_ in t2))==WIDTH,
          "unique top8 including EOS when ranked")
    options=[]
    greedy_second=t2[0][0]
    greedy_third=None
    for second, logp_second in t2:
        if second==EOS:
            options.append({"continuation_native_ids":[second],
                            "mean_log_probability":logp_second,
                            "sum_log_probability":logp_second,
                            "EOS_terminal_at_second":True,
                            "second_native_id":second})
            continue
        lp3=fetch(prefix+(second,))
        third_values,third_ids=torch.topk(lp3,THIRD_WIDTH,sorted=True)
        for j,(third,lp) in enumerate(zip(third_ids,third_values)):
            token=int(third.item())
            score=float(lp.item())
            if second==greedy_second and j==0:
                greedy_third=token
            options.append({"continuation_native_ids":[second,token],
                            "mean_log_probability":(logp_second+score)/HORIZON,
                            "sum_log_probability":logp_second+score,
                            "EOS_terminal_at_second":False,
                            "second_native_id":second})
    check(len(options)==WIDTH*THIRD_WIDTH-(THIRD_WIDTH-1)*sum(x[0]==EOS for x in t2)
          and calls==1+sum(x[0]!=EOS for x in t2)
          and (greedy_second==EOS or greedy_third is not None),
          "strict 8 second branches, 2 continuation probes each, EOS unmasked")
    # Pre-registered deterministic two-token average negative log-likelihood
    # per actual token. This is a heuristic, NOT verified or learned quality.
    best=sorted(options,
                key=lambda o:(-o["mean_log_probability"],
                              tuple(o["continuation_native_ids"])))[0]
    greedy=[greedy_second] if greedy_second==EOS else [greedy_second,greedy_third]
    return {
        "mode":"TRUE_NATIVE49152_LOGICAL_PARALLEL_BEAM_SEQUENTIAL_CPU_FORWARDS",
        "top8_second_native_ids":[second for second,_ in t2],
        "second_top8_native_logprobs":[round(value,8) for _,value in t2],
        "own_greedy_continuation_2native_BPE":greedy,
        "lookahead_selected_continuation_2native_BPE":best["continuation_native_ids"],
        "lookahead_changes_greedy_second_BPE":best["continuation_native_ids"][0]!=greedy[0],
        "lookahead_changes_greedy_second_or_third_BPE":best["continuation_native_ids"]!=greedy,
        "lookahead_score_mean_logprob":best["mean_log_probability"],
        "actual_candidate_count":len(options),
        "actual_original43M_native_forward_calls":calls,
        "EOS_2_allowed":True,
        "HORIZON":HORIZON,"WIDTH":WIDTH,"THIRD_WIDTH":THIRD_WIDTH,
        "no_gold_used_in_search":True,
        "no_weights_or_logits_modified":True,
        "candidate_generation_is_sequential_CPU_not_parallel_wallclock":True,
    }
