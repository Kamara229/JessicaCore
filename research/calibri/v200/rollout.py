"""v200 TRAIN-only bounded greedy sequence rollout (not a quality certification).

All trainable values are existing packed 135M model codebooks/indices.
Greedy prefixes make the surrogate face the tokens it actually generates;
full-answer tests and independent FP32 checks are held out until sealing.
"""
from collections import Counter
import torch
from torch.nn import functional as F
from v176.operators import clear

def _sample(pair,side):
    if pair.get("split")!="TRAIN" or side not in ("left","right"):
        raise AssertionError("Only precommitted opposing TRAIN is permitted")
    x=pair[side]
    a,b=x["prompt_ids"],x["targets"]
    if a.ndim!=1 or b.ndim!=1 or len(b)==0:
        raise ValueError("Missing complete TRAIN answer/EOS")
    if int(b[0])!=int(x["first_expected_id"]):
        raise AssertionError("Frozen first TRAIN token drift")
    return a,b

def paired_greedy_rollout_loss(student,pair,*,steps_cap=4):
    """Backprop CE over own detached argmax prefixes on both opposing answers.

    Targets contain the complete gold answer and EOS. No sampling outside TRAIN,
    no changes to the packed source storage or inference-time model.
    """
    if not 1<=steps_cap<=12:raise ValueError("Invalid rollout horizon")
    losses=[];predicted={};expected={}
    for side in ("left","right"):
        prompt,gold=_sample(pair,side)
        context=prompt.detach().clone()[None]
        side_losses=[];tokens=[]
        for k in range(min(steps_cap,len(gold))):
            logits=student(input_ids=context,use_cache=False).logits[0,-1]
            if not torch.isfinite(logits).all():
                raise AssertionError("Nonfinite TRAIN rollout")
            side_losses.append(F.cross_entropy(logits[None,:],gold[k].reshape(1)))
            token=logits.argmax().detach().reshape(1,1)
            tokens.append(int(token))
            context=torch.cat((context,token),dim=1)
        losses.append(torch.stack(side_losses).mean())
        predicted[side]=tokens
        expected[side]=[int(v) for v in gold[:steps_cap]]
    if expected["left"][0]==expected["right"][0]:
        raise AssertionError("TRAIN opposite pair not contradictory")
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss):raise AssertionError("Nonfinite rollout gradient")
    return loss,{"TRAIN_pair_id":pair["pair_id"],"greedy_predicted":predicted,
                 "TRAIN_gold_complete_sequence_prefixes":expected,
                 "generated_prefix_not_teacher_forced":True,"QA_or_VALID_used":False}

@torch.no_grad()
def probe_greedy_train(student,sources,pairs,*,max_new_tokens=6):
    """Generate actual token sequences on frozen TRAIN, not on QA/VALID."""
    if not 2<=max_new_tokens<=12 or not pairs:
        raise ValueError("Bounded nonempty TRAIN rollout required")
    result={}
    try:
        for pair in pairs:
            if pair.get("split")!="TRAIN":
                raise AssertionError("HELDOUT or QA attempted as TRAIN gate")
            entry={"domain":pair["domain"],"sides":{}}
            for side in ("left","right"):
                prompt,gold=_sample(pair,side)
                context=prompt.detach().clone()[None]
                eos=int(gold[-1]);generated=[]
                for _ in range(max_new_tokens):
                    clear(sources)
                    logits=student(input_ids=context,use_cache=False).logits[0,-1]
                    if not torch.isfinite(logits).all():
                        raise AssertionError("Nonfinite generated TRAIN sequence")
                    token=int(logits.argmax());generated.append(token)
                    if token==eos:break
                    context=torch.cat((context,context.new_tensor([[token]])),dim=1)
                expected=[int(t) for t in gold]
                prefix=0
                for a,b in zip(generated,expected):
                    if a!=b:break
                    prefix+=1
                entry["sides"][side]={
                    "gold_tokens_with_EOS":expected,"generated_tokens":generated,
                    "exact_with_EOS":generated==expected,
                    "correct_prefix_tokens":prefix,
                    "adjacent_repetition_excess":sum(
                        a==b for a,b in zip(generated[:-1],generated[1:])),
                    "stopped_at_EOS":bool(generated and generated[-1]==eos)}
            result[pair["pair_id"]]=entry
    finally:
        clear(sources)
    return result

def summarize_sequences(probes):
    if not probes:raise ValueError("Empty TRAIN sequence samples")
    exact=set();by_domain=Counter();repetition=prefixes=0
    for pair_id, row in probes.items():
        if set(row["sides"])!={"left","right"}:
            raise ValueError("Opposing sides incomplete")
        for side in ("left","right"):
            v=row["sides"][side]
            if v["exact_with_EOS"]:
                exact.add((pair_id,side));by_domain[row["domain"]]+=1
            prefixes+=v["correct_prefix_tokens"]
            repetition+=v["adjacent_repetition_excess"]
    return {"exact":exact,"per_domain_exact":dict(by_domain),
            "prefix_tokens":prefixes,"repetition_excess":repetition}

def sequence_no_regression_gate(immutable,candidate):
    """No destruction of original full correct TRAIN responses or worse loops."""
    if set(immutable)!=set(candidate):
        raise ValueError("Frozen complete TRAIN pair identities drift")
    for key in immutable:
        a,b=immutable[key],candidate[key]
        if a["domain"]!=b["domain"] or set(a["sides"])!=set(b["sides"]):
            raise ValueError("Frozen TRAIN pair sides/domain changed")
        for side in ("left","right"):
            if a["sides"][side]["gold_tokens_with_EOS"]!=b["sides"][side]["gold_tokens_with_EOS"]:
                raise ValueError("Gold complete TRAIN answers changed")
    old,new=summarize_sequences(immutable),summarize_sequences(candidate)
    missing=sorted(old["exact"]-new["exact"])
    worse_prefix=new["prefix_tokens"]<old["prefix_tokens"]
    worse_repeats=new["repetition_excess"]>old["repetition_excess"]
    domains={d:{"before":v,"after":new["per_domain_exact"].get(d,0)}
             for d,v in old["per_domain_exact"].items()
             if new["per_domain_exact"].get(d,0)<v}
    return {"ACCEPT_FULL_SEQUENCE_TRAIN_ONLY":not (
                missing or worse_prefix or worse_repeats or domains),
            "lost_original_complete_TRAIN_answers":missing,
            "correct_prefix_before":old["prefix_tokens"],
            "correct_prefix_after":new["prefix_tokens"],
            "adjacent_repeats_before":old["repetition_excess"],
            "adjacent_repeats_after":new["repetition_excess"],
            "prefix_regressed":worse_prefix,"repetition_regressed":worse_repeats,
            "complete_exact_domain_regressions":domains,
            "TRAIN_only_NOT_model_parity":True}
