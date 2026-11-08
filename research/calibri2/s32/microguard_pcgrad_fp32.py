"""CALIBRI S32: 36M matched dense FP32, fine-grained protected TRAIN learning.

S31 reverted all eight 24-step blocks: 6/8 old TRAIN exact+EOS answers
were protected, but none of the new learning could survive. S32 freezes
a different TRAIN-only mechanism: 96 attempted steps, 2-step rollback
windows, 3:1 new-vs-old curriculum, smaller AdamW LR, gradient surgery
against a historical TRAIN pair, and full exact-EOS old TRAIN replay gate.

This is a single predeclared experiment; no S29 unseen QA nor previously
inspected S26 QA is imported or scored. Guard questions are *TRAIN*, so a
guard passing proves only preservation of those selected training answers.
No claim of unseen generalization, compression, or production change.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from v186.pairs import build_pairs
from v174.run import save

SOURCE_SHA=s30.SOURCE_SHA
S29_SEALED_SHA=s30.S29_FROZEN_SHA
STEPS=96
BLOCK_STEPS=2
LR=0.000025
OLD_GRAD_BLEND=.65
CURRICULUM=("new","new","old","new")
assert STEPS%BLOCK_STEPS==0


def ids_of_full_correct(probe,domain):
    return {r["id"] for r in probe["answers"]
            if r["kind"]==domain and r["strict_correct_full_EOS"]}


def old_train_probe(model,tok,old):
    # Strict identity of original 8 historical TRAIN answers, no prefix credit.
    result=[]
    for pair in old[:4]:
        for side in ("left","right"):
            row=pair[side]
            actual=s30.greedy_actual(model,tok,row["question"],max_new=10)
            gold=tok.encode(row["answer"],add_special_tokens=False)+[tok.eos_token_id]
            result.append({
                "id":f'{pair["pair_id"]}_{side}',
                "kind":"historical_TRAIN",
                "full_correct":actual==gold,
                "actual_text":tok.decode(actual,skip_special_tokens=True),
                "expected_tokens_EOS":gold,
                "actual_tokens":actual
            })
    if len(result)!=8:raise AssertionError("Expected exactly 8 old TRAIN probes")
    return result


def guarded_old_ids(probe):
    return {r["id"] for r in probe if r["full_correct"]}


def grad_step(model,opt,params,primary,old_guard,new_task):
    """PCGrad against an *old TRAIN* anchor; full 36M dense FP32 update."""
    opt.zero_grad(set_to_none=True)
    first=s30.train_loss(model,primary,new_task)
    if not torch.isfinite(first):raise AssertionError("Nonfinite primary TRAIN loss")
    objective_start=float(first.detach())
    first.backward()
    gp=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for p in params]
    opt.zero_grad(set_to_none=True)
    other=s30.train_loss(model,old_guard,False)
    if not torch.isfinite(other):raise AssertionError("Nonfinite old TRAIN loss")
    old_start=float(other.detach())
    other.backward()
    go=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for p in params]
    dot=sum((x.double()*y.double()).sum() for x,y in zip(gp,go))
    norm_old=sum((y.double()*y.double()).sum() for y in go)
    if not norm_old>0 or not torch.isfinite(norm_old):
        raise AssertionError("Zero/nonfinite old skill gradient")
    projected=bool(dot<0)
    coef=float(dot/norm_old) if projected else 0.
    for p,g,h in zip(params,gp,go):
        # Protect the old objective from conflicting new gradient, then add
        # a small positive old-skill replay gradient for both task types.
        p.grad=(g-coef*h+OLD_GRAD_BLEND*h).detach()
    clipped=float(torch.nn.utils.clip_grad_norm_(
        params,1.,error_if_nonfinite=True))
    if not clipped>0:raise AssertionError("No real full FP32 gradient")
    opt.step()
    return {"primary_loss_before":objective_start,
            "old_train_loss_before":old_start,
            "grad_conflict":projected,"old_gradient_projection":coef,
            "full_grad_norm_before_clipping":clipped}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(3030)
    torch.set_num_threads(2)
    raw=args.source.read_bytes()
    if len(raw)!=SOURCE_BYTES or hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise AssertionError("S32 source must equal S26 final 5-bit immutable CB59")
    cb=CB59(raw)
    if cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S32 source model unique parameter count mismatch")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    assert len(new)==32 and len(old)==24
    model=s30.dense_from_paid(cb)
    max_diff=s30.double_check_original_logits(cb,model,tok)
    initial_sha=s30.snapshot_sha(model)
    before=s30.train_probe(model,tok,new,old)
    previous=old_train_probe(model,tok,old)
    guarded=guarded_old_ids(previous)
    if len(guarded)!=6:
        raise AssertionError("S32 old exact-EOS baseline changed from S30/S31")
    params=list(model.parameters())
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.)
    task_count=Counter()
    attempts=[]
    accept=0
    reject=0
    effective=0
    conflicts=0
    for block in range(STEPS//BLOCK_STEPS):
        # Tiny two-step trust windows; rollback *optimizer moments* as well.
        saved_weights=copy.deepcopy(model.state_dict())
        saved_optimizer=copy.deepcopy(opt.state_dict())
        local=[]
        for k in range(BLOCK_STEPS):
            step=block*BLOCK_STEPS+k
            kind=CURRICULUM[step%len(CURRICULUM)]
            pos=task_count[kind]
            task_count[kind]+=1
            pair=(new[pos%len(new)] if kind=="new" else old[pos%len(old)])
            old_anchor=old[(step*5)%len(old)]
            item=grad_step(model,opt,params,pair,old_anchor,kind=="new")
            local.append({"step":step+1,"kind":kind,
                          "train_pair_id":pair["pair_id"],
                          "old_train_anchor_id":old_anchor["pair_id"],**item})
            conflicts+=int(item["grad_conflict"])
        candidate=old_train_probe(model,tok,old)
        retained=guarded_old_ids(candidate)
        passed=guarded.issubset(retained)
        if passed:
            accept+=1
            effective+=BLOCK_STEPS
        else:
            model.load_state_dict(saved_weights)
            opt.load_state_dict(saved_optimizer)
            reject+=1
        attempts.append({
            "block":block+1,"accepted":passed,
            "protected_old_TRAIN_full_EOS":len(guarded),
            "candidate_old_TRAIN_full_EOS":len(retained),
            "lost_protected_old_TRAIN_answer_ids":sorted(guarded-retained),
            "optimizer_and_weights_rolled_back":not passed,
            "actual_two_gradient_updates":local,
        })
        if (block+1)%4==0 or block==0:
            print(json.dumps({
                "event":"S32_MICROGUARD_FP32_TRAIN_PROGRESS",
                "block":block+1,
                "accepted":accept,"rejected":reject,
                "FP32_accepted_steps":effective,
                "old_train_candidate_correct":len(retained),
                "projection_conflicts":conflicts,
            },ensure_ascii=False),flush=True)
        del saved_weights,saved_optimizer
    after=s30.train_probe(model,tok,new,old)
    last=old_train_probe(model,tok,old)
    assert guarded.issubset(guarded_old_ids(last))
    final_sha=s30.snapshot_sha(model)
    report={
        "stage":"S32_MATCHED_36M_FP32_TWO_STEP_PCGRAD_SKILL_GUARD",
        "S26_CB59_source_sha256":SOURCE_SHA,
        "original_source_36M_parameters":SOURCE_PARAMS,
        "initial_paid_to_dense_max_abs_logit_diff":max_diff,
        "dense_FP32_not_compressed_bpp":True,
        "frozen_S29_SHA256_not_read":S29_SEALED_SHA,
        "S26_inspected_QA_not_read":True,
        "QA_never_used_for_optimization_or_selection":True,
        "TRAIN_only_outcomes_not_generalization":True,
        "step_attempts":STEPS,
        "microguard_block_steps":BLOCK_STEPS,
        "PCGrad_OLD_REPLAY_BLEND":OLD_GRAD_BLEND,
        "learning_rate":LR,
        "accepted_microblocks":accept,
        "rejected_and_reverted_microblocks":reject,
        "effective_accepted_full_FP32_gradient_updates":effective,
        "number_conflicting_gradients_projected":conflicts,
        "before_original_36M_FP32_SHA256":initial_sha,
        "after_original_36M_FP32_SHA256":final_sha,
        "old_protected_TRAIN_full_EOS_ids":sorted(guarded),
        "all_six_originally_correct_old_TRAIN_answers_preserved":True,
        "old_TRAIN_answers_before":previous,
        "old_TRAIN_answers_after":last,
        "new_and_old_TRAIN_probe_before":before,
        "new_and_old_TRAIN_probe_after":after,
        "every_train_step_and_rollback":attempts,
    }
    save(args.out/"ACTUAL_S32_36M_FP32_MICROGUARD_TRAIN.json",report)
    print(json.dumps({
        "stage":report["stage"],
        "accepted_blocks":accept,"rejected_blocks":reject,
        "new_train_before":before["per_kind"]["new_noun_TRAIN"],
        "new_train_after":after["per_kind"]["new_noun_TRAIN"],
        "old_train_before":before["per_kind"]["historical_TRAIN"],
        "old_train_after":after["per_kind"]["historical_TRAIN"],
        "same_weights":initial_sha==final_sha,
        "S29_QA_untouched":True},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
