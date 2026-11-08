"""CALIBRI2 S13: read-only broad TRAIN retention audit of S12 transaction guard.

S12 accepted 24/24 candidate updates because each update improved its
NEW TRAIN pair while not worsening one rotating OLD TRAIN guard. This
does not imply retention across a larger skill set. This audit tests 32
predeclared old TRAIN pairs (all 24 original + first 8 S3 binding) and 24
predeclared new TRAIN pairs across colors, numbers and phrase copying.
It compares exact full-answer+EOS greedy generations as well as
TRAIN-loss differences for real S7 and the physically sealed S12
checkpoint; no QA/VALID/TEST, no weight updates or checkpoint selection.

S13's result is TRAIN-retention evidence, not independent generalization,
not FP32 parity, not an S11 quality result or a proposed inference sidecar.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v186.pairs import build_pairs
from v200.rollout import probe_greedy_train
from calibri2.s3.diverse_task_replay import make_new_pairs as s3_pairs
from calibri2.s11.green_tunnel_protected import new_train_pairs
from calibri2.s8.broad_replay import (
    SOURCE_SHA,SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC,weighted_pair_loss
)
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from v174.run import save

SEALED_S12="f8666b6815a9f7cc966550260706dd2632c0863ce45c2581e7c0a44a7a2e084a"
OLD_TRAIN_PER_GROUP=(24,8)
NEW_TRAIN_PER_GROUP={"color":8,"numeric":8,"copy":8}
MAX_GREEDY=12


def load_model(path,sha):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha or len(raw)!=SOURCE_BYTES:
        raise AssertionError("S13 cannot audit a nonsealed CB59 input")
    cb=CB59(raw)
    if cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Incorrect architecture source")
    if 8*len(raw)/SOURCE_PARAMS>.7:
        raise AssertionError("Source bpp exceeds agreed physical cap")
    return create_small_paid_model(cb,build_config(MODEL_SPEC))


def frozen_training_subset(tok):
    old=build_pairs(tok,"TRAIN")
    s3=s3_pairs(tok)
    new=new_train_pairs(tok)
    if len(old)!=24 or len(s3)!=64:
        raise AssertionError("Old TRAIN source changed")
    pairs=old+s3[:OLD_TRAIN_PER_GROUP[1]]
    for domain,count in NEW_TRAIN_PER_GROUP.items():
        pairs.extend(new[domain][:count])
    if len(pairs)!=sum(OLD_TRAIN_PER_GROUP)+sum(NEW_TRAIN_PER_GROUP.values()):
        raise AssertionError("TRAIN retention probe selection drift")
    if len({p["pair_id"] for p in pairs})!=len(pairs):
        raise AssertionError("Cross-set TRAIN IDs are not unique")
    frozen=[{"pair_id":x["pair_id"],"domain":x["domain"],
        "question":[x[s]["question"] for s in ("left","right")],
        "gold":[x[s]["answer"] for s in ("left","right")]} for x in pairs]
    sha=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    return pairs,sha


def measure(path,sha,pairs):
    model,sources=load_model(path,sha)
    model.eval()
    # TRAIN CE is not greedy accuracy: record both, do not conflate.
    ce={}
    with torch.inference_mode():
        for pair in pairs:
            clear(sources)
            ce[pair["pair_id"]]=float(weighted_pair_loss(model,pair).detach())
    clear(sources)
    answers=probe_greedy_train(model,sources,pairs,max_new_tokens=MAX_GREEDY)
    rows={}
    counts=defaultdict(lambda:{"correct":0,"total":0,"first":0})
    for pair in pairs:
        for side in ("left","right"):
            found=answers[pair["pair_id"]]["sides"][side]
            generated=found["generated_tokens"]
            gold=found["gold_tokens_with_EOS"]
            exact=bool(found["exact_with_EOS"])
            first=bool(generated and gold and generated[0]==gold[0])
            label=f"{pair['pair_id']}::{side}"
            rows[label]={
                "pair_id":pair["pair_id"],"side":side,
                "domain":pair["domain"],
                "question":pair[side]["question"],
                "expected_answer":pair[side]["answer"],
                "expected_token_ids_with_EOS":gold,
                "actual_generated_ids":generated,
                "exact_full_answer_EOS":exact,
                "first_token_correct":first,
            }
            bucket=counts[pair["domain"]]
            bucket["correct"]+=int(exact)
            bucket["total"]+=1
            bucket["first"]+=int(first)
    return {"physical_archive_SHA256":sha,
            "per_domain":dict(counts),
            "all_112_actual_TRAIN_answers":rows,
            "per_pair_weighted_TRAIN_loss":ce}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("s7",type=Path)
    p.add_argument("s12",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    frozen,prompt_sha=frozen_training_subset(tok)
    before=measure(args.s7,SOURCE_SHA,frozen)
    after=measure(args.s12,SEALED_S12,frozen)
    losses=Counter()
    gains=Counter()
    unchanged=Counter()
    changed=[]
    for key,prev in before["all_112_actual_TRAIN_answers"].items():
        nxt=after["all_112_actual_TRAIN_answers"][key]
        if (prev["expected_token_ids_with_EOS"]!=nxt["expected_token_ids_with_EOS"]
           or prev["question"]!=nxt["question"]):
            raise AssertionError("Gold TRAIN questions changed between model evaluations")
        b=prev["exact_full_answer_EOS"]
        a=nxt["exact_full_answer_EOS"]
        domain=prev["domain"]
        if b and not a:losses[domain]+=1
        if not b and a:gains[domain]+=1
        if b==a:unchanged[domain]+=1
        if prev["actual_generated_ids"]!=nxt["actual_generated_ids"]:
            changed.append({"pair_side":key,"domain":domain,
               "before_answer_ids":prev["actual_generated_ids"],
               "after_answer_ids":nxt["actual_generated_ids"],
               "was_correct":b,"now_correct":a})
    before_count=sum(v["correct"] for v in before["per_domain"].values())
    after_count=sum(v["correct"] for v in after["per_domain"].values())
    if before_count - sum(losses.values())+sum(gains.values())!=after_count:
        raise AssertionError("Nonconservation of actual strict TRAIN correctness")
    diffs={}
    for p in frozen:
        k=p["pair_id"]
        old=before["per_pair_weighted_TRAIN_loss"][k]
        new=after["per_pair_weighted_TRAIN_loss"][k]
        diffs[k]={"domain":p["domain"],"S7_TRAIN_loss":old,
                  "S12_TRAIN_loss":new,"difference":new-old}
    result={
        "stage":"CALIBRI2_S13_READONLY_MULTI_GUARD_TRAIN_RETENTION_AFTER_S12_24",
        "source_S7_SHA256":SOURCE_SHA,"source_S12_SHA256":SEALED_S12,
        "physical_bytes":SOURCE_BYTES,"unique_original_params":SOURCE_PARAMS,
        "complete_physical_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
        "same_training_prompt_SHA256":prompt_sha,
        "tested_old_TRAIN_pairs":sum(OLD_TRAIN_PER_GROUP),
        "tested_new_TRAIN_pairs":sum(NEW_TRAIN_PER_GROUP.values()),
        "train_answers_before":before_count,
        "train_answers_after":after_count,
        "total_actual_TRAIN_answers":2*len(frozen),
        "previously_correct_lost_by_domain":dict(losses),
        "previously_wrong_gained_by_domain":dict(gains),
        "actual_changed_greedy_token_sequences":len(changed),
        "changed_answers":changed,
        "per_PAIR_actual_loss_differences":diffs,
        "before":before,"after":after,
        "all_evaluated_examples_are_TRAIN_only":True,
        "no_optimizer_or_physical_weight_mutations":True,
        "S12_guard_is_one_rotating_TRAIN_pair_not_all_pairs":True,
        "not_an_independent_QUALITY_or_FP32_test":True,
        "not_a_claim_of_protected_generalization":True
    }
    save(args.out/"ACTUAL_CALIBRI2_S13_ALL_112_TRAIN_ANSWER_RETENTION.json",result)
    print(json.dumps({"event":"S13_READONLY_MULTI_TRAIN_RETENTION_DONE",
        "S7_correct":before_count,"S12_correct":after_count,
        "total":2*len(frozen),"lost":sum(losses.values()),
        "gained":sum(gains.values()),
        "lost_by_domain":dict(losses),
        "gained_by_domain":dict(gains),
        "changed_actual_sequences":len(changed),
        "physical_bpp":result["complete_physical_bpp"]}),flush=True)


if __name__=="__main__":
    main()
