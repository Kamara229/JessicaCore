"""CALIBRI2 S8: precommitted 192 NEW independent full-answer/EOS cases +16 open.

Do not import this module into TRAIN, inspect the benchmark to change steps,
tune learning rate or select a model. Run exactly once read-only on S7 baseline
and exactly once on S8 final 36M packed CB59 archive. QA output cannot prove
general language understanding or FP32 parity. Exact token sequences retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoTokenizer
from direct_codebook_v59 import CB59
from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.fresh_heldout import generate
from calibri2.s8.broad_replay import (
    SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC,make_broad_train
)
from v186.pairs import build_pairs
from calibri2.s3.diverse_task_replay import make_new_pairs as s3_pairs
from calibri2.s6.train_29m import new_pairs as s4_pairs
from v174.run import save

UNSEEN_ADJ=("pale","striped","mossy","winding","polished","shaded",
            "sparkling","dusty","curved","gentle","sleepy","dappled")
UNSEEN_NOUN=("fountain","orchid","ladder","seashell","pebble","canyon",
             "feather","sailboat","cupboard","lighthouse","pumpkin","coral")
FREE=(
 "Explain in one sentence why people need clean water.",
 "Why do trees need sunlight?",
 "What can a small flashlight help someone do?",
 "Explain why a bridge is useful.",
 "Write a short sentence about a gentle rain.",
 "Why does winter weather often feel cold?",
 "What does a lighthouse help people with?",
 "Why is it helpful to learn to read?",
 "Write a sentence about a yellow flower.",
 "How can a person keep safe on a slippery floor?",
 "Why do people sometimes use gloves?",
 "What is a telephone useful for?",
 "Explain how a bicycle helps a rider.",
 "Write two connected sentences about a book.",
 "What does a school library contain?",
 "What does a person need for a short walk?",
)

def suite():
    cases=[]
    def add(kind,i,q,a):
        cases.append({"id":f"{kind}_{i:03d}","domain":kind,
                      "question":q,"gold":a})
    for i in range(48):
        adj=UNSEEN_ADJ[i%12]
        noun=UNSEEN_NOUN[(i*5 + i//12)%12]
        phrase=adj+" "+noun
        add("two_word_generalization",i,
            f"A printed card for QAHeldout{i:03d} reads '{phrase}'. "
            "Write exactly the two words on the card, with no explanation.",phrase)
        phrase3=adj+" "+noun+" slowly"
        add("three_word_generalization",i,
            f"At a rehearsal, QAHeldout{i:03d} sees the phrase '{phrase3}'. "
            "Repeat all three words exactly.",phrase3)
        owner=f"KayEval{i:03d}"
        other=f"LeoEval{i:03d}"
        col="blue" if i%2 else "red"
        other_color="red" if i%2 else "blue"
        add("relation_grounding",i,
            f"In an imaginary shop, {owner} holds the {col} bag; "
            f"{other} holds the {other_color} bag. "
            f"Which bag color belongs to {owner}? One word.",col)
        x=7007+i*59
        y=x+11+(i%11)
        yes=i%2==0
        hi,lo=(y,x) if yes else (x,y)
        add("number_comparison",i,
            f"Compare the numbers {hi} and {lo}. "
            "Is the first number greater? Answer Yes or No.",
            "Yes" if yes else "No")
    if len(cases)!=192 or len({v["question"] for v in cases})!=192:
        raise AssertionError("Broken fixed 192 objective heldout")
    return cases

def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--label",required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=args.model.read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    cb=CB59(raw)
    if len(raw)!=SOURCE_BYTES or cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Only real physical 36M CB59 accepted")
    if 8*len(raw)/SOURCE_PARAMS>.700:
        raise AssertionError("Sealed 36M source exceeded 0.7 physical bpp")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    training=(build_pairs(tok,"TRAIN")+s3_pairs(tok)+s4_pairs(tok)+make_broad_train(tok))
    banned={p[side]["question"] for p in training for side in ("left","right")}
    cases=suite()
    if banned.intersection(x["question"] for x in cases):
        raise AssertionError("TRAIN and frozen QA share exact prompts")
    if len(set(FREE))!=16 or any(x in banned for x in FREE):
        raise AssertionError("Heldout free question in source TRAIN")
    suite_sha=hashlib.sha256(json.dumps([cases,FREE],sort_keys=True).encode()).hexdigest()
    cfg=build_config(MODEL_SPEC)
    model,sources=create_small_paid_model(cb,cfg)
    model.eval()
    rights=Counter();first=Counter();denom=Counter()
    rows=[]
    for i,x in enumerate(cases):
        target=tok.encode(x["gold"],add_special_tokens=False)+[tok.eos_token_id]
        if not 2<=len(target)<=14:
            raise AssertionError("Unscorable complete gold")
        observed=generate(model,sources,tok,x["question"],14)
        exact=observed==target
        firstok=bool(observed) and observed[0]==target[0]
        rights[x["domain"]]+=int(exact)
        first[x["domain"]]+=int(firstok)
        denom[x["domain"]]+=1
        rows.append({**x,"generated_ids":observed,
                     "actual_text":tok.decode(observed,skip_special_tokens=True),
                     "expected_ids":target,"strict_full_EOS":exact,
                     "first_token_correct":bool(firstok)})
        if (i+1)%48==0:
            print(json.dumps({"event":"CALIBRI2_S8_FRESH_PROGRESS","label":args.label,
                    "checked":i+1,"correct":sum(rights.values())}),flush=True)
    answers=[]
    for q in FREE:
        tokens=generate(model,sources,tok,q,26)
        answers.append({"question":q,"answer":tok.decode(tokens,skip_special_tokens=True),
                        "tokens":tokens,"stopped_at_EOS":tokens[-1]==tok.eos_token_id if tokens else False})
    result={"stage":"CALIBRI2_S8_FRESH_192_FULL_EOS_AND_16_OPEN",
            "model_label":args.label,"physical_SHA256":sha,
            "physical_bytes":len(raw),"physical_bpp":8*len(raw)/SOURCE_PARAMS,
            "suite_sha256":suite_sha,
            "objective_total":len(rows),"full_EOS_correct":sum(rights.values()),
            "first_token_correct":sum(first.values()),
            "domains":[{"name":d,"full_EOS_correct":rights[d],
                        "first_token_correct":first[d],"total":denom[d]}
                        for d in sorted(denom)],
            "all_actual_192_answers":rows,"all_actual_16_open_answers":answers,
            "QA_not_used_to_train_or_select_checkpoint":True,
            "no_teacher_or_original_FP32_weights":True,
            "not_comparable_to_original_135M_FP32_quality_yet":True}
    save(args.out/"ACTUAL_CALIBRI2_S8_FROZEN_192_AND_OPEN.json",result)
    print(json.dumps({k:v for k,v in result.items()
        if k not in ("all_actual_192_answers","all_actual_16_open_answers")}),flush=True)

if __name__=="__main__":
    main()
