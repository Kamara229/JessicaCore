"""CALIBRI2 S11 independent, *precommitted*, 192+16 real-answer benchmark.

Use once on the immutable S7 source and once on a single sealed S11 model,
with no QA-informed training, checkpoints or hyperparameter changes. This
suite does NOT reuse prompts from previous S8 or S9 QA. Its instruction
families/words differ from S11 TRAIN; fully novel long compositional cases
are included. Both models must pass physical CB59 budget, 36M schema, SHA.
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
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC
from calibri2.s11.green_tunnel_protected import new_train_pairs
from calibri2.s3.diverse_task_replay import make_new_pairs as s3_pairs
from calibri2.s6.train_29m import new_pairs as s4_pairs
from v186.pairs import build_pairs
from v174.run import save

TEST_ADJ=("rusty","woven","shimmering","ancient","marble","velvet","round",
          "hollow","crimson","azure","starlit","twisted")
TEST_NOUN=("compass","teapot","camera","scarf","ladle","bookcase",
           "suitcase","windmill","rocket","backpack","microscope","umbrella")
COLORS=("red","blue","green","yellow")
FREE=(
 "Tell me why a compass may be useful during a hike.",
 "Explain why people use an umbrella in the rain.",
 "Write a short description of a bridge.",
 "Why should one take care of a garden?",
 "What is a telescope used for?",
 "Write a short sentence about a peaceful lake.",
 "How can people share useful information?",
 "Why might a windmill move when wind blows?",
 "Why should children learn how to read?",
 "Describe a short trip to a nearby village.",
 "How does a person benefit from a warm coat?",
 "Why can a map help someone find a place?",
 "What does it mean to compare two numbers?",
 "Write two sentences about a small sailboat.",
 "Describe the difference between a book and a notebook.",
 "What might a person use a flashlight for?",
)

def cases():
    rows=[]
    def append(category,i,q,gold):
        rows.append({"id":f"S11{category}{i:03d}",
                     "domain":category,"question":q,"gold":gold})
    for i in range(64):
        c=COLORS[(i*3+i//8)%4]
        opposite=COLORS[(i*3+i//8+1)%4]
        a=f"EvalNiko{i:03d}"
        b=f"EvalLina{i:03d}"
        # Never a direct substring of the known S8 heldout, nor of S11 TRAIN.
        append("color_rebinding",i,
          f"At an exhibit, {a} chose the {c} painting and {b} chose the "
          f"{opposite} painting. Tell me the painting color chosen by {a}. One word.",
          c)
    for i in range(48):
        x=35017+127*i
        d=2+i%31
        first_is_bigger= i%2==0
        a,b=(x+d,x) if first_is_bigger else (x,x+d)
        append("unseen_numeric",i,
          f"Two values appear in a chart: {a}, then {b}. "
          f"Does the earlier value exceed the later value? Reply Yes or No.",
          "Yes" if first_is_bigger else "No")
    for i in range(48):
        phrase=TEST_ADJ[(i*5)%12]+" "+TEST_NOUN[(i*7+i//12)%12]
        append("unseen_two_word",i,
          f"On the festival sign, the words are '{phrase}'. "
          "Reproduce that entire label, in order, as your answer.",
          phrase)
    for i in range(32):
        phrase=TEST_ADJ[(i*7)%12]+" "+TEST_NOUN[(i*3+i//12)%12]+" softly"
        append("unseen_three_word",i,
          f"These words occur on a sign: '{phrase}'. "
          "Please type the three-word inscription exactly.",
          phrase)
    if len(rows)!=192 or len({x["question"] for x in rows})!=192:
        raise AssertionError("S11 independent set has missing or duplicate cases")
    if len(FREE)!=16 or len(set(FREE))!=16:
        raise AssertionError("S11 independent free-generation test is broken")
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--label",required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    blob=a.archive.read_bytes()
    cb=CB59(blob)
    if len(blob)!=SOURCE_BYTES or cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S11 QA requires a sealed 36M physical 5bit file")
    bpp=8*len(blob)/SOURCE_PARAMS
    if bpp>.700:
        raise AssertionError("S11 physically unpaid weight archive")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    historical=build_pairs(tok,"TRAIN")+s3_pairs(tok)+s4_pairs(tok)
    new=new_train_pairs(tok)
    all_training=historical+[x for group in new.values() for x in group]
    banned={p[s]["question"] for p in all_training for s in ("left","right")}
    heldout=cases()
    if any(x["question"] in banned for x in heldout) or any(x in banned for x in FREE):
        raise AssertionError("TRAIN/QA exact prompt collision")
    suite_sha=hashlib.sha256(json.dumps([heldout,FREE],sort_keys=True).encode()).hexdigest()
    model,sources=create_small_paid_model(cb,build_config(MODEL_SPEC))
    rows=[]
    domain=Counter(); denominators=Counter(); first=Counter()
    for n,case in enumerate(heldout):
        gold=tok.encode(case["gold"],add_special_tokens=False)+[tok.eos_token_id]
        if len(gold)>18:
            raise AssertionError("S11 answer does not fit strict max_new_tokens")
        guess=generate(model,sources,tok,case["question"],18)
        correct=guess==gold
        same_first=bool(guess) and guess[0]==gold[0]
        rows.append({**case,"generated_ids":guess,
                     "decoded_full_answer":tok.decode(guess,skip_special_tokens=True),
                     "gold_ids":gold,"full_EOS_correct":correct,
                     "first_token_correct":same_first})
        domain[case["domain"]]+=int(correct)
        denominators[case["domain"]]+=1
        first[case["domain"]]+=int(same_first)
        if (n+1)%48==0:
            print(json.dumps({"event":"S11_INDEPENDENT_PROGRESS","label":a.label,
                              "tested":n+1,"correct":sum(domain.values())}),flush=True)
    open_results=[]
    for q in FREE:
        generated=generate(model,sources,tok,q,30)
        open_results.append({"question":q,
              "tokens":generated,"full_text":tok.decode(generated,skip_special_tokens=True),
              "EOS":bool(generated) and generated[-1]==tok.eos_token_id})
    report={
      "stage":"S11_INDEPENDENT_192_STRICT_EOS_16_OPEN",
      "label":a.label,"model_sha256":hashlib.sha256(blob).hexdigest(),
      "suite_sha256":suite_sha,
      "unique_params":SOURCE_PARAMS,
      "all_physical_bytes":len(blob),"physical_bpp":bpp,
      "strict_total_correct":sum(domain.values()),
      "first_token_correct":sum(first.values()),
      "total":len(rows),
      "per_domain":[{"domain":k,"strict":domain[k],"first":first[k],
                     "total":denominators[k]} for k in sorted(denominators)],
      "real_192_full_greedy_18_token_answers":rows,
      "real_16_freeform_30_token_answers":open_results,
      "no_QA_in_TRAIN_or_checkpoint_selection":True,
      "not_a_FP32_parity_test":True,
      "new_S11_suite_not_comparable_in_denominator_to_previous_S8_suite":True,
    }
    save(a.out/"ACTUAL_CALIBRI2_S11_FROZEN_192_AND_16_OPEN.json",report)
    print(json.dumps({k:v for k,v in report.items()
      if k not in ("real_192_full_greedy_18_token_answers",
                   "real_16_freeform_30_token_answers")}),flush=True)


if __name__=="__main__":
    main()
