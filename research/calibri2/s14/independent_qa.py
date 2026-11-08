"""CALIBRI2 S14 independent, *precommitted*, 192+16 real-answer benchmark.

Use once on the immutable S7 source and once on a single sealed S14 model,
with no QA-informed training, checkpoints or hyperparameter changes. This
suite does NOT reuse prompts from previous S8 or S9 QA. Its instruction
families/words differ from S14 TRAIN; fully novel long compositional cases
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

TEST_ADJ=("glittering","tattered","ceramic","whispering","brushed","spotted","woven",
          "cloudy","cobalt","bronze","crinkled","breezy")
TEST_NOUN=("hourglass","kettle","binoculars","mural","pyramid","drum",
           "thermos","trophy","ladder","windsock","magnifier","carousel")
COLORS=("red","blue","green","yellow")
FREE=(
 "How could a thermos keep a drink useful on a journey?",
 "Explain why a person could choose to wear a helmet.",
 "Describe a colorful poster in a short sentence.",
 "What might be useful about a magnifying glass?",
 "Write two brief sentences about a quiet railway station.",
 "Why does a bicycle wheel need to turn?",
 "How could a person protect a book from rain?",
 "What can a gardener do during a dry summer?",
 "Why is a clock sometimes useful?",
 "Describe a trip to a museum in one sentence.",
 "How does a person use binoculars?",
 "What can a firefighter do to help people?",
 "Explain how a painter chooses a color.",
 "Tell a simple story about a little house and a tree.",
 "What is an hourglass for?",
 "Describe what might happen when a bell rings.",
)

def cases():
    rows=[]
    def append(category,i,q,gold):
        rows.append({"id":f"S14{category}{i:03d}",
                     "domain":category,"question":q,"gold":gold})
    for i in range(64):
        c=COLORS[(i*3+i//8)%4]
        opposite=COLORS[(i*3+i//8+1)%4]
        a=f"EvalNiko{i:03d}"
        b=f"EvalLina{i:03d}"
        # Never a direct substring of the known S8 heldout, nor of S14 TRAIN.
        append("color_rebinding",i,
          f"During a parade, {a} carried a {c} banner while {b} carried a "
          f"{opposite} banner. Give the banner color carried by {a}, one word.",
          c)
    for i in range(48):
        x=68041+113*i
        d=2+i%31
        first_is_bigger= i%2==0
        a,b=(x+d,x) if first_is_bigger else (x,x+d)
        append("unseen_numeric",i,
          f"Two measurements are written down: {a} followed by {b}. "
          "Is the initial measurement larger than the second one? Reply Yes or No.",
          "Yes" if first_is_bigger else "No")
    for i in range(48):
        phrase=TEST_ADJ[(i*5)%12]+" "+TEST_NOUN[(i*7+i//12)%12]
        append("unseen_two_word",i,
          f"An archivist finds the inscription '{phrase}' on an old plaque. "
          "Write just the inscription's two words exactly.",
          phrase)
    for i in range(32):
        phrase=TEST_ADJ[(i*7)%12]+" "+TEST_NOUN[(i*3+i//12)%12]+" softly"
        append("unseen_three_word",i,
          f"Here is a three-word note: '{phrase}'. "
          "Transcribe that whole note without changing any word.",
          phrase)
    if len(rows)!=192 or len({x["question"] for x in rows})!=192:
        raise AssertionError("S14 independent set has missing or duplicate cases")
    if len(FREE)!=16 or len(set(FREE))!=16:
        raise AssertionError("S14 independent free-generation test is broken")
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
        raise AssertionError("S14 QA requires a sealed 36M physical 5bit file")
    bpp=8*len(blob)/SOURCE_PARAMS
    if bpp>.700:
        raise AssertionError("S14 physically unpaid weight archive")
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
            raise AssertionError("S14 answer does not fit strict max_new_tokens")
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
            print(json.dumps({"event":"S14_INDEPENDENT_PROGRESS","label":a.label,
                              "tested":n+1,"correct":sum(domain.values())}),flush=True)
    open_results=[]
    for q in FREE:
        generated=generate(model,sources,tok,q,30)
        open_results.append({"question":q,
              "tokens":generated,"full_text":tok.decode(generated,skip_special_tokens=True),
              "EOS":bool(generated) and generated[-1]==tok.eos_token_id})
    report={
      "stage":"S14_INDEPENDENT_192_STRICT_EOS_16_OPEN",
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
      "different_frozen_suite_from_S8_and_S11":True,
    }
    save(a.out/"ACTUAL_CALIBRI2_S14_FROZEN_192_AND_16_OPEN.json",report)
    print(json.dumps({k:v for k,v in report.items()
      if k not in ("real_192_full_greedy_18_token_answers",
                   "real_16_freeform_30_token_answers")}),flush=True)


if __name__=="__main__":
    main()
