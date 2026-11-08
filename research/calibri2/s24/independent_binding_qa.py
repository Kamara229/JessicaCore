"""S24 independently frozen 96 new color+noun BINDING QA, not TRAIN.

Goal: after S22/S23, test whether 36M can pick a correct *noun* after
learning continuation and keep previous skills. 8 withheld COLOR+NEW-NOUN
combinations from 6 new objects, 12 question variants per combo.
The entire S24 TRAIN source is forbidden from using these 8 exact
color+noun combinations. Do not use these questions or their answers
for gradients, hyperparameter selection, stopping or checkpoint choice.
Unlike old S20 QA, these new lexical combinations were never evaluated
when the S24 training program was fixed. Responses are actual greedy
generated tokens, checked in full including EOS; first-slot-only credit
does NOT count as correct. No 135M FP32 comparison or teacher.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.fresh_heldout import generate
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from v174.run import save

COLORS=("red","blue","green","yellow")
NEW_NOUNS=("plate","scarf","clock","spoon","pencil","bottle")
NOUNS=("cup","book","bag","lamp","chair","hat")+NEW_NOUNS
FREE_QUESTIONS=(
 "In two words, describe a yellow plate that you saw.",
 "What two words describe the green scarf in the drawing?",
 "A blue clock is hanging on the wall. Describe it using its color and object.",
 "A red spoon lies near a green bottle. What is the spoon?",
 "A small child holds a yellow pencil. Describe the pencil in two words.",
 "At a picnic, there is a green bottle on a table. Name that item in two words.",
 "The clock in front of you is blue. What is it? Use two words.",
 "One person carries a red plate and another person carries a yellow plate. What is the first person's item?",
 "Explain why a spoon may be useful, in one short sentence.",
 "Describe what you can do with a pencil in one sentence.",
 "A glass falls on the floor. What could happen? Answer in a short sentence.",
 "What is 17 plus 5? Say the number.",
 "A scarf is red and a hat is green. Give the scarf's color and name as two words.",
 "A book can be read while a bottle holds water. Describe what each item does.",
 "Tell a short story about a lamp and a clock.",
 "Describe a plate and a spoon together in a simple sentence.",
)

def unseen():
    combinations=[(ci,oi) for ci in range(4) for oi in range(6,12)
                  if (ci+oi)%3==0]
    if len(combinations)!=8 or len(set(combinations))!=8:
        raise AssertionError("S24 expects 8 new vocabulary withheld combinations")
    return combinations

def cases():
    rows=[]
    for k,(ci,oi) in enumerate(unseen()):
        color=COLORS[ci];obj=NOUNS[oi]
        foil_color=COLORS[(ci+1+(k%2))%4]
        foil_noun=NOUNS[6+((oi-6+1)%6)]
        for variation in range(12):
            person=f"QaRami{k:02d}{variation:02d}"
            other=f"QaVeli{k:02d}{variation:02d}"
            if variation%4==0:
                question=(f"During a parade, {person} carries a {color} {obj} "
                  f"while {other} carries a {foil_color} {foil_noun}. "
                  f"Write the two-word color-and-object label for {person}.")
            elif variation%4==1:
                question=(f"An art exhibit description says that a {color} {obj} "
                  f"is next to {person}, while a {foil_color} {foil_noun} "
                  f"is next to {other}. Give the color and object beside {person}, two words.")
            elif variation%4==2:
                question=(f"An event roster pairs {person} with '{color} {obj}' "
                  f"and {other} with '{foil_color} {foil_noun}'. "
                  f"Return exactly two words identifying {person}'s assigned item.")
            else:
                question=(f"Look at the two entries. {other}: {foil_color} {foil_noun}. "
                  f"{person}: {color} {obj}. What is {person}'s item? "
                  f"Give its color and noun, in that order.")
            rows.append({"id":f"S24_NEW_{k:02d}_{variation:02d}",
              "domain":"new_heldout_color_noun",
              "question":question,"gold":f"{color} {obj}",
              "new_object_index":oi,"withheld_pair_index":k})
    for i in range(24):
        owner=f"QaLina{i:02d}"
        c=COLORS[(i*3+i//4)%4]
        other=COLORS[(i*3+i//4+1)%4]
        rows.append({"id":f"S24_COLOR_{i:02d}",
          "domain":"color_one_word_control",
          "question":f"On a stage, {owner} holds a {c} flag, "
                     f"and a second person holds a {other} flag. "
                     f"Say the color of {owner}'s flag, one word.",
          "gold":c})
    for i in range(24):
        b=7003+73*i;d=3+i%13
        yes=i%2==0
        x,y=(b+d,b) if yes else (b,b+d)
        rows.append({"id":f"S24_NUM_{i:02d}",
          "domain":"number_one_word_control",
          "question":f"A chart displays {x} as value A and {y} as value B. "
                     f"Is A greater than B? Reply Yes or No.",
          "gold":"Yes" if yes else "No"})
    if len(rows)!=144 or len({x["question"] for x in rows})!=144:
        raise AssertionError("Independent S24 QA has incomplete or duplicate questions")
    if len(FREE_QUESTIONS)!=16:raise AssertionError("Open questions missing")
    return rows

def score(rows):
    stats=defaultdict(Counter)
    predictions=Counter()
    expression=re.compile(r"^\s*(red|blue|green|yellow)\s+"
       r"(cup|book|bag|lamp|chair|hat|plate|scarf|clock|spoon|pencil|bottle)\s*[.!]?\s*$",re.I)
    for row in rows:
        d=stats[row["domain"]];d["n"]+=1
        d["strict_full_EOS"]+=int(row["strict_full_EOS"])
        d["correct_first_token"]+=int(row["correct_first_token"])
        d["EOS_emitted"]+=int(row["EOS_emitted"])
        if row["domain"]!="new_heldout_color_noun":continue
        m=expression.fullmatch(row["actual_text"])
        gold=row["gold"].split()
        first=bool(m and m.group(1).lower()==gold[0])
        noun=bool(m and m.group(2).lower()==gold[1])
        d["valid_two_word"]+=int(m is not None)
        d["correct_color_with_valid_format"]+=int(first)
        d["correct_noun_with_valid_format"]+=int(noun)
        d["both_semantic_words_correct"]+=int(first and noun)
        d["color_correct_noun_wrong"]+=int(first and not noun)
        d["noun_correct_color_wrong"]+=int(noun and not first)
        predictions[row["actual_text"].strip()]+=1
    return dict(stats),predictions.most_common(16)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--label",required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=a.archive.read_bytes()
    archive=CB59(raw)
    if len(raw)!=SOURCE_BYTES or archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S24 requires fixed physical 36M CB59")
    bpp=8*len(raw)/SOURCE_PARAMS
    if bpp>.7:raise AssertionError("S24 physically unpaid model")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases_now=cases()
    qa_sha=hashlib.sha256(json.dumps(
       [cases_now,FREE_QUESTIONS],sort_keys=True).encode()).hexdigest()
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    model.eval()
    real=[]
    for idx,case in enumerate(cases_now):
        gold=tok.encode(case["gold"],add_special_tokens=False)+[tok.eos_token_id]
        if len(gold)>14:raise AssertionError("S24 target too long for 14-token cap")
        predicted=generate(model,sources,tok,case["question"],14)
        real.append({
          **case,
          "expected_token_ids_full_EOS":gold,"actual_token_ids":predicted,
          "actual_text":tok.decode(predicted,skip_special_tokens=True),
          "strict_full_EOS":predicted==gold,
          "correct_first_token":bool(predicted) and predicted[0]==gold[0],
          "EOS_emitted":bool(predicted) and predicted[-1]==tok.eos_token_id
        })
        if (idx+1)%48==0:
            print(json.dumps({"event":"S24_FROZEN_QA_PROGRESS",
                "label":a.label,"checked":idx+1,
                "complete_correct":sum(v["strict_full_EOS"] for v in real)}),flush=True)
    opened=[]
    for question in FREE_QUESTIONS:
        generated=generate(model,sources,tok,question,32)
        opened.append({"question":question,
          "actual_answer":tok.decode(generated,skip_special_tokens=True),
          "actual_tokens":generated,
          "EOS":bool(generated) and generated[-1]==tok.eos_token_id,
          "requires_human_coherence_review":True})
    per_domain,common=score(real)
    result={
      "stage":"S24_PRECOMMITTED_NOVEL_96_NOUN_BINDINGS_48_CONTROLS_16_OPEN",
      "label":a.label,
      "physical_archive_SHA256":hashlib.sha256(raw).hexdigest(),
      "complete_bytes":len(raw),"physical_bpp":bpp,
      "unique_params":SOURCE_PARAMS,"frozen_suite_sha256":qa_sha,
      "withheld_new_object_combinations":unseen(),
      "no_QA_used_for_training_or_model_selection":True,
      "strict_full_EOS_only_claims_correct_complete_answer":True,
      "total_cases":len(real),
      "total_correct_full_EOS":sum(x["strict_full_EOS"] for x in real),
      "per_domain":per_domain,"common_real_answers":common,
      "every_144_actual_answer_and_gold":real,
      "every_16_open_actual_answer":opened,
      "no_original_135m_FP32_teacher":True,
    }
    save(a.out/"ACTUAL_CALIBRI2_S24_NOVEL_144_16_ANSWERS.json",result)
    print(json.dumps({key:value for key,value in result.items()
       if key not in ("every_144_actual_answer_and_gold",
                      "every_16_open_actual_answer","common_real_answers")}),flush=True)

if __name__=="__main__":
    main()
