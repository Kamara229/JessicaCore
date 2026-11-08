"""CALIBRI2 S20: new, precommitted, independent MULTIWORD benchmark.

The owner wants model-size-appropriate coherent answers, NOT hard FP32
135M vs 36M parity nor yes/no guessing. Main metric: 96 unseen 2-word
color-object BINDINGS (exact FULL greedy answer + EOS), with same known
four color and six object words but 8 entirely unseen color/object
combinations. Then 24 simple color and 24 numeric controls and 16 open
prompts (open responses are not automatically declared semantically
correct). The new QA question forms and people are absent from TRAIN.
No scorer or QA suite imported into S20 TRAIN, no QA checkpoint selection.

2-word exact metrics require correctness of BOTH generated words and EOS,
and record first/second lexical token and EOS separately for diagnosis.
Words may span several tokenizer IDs; use gold-token prefix length rather
than assuming one word equals one tokenizer ID. Also inspect specific
wrong answers to establish whether output stops after one word.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

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
OBJECTS=("cup","book","bag","lamp","chair","hat")
MULTI_TOTAL=96
CONTROLS_TOTAL=48
FREE_PROMPTS=(
 "What two words describe a red cup?",
 "In one short sentence, describe a blue book.",
 "Which is bigger, 20 or 15? Give a short answer.",
 "Write a short phrase containing the word green.",
 "A yellow chair is on the floor. Describe it.",
 "Give a two-word description of a red hat.",
 "A blue lamp is on the desk. Describe it in two words.",
 "What is a green bag? Reply with a short phrase.",
 "Repeat exactly: yellow book.",
 "Write two connected words about a cup.",
 "Is 12 larger than 24? Reply and give a reason.",
 "Describe one thing a lamp can do.",
 "What does a book contain? Use a short sentence.",
 "A person carries a green cup. Describe the item.",
 "Make a sentence containing 'blue' and 'hat'.",
 "A red chair is beside a blue book. Name the chair in two words.",
)


def unseen_combinations():
    result=[(ci,oi) for ci in range(len(COLORS))
            for oi in range(len(OBJECTS)) if (ci+oi)%3==0]
    if len(result)!=8 or len(set(result))!=8:
        raise AssertionError("Exactly 8 precommitted, withheld two-word combinations")
    return result


def cases():
    rows=[]
    held=unseen_combinations()
    for n,(ci,oi) in enumerate(held):
        gold=f"{COLORS[ci]} {OBJECTS[oi]}"
        contrast_col=COLORS[(ci+1)%4]
        contrast_obj=OBJECTS[(oi+1)%6]
        for variant in range(12):
            person=f"EvalNara{n:02d}{variant:02d}"
            another=f"EvalSumi{n:02d}{variant:02d}"
            if variant%4==0:
                prompt=(f"At a village fair, {person} carries the {gold}, "
                        f"while {another} carries the {contrast_col} "
                        f"{contrast_obj}. What are the two words describing "
                        f"{person}'s item? Use just those two words.")
            elif variant%4==1:
                prompt=(f"At a theater, a {gold} is given to {person} and "
                        f"a {contrast_col} {contrast_obj} is given to "
                        f"{another}. Identify {person}'s object using "
                        f"its color and noun, two words only.")
            elif variant%4==2:
                prompt=(f"An exhibition label associates {person} with a "
                        f"{gold}; another label associates {another} with "
                        f"a {contrast_col} {contrast_obj}. Reply with the "
                        f"two-word color-and-object description for {person}.")
            else:
                prompt=(f"Two packages are marked: {person} - {gold}, "
                        f"{another} - {contrast_col} {contrast_obj}. "
                        f"What exact two-word description is next to "
                        f"{person}'s name?")
            rows.append({"id":f"S20_UNSEEN_{n:02d}_{variant:02d}",
              "kind":"unseen_color_object_binding",
              "question":prompt,"gold":gold,
              "withheld_color":COLORS[ci],"withheld_object":OBJECTS[oi],
              "combo_index":n})
    for i in range(24):
        ci=(i*3+i//4)%len(COLORS)
        color=COLORS[ci]
        other=COLORS[(ci+1)%len(COLORS)]
        owner=f"EvalLeo{i:02d}"
        rows.append({
          "id":f"S20_COLOR_CTRL_{i:02d}",
          "kind":"color_single_word_control",
          "question":f"On the stage {owner} holds a {color} candle and a "
                     f"friend holds a {other} candle. What color is "
                     f"{owner}'s candle? Reply with one color word.",
          "gold":color})
    for i in range(24):
        x=2231+83*i
        delta=5+i%17
        yes=i%2==0
        left,right=(x+delta,x) if yes else (x,x+delta)
        rows.append({
          "id":f"S20_NUMBER_CTRL_{i:02d}",
          "kind":"numeric_single_word_control",
          "question":f"Exercise asks whether {left} is larger than "
                     f"{right}. Reply Yes or No.",
          "gold":"Yes" if yes else "No"})
    if len(rows)!=MULTI_TOTAL+CONTROLS_TOTAL or len(set(x["question"] for x in rows))!=len(rows):
        raise AssertionError("S20 QA coverage changed")
    if sum(x["kind"]=="unseen_color_object_binding" for x in rows)!=MULTI_TOTAL:
        raise AssertionError("No 96 genuinely withheld two-word combinations")
    if len(FREE_PROMPTS)!=16:
        raise AssertionError("Missing 16 actual open generation cases")
    return rows


def metrics(rows):
    groups=defaultdict(lambda:{
        "n":0,"full_plus_EOS":0,"first_gold_token":0,
        "all_answer_tokens_before_EOS":0,
        "EOS_correct_after_all_words":0,
        "stopped_too_early":0,
    })
    errors=Counter()
    for case in rows:
        d=groups[case["kind"]]
        d["n"]+=1
        for k in ("full_plus_EOS","first_gold_token","all_answer_tokens_before_EOS",
                  "EOS_correct_after_all_words","stopped_too_early"):
            d[k]+=int(case[k])
        if not case["full_plus_EOS"]:
            errors[case["actual_text"]]+=1
    return dict(groups),errors.most_common(20)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--label",required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=args.archive.read_bytes()
    archive=CB59(raw)
    if len(raw)!=SOURCE_BYTES or archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Exact physical 36M CB59 format required")
    bpp=8*len(raw)/SOURCE_PARAMS
    if bpp>.700:
        raise AssertionError("S20 physical bit budget drift")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    suite=cases()
    suite_sha=hashlib.sha256(json.dumps([suite,FREE_PROMPTS],
                          sort_keys=True).encode()).hexdigest()
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    model.eval()
    outputs=[]
    for i,case in enumerate(suite):
        gold=tokenizer.encode(case["gold"],add_special_tokens=False)
        expected=gold+[tokenizer.eos_token_id]
        if not 2<=len(expected)<=14:
            raise AssertionError("Gold answer is longer than S20 max scoring cap")
        pred=generate(model,sources,tokenizer,case["question"],14)
        full=pred==expected
        first=bool(pred and pred[0]==gold[0])
        all_tokens=len(pred)>=len(gold) and pred[:len(gold)]==gold
        correct_eos=all_tokens and len(pred)>len(gold) and pred[len(gold)]==tokenizer.eos_token_id
        early=tokenizer.eos_token_id in pred[:len(gold)]
        outputs.append({
            **case,"expected_token_ids_plus_EOS":expected,
            "actual_token_ids":pred,
            "actual_text":tokenizer.decode(pred,skip_special_tokens=True),
            "full_plus_EOS":full,
            "first_gold_token":first,
            "all_answer_tokens_before_EOS":all_tokens,
            "EOS_correct_after_all_words":correct_eos,
            "stopped_too_early":early,
            "EOS_emitted":bool(pred) and pred[-1]==tokenizer.eos_token_id
        })
        if (i+1)%48==0:
            done=sum(z["full_plus_EOS"] for z in outputs)
            print(json.dumps({"event":"S20_FROZEN_144_PROGRESS",
                "label":args.label,"checked":i+1,"correct":done}),flush=True)
    opened=[]
    for q in FREE_PROMPTS:
        a=generate(model,sources,tokenizer,q,26)
        opened.append({
           "question":q,"actual_text":tokenizer.decode(a,skip_special_tokens=True),
           "actual_token_ids":a,
           "EOS":bool(a) and a[-1]==tokenizer.eos_token_id,
           "manual_coherence_review_required":True,
        })
    scored,common=metrics(outputs)
    report={
        "stage":"S20_INDEPENDENT_96_NEW_TWO_WORD_48_SIMPLE_CONTROL_16_OPEN",
        "model_label":args.label,"model_sha256":hashlib.sha256(raw).hexdigest(),
        "unique_parameters":SOURCE_PARAMS,"complete_bytes":len(raw),
        "physical_bpp":bpp,
        "precommitted_suite_sha256":suite_sha,
        "physical_new_combinations_held_out_in_training":8,
        "two_word_cases":MULTI_TOTAL,
        "single_word_controls":CONTROLS_TOTAL,
        "total_cases":len(outputs),
        "total_full_EOS_correct":sum(z["full_plus_EOS"] for z in outputs),
        "per_domain":scored,"most_common_mistaken_answers":common,
        "actual_144_complete_greedy_answers":outputs,
        "actual_16_open_answers":opened,
        "first_gold_token_is_NOT_sufficient_for_quality":True,
        "strict_answer_must_contain_BOTH_words_and_EOS":True,
        "no_QA_used_for_training_or_checkpoint_selection":True,
        "no_FP32_teacher_OR_baseline_required":True,
        "not_a_general_language_intelligence_proof":True,
    }
    save(args.out/"ACTUAL_CALIBRI2_S20_96_NEW_MULTIWORD_48_CONTROL_16_OPEN.json",report)
    print(json.dumps({k:v for k,v in report.items()
      if k not in ("actual_144_complete_greedy_answers",
                   "actual_16_open_answers","most_common_mistaken_answers")}),flush=True)


if __name__=="__main__":
    main()
