"""CALIBRI2 S15: user-requested high-quality, multi-attribute answer evaluation.

The previously used 192-question benchmarks were dominated by Yes/No
or single color extraction: responding "Yes" repeatedly could falsely
look like model progress. This predeclared suite requires *both* a real
binary judgment and a correctly bound color within the same generated
response ("Yes, blue." or "No, red."). The two fields vary orthogonally:
192 cases = 4 reasoning domains * 24 independently constructed situations
* a counterfactual Yes/No twin. There are precisely 24 instances of each
of the 8 combinations (Yes/No)*(red/blue/green/yellow); the always-Yes
and always-a-color shortcut gets at most 24/192 = 12.5%. A whole
counterfactual pair can score only if both answers are right, with the
color consistent when the binary proposition flips.

Each question is self-contained and has a verified deterministic gold.
All test prompts are disjoint from fixed TRAIN and previous S8/S14 QA.
Record STRICT complete token sequence+EOS and separately both semantic
slots allowing punctuation variation. Store every actual generated
response, full refusal/extra-token behavior, counterfactual consistency,
and 16 real 2-clause open compositions. Never use this suite for training,
early stop, choosing indices, selecting final compressed checkpoint or
model architecture, or statistical claims about unseen domains.
No original pretrained FP32 teacher/weights, no LIVE code changes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer
from direct_codebook_v59 import CB59
from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.fresh_heldout import generate
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from v174.run import save

COLORS=("red","blue","green","yellow")
PAIRS_PER_DOMAIN=24
DOMAIN_NAMES=("number_plus_color","ownership_plus_color",
              "count_plus_color","transfer_plus_color")
ANSWER_CAP=26
OPEN_CAP=44
WORDS1=("Meren","Liora","Calen","Viona","Tovin","Arden")
WORDS2=("Sella","Brina","Yarin","Pelen","Oriva","Narin")
OPEN_QUALITY=(
 "Describe why a boat might need a map, and give one example of how to use it. Use two short sentences.",
 "If a library has no lights at night, what is one problem and one practical solution? Answer in two sentences.",
 "Explain how a bicycle wheel helps a rider and name one safety precaution.",
 "Give one reason to keep drinking water clean and one thing that can contaminate it.",
 "A cup is red and a bowl is blue. Explain which is blue and whether the cup is blue.",
 "A child has two books and receives three more. State the total and explain the addition.",
 "Describe what happens when ice warms and why it changes.",
 "State whether five is larger than eight, then explain the comparison using one sentence.",
 "A gardener plants a seed and then waters it. Describe the sequence and why watering is useful.",
 "Write two connected sentences about a bus arriving at a stop and passengers boarding.",
 "A person gave a green umbrella to a friend. Who holds the umbrella now, and what color is it?",
 "Explain the difference between a thermometer and a clock, giving the purpose of each.",
 "Two baskets contain four and seven apples. Identify the larger basket and the difference in count.",
 "A street is flooded and someone needs to cross. Describe one safer choice and why it helps.",
 "A dog is under a table and a cat is on a chair. State where each animal is, in a sentence.",
 "Write a short explanation connecting rain, rivers, and the water cycle.",
)


def make_cases():
    result=[]
    def add(domain,index,flip,question,answer_flag,color):
        identifier=f"S15_{domain}_{index:02d}_{flip}"
        expected=f"{'Yes' if answer_flag else 'No'}, {color}"
        result.append({"id":identifier,"twin_id":f"S15_{domain}_{index:02d}",
           "domain":domain,"twin_variant":flip,"question":question,
           "expected_flag":"Yes" if answer_flag else "No",
           "expected_color":color,"gold":expected})
    for domain in DOMAIN_NAMES:
        for i in range(PAIRS_PER_DOMAIN):
            # Every category has 6 situations per color. Every one of the
            # 8 combinations occurs 6 times per task = 24 across tasks.
            color=COLORS[i%4]
            wrong=COLORS[(i+1+(i//4)%2)%4]
            if color==wrong:
                raise AssertionError("Invalid counterfactual color foil")
            a=f"{WORDS1[i%6]}{i:03d}"
            b=f"{WORDS2[(i*3+i//6)%6]}{i:03d}"
            num=4097+i*43
            difference=5+(i%13)
            for flip in (0,1):
                yes=(flip==0)
                if domain=="number_plus_color":
                    hi,lo=(num+difference,num) if yes else (num,num+difference)
                    # Ask two unrelated but simultaneous checks so copying
                    # the literal color or answering only Yes cannot pass.
                    text=(f"In the field log, {a} owns a {color} backpack while "
                          f"{b} owns a {wrong} backpack. The two readings are "
                          f"{hi} followed by {lo}. Is the first reading larger? "
                          f"Also state {a}'s backpack color. Answer in exactly "
                          f"this form: Yes, color or No, color.")
                elif domain=="ownership_plus_color":
                    test_color=color if yes else wrong
                    text=(f"A record states {a} carries the {color} lantern; "
                          f"{b} carries the {wrong} lantern. Does {a} carry "
                          f"the {test_color} lantern? Also give the lantern "
                          f"color that belongs to {a}. Respond with two parts: "
                          f"Yes, color or No, color.")
                elif domain=="count_plus_color":
                    x=14+(i%17)
                    y=x+(i%7)+1
                    count_a,count_b=(y,x) if yes else (x,y)
                    text=(f"At the stall, {a} has {count_a} tickets and {b} "
                          f"has {count_b} tickets. {b} keeps a {color} pouch, "
                          f"while {a} keeps a {wrong} pouch. Does {a} have "
                          f"more tickets than {b}? Then provide {b}'s pouch "
                          f"color. Format: Yes, color or No, color.")
                else:
                    # One explicit handoff between two participants. Flip
                    # only the subject of the possession question.
                    asked=b if yes else a
                    text=(f"Initially {a} holds a {color} badge. Then {a} "
                          f"hands that same badge to {b}, and nobody else "
                          f"touches it. After the handoff, does {asked} have "
                          f"the badge? Give its color too. Write only "
                          f"Yes, color or No, color.")
                add(domain,i,flip,text,yes,color)
    if len(result)!=192 or len({x["id"] for x in result})!=192:
        raise AssertionError("S15 192 composite test rows missing")
    if len(set(x["question"] for x in result))!=192:
        raise AssertionError("S15 composite test prompts duplicated")
    classes=Counter(x["gold"] for x in result)
    for flag in ("Yes","No"):
        for color in COLORS:
            if classes[f"{flag}, {color}"]!=24:
                raise AssertionError("S15 outcome shortcuts not exactly balanced")
    groups=defaultdict(list)
    for r in result:
        groups[r["twin_id"]].append(r)
    if len(groups)!=96:
        raise AssertionError("Wrong number of counterfactual pairs")
    for case_id,rows in groups.items():
        if len(rows)!=2 or rows[0]["expected_flag"]==rows[1]["expected_flag"]:
            raise AssertionError("Twinning did not invert judgment "+case_id)
        if rows[0]["expected_color"]!=rows[1]["expected_color"]:
            raise AssertionError("Twinning mutated the second answer label "+case_id)
    return result


def parse_two_slots(response):
    # Normalize only formatting of exactly two simple words. No partial
    # substring credit, no hallucinated filler or extra clause credit.
    cleaned=response.strip()
    matched=re.fullmatch(
        r"\s*(Yes|No)\s*[,;:\-]?\s*(red|blue|green|yellow)\s*[.!]?\s*",
        cleaned,re.IGNORECASE)
    if matched:
        return matched.group(1).capitalize(),matched.group(2).lower()
    # Diagnostic only: find first Yes/No and color even with extraneous
    # text. These are explicitly NOT credited as a valid complete answer.
    flags=re.findall(r"\b(Yes|No)\b",cleaned,re.IGNORECASE)
    colors=re.findall(r"\b(red|blue|green|yellow)\b",cleaned,re.IGNORECASE)
    if len(flags)==1 or len(colors)==1:
        return (flags[0].capitalize() if len(flags)==1 else None,
                colors[0].lower() if len(colors)==1 else None)
    return None,None


def compare_rows(rows):
    domains=defaultdict(lambda:{"count":0,"strict_full_EOS":0,
                                 "both_slots_valid_and_correct":0,
                                 "flag_correct":0,"color_correct":0})
    pairs=defaultdict(list)
    confusion=Counter()
    invalid=Counter()
    for case in rows:
        d=domains[case["domain"]]
        d["count"]+=1
        for k in ("strict_full_EOS","both_slots_valid_and_correct",
                  "flag_correct","color_correct"):
            d[k]+=int(case[k])
        confusion[f"{case['gold']} -> {case['actual_text']}"]+=1
        if not case["valid_two_slot_format"]:
            invalid[case["domain"]]+=1
        pairs[case["twin_id"]].append(case)
    paired_full=paired_semantic=0
    pair_details=[]
    for pair,twins in sorted(pairs.items()):
        if len(twins)!=2:
            raise AssertionError("Missing counterfactual twin")
        t=sorted(twins,key=lambda x:x["twin_variant"])
        both_strict=all(z["strict_full_EOS"] for z in t)
        both_semantic=all(z["both_slots_valid_and_correct"] for z in t)
        paired_full+=int(both_strict)
        paired_semantic+=int(both_semantic)
        pair_details.append({
          "twin_id":pair,"domain":t[0]["domain"],
          "both_strict_correct":both_strict,
          "both_semantic_correct":both_semantic,
          "gold_first":t[0]["gold"],"gold_flipped":t[1]["gold"],
          "actual_first":t[0]["actual_text"],
          "actual_flipped":t[1]["actual_text"],
        })
    allcorrect=sum(x["strict_full_EOS"] for x in rows)
    semantic=sum(x["both_slots_valid_and_correct"] for x in rows)
    return {"strict_full_plus_EOS":allcorrect,
            "semantic_two_slots_both_correct":semantic,
            "strict_twin_pairs_both_correct":paired_full,
            "semantic_twin_pairs_both_correct":paired_semantic,
            "twin_pairs_total":96,
            "simple_shortcut_always_Yes_and_same_color_max":24,
            "simple_shortcut_fraction":.125,
            "per_domain":dict(domains),
            "invalid_format_by_domain":dict(invalid),
            "most_common_mismatches":dict(confusion.most_common(20)),
            "all_96_counterfactual_twin_comparisons":pair_details}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("archive",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--label",required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=args.archive.read_bytes()
    cb=CB59(raw)
    if len(raw)!=SOURCE_BYTES or cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S15 requires exactly the same paid 36M CB59 architecture")
    bpp=8*len(raw)/SOURCE_PARAMS
    if bpp>.7:
        raise AssertionError("S15 physical 0.7bpp full archive budget violated")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    suite=make_cases()
    suite_hash=hashlib.sha256(json.dumps([suite,OPEN_QUALITY],
                                sort_keys=True).encode()).hexdigest()
    if len(OPEN_QUALITY)!=16 or len(set(OPEN_QUALITY))!=16:
        raise AssertionError("S15 16 complex open cases missing")
    model,sources=create_small_paid_model(cb,build_config(MODEL_SPEC))
    model.eval()
    responses=[]
    for n,case in enumerate(suite):
        gold=tok.encode(case["gold"],add_special_tokens=False)+[tok.eos_token_id]
        if not 2<=len(gold)<=ANSWER_CAP:
            raise AssertionError("One new composite gold answer cannot be scored")
        got=generate(model,sources,tok,case["question"],ANSWER_CAP)
        actual=tok.decode(got,skip_special_tokens=True)
        formatted=bool(re.fullmatch(
              r"\s*(Yes|No)\s*[,;:\-]?\s*(red|blue|green|yellow)\s*[.!]?\s*",
              actual,re.I))
        flag,col=parse_two_slots(actual)
        bool_correct=flag==case["expected_flag"]
        color_correct=col==case["expected_color"]
        responses.append({
           **case,"expected_ids_plus_EOS":gold,"actual_ids":got,
           "actual_text":actual,
           "EOS_emitted":bool(got) and got[-1]==tok.eos_token_id,
           "strict_full_EOS":got==gold,
           "valid_two_slot_format":formatted,
           "flag_correct":bool_correct,
           "color_correct":color_correct,
           "both_slots_valid_and_correct":formatted and bool_correct and color_correct,
        })
        if (n+1)%48==0:
            print(json.dumps({
              "event":"CALIBRI2_S15_COMPOSITE_QA_PROGRESS","label":args.label,
              "checked":n+1,
              "strict":sum(x["strict_full_EOS"] for x in responses),
              "both_semantic":sum(x["both_slots_valid_and_correct"] for x in responses)
            }),flush=True)
    open_rows=[]
    for prompt in OPEN_QUALITY:
        real=generate(model,sources,tok,prompt,OPEN_CAP)
        open_rows.append({
          "question":prompt,
          "actual_tokens":real,
          "actual_answer":tok.decode(real,skip_special_tokens=True),
          "EOS_emitted":bool(real) and real[-1]==tok.eos_token_id,
          "manual_coherence_assessment_required":True,
        })
    scored=compare_rows(responses)
    result={
      "stage":"CALIBRI2_S15_USER_REQUESTED_192_COMPOSITE_PLUS_16_OPEN",
      "model_label":args.label,
      "physical_SHA256":hashlib.sha256(raw).hexdigest(),
      "full_paid_model_bytes":len(raw),"physical_bpp":bpp,
      "original_parameter_count":SOURCE_PARAMS,
      "independent_immutable_qa_suite_sha256":suite_hash,
      "total_composite_questions":len(responses),
      "total_counterfactual_pairs":96,
      "balance_each_of_eight_gold_combinations":24,
      "expected_Yes_No_and_four_colors_are_orthogonal":True,
      "no_binary_or_fixed_label_shortcut_above_12_5pct":True,
      **scored,
      "real_192_question_expected_vs_actual":responses,
      "real_16_multi_clause_open_answers":open_rows,
      "open_answers_not_claimed_correct_by_auto_metric":True,
      "not_dedicated_general_language_comprehension_measure":True,
      "this_suite_never_used_for_ANY_model_training_or_selection":True,
      "no_FP32_teacher_or_original_pretrained_weights":True,
      "no_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    save(args.out/"ACTUAL_CALIBRI2_S15_MULTIATTRIBUTE_192_QA_16_OPEN.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in (
         "real_192_question_expected_vs_actual",
         "real_16_multi_clause_open_answers",
         "all_96_counterfactual_twin_comparisons",
         "most_common_mismatches"
    )}),flush=True)


if __name__=="__main__":
    main()
