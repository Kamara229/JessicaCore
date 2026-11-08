"""S19: FAIR, NEW two-slot combined answers for canonical FP32 and compressed 36M.

The real S15/S16 tests returned 0/192 for BOTH compressed CALIBRI and
original pretrained 135M FP32; forensic S18 shows an avoidable prompt
confound: the text "Yes, color or No, color" was repeated literally in
many FP32 answers as "Yes, color." S19 commits a brand-new, different
96-question set with clear instructions that never contain the literal
placeholder "Yes, color" or "No, color". A question still requires BOTH
binary truth AND actual object color, with counterfactual false twins.
48 pairs × 2 = 96, four domains, each of 8 label combinations appears
exactly 12 times. A fixed "Yes, red" guess scores <=12/96=12.5%.
This test runs on the original FP32 pretrained weights only as a
reference, never as a trainer/teacher for Calibri.

Same full question/gold/tokenizer/greedy CPU scoring for FP32 and CB59.
No QA used to choose models or alter their weights; checkpoints have
exact SHA and fully paid 0.7bpp physical budget assertions.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s3.fresh_heldout import generate
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import (
    SOURCE_SHA,SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
)
from v174.run import save

COLORS=("red","blue","green","yellow")
KINDS=("numerical","object_assignment","quantities","transfer")
PAIRS_PER_KIND=12
ANSWERS_PER_CASE=22
INSTRUCTION=(
 "Reply with exactly two items, separated by a comma. "
 "The first item must be Yes or No. "
 "The second item must be the real color of the specified object. "
 "Do not add any explanation."
)
# Explicitly refuse the original misleading generic phrase in instructions.
assert "Yes, color" not in INSTRUCTION and "No, color" not in INSTRUCTION


def suite():
    result=[]
    for kind in KINDS:
        for index in range(PAIRS_PER_KIND):
            original=COLORS[index%4]
            contrast=COLORS[(index%4+1+(index//4)%2)%4]
            first=f"ElinCheck{index:03d}"
            second=f"YaroCheck{index:03d}"
            big=51001+index*71
            gap=4+(index%7)
            for flipped in (0,1):
                yes=flipped==0
                if kind=="numerical":
                    hi,lo=(big+gap,big) if yes else (big,big+gap)
                    query=(
                        f"A museum label says {first} has a {original} notebook "
                        f"while {second} has a {contrast} notebook. "
                        f"A measurement log shows {hi} then {lo}. "
                        "Is the first measurement greater than the second? "
                        f"What is {first}'s notebook color?"
                    )
                elif kind=="object_assignment":
                    proposed=original if yes else contrast
                    query=(
                        f"At a festival, {first} holds a {original} balloon and "
                        f"{second} holds a {contrast} balloon. "
                        f"Does {first} hold the {proposed} balloon? "
                        f"What is the actual color of {first}'s balloon?"
                    )
                elif kind=="quantities":
                    high=8+index
                    low=4+index
                    left,right=(high,low) if yes else (low,high)
                    query=(
                        f"In the workshop, {first} has {left} nails and "
                        f"{second} has {right} nails. "
                        f"{first} has a {contrast} bag; {second} has a "
                        f"{original} bag. Does {first} have more nails than "
                        f"{second}? What is {second}'s bag color?"
                    )
                else:
                    queried=second if yes else first
                    query=(
                        f"{first} initially has a {original} umbrella. "
                        f"{first} gives it to {second}; nobody else moves it. "
                        f"Afterwards, is the umbrella held by {queried}? "
                        "What color is the umbrella?"
                    )
                row={
                    "id":f"S19_{kind}_{index:02d}_{flipped}",
                    "pair":f"S19_{kind}_{index:02d}",
                    "kind":kind,"variant":flipped,
                    "question":query+" "+INSTRUCTION,
                    "expected_truth":"Yes" if yes else "No",
                    "expected_color":original,
                    "gold":f"{'Yes' if yes else 'No'}, {original}",
                }
                result.append(row)
    if len(result)!=96 or len({r["question"] for r in result})!=96:
        raise AssertionError("S19 new QA not exactly 96 unique questions")
    assert Counter(r["gold"] for r in result)=={
        f"{flag}, {color}":12 for flag in ("Yes","No") for color in COLORS}
    by_pair=defaultdict(list)
    for r in result:by_pair[r["pair"]].append(r)
    if len(by_pair)!=48 or any(len(pair)!=2 or
        pair[0]["expected_truth"]==pair[1]["expected_truth"] or
        pair[0]["expected_color"]!=pair[1]["expected_color"]
        for pair in by_pair.values()):
        raise AssertionError("S19 paired counterfactual invalid")
    return result


def format_slots(actual):
    match=re.fullmatch(
        r"\s*(yes|no)\s*,\s*(red|blue|green|yellow)\s*[.!]?\s*",
        actual,re.IGNORECASE)
    return (match.group(1).capitalize(),match.group(2).lower()) if match else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--engine",choices=("fp32_135m_original","physical_cb59_36m"),required=True)
    parser.add_argument("--physical-archive",type=Path)
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(2119)
    torch.use_deterministic_algorithms(True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    if a.engine=="fp32_135m_original":
        if a.physical_archive:
            raise AssertionError("FP32 reference must not load compressed model")
        model=AutoModelForCausalLM.from_pretrained(
            MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
            attn_implementation="eager").cpu().eval()
        params=sum(p.numel() for p in model.parameters())
        if params!=134515008 or any(p.dtype!=torch.float32 for p in model.parameters()):
            raise AssertionError("Not the real 135M fully FP32 original model")
        sources={}
        model_sha=MODEL_SHA
        bpp=None
    else:
        if a.physical_archive is None:
            raise AssertionError("Compressed test requires a real physical CB59 source")
        packed=a.physical_archive.read_bytes()
        if hashlib.sha256(packed).hexdigest()!=SOURCE_SHA:
            raise AssertionError("S19 compare against exact sealed 36M S7")
        archive=CB59(packed)
        params=archive.meta["original_unique_parameter_count"]
        if params!=SOURCE_PARAMS or len(packed)!=SOURCE_BYTES:
            raise AssertionError("Compressed source format did not preserve size")
        bpp=8*len(packed)/params
        if bpp>.7:
            raise AssertionError("Physical CB59 model above bit budget")
        model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
        model.eval()
        model_sha=SOURCE_SHA
    qs=suite()
    qa_sha=hashlib.sha256(json.dumps(qs,sort_keys=True).encode()).hexdigest()
    answers=[]
    for i,row in enumerate(qs):
        gold_ids=tok.encode(row["gold"],add_special_tokens=False)+[tok.eos_token_id]
        got=generate(model,sources,tok,row["question"],ANSWERS_PER_CASE)
        actual=tok.decode(got,skip_special_tokens=True)
        slots=format_slots(actual)
        answers.append({
          **row,"expected_ids":gold_ids,"actual_token_ids":got,
          "actual":actual,
          "generated_EOS":bool(got) and got[-1]==tok.eos_token_id,
          "strict_EOS":gold_ids==got,
          "valid_two_slot":slots is not None,
          "both_fields_correct":slots==(row["expected_truth"],row["expected_color"]),
          "truth_correct":bool(slots and slots[0]==row["expected_truth"]),
          "color_correct":bool(slots and slots[1]==row["expected_color"]),
        })
        if (i+1)%24==0:
            print(json.dumps({
              "event":"S19_NEW_FAIR_COMPOSITE_PROGRESS",
              "engine":a.engine,"count":i+1,
              "both_valid":sum(v["both_fields_correct"] for v in answers),
              "strict_EOS":sum(v["strict_EOS"] for v in answers)
            }),flush=True)
    by_domain=defaultdict(lambda:Counter())
    groups=defaultdict(list)
    for item in answers:
        d=by_domain[item["kind"]]
        for key in ("both_fields_correct","strict_EOS","valid_two_slot",
                    "truth_correct","color_correct"):
            d[key]+=int(item[key])
        groups[item["pair"]].append(item)
    paired=sum(int(all(x["both_fields_correct"] for x in pair)) for pair in groups.values())
    report={
      "stage":"S19_NEW_BALANCED_96_USER_TWO_FIELD_FAIR_FORMAT",
      "model_engine":a.engine,"model_SHA":model_sha,
      "original_parameter_count":params,
      "physically_paid_bpp_or_none_for_original_fp32":bpp,
      "identical_suite_SHA256":qa_sha,
      "fresh_question_count":len(answers),
      "paired_counterfactual_groups":len(groups),
      "each_yesno_plus_color_combination_frequency":12,
      "fixed_prediction_shortcut_MAX":12,
      "fully_correct_two_slot_answers":sum(v["both_fields_correct"] for v in answers),
      "strict_token_sequence_plus_EOS":sum(v["strict_EOS"] for v in answers),
      "valid_two_slot_form":sum(v["valid_two_slot"] for v in answers),
      "correct_first_slot_given_valid_format":sum(v["truth_correct"] for v in answers),
      "correct_second_slot_given_valid_format":sum(v["color_correct"] for v in answers),
      "both_twins_correct":paired,
      "by_domain":dict(by_domain),
      "most_common_verbatim_answers":Counter(x["actual"] for x in answers).most_common(12),
      "all_96_actual_answers":answers,
      "no_QA_in_weight_training_or_model_selection":True,
      "no_FP32_teacher_transfer":True,
      "test_prompt_does_not_literalize_color_placeholder":True,
      "distinct_model_architecture_36m_versus_135m":True,
    }
    save(a.output/f"ACTUAL_CALIBRI2_S19_{a.engine}_96_TWO_SLOTS.json",report)
    print(json.dumps({
      "event":"S19_NEW_FAIR_96_TWO_SLOT_FINAL",
      "engine":a.engine,"model_SHA":model_sha,
      "correct_both":report["fully_correct_two_slot_answers"],
      "strict_EOS":report["strict_token_sequence_plus_EOS"],
      "format":report["valid_two_slot_form"],
      "paired":paired,"by_domain":dict(by_domain),
      "top_answers":report["most_common_verbatim_answers"][:8],
      "actual_example_pairs":[
          {"question":x["question"],"gold":x["gold"],"actual":x["actual"]}
          for x in answers[:4]]
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
