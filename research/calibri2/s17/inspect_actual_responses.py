"""CALIBRI2 S17: explain S15 composite-answer errors with actual model outputs.

Read only the final saved JSON results of the precommitted 192+16 S15
heldout suite. No model code, no weights, no prompts generated anew,
no retraining or QA-based checkpoint selection. Print a short falsifiable
sample of real questions, expected vs generated words and 16 open answers.
Do not mistake 0 joint score for absence of all component understanding.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

EXPECTED_SHA="3ddb7bf64779c47990f1b729cb6a59a3b2a0ae9f3d0f1b575317c279ac5cb8a0"
MODELS=("S7","S8","S12")
MODEL_SHA={
  "S7":"70512bc374c1846631f635998f9e035e2e25f147da185c2b7b0dd663799d9f8a",
  "S8":"386054d9fe2abfa5f43df5f9ae7292629fade5a2e527e64b46325d5201d4c8f8",
  "S12":"f8666b6815a9f7cc966550260706dd2632c0863ce45c2581e7c0a44a7a2e084a",
}
SAMPLE_IDS=(
 "S15_number_plus_color_00_0","S15_number_plus_color_00_1",
 "S15_ownership_plus_color_01_0","S15_ownership_plus_color_01_1",
 "S15_count_plus_color_02_0","S15_count_plus_color_02_1",
 "S15_transfer_plus_color_03_0","S15_transfer_plus_color_03_1",
 "S15_number_plus_color_08_0","S15_number_plus_color_08_1",
 "S15_ownership_plus_color_09_0","S15_ownership_plus_color_09_1",
)


def read(path,name):
    r=json.loads(path.read_text())
    if r["independent_immutable_qa_suite_sha256"]!=EXPECTED_SHA:
        raise AssertionError("S15 QA suite SHA drift")
    if r["physical_SHA256"]!=MODEL_SHA[name]:
        raise AssertionError("Unexpected sealed physical checkpoint")
    rows=r["real_192_question_expected_vs_actual"]
    expected={f"S15_{domain}_{i:02d}_{flip}"
              for domain in ("number_plus_color","ownership_plus_color",
                             "count_plus_color","transfer_plus_color")
              for i in range(24) for flip in (0,1)}
    if len(rows)!=192 or {x["id"] for x in rows}!=expected:
        raise AssertionError("Not all actual frozen QA answers present")
    if len(r["real_16_multi_clause_open_answers"])!=16:
        raise AssertionError("Missing 16 open prompts")
    if r["semantic_two_slots_both_correct"]!=0:
        raise AssertionError("Actual S15 joint quality changed unexpectedly")
    return r,{x["id"]:x for x in rows}


def main():
    a=argparse.ArgumentParser()
    for name in MODELS:
        a.add_argument(name.lower(),type=Path)
    a.add_argument("out",type=Path)
    args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    by_name={}
    for name in MODELS:
        report,rows=read(getattr(args,name.lower()),name)
        categories=Counter()
        invalid=Counter()
        for x in rows.values():
            actual=x["actual_text"].strip()
            categories[actual]+=1
            if not x["valid_two_slot_format"]:
                invalid[x["domain"]]+=1
        by_name[name]={"report":report,"rows":rows,
                       "frequent_actual_answers":categories.most_common(12),
                       "invalid_by_domain":dict(invalid)}
    examples=[]
    for case_id in SAMPLE_IDS:
        row=by_name["S7"]["rows"][case_id]
        example={"id":case_id,"domain":row["domain"],
                 "question":row["question"],"gold":row["gold"]}
        for name in MODELS:
            other=by_name[name]["rows"][case_id]
            if other["gold"]!=row["gold"]:
                raise AssertionError("Gold answers not aligned")
            example[name+"_actual"]=other["actual_text"]
            example[name+"_EOS"]=other["EOS_emitted"]
            example[name+"_flag_correct"]=other["flag_correct"]
            example[name+"_color_correct"]=other["color_correct"]
        examples.append(example)
    open_samples=[]
    for i in (0,1,4,5,8,10,11,15):
        first=by_name["S7"]["report"]["real_16_multi_clause_open_answers"][i]
        sample={"question":first["question"]}
        for name in MODELS:
            item=by_name[name]["report"]["real_16_multi_clause_open_answers"][i]
            if item["question"]!=first["question"]:
                raise AssertionError("Open question mismatch")
            sample[name+"_answer"]=item["actual_answer"]
        open_samples.append(sample)
    summary={
        "stage":"S17_ACTUAL_S15_COMPLEX_192_16_REAL_RESPONSE_DIAGNOSTIC",
        "same_frozen_suite_sha":EXPECTED_SHA,
        "measured_correct_full_EOS":{
            name:by_name[name]["report"]["strict_full_plus_EOS"] for name in MODELS},
        "measured_correct_semantic_both":{
            name:by_name[name]["report"]["semantic_two_slots_both_correct"]
            for name in MODELS},
        "measured_independent_slot_counts":{
            name:{
                "truth_flag":sum(row["flag_correct"] for row in by_name[name]["rows"].values()),
                "color":sum(row["color_correct"] for row in by_name[name]["rows"].values()),
            } for name in MODELS
        },
        "actual_prediction_modes":{
            name:by_name[name]["frequent_actual_answers"] for name in MODELS},
        "12_fixed_counterfactual_examples":examples,
        "8_open_qualitative_actual_answer_comparisons":open_samples,
        "interpretation":"The frozen suite measures two simultaneously correct"
            " components. A 0/192 joint score does not imply zero partial"
            " word knowledge; check actual generated answers. FP32 is separately"
            " evaluated by S16 to establish whether this task is achievable"
            " by the base pretrained 135M model.",
        "read_only":True,
        "no_QA_use_for_training_or_checkpoint_selection":True,
        "no_production_files_changed":True,
    }
    result=args.out/"ACTUAL_S17_S15_12_EXAMPLES_AND_8_OPEN.json"
    result.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
      "event":"S17_REPORT_REAL_COMPOSITE_ANSWERS",
      "common_predictions":summary["actual_prediction_modes"],
      "slot_counts":summary["measured_independent_slot_counts"],
      "example_pairs":examples[:4],
      "freeform_examples":open_samples[:3],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
