"""Read-only forensic review of already generated 135M FP32 results on S15.
No weights, no model inference, no training, no QA-informed checkpoint selection.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path

EXPECTED_SUITE_SHA="3ddb7bf64779c47990f1b729cb6a59a3b2a0ae9f3d0f1b575317c279ac5cb8a0"
EXPECTED_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"

def main():
 p=argparse.ArgumentParser()
 p.add_argument("original_fp32_s16_json",type=Path)
 p.add_argument("out",type=Path)
 args=p.parse_args()
 result=json.loads(args.original_fp32_s16_json.read_text())
 if result["identical_S15_new_192_and_16_open_qa_suite_SHA256"]!=EXPECTED_SUITE_SHA:
  raise AssertionError("FP32 suite must be exactly S15 new two-slot QA")
 if result["model"]!=EXPECTED_MODEL or result["original_FP32_unique_params"]!=134515008:
  raise AssertionError("Not canonical original FP32 pretrained model")
 rows=result["all_192_real_generated_answers"]
 open_rows=result["all_16_real_compositional_open_answers"]
 if len(rows)!=192 or len(open_rows)!=16:
  raise AssertionError("Expected 192 actual composite and 16 open responses")
 counts=Counter(row["actual_text"].strip() for row in rows)
 domains=defaultdict(lambda:dict(n=0,yes_no=0,color=0,valid_format=0,both=0,exact=0,eos=0))
 sample=[]
 for row in rows:
  domain=row["domain"]; stats=domains[domain]
  stats["n"]+=1
  stats["yes_no"]+=int(row["flag_correct"])
  stats["color"]+=int(row["color_correct"])
  stats["valid_format"]+=int(row["valid_two_slot_format"])
  stats["both"]+=int(row["both_slots_valid_and_correct"])
  stats["exact"]+=int(row["strict_full_EOS"])
  stats["eos"]+=int(row["EOS_emitted"])
  if row["id"].endswith("_00_0") or row["id"].endswith("_00_1") or row["id"].endswith("_07_0") or row["id"].endswith("_07_1"):
   sample.append({
    "id":row["id"],"question":row["question"],"gold":row["gold"],
    "actual":row["actual_text"],"eos":row["EOS_emitted"],
    "binary_correct":row["flag_correct"],"color_correct":row["color_correct"],
    "both_correct":row["both_slots_valid_and_correct"]})
 if len(sample)!=16:
  raise AssertionError("Missed paired actual examples")
 finished={
  "stage":"S18_CROSSCHECK_ORIGINAL_FP32_TWO_SLOT_S15_REAL_OUTPUTS",
  "same_new_S15_suite_sha":EXPECTED_SUITE_SHA,
  "original_FP32_model":EXPECTED_MODEL,
  "original_FP32_unique_params":134515008,
  "total_tested":len(rows),
  "complete_answer_both_fields":sum(s["both"] for s in domains.values()),
  "exact_full_answer_EOS":sum(s["exact"] for s in domains.values()),
  "either_field_alone_no_credit_for_full_answer":True,
  "correct_binary_field_on_192":sum(s["yes_no"] for s in domains.values()),
  "correct_color_field_on_192":sum(s["color"] for s in domains.values()),
  "generated_end_of_sequence_on_192":sum(s["eos"] for s in domains.values()),
  "generated_two_field_parseable_on_192":sum(s["valid_format"] for s in domains.values()),
  "by_domain":dict(domains),
  "most_common_actual_answers":counts.most_common(25),
  "actual_16_frozen_counterfactual_examples":sample,
  "actual_16_open_compositions":[{"question":x["question"],"response":x["actual_answer"],"EOS":x["EOS_emitted"]} for x in open_rows],
  "no_model_load_or_training_or_weight_export":True,
  "diagnostic_not_model_selection":True,
 }
 args.out.mkdir(parents=True,exist_ok=True)
 (args.out/"ACTUAL_S18_ORIGINAL_FP32_135M_192_COMPOSITE_RESPONSE_AUDIT.json").write_text(json.dumps(finished,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps({
   "event":"S18_ORIGINAL_FP32_S15_NEW_TWO_FIELD_REAL_OUTPUTS",
   "whole":finished["complete_answer_both_fields"],
   "binary":finished["correct_binary_field_on_192"],
   "color":finished["correct_color_field_on_192"],
   "EOS":finished["generated_end_of_sequence_on_192"],
   "format":finished["generated_two_field_parseable_on_192"],
   "frequency":finished["most_common_actual_answers"][:10],
   "samples":sample[:8],
   "open_samples":finished["actual_16_open_compositions"][:4]
  },ensure_ascii=False),flush=True)

if __name__=="__main__":
 main()
