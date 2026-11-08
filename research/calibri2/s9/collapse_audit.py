"""CALIBRI2 S9: read-only, exact-same-suite real-answer collapse audit.

Original S8 baseline and final already evaluated the same frozen 192/16 suite
in the committed S8 CI run. This step does NOT re-run training, select weights,
look at QA to change training, or infer mechanisms from observed correlation.
It measures exact FULL-EOS losses by task and inspects whether relation/color
answers turned into a single default, a label inversion or unrelated output.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

EXPECTED_SUITE = "c859b1b6da538f415cf083bd8a8b3db95fd396d1a8220193070a04bb00cfe542"
EXPECTED_S7 = "70512bc374c1846631f635998f9e035e2e25f147da185c2b7b0dd663799d9f8a"
EXPECTED_S8 = "386054d9fe2abfa5f43df5f9ae7292629fade5a2e527e64b46325d5201d4c8f8"
LABELS = ("red", "blue", "Yes", "No")

def assert_and_map(report, expected_sha):
    if report["suite_sha256"]!=EXPECTED_SUITE or report["physical_SHA256"]!=expected_sha:
        raise AssertionError("Frozen suite, immutable source or final weight SHA changed")
    if report["objective_total"]!=192 or len(report["all_actual_192_answers"])!=192:
        raise AssertionError("Must examine all 192 existing QA questions, not a sample")
    if len(report["all_actual_16_open_answers"])!=16:
        raise AssertionError("Missing all 16 open-answer actual generations")
    rows=report["all_actual_192_answers"]
    result={x["id"]:x for x in rows}
    if len(result)!=192:
        raise AssertionError("Duplicate QA IDs")
    return result

def output_bucket(x):
    """Label-only view, no new model inference."""
    normalized=" ".join(x["actual_text"].strip().split()).rstrip(".,! ").lower()
    for k in LABELS:
        if normalized==k.lower():
            return k
    if not normalized:
        return "(empty)"
    return "(other)"

def analyze(before,after):
    a=assert_and_map(before,EXPECTED_S7)
    b=assert_and_map(after,EXPECTED_S8)
    if set(a)!=set(b):
        raise AssertionError("Independent QA case IDs changed")
    comparisons=[]
    confusion=defaultdict(Counter)
    task_transitions=defaultdict(Counter)
    per_task=defaultdict(lambda:{"before_correct":0,"after_correct":0,
                                 "lost":0,"gained":0,"unchanged_correct":0,
                                 "unchanged_wrong":0,"before_first":0,"after_first":0})
    output_before=defaultdict(Counter)
    output_after=defaultdict(Counter)
    for case_id in sorted(a):
        x,y=a[case_id],b[case_id]
        if (x["domain"],x["question"],x["gold"],x["expected_ids"])!= \
           (y["domain"],y["question"],y["gold"],y["expected_ids"]):
            raise AssertionError("Answers are not on identical gold QA examples")
        domain=x["domain"]
        old_ok=bool(x["strict_full_EOS"])
        new_ok=bool(y["strict_full_EOS"])
        if old_ok and not new_ok:
            transition="lost_correct"
        elif not old_ok and new_ok:
            transition="gained_correct"
        elif old_ok:
            transition="unchanged_correct"
        else:
            transition="unchanged_wrong"
        counts=per_task[domain]
        counts["before_correct"]+=int(old_ok)
        counts["after_correct"]+=int(new_ok)
        counts[transition.replace("_correct","") if False else 
               {"lost_correct":"lost","gained_correct":"gained",
                "unchanged_correct":"unchanged_correct",
                "unchanged_wrong":"unchanged_wrong"}[transition]]+=1
        counts["before_first"]+=int(x["first_token_correct"])
        counts["after_first"]+=int(y["first_token_correct"])
        task_transitions[domain][transition]+=1
        old_bucket=output_bucket(x)
        new_bucket=output_bucket(y)
        output_before[domain][old_bucket]+=1
        output_after[domain][new_bucket]+=1
        if domain in ("relation_grounding","number_comparison"):
            confusion[domain][f"gold={x['gold']}|S7={old_bucket}|S8={new_bucket}"]+=1
        comparisons.append({
            "id":case_id,"domain":domain,"gold":x["gold"],
            "S7_answer":x["actual_text"],"S8_answer":y["actual_text"],
            "S7_generated_ids":x["generated_ids"],
            "S8_generated_ids":y["generated_ids"],
            "S7_strict_correct":old_ok,"S8_strict_correct":new_ok,
            "transition":transition,"changed_generated_tokens":
                x["generated_ids"]!=y["generated_ids"]})
    if sum(x["before_correct"] for x in per_task.values())!=50:
        raise AssertionError("Unexpected S7 baseline 50/192")
    if sum(x["after_correct"] for x in per_task.values())!=22:
        raise AssertionError("Unexpected S8 final 22/192")
    open_alignment=[]
    for left,right in zip(before["all_actual_16_open_answers"],
                          after["all_actual_16_open_answers"]):
        if left["question"]!=right["question"]:
            raise AssertionError("Open QA question changed")
        open_alignment.append({
            "question":left["question"],
            "S7_answer":left["answer"],
            "S8_answer":right["answer"],
            "same_exact_tokens":left["tokens"]==right["tokens"],
            "S7_stopped_at_EOS":left["stopped_at_EOS"],
            "S8_stopped_at_EOS":right["stopped_at_EOS"]})
    losses=[x for x in comparisons if x["transition"]=="lost_correct"]
    gains=[x for x in comparisons if x["transition"]=="gained_correct"]
    expected_losses=sum(v["before_correct"]-v["after_correct"] for v in per_task.values())
    if len(losses)-len(gains)!=expected_losses:
        raise AssertionError("Nonconservation of strict correct-response outcomes")
    report={
       "stage":"CALIBRI2_S9_READ_ONLY_S7_VS_S8_ANSWER_COLLAPSE_DIAGNOSTIC",
       "observed_s8_qa_drop_full_EOS":{"before":50,"after":22,"change":-28,
             "losses":len(losses),"gains":len(gains),
             "total":192,"measurement_not_hypothesis":True},
       "tested_same_suite_sha256":EXPECTED_SUITE,
       "S7_model_sha256":EXPECTED_S7,"S8_model_sha256":EXPECTED_S8,
       "by_domain":dict(sorted(per_task.items())),
       "answer_class_distribution_before":{k:dict(v) for k,v in output_before.items()},
       "answer_class_distribution_after":{k:dict(v) for k,v in output_after.items()},
       "gold_vs_predicted_confusion":{k:dict(v) for k,v in confusion.items()},
       "domain_transitions":{k:dict(v) for k,v in task_transitions.items()},
       "all_192_case_transitions":comparisons,
       "all_16_open_qa_side_by_side":open_alignment,
       "read_only_existing_artifacts":True,
       "no_new_weights_changed_or_generated":True,
       "QA_not_used_to_choose_a_checkpoint_or_hyperparameters":True,
       "cause_not_established_by_this_diagnostic":True,
       "working_hypothesis_only":
          "Shared 85 paid dictionaries may have lost task-discriminative features after broad TRAIN. "
          "The same prompts can also fail from output tokenization/answer format; audit distributions before diagnosing mechanism.",
       "no_live_Render_Supabase_Jessica_APK_changes":True,
    }
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("before",type=Path)
    ap.add_argument("after",type=Path)
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    before=json.loads(args.before.read_text())
    after=json.loads(args.after.read_text())
    result=analyze(before,after)
    args.out.mkdir(parents=True,exist_ok=True)
    output=args.out/"ACTUAL_CALIBRI2_S9_192_CORRECTNESS_AND_CONFUSION.json"
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
       "stage":result["stage"],
       "before":50,"after":22,"lost":result["observed_s8_qa_drop_full_EOS"]["losses"],
       "gained":result["observed_s8_qa_drop_full_EOS"]["gains"],
       "by_domain":result["by_domain"],
       "before_answer_distribution":result["answer_class_distribution_before"],
       "after_answer_distribution":result["answer_class_distribution_after"],
       "confusion":result["gold_vs_predicted_confusion"],
       "open_changed":sum(not x["same_exact_tokens"] for x in result["all_16_open_qa_side_by_side"]),
    }),flush=True)

if __name__=="__main__":
    main()
