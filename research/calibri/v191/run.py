"""v191 read-only posthoc v190 full-utterance format analysis."""
import argparse
import json
from collections import Counter
from pathlib import Path

from v174.run import save
from v190.cases import cases
from .protocol import ARMS,EXPECTED_PER_ARM
from .grammar import grade_structured


def audit(original):
    if not original.get("complete"):
        raise ValueError("Source v190 original model audit is not completed")
    fixed=cases()
    if len(fixed)!=EXPECTED_PER_ARM:
        raise AssertionError("Frozen 64 source questions drifted")
    frozen=original["locked_before_original_model_inference"]
    if frozen["all_64_questions_golds_hashes"]!=fixed:
        raise AssertionError("The v190 source question IDs, domains, golds or hashes differ")
    result={}
    for arm in ARMS:
        rows=original["original_FP32_64_full_verbatim_generations_each"][arm]
        if len(rows)!=EXPECTED_PER_ARM or [x["case_id"] for x in rows]!=list(range(EXPECTED_PER_ARM)):
            raise AssertionError("Missing or reordered original generated model responses")
        before=Counter()
        after=Counter()
        additions=[]
        full=[]
        for case,row in zip(fixed,rows):
            if (row["question"]!=case["question"] or
                row["gold"]!=case["gold"] or
                row["domain"]!=case["domain"] or
                row["strict_grade"]["case_id"]!=case["case_id"]):
                raise AssertionError("Model answer and original case mismatch")
            old=row["strict_grade"]["machine_status"]
            if old not in ("machine_correct","machine_wrong","ungraded"):
                raise ValueError("v190 original grade altered")
            new=grade_structured(case,row)
            before[old]+=1
            after[new["machine_status"]]+=1
            merged={"case_id":case["case_id"],"domain":case["domain"],
                "question":case["question"],"gold":case["gold"],
                "original_verbatim":row["response"]["verbatim_text"],
                "original_v190_grade":old,"posthoc_v191_grade":new}
            full.append(merged)
            if new["changed_from_original_v190"]:
                if old!="ungraded":
                    raise AssertionError("Original nonungraded result was changed by exploratory parsing")
                additions.append(merged)
        if sum(before.values())!=EXPECTED_PER_ARM or sum(after.values())!=EXPECTED_PER_ARM:
            raise AssertionError("Lost original source replies")
        by_domain={}
        for dom in sorted(set(x["domain"] for x in fixed)):
            selection=[x for x in full if x["domain"]==dom]
            by_domain[dom]={"n":len(selection),
                "v190_precommitted_correct":sum(x["original_v190_grade"]=="machine_correct" for x in selection),
                "v191_posthoc_correct":sum(x["posthoc_v191_grade"]["machine_status"]=="machine_correct" for x in selection),
                "v191_posthoc_wrong":sum(x["posthoc_v191_grade"]["machine_status"]=="machine_wrong" for x in selection),
                "v191_posthoc_ungraded":sum(x["posthoc_v191_grade"]["machine_status"]=="ungraded" for x in selection)}
        result[arm]={"v190_precommitted_original_counts":dict(before),
            "v191_exploratory_posthoc_counts":dict(after),
            "changes_from_original_ungraded":len(additions),
            "posthoc_only_changed_cases":additions,
            "all_64_original_and_regraded":full,
            "by_domain":by_domain}
    return {"complete":True,
        "scientific_status":"RETROSPECTIVE POSTHOC PARSER AUDIT NOT INDEPENDENT QA",
        "input_source":"complete unchanged v190 original source output #13446560270",
        "input_models":list(ARMS),"input_cases_per_model":EXPECTED_PER_ARM,
        "raw_answer_gold_and_original_grade_unmodified":True,
        "v190_primary_quality_claims_remain_unmodified":True,
        "this_is_not_an_independent_precommitted_semantic_score":True,
        "future_grader_must_be_frozen_before_NEW_unseen_dataset":True,
        "private_v89_TEST_unread":True,
        "no_model_weights_or_native_runtime_modified":True,
        "no_Jessica_model_tests":True,
        "comparison":result}


def run(source,target):
    source=Path(source)
    modeldata=json.loads(source.read_text())
    out=audit(modeldata)
    target.mkdir(parents=True,exist_ok=True)
    save(target/"v191_retrospective_128_raw_original_answers_audit.json",out)
    print(json.dumps({"stage":"posthoc_audit_complete",
        "original_vs_exploratory":{
          arm:{"original":row["v190_precommitted_original_counts"],
               "posthoc":row["v191_exploratory_posthoc_counts"],
               "newly_recognized":row["changes_from_original_ungraded"],
               "domains":row["by_domain"]}
          for arm,row in out["comparison"].items()},
        "not_independent_test":True}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.source,args.output)
