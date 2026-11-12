#!/usr/bin/env python3
"""S164: fail-closed prompt-only owner binding, OFFLINE research diagnostic.

This does not train or change any neural weights. Successful lexical template
matching does not establish a model improvement or held-out generalization.
"""
import json, re, sys
from pathlib import Path

PREFIX=re.compile(r"^TRAIN S82 [^:]+:\s*")
O=r"([A-Za-z][A-Za-z0-9]*)"
P=r"([a-z]+ [a-z]+)"
PATTERNS=[
    re.compile(rf"^The record for {O} displays {P}\. The separate record for {O} displays {P}\. Return the two-word value of {O}\.$"),
    re.compile(rf"^Worker {O} has key {P}; worker {O} instead has key {P}\. State exactly the assigned two words for {O}\.$"),
    re.compile(rf"^Under {O} the color-item phrase is {P}, whereas under {O} it is {P}\. Copy the complete color and item for {O}\.$"),
    re.compile(rf"^Package owner {O} received {P}\. Package owner {O} received {P}\. What is the exact two-word label on {O}'s package\?$"),
]

def owner_bound_copy(question):
    # STRICTLY prompt-only: neither gold nor expected token IDs enter the selector.
    prompt=PREFIX.sub("",question,count=1)
    for pattern in PATTERNS:
        match=pattern.fullmatch(prompt)
        if not match: continue
        owner1,phrase1,owner2,phrase2,requested=match.groups()
        if owner1==owner2 or phrase1==phrase2: return None
        if requested==owner1: return phrase1
        if requested==owner2: return phrase2
        return None
    return None

def main():
    if len(sys.argv)!=3:
        raise SystemExit("Usage: S164.py S163_ATLAS.json OUTPUT_DIR")
    atlas=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    assert atlas["parent_run"]==43278364923
    assert atlas["model_accepted"] is False and atlas["QA_TEST_never_run"]
    rows=atlas["all_64_cases"]
    assert len(rows)==64 and len({r["id"] for r in rows})==64
    failed_ids={r["id"] for r in atlas["residual_17_cases"]}
    assert len(failed_ids)==17
    results=[]
    for row in rows:
        # No gold passed to the selector: score ONLY after owner_bound_copy.
        candidate=owner_bound_copy(row["question"])
        results.append({
            "id":row["id"],"same_color":row["same_color"],
            "parent_route":row["route"],
            "original_model_correct":row["candidate_correct"],
            "offline_parser_correct":candidate==row["answer"],
            "offline_parser_abstained":candidate is None,
            "original_model_failed":row["id"] in failed_ids,
        })
    negative=[
        "TRAIN S82 inventory: The record for Folemar displays green desk. The separate record for Bintan displays green lamp. Return the two-word value of Stranger.",
        "TRAIN S82 inventory: The record for Folemar displays green desk. The separate record for Folemar displays green lamp. Return the two-word value of Folemar.",
        "TRAIN S82 inventory: The record for Folemar displays green desk. The separate record for Bintan displays green desk. Return the two-word value of Folemar.",
        "TRAIN S82 inventory: The record for Folemar displays green desk. The separate record for Bintan displays green lamp. Return the two-word value of Folemar. extra",
        "unknown question",
    ]
    fail_closed=all(owner_bound_copy(x) is None for x in negative)
    count_correct=sum(r["offline_parser_correct"] for r in results)
    recovered=sum(r["original_model_failed"] and r["offline_parser_correct"] for r in results)
    lost=sum(r["original_model_correct"] and not r["offline_parser_correct"] for r in results)
    assert count_correct==64 and recovered==17 and lost==0 and fail_closed
    out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    report={
        "stage":"S164_PROMPT_ONLY_OWNER_BINDING_TEMPLATE_DIAGNOSTIC",
        "immutable_S163_run":43280941993,
        "original_model_correct":47,
        "original_model_wrong":17,
        "offline_template_parser_correct":count_correct,
        "offline_template_parser_recovers_prior_errors":recovered,
        "offline_template_parser_loses_prior_correct":lost,
        "fail_closed_negative_cases":len(negative),
        "fail_closed":fail_closed,
        "gold_never_used_at_inference":True,
        "no_model_training_or_integrated_candidate":True,
        "no_new_J7_or_CPU_or_bpp_measurement":True,
        "no_QA_TEST":True,"no_LIVE_changes":True,
        "model_accepted":False,
        "novel_question_generalization_not_established":True,
        "diagnostics":results,
    }
    (out/"S164_OWNER_BINDING_OFFLINE_64_PROBE.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({
        "event":"S164_PROMPT_ONLY_OWNER_BINDING_OFFLINE_AUDIT",
        "S163_model_correct":47,"offline_parser_correct":count_correct,
        "previous_errors_covered":recovered,"previous_correct_lost":lost,
        "fail_closed":fail_closed,"no_model_promotion":True,
    }),flush=True)

if __name__=="__main__":
    main()
