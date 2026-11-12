#!/usr/bin/env python3
"""S173: predeclared unseen TRAIN-derived owner frames + adversarial negatives.

Standalone source-only parser test, NOT weight training/neural generalization.
Uses immutable S168 TRAIN-derived artifact; gold is read ONLY after inference.
No QA/TEST, no ordinary Jessica tests, no LIVE / Render / Supabase changes.
"""
import json
import re
import sys
import hashlib
from pathlib import Path
from calibri2.s169.owner_segment_offline import owner_value, VALUE, NAME

TRAIN_OWNERS=("Folemar","Bintan","Nerilo","Kassel","Voltari","Meyven","Renavi","Calori")
# Deliberately new frames not in either the S164 or S168 templates.
NOVEL_TRAIN_FRAMES=(
    "TRAIN S82 field log: {ask} has {want}; {other} has {alt}. Identify the two-word tag for {ask}.",
    "TRAIN S82 auditor note: For {other} the item is {alt}. For {ask} the item is {want}. Reply with the label assigned to {ask}.",
    "TRAIN S82 dispatch: {alt} goes to {other}; {want} goes to {ask}. Which two-word phrase is attached to {ask}?",
    "TRAIN S82 delivery: {other} receives {alt} and {ask} receives {want}. Tell me the assigned label of {ask}.",
)

def context_from_s168_question(question):
    assert question.startswith("TRAIN S82 ")
    body=question.rsplit(". ",1)[-1]
    owner_names=[x for x in NAME.findall(body) if x in TRAIN_OWNERS]
    assert len(owner_names)==1
    asked=owner_names[0]
    present=[x for x in TRAIN_OWNERS if re.search(r"\\b"+re.escape(x)+r"\\b",question)]
    assert len(present)==2 and asked in present
    other=next(x for x in present if x!=asked)
    source=owner_value(question)
    assert source is not None
    phrases=VALUE.findall(question)
    assert len(phrases)==2 and len(set(phrases))==2 and source in phrases
    alternative=next(x for x in phrases if x!=source)
    return asked,other,source,alternative

def synthesize_frozen_cases(rows):
    assert len(rows)==64 and len(set(r["id"] for r in rows))==64
    positives=[]
    negatives=[]
    for i,row in enumerate(sorted(rows,key=lambda r:r["id"])):
        ask,other,want,alt=context_from_s168_question(row["question"])
        question=NOVEL_TRAIN_FRAMES[i%len(NOVEL_TRAIN_FRAMES)].format(
            ask=ask,other=other,want=want,alt=alt)
        positives.append({"source_id":row["id"],"question":question,
                          "posthoc_answer":row["answer"],
                          "seed_source_sha256":hashlib.sha256(row["question"].encode()).hexdigest()})
        if i<8:
            q=f"TRAIN S82 negation: {ask} does not have {want}; {other} has {alt}. Give the tag for {ask}."
            kind="negated_ownership"
        elif i<16:
            q=f"TRAIN S82 revoked: {ask} once had {want}, but that assignment is revoked; {other} has {alt}. Give the tag for {ask}."
            kind="revoked_ownership"
        elif i<24:
            q=f"TRAIN S82 duplicate: {ask} has {want}; {ask} has {alt}. Give the tag for {ask}."
            kind="ambiguous_same_owner"
        elif i<32:
            q=f"TRAIN S82 unknown: {ask} has {want}; {other} has {alt}. Give the tag for Nonexistent."
            kind="unknown_owner"
        else:
            continue
        negatives.append({"kind":kind,"source_id":row["id"],"question":q})
    assert len(positives)==64 and len(negatives)==32
    all_questions=[x["question"] for x in positives]+[x["question"] for x in negatives]
    assert len(set(all_questions))==96
    assert not (set(all_questions)&set(r["question"] for r in rows))
    return positives,negatives

def main():
    if len(sys.argv)!=3:
        raise SystemExit("S173 needs immutable S168 artifact JSON + result directory")
    source=json.loads(Path(sys.argv[1]).read_text())
    assert source["stage"]=="S168_NOVEL_TRAIN_FRAMES_GENERALIZATION_DIAGNOSTIC"
    assert source["parent_run"]==43287922683
    assert source["no_QA_TEST"] is True and source["model_accepted"] is False
    positives,negatives=synthesize_frozen_cases(source["cases"])
    # INFERENCE: only natural-language question reaches parser, never gold.
    selected_pos=[owner_value(x["question"]) for x in positives]
    selected_neg=[owner_value(x["question"]) for x in negatives]
    # SCORE POSTHOC (not used by owner_value or generation of candidate outputs).
    pos_report=[{
        "case_id":x["source_id"],"question":x["question"],
        "candidate":cand,"expected_posthoc_only":x["posthoc_answer"],
        "correct":cand==x["posthoc_answer"],
        "abstained":cand is None
    } for x,cand in zip(positives,selected_pos)]
    neg_report=[{
        "case_id":x["source_id"],"kind":x["kind"],
        "question":x["question"],"candidate":cand,
        "abstained":cand is None
    } for x,cand in zip(negatives,selected_neg)]
    summary={
        "stage":"S173_PRECOMMITTED_NOVEL_TRAIN_FRAMES_AND_NEGATIVES_OFFLINE",
        "source_S168_GitHub_run":43289943629,
        "novel_train_questions":64,
        "novel_train_parser_correct":sum(x["correct"] for x in pos_report),
        "novel_train_abstained":sum(x["abstained"] for x in pos_report),
        "adversarial_negatives":32,
        "adversarial_abstained":sum(x["abstained"] for x in neg_report),
        "dangerous_false_positives":sum(not x["abstained"] for x in neg_report),
        "adversarial_categories":sorted(set(x["kind"] for x in neg_report)),
        "fully_safe_heuristic":all(x["abstained"] for x in neg_report),
        "cases":pos_report,
        "negative_cases":neg_report,
        "no_gold_used_in_selector":True,
        "QA_TEST_never_run":True,
        "no_model_weight_updates":True,
        "no_real_neural_J7_re_run":True,
        "neural_generalization_not_demonstrated":True,
        "model_accepted":False,
        "LIVE_Jessica_Render_Supabase_untouched":True,
    }
    target=Path(sys.argv[2]);target.mkdir(parents=True,exist_ok=True)
    (target/"S173_NOVEL_TRAIN_64_AND_ADVERSARIAL_32_DIAGNOSTIC.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True))
    print(json.dumps({k:summary[k] for k in (
        "stage","novel_train_questions","novel_train_parser_correct",
        "novel_train_abstained","adversarial_negatives","adversarial_abstained",
        "dangerous_false_positives","fully_safe_heuristic","model_accepted")},
        sort_keys=True),flush=True)
    # Report limitations without hiding failure behind workflow SUCCESS.
    assert len(pos_report)==64 and len(neg_report)==32

if __name__=="__main__":
    main()
