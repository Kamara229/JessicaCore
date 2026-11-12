#!/usr/bin/env python3
"""S174: research-only offline vetoes for negative owner claims; new conjunctive frame.

NOT neural training or a proof of out-of-distribution generalization. No QA/TEST,
LIVE, Jessica chat, Green Tunnel or backend changes. Selector sees ONLY prompt.
Freeze the S173 64+32 source set, and add an independent set of 32 negatives.
"""
import json
import re
import sys
from pathlib import Path
from calibri2.s169.owner_segment_offline import owner_value, PREFIX, failclosed_checks
from calibri2.s173.novel_train_owner_adversarial_offline import (
    synthesize_frozen_cases, context_from_s168_question, TRAIN_OWNERS,
)

# If ownership is disputed, rescinded or negated, ALWAYS abstain:
# understanding negative assertions was not established by S169's parser.
NEGATIVE_CLAIMS = re.compile(
    r"\b(?:does\s+not|do\s+not|did\s+not|not\s+assigned|"
    r"not\s+owned|never|no\s+longer|revoked|cancelled|canceled|"
    r"withdrawn|former|previously|once\s+had|"
    r"reassigned|replaced|superseded|except|unless)\b", re.I)
# Only two supported full positive source constructions, with two *separate*
# owner/value clauses. Do not interpret loose "and" in an unrelated sentence.
CONJUNCTIVE_TWO_RECEIVERS = re.compile(
    r"\s+and\s+(?=[A-Z][a-z]{2,}\s+receives\s+"
    r"(?:red|green|blue|yellow)\s+[a-z]+\b)")

OWNER_WORD=re.compile(r"\b[A-Z][a-z]{2,}\b")

def safe_owner_value(question):
    if not isinstance(question,str) or not question.startswith("TRAIN S82 "):
        return None
    plain=PREFIX.sub("",question,count=1)
    if ". " not in plain:
        return None
    source,request=plain.rsplit(". ",1)
    # Conservative one-owner request, no multi-owner answer instructions.
    requested={word for word in OWNER_WORD.findall(request)
               if word in TRAIN_OWNERS}
    if len(requested)!=1:
        return None
    # Reject conflicting ownership claims, not just a missing second span.
    if NEGATIVE_CLAIMS.search(source):
        return None
    fixed=CONJUNCTIVE_TWO_RECEIVERS.sub("; ",source)
    if fixed!=source:
        # Only one two-receiver conjunction; no conjunctions left.
        if not re.fullmatch(
            r"\s*[A-Z][a-z]{2,}\s+receives\s+"
            r"(?:red|green|blue|yellow)\s+[a-z]+;\s*"
            r"[A-Z][a-z]{2,}\s+receives\s+"
            r"(?:red|green|blue|yellow)\s+[a-z]+\s*",fixed):
            return None
    return owner_value("TRAIN S82 checked: "+fixed+". "+request)


def new_adversarial(rows):
    cases=[]
    for i,row in enumerate(sorted(rows,key=lambda r:r["id"])[:32]):
        ask,other,want,alt=context_from_s168_question(row["question"])
        if i<8:
            kind="explicit_not_assigned"
            q=f"TRAIN S82 status: {ask} is not assigned {want}; {other} has {alt}. Give the tag for {ask}."
        elif i<16:
            kind="previously_reassigned"
            q=f"TRAIN S82 change: {ask} previously held {want} but it was reassigned; {other} has {alt}. Give the tag for {ask}."
        elif i<24:
            kind="double_requested_owner"
            q=f"TRAIN S82 request: {ask} has {want}; {other} has {alt}. Give both owners' tags for {ask} and {other}."
        else:
            kind="third_conjunctive_assignment"
            q=f"TRAIN S82 triple: {ask} receives {want} and {other} receives {alt} and Voltari receives green extra. Give the tag for {ask}."
        cases.append({"kind":kind,"source_id":row["id"],"question":q})
    assert len(cases)==32 and len({c["question"] for c in cases})==32
    return cases


def main():
    if len(sys.argv)!=3:
        raise SystemExit("Expected pinned S168 report and outputs folder")
    source=json.loads(Path(sys.argv[1]).read_text())
    assert source["stage"]=="S168_NOVEL_TRAIN_FRAMES_GENERALIZATION_DIAGNOSTIC"
    assert len(source["cases"])==64 and source["model_accepted"] is False
    positive,negative=synthesize_frozen_cases(source["cases"])
    extra=new_adversarial(source["cases"])
    # No label or reference answer is read by selector; only question text.
    p=[safe_owner_value(case["question"]) for case in positive]
    n=[safe_owner_value(case["question"]) for case in negative]
    x=[safe_owner_value(case["question"]) for case in extra]
    original=[safe_owner_value(case["question"]) for case in source["cases"]]
    reference_positives,reference_negatives=failclosed_checks()
    # Scoring only after ALL inference calls.
    score=[{"id":case["source_id"],"question":case["question"],
            "candidate":candidate,"correct":candidate==case["posthoc_answer"],
            "posthoc_expected":case["posthoc_answer"]}
           for case,candidate in zip(positive,p)]
    controls=[{"id":case["source_id"],"kind":case["kind"],
               "question":case["question"],"candidate":candidate,
               "abstained":candidate is None}
              for case,candidate in zip([*negative,*extra],[*n,*x])]
    preserved=[{"id":case["id"],"posthoc_correct":candidate==case["answer"]}
               for case,candidate in zip(source["cases"],original)]
    report={
        "stage":"S174_NEGATION_VETO_CONJUNCTION_OFFLINE_ONLY",
        "source_S168_run":43289943629,
        "source_S173_run":43297453550,
        "source_S172_real_J7_run":43295694441,
        "new_form_64_correct":sum(s["correct"] for s in score),
        "new_form_64_abstained":sum(v is None for v in p),
        "adversarial_total":len(controls),
        "adversarial_abstained":sum(c["abstained"] for c in controls),
        "dangerous_false_accepts":sum(not c["abstained"] for c in controls),
        "frozen_S168_64_correct":sum(c["posthoc_correct"] for c in preserved),
        "frozen_S168_64_abstained":sum(v is None for v in original),
        "safe_offline_prompt_heuristic":(
            all(s["correct"] for s in score)
            and all(c["abstained"] for c in controls)
            and all(c["posthoc_correct"] for c in preserved)),
        "synthetic_S169_positive_count":sum(reference_positives),
        "synthetic_S169_negative_count":sum(reference_negatives),
        "cases":score,"negative_cases":controls,"original_S168_cases":preserved,
        "not_integrated_neural_runtime":True,"not_neural_generalization":True,
        "no_QA_TEST":True,"Green_Tunnel_weights_unchanged":True,
        "Jessica_Render_Supabase_LIVE_untouched":True,
        "model_accepted":False,
    }
    out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    (out/"S174_64_NEW_TRAIN_PLUS_64_NEGATIVES_OFFLINE.json").write_text(
        json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2))
    print(json.dumps({key:report[key] for key in (
        "stage","new_form_64_correct","new_form_64_abstained",
        "adversarial_total","adversarial_abstained",
        "dangerous_false_accepts","frozen_S168_64_correct",
        "safe_offline_prompt_heuristic","model_accepted")},
        sort_keys=True),flush=True)
    assert len(score)==64 and len(controls)==64 and len(preserved)==64


if __name__=="__main__":
    main()
