"""Conservative full-answer grader for new frozen 64 public development prompts.

Key rule: an answer is machine-correct only if the *whole answer*
unambiguously expresses the expected token/number with an allowed
neutral lead-in. An incidental gold word, negated correct answer,
or verbose clause is NOT automatically correct. If recognized as
a different valid same-domain answer it is wrong, otherwise UNGRADED.
The model receives no grade or prompt examples during any training.
"""
import re
from .cases import FACTS

MIXED_PREFIX=r"(?:the\s+answer\s*(?:is|:)\s*|answer\s*:\s*|the\s+result\s+is\s*|result\s*:\s*|it\s+is\s*|it's\s*)?"
COLOR_LEX=frozenset(("purple","orange","green","blue","yellow","pink","brown","black",
                     "gray","white","red","turquoise","silver","violet","gold","maroon"))
FACT_LEX=frozenset(a.casefold() for _,a in FACTS)


def _one_value(value):
    value=re.sub(r"\s+"," ",value.strip().casefold())
    value=value.strip(" \t\r\n\"'").rstrip(".! ").strip()
    for rule in (r"^the\s+answer\s+(?:is|:)\s*",
                 r"^answer\s*:\s*",
                 r"^the\s+result\s+is\s*",
                 r"^result\s*:\s*",
                 r"^it\s+is\s*",
                 r"^it's\s*"):
        value=re.sub(rule,"",value,count=1)
    return value.strip(" \t\r\n\"'").rstrip(".! ").strip()


def grade(case,response):
    raw=response.get("verbatim_text","")
    if not isinstance(raw,str):
        raise ValueError("Source tokenizer did not return model text")
    answer=_one_value(raw)
    gold=str(case["gold"]).casefold()
    domain=case["domain"]
    if domain=="arithmetic":
        recognized=bool(re.fullmatch(r"[-+]?[0-9]+",answer))
    elif domain=="opposite_yesno":
        recognized=answer in ("yes","no")
    elif domain=="fictional_binding":
        recognized=answer in COLOR_LEX
    elif domain=="factual_short":
        recognized=answer in FACT_LEX
    else:raise ValueError("Unknown original source QA domain")
    if recognized:
        status="machine_correct" if answer==gold else "machine_wrong"
    else:
        status="ungraded"
    return {"case_id":case["case_id"],"domain":domain,
        "gold":case["gold"],"raw_text":raw,
        "recognized_answer":answer if recognized else None,
        "machine_status":status,
        "machine_correct":True if status=="machine_correct" else
                          False if status=="machine_wrong" else None,
        "nonexact_gold_occurrence_ungraded":status=="ungraded" and
                           bool(re.search(r"\b"+re.escape(gold)+r"\b",raw.casefold()))}


def summarize(cases,graded):
    if len(cases)!=len(graded):
        raise AssertionError("Incomplete actual original model response audit")
    lookup={x["case_id"]:x for x in graded}
    result={"total":len(graded),
        "confirmed":sum(x["machine_status"]=="machine_correct" for x in graded),
        "parsed_wrong":sum(x["machine_status"]=="machine_wrong" for x in graded),
        "ungraded":sum(x["machine_status"]=="ungraded" for x in graded),
        "gold_word_appears_but_ungraded":sum(
            x["nonexact_gold_occurrence_ungraded"] for x in graded),
        "domains":{},"paired":{}}
    for domain in sorted(set(x["domain"] for x in cases)):
        rows=[lookup[x["case_id"]] for x in cases if x["domain"]==domain]
        result["domains"][domain]={
            "n":len(rows),"machine_confirmed":sum(x["machine_correct"] is True for x in rows),
            "machine_wrong":sum(x["machine_correct"] is False for x in rows),
            "ungraded":sum(x["machine_correct"] is None for x in rows)}
    for domain in ("opposite_yesno","fictional_binding"):
        groups={}
        for case in cases:
            if case["domain"]==domain:
                groups.setdefault(case["pair_id"],[]).append(lookup[case["case_id"]])
        if any(len(rows)!=2 for rows in groups.values()):
            raise AssertionError("Missing opposing balanced QA case")
        result["paired"][domain]={
            "total_pairs":len(groups),
            "both_machine_correct":sum(all(x["machine_correct"] is True for x in rows)
                                       for rows in groups.values()),
            "one_correct_only":sum(sum(x["machine_correct"] is True for x in rows)==1
                                   for rows in groups.values()),
            "no_proven_pair_success":sum(not all(x["machine_correct"] is True for x in rows)
                                         for rows in groups.values())}
    if result["confirmed"]+result["parsed_wrong"]+result["ungraded"]!=len(graded):
        raise AssertionError("Invalid answer status conservation")
    return result
