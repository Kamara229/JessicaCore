"""v192 locked grading facade: whole-answer machine status and pair accounting.

Truth label is consulted ONLY after extracting candidate value from the
actual answer and the original question. No gold-word scanning or substring
match. Prefixes, domain grammars and status rules defined before new QA.
"""
from collections import Counter,defaultdict

from . import contract as C
from .recognize import recognize


def grade(case,answer):
    """Return conservative machine result; raw answer preserved without changes."""
    for k in ("case_id","domain","question","gold"):
        if k not in case:raise ValueError("Incomplete frozen QA case: "+k)
    if case["domain"] not in C.DOMAINS:
        raise ValueError("Unsupported QA domain")
    result=recognize(case["domain"],case["question"],answer)
    if result is None:
        status="ungraded"
        candidate=None
        basis="whole_utterance_unparsed"
    else:
        candidate,basis=result
        gold=str(case["gold"]).casefold().strip()
        if case["domain"]=="arithmetic":
            if not gold.lstrip("+-").isdigit():
                raise ValueError("Arithmetic gold is not integer")
            gold=str(int(gold))
        status="machine_correct" if candidate.casefold()==gold else "machine_wrong"
    return {"case_id":case["case_id"],"domain":case["domain"],
        "grader_version":C.VERSION,"raw":answer,
        "candidate":candidate,"recognition_basis":basis,
        "machine_status":status,
        "machine_correct":True if status=="machine_correct" else
                           False if status=="machine_wrong" else None}


def summarize(cases,answers):
    if len(cases)!=len(answers) or not cases:raise ValueError("Missing original source QA replies")
    if len(set(x["case_id"] for x in cases))!=len(cases):
        raise AssertionError("Duplicate QA source case ID")
    if set(answers)!=set(c["case_id"] for c in cases):
        raise AssertionError("Actual original generated answers missing/extra")
    graded={}
    for c in cases:
        text=answers[c["case_id"]]
        graded[c["case_id"]]=grade(c,text)
    counts=Counter(row["machine_status"] for row in graded.values())
    if sum(counts.values())!=len(cases):
        raise AssertionError("Lost a model output")
    per_domain={}
    for domain in C.DOMAINS:
        rows=[x for x in graded.values() if x["domain"]==domain]
        if rows:
            sub=Counter(x["machine_status"] for x in rows)
            per_domain[domain]={"n":len(rows),"machine_correct":sub["machine_correct"],
                "machine_wrong":sub["machine_wrong"],"ungraded":sub["ungraded"]}
    pairs=defaultdict(list)
    for c in cases:
        if c.get("pair_id"):
            pairs[(c["domain"],c["pair_id"])].append(c)
    per_pair={}
    for (domain,pair),rows in sorted(pairs.items()):
        if len(rows)!=2 or rows[0]["gold"].casefold()==rows[1]["gold"].casefold():
            raise AssertionError("Invalid opposing test pair; needs two contrasting golds")
        statuses=[graded[x["case_id"]]["machine_status"] for x in rows]
        per_pair.setdefault(domain,[]).append({
             "pair_id":pair,"both_verified":all(s=="machine_correct" for s in statuses),
             "member_cases":[x["case_id"] for x in rows],"member_statuses":statuses})
    summary={"grader_version":C.VERSION,"n":len(graded),
        "machine_confirmed":counts["machine_correct"],
        "machine_recognized_wrong":counts["machine_wrong"],
        "ungraded_not_known_wrong":counts["ungraded"],
        "domain_totals":per_domain,
        "opposite_pairs":{dom:{"total":len(rows),
             "both_correct":sum(x["both_verified"] for x in rows),
             "individual_verdicts":rows} for dom,rows in per_pair.items()},
        "verbatim_answers_and_grades":[graded[c["case_id"]] for c in cases],
        "no_llm_based_judging":True,"no_model_training_or_compression":True}
    return summary
