"""Audit v96's false-negative exact-prefix judge without changing source models.

This report is EVIDENCE, not a trained answer grader: a key appearing in
the FIRST visible response is necessary evidence, NOT proven semantic truth.
Never scan second/third chat turns, never use as a model-selection metric.
"""
from __future__ import annotations
import json,re
from pathlib import Path

ROLE_LINE=re.compile(r"(?im)^\s*(?:system|user|assistant)\s*$")
START_LINE=re.compile(r"(?im)^(?:\s*(?:assistant|a:)\s*\n)+")
def first_response(text):
    text=(text or "").lstrip()
    # Strip a leading 'assistant' or 'A:' role heading, but not content.
    while True:
        m=re.match(r"(?i)^(assistant|a:)\s*\n",text)
        if not m:break
        text=text[m.end():].lstrip()
    # Never award a response for a word in a later fake user/system turn.
    m=ROLE_LINE.search(text)
    if m:text=text[:m.start()]
    return text.strip()

def leading_reference_signal(text,reference):
    first=first_response(text)
    # Limit to first written answer sentence, not arbitrary later output.
    head=re.split(r"(?<=[.!?])\s+",first,1)[0][:180]
    # Whole-token presence only; '122' is not '22'.
    matches=[r for r in reference if re.search(r"(?i)(?<![a-z0-9])"+re.escape(str(r))+r"(?![a-z0-9])",head)]
    return {"first_answer_excerpt":head,
            "reference_present_in_first_answer_sentence":bool(matches),
            "reference_match_candidates":matches,
            "not_proof_of_correctness_without_semantic_judgement":True}

def audit(report):
    if report.get("experiment")!="CALIBRI_v96_original_answer_and_semantic_content_canary_audit":
        raise AssertionError("only exact frozen v96 model-inference artifact")
    if report.get("previous_official_TEST_never_loaded") is not True:
        raise AssertionError("independent TEST provenance compromised")
    models=report["full_original_models_genuine_greedy_decoding"]
    rows={};summary={}
    for name,data in models.items():
        all_rows=[]
        for item in data["questions"]:
            visible=leading_reference_signal(item["actual_generated_text"],item["graded"]["answer_reference"])
            all_rows.append({"prompt_SHA":item["prompt_SHA"],"strict_prefix_correct":item["graded"]["correct"],
                             "visible_reference_signal":visible})
        rows[name]=all_rows
        summary[name]={"n":len(all_rows),
            "strict_prefix_correct":sum(x["strict_prefix_correct"] for x in all_rows),
            "reference_present_in_first_sentence":sum(x["visible_reference_signal"]["reference_present_in_first_answer_sentence"] for x in all_rows)}
    if len({tuple(x["prompt_SHA"] for x in v) for v in rows.values()})!=1:
        raise AssertionError("nonpaired original model prompts")
    return {"experiment":"CALIBRI_v96_scoring_false_negative_audit_only",
        "source_experiment":report["experiment"],
        "counts":summary,"first_answer_excerpts":rows,
        "previous_scientific_conclusion":"v96 original exact-prefix scorer reported 0/8 on all three models",
        "correct_interpretation":"FP32 visibly contains answers later in first response, so 0/8 was a grading artifact; reference presence is not a substitute for semantic correctness.",
        "new_models_trained_or_reweighted":False,
        "source_test_reopened":False,
        "external_official_TEST_not_read":True}

def run(source:Path,out:Path):
    result=audit(json.loads(source.read_text()))
    out.mkdir(parents=True,exist_ok=True)
    dest=out/"v96_first_answer_reference_audit.json"
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result["counts"],ensure_ascii=False),flush=True)
    return result
