#!/usr/bin/env python3
"""CALIBRI S169: source-only segmented owner→two-token phrase offline analysis.

This is a deterministic research grammar heuristic, NOT a learned generalizing
neural network. No model loading/weight updates, no QA/TEST, no LIVE, no Jessica.
No ground-truth answer or expected token ids are accessible inside the selector.
"""
import argparse
import json
import re
from pathlib import Path

PREFIX = re.compile(r"^TRAIN S82 [^:]+:\s*")
NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
VALUE = re.compile(r"(?<![A-Za-z])(?:red|green|blue|yellow) [a-z]+\b")
SEPARATOR = re.compile(r"\s*;\s*|\.\s+|,\s+(?:while|whereas)\s+")

def owner_value(question):
    """Choose an owner from the question, bind it to one uniquely scoped phrase.

    Fail closed if no exact requested owner, two distinct source values,
    unambiguous segmentation, or exactly one value belonging to that owner.
    """
    if not isinstance(question, str):
        return None
    clean = PREFIX.sub("", question, count=1)
    if ". " not in clean:
        return None
    source, request = clean.rsplit(". ", 1)
    # An owner name must be the final named word of the request.
    req_names = NAME.findall(request)
    if not req_names:
        return None
    asked = req_names[-1]
    if not re.search(r"\b" + re.escape(asked) + r"(?:\x27s package)?[.?]?$", request):
        return None
    segments = [s.strip() for s in SEPARATOR.split(source) if s.strip()]
    all_values = [m.group(0) for m in VALUE.finditer(source)]
    # Exactly two DISTINCT explicit color/item labels, no hidden choice.
    if len(all_values) != 2 or len(set(all_values)) != 2:
        return None
    matches = []
    for seg in segments:
        if not re.search(r"\b" + re.escape(asked) + r"\b", seg):
            continue
        values = VALUE.findall(seg)
        if len(values) != 1:
            return None
        # Exact source alignment: a single requested owner and a single phrase.
        matches.append(values[0])
    if len(matches) != 1:
        return None
    return matches[0]


def failclosed_checks():
    positive = [
        ("TRAIN S82 inventory: The record for Folemar displays green desk. The separate record for Bintan displays green lamp. Return the two-word value of Folemar.", "green desk"),
        ("TRAIN S82 workshop note: Worker Bintan has key blue wheel; worker Folemar instead has key blue tent. State exactly the assigned two words for Folemar.", "blue tent"),
        ("TRAIN S82 permit index: Under Folemar the color-item phrase is red tree, whereas under Bintan it is red book. Copy the complete color and item for Folemar.", "red tree"),
        ("TRAIN S82 packing label: Package owner Bintan received yellow river. Package owner Folemar received yellow shoe. What is the exact two-word label on Folemar's package?", "yellow shoe"),
        ("TRAIN S82 inventory: Folemar has the color-item label green desk, while Bintan has green lamp. Give the two words recorded for Folemar.", "green desk"),
        ("TRAIN S82 workshop note: The key for worker Folemar is blue wheel; the key for worker Bintan is blue tent. Write the full two-word key of Bintan.", "blue tent"),
        ("TRAIN S82 permit index: Bintan is listed with red tree; Folemar is listed with red book. Which two-word color-item belongs to Folemar?", "red book"),
        ("TRAIN S82 packing label: The package marked yellow river belongs to Bintan; yellow shoe belongs to Folemar. State the color and item owned by Folemar.", "yellow shoe"),
    ]
    negatives = [
        "TRAIN S82 inventory: Folemar has green desk; Bintan has green lamp. Give the two words recorded for Stranger.",
        "TRAIN S82 inventory: Folemar has green desk; Folemar has green lamp. Give the two words recorded for Folemar.",
        "TRAIN S82 inventory: Folemar has green desk; Bintan has green desk. Give the two words recorded for Folemar.",
        "TRAIN S82 inventory: Folemar has green desk; Bintan has green lamp. Give the two words recorded for Folemar. extra",
        "TRAIN S82 inventory: Folemar has green desk and blue desk; Bintan has green lamp. Give the two words recorded for Folemar.",
        "TRAIN S82 inventory: Folemar has green desk; Bintan has green lamp; Renavi has yellow shoe. Give the two words recorded for Folemar.",
        "unknown question",
    ]
    positive_results = [owner_value(q) == a for q, a in positive]
    negative_results = [owner_value(q) is None for q in negatives]
    return positive_results, negative_results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("s168_report",type=Path)
    p.add_argument("outdir",type=Path)
    args=p.parse_args()
    source=json.loads(args.s168_report.read_text(encoding="utf-8"))
    assert source["stage"]=="S168_NOVEL_TRAIN_FRAMES_GENERALIZATION_DIAGNOSTIC"
    assert source["parent_run"]==43287922683
    assert source["no_QA_TEST"] and source["no_LIVE"]
    assert source["model_accepted"] is False
    rows=source["cases"]
    assert len(rows)==64 and len({r["id"] for r in rows})==64
    positive_results,negative_results=failclosed_checks()
    # The selector sees ONLY the question. Answers are scored AFTER extraction.
    candidates=[owner_value(row["question"]) for row in rows]
    scored=[{
        "id":row["id"],
        "parent_neural_full_correct":row["parent_correct"],
        "S167_candidate_neural_full_correct":row["candidate_correct"],
        "S169_offline_candidate":candidate,
        "S169_offline_correct":candidate==row["answer"],
        "S169_offline_abstained":candidate is None,
    } for row,candidate in zip(rows,candidates)]
    correct=sum(r["S169_offline_correct"] for r in scored)
    abstain=sum(r["S169_offline_abstained"] for r in scored)
    report={
        "stage":"S169_SOURCE_SEGMENT_OWNER_BINDING_OFFLINE_ONLY",
        "S168_immutable_run":43289943629,
        "S168_neural_full_EOS_correct":source["candidate_full_correct"],
        "S169_offline_parser_exact_phrase_correct":correct,
        "S169_offline_parser_abstains":abstain,
        "synthetic_positive_passed":sum(positive_results),
        "synthetic_positive_count":len(positive_results),
        "negative_abstention_passed":sum(negative_results),
        "negative_abstention_count":len(negative_results),
        "S169_only_source_question_at_inference":True,
        "no_gold_in_selector":True,
        "not_integrated_with_neural_runtime":True,
        "not_neural_generalization_proof":True,
        "no_new_training_or_gradients":True,
        "no_QA_TEST_or_LIVE_or_Jessica":True,
        "model_accepted":False,
        "cases":scored
    }
    args.outdir.mkdir(parents=True,exist_ok=True)
    (args.outdir/"S169_64_CASE_SOURCE_ONLY_OFFLINE_DIAGNOSTIC.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({k:report[k] for k in (
        "stage","S168_neural_full_EOS_correct",
        "S169_offline_parser_exact_phrase_correct",
        "S169_offline_parser_abstains","synthetic_positive_passed",
        "negative_abstention_passed","model_accepted")},sort_keys=True),flush=True)
    assert all(positive_results), "S169 source grammar failed known positive example"
    assert all(negative_results), "S169 fail-closed negative example failed"
    # Don't gate on 64/64; REPORT misses and learn from them without gold-leak.
if __name__=="__main__":
    main()
