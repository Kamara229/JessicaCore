"""Read-only independent S24 vs unmatched pretrained 135M FP32 audit.

Explains the misleading strict 0/96 reference score: public FP32 may
generate a correct color+noun WITHIN a longer reply but fail the strict
two-word-and-EOS format. Target phrase *presence* is not correctness,
and may even occur as part of a copied prompt/foil. Never use it for
learning, selecting a checkpoint, or asserting Green Tunnel quality.

Frozen QA and tokenizer are unmodified. No model loading, tests, LIVE.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from calibri2.s27.posthoc_frozen_answer_audit import (
    QA_SHA, SEALED_S24_SHA, NOVEL, DOMAINS,
    EXPECTED_BYTES, PARAMS, VALID_TWO, ensure_rows, read_json
)

def contains_exact_gold_phrase(row):
    gold=row["gold"]
    if not isinstance(gold,str) or not re.fullmatch(
        r"(red|blue|green|yellow) (fork|glove|shirt|key)",gold):
        raise AssertionError("Unexpected sealed novel gold phrase")
    # Only complete word boundaries, not arbitrary substrings.
    return bool(re.search(r"(?<![A-Za-z])"+re.escape(gold)+r"(?![A-Za-z])",
                          row["actual_text"],flags=re.IGNORECASE))

def audit(before,external):
    a,opened_a=ensure_rows(before,"sealed_S24_96_baseline")
    b,opened_b=ensure_rows(external,"external_pretrained_135M_FP32")
    if (before.get("frozen_suite_sha256")!=QA_SHA or
        external.get("S26_precommitted_frozen_qa_SHA256")!=QA_SHA or
        before.get("physical_archive_SHA256")!=SEALED_S24_SHA or
        before.get("complete_bytes")!=EXPECTED_BYTES or
        before.get("unique_params")!=PARAMS):
        raise AssertionError("S24 source or independent frozen QA mismatch")
    if (external.get("same_architecture_or_trained_initial_weights_as_36M") is not False or
        external.get("is_not_an_FP32_teacher_for_S26") is not True):
        raise AssertionError("Unmatched public FP32 misidentified as 36M")
    if set(a)!=set(b):raise AssertionError("Different fixed 144 QA IDs")
    for key in a:
        for field in ("question","gold","domain","expected_token_ids_full_EOS"):
            if a[key].get(field)!=b[key].get(field):
                raise AssertionError("Independent QA rows modified")
    ids=sorted(k for k,x in a.items() if x["domain"]==NOVEL)
    if len(ids)!=96:raise AssertionError("Expected 96 independent novel cases")
    rows=[]
    for key in ids:
        prev,pretrained=a[key],b[key]
        rows.append({
            "id":key,"gold":prev["gold"],
            "S24_strict_correct":prev["strict_full_EOS"],
            "public135m_FP32_strict_correct":pretrained["strict_full_EOS"],
            "S24_target_phrase_mentioned":contains_exact_gold_phrase(prev),
            "public135m_FP32_target_phrase_mentioned":contains_exact_gold_phrase(pretrained),
            "S24_exact_two_word_format":bool(VALID_TWO.fullmatch(prev["actual_text"])),
            "public135m_FP32_exact_two_word_format":bool(VALID_TWO.fullmatch(pretrained["actual_text"])),
            "S24_generated_text":prev["actual_text"],
            "public135m_FP32_generated_text":pretrained["actual_text"],
            "S24_emitted_EOS":prev.get("EOS_emitted"),
            "public135m_FP32_emitted_EOS":pretrained.get("EOS_emitted")
        })
    controls={}
    for domain,expected in DOMAINS.items():
        ids_in_domain=[k for k in a if a[k]["domain"]==domain]
        controls[domain]={
           "n":expected,
           "S24_strict":sum(bool(a[k]["strict_full_EOS"]) for k in ids_in_domain),
           "unmatched_pretrained_FP32_strict":
               sum(bool(b[k]["strict_full_EOS"]) for k in ids_in_domain),
        }
    return {
        "stage":"S27_FROZEN_PRE_S26_PUBLIC_FP32_STRICT_VS_TARGET_MENTION",
        "frozen_QA_SHA256":QA_SHA,
        "complete_S24_bpp":8*EXPECTED_BYTES/PARAMS,
        "unmatched_pretrained_FP32_is_NOT_compression_retention_measure":True,
        "target_phrase_mentioned_is_NOT_answer_correctness":True,
        "exact_full_two_word_plus_EOS_is_primary_metric":True,
        "models":("S24_sealed_36M_random_init_trained",
                  "135M_separately_pretrained_public_FP32"),
        "frozen_control_strict":controls,
        "novel": {
            "n":len(rows),
            "S24_full_correct":sum(r["S24_strict_correct"] for r in rows),
            "public_pretrained_FP32_full_correct":sum(r["public135m_FP32_strict_correct"] for r in rows),
            "S24_gold_phrase_mentioned_anywhere":sum(r["S24_target_phrase_mentioned"] for r in rows),
            "public_pretrained_FP32_gold_phrase_mentioned_anywhere":sum(r["public135m_FP32_target_phrase_mentioned"] for r in rows),
            "S24_two_word_format":sum(r["S24_exact_two_word_format"] for r in rows),
            "public_pretrained_FP32_two_word_format":sum(r["public135m_FP32_exact_two_word_format"] for r in rows),
            "S24_EOS":sum(bool(r["S24_emitted_EOS"]) for r in rows),
            "public_pretrained_FP32_EOS":sum(bool(r["public135m_FP32_emitted_EOS"]) for r in rows),
        },
        "all_96_real_observed_answers":rows,
        "all_16_open_real_observed_answers":[{
           "question":opened_a[i].get("question"),
           "S24_text":opened_a[i]["actual_answer"],
           "public_FP32_text":opened_b[i]["actual_answer"],
        } for i in range(16)],
        "no_s26_training_or_selection_from_this_result":True,
        "all_negative_results_preserved":True
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s24",type=Path,required=True)
    p.add_argument("--unmatched-fp32",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit(read_json(a.s24),read_json(a.unmatched_fp32))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"stage":result["stage"],
        "novel":result["novel"],
        "frozen_control_strict":result["frozen_control_strict"],
        "NOT_semantic_parity":"public_135M_FP32_is_unmatched",
        "no_heldout_training":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
