"""CALIBRI S87-J4: read-only TRAIN-only first/second/EOS error attribution.

Consumes only the existing immutable S87-J4 456-authorized-TRAIN guard
report. Never reads prompts or answers from any inspected independent QA.
No model inference, training, model selection, promotion or LIVE changes.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(observed: list[int], gold: list[int]) -> str:
    if not observed or observed[0] != gold[0]:
        return "wrong_first_token"
    if observed[:len(gold) - 1] != gold[:-1]:
        return "correct_first_wrong_later_content"
    if observed != gold:
        return "correct_content_wrong_eos_or_trailing"
    return "exact_full_plus_eos"


def tally(rows, key: str):
    stages = collections.Counter()
    first = second = body = exact = 0
    expected_token_lengths = collections.Counter()
    for row in rows:
        gold = row["target_ids"]
        observed = row[key]
        if not isinstance(gold, list) or not isinstance(observed, list):
            raise AssertionError("Missing expected/observed token arrays")
        if len(gold) < 2:
            raise AssertionError("Expected at least one content token and EOS")
        expected_token_lengths[str(len(gold))] += 1
        stages[classify(observed, gold)] += 1
        first += bool(observed and observed[0] == gold[0])
        second += bool(len(gold) >= 3 and len(observed) >= 2
                       and observed[0] == gold[0] and observed[1] == gold[1])
        body += observed[:len(gold) - 1] == gold[:-1]
        exact += observed == gold
    if sum(stages.values()) != len(rows):
        raise AssertionError("Stage counts must partition TRAIN rows")
    return {
        "total": len(rows),
        "stages": dict(stages),
        "first_correct": first,
        "first_and_second_correct": second,
        "all_content_correct": body,
        "full_plus_eos_correct": exact,
        "conditional_second_given_correct_first": second / first if first else None,
        "conditional_full_given_correct_content": exact / body if body else None,
        "expected_token_lengths": dict(expected_token_lengths),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("guard_report", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    report = json.loads(a.guard_report.read_text())
    if report.get("stage") != "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT":
        raise AssertionError("This diagnostic requires the immutable 456 TRAIN guard")
    if not report.get("all_inspected_QA_embargoed"):
        raise AssertionError("Prior inspected QA embargo must stay active")
    rows = report["all_456_per_identity_results"]
    if len(rows) != 456 or len({r["id"] for r in rows}) != 456:
        raise AssertionError("TRAIN cohort not exactly 456 unique identities")
    by_family = collections.defaultdict(list)
    changed = collections.Counter()
    for row in rows:
        old, new, gold = row["S82_ids"], row["S87_ids"], row["target_ids"]
        family = row["id"].split("/")[0]
        by_family[family].append(row)
        if bool(old == gold) != bool(row["old_full"]):
            raise AssertionError("S82 previous correctness flag inconsistent")
        if bool(new == gold) != bool(row["new_full"]):
            raise AssertionError("J4 correctness flag inconsistent")
        if old != new:
            changed[family] += 1
    expected = {"S30": 64, "S54": 64, "history": 8,
                "S69": 32, "S73": 32, "S82_new": 256}
    if {k: len(v) for k, v in by_family.items()} != expected:
        raise AssertionError("TRAIN provenance family counts changed")
    attribution = {
        "stage": "S87_J4_TRAIN_ONLY_ERROR_ATTRIBUTION_READ_ONLY",
        "source_guard_sha256": sha256(a.guard_report),
        "no_QA_used": True,
        "QA_permanently_embargoed": True,
        "no_model_inference_or_weight_changes": True,
        "original_456_guard_retention": report["full_safety_guard"],
        "physical_bpp": report["all_inclusive_bpp"],
        "pytorch_cpu_speed_ratio": report[
            "cpu_speed_ratio_48_postwarmup_per_token_median_S82_over_S87"],
        "old": tally(rows, "S82_ids"),
        "j4": tally(rows, "S87_ids"),
        "changed_output_ids_by_family": dict(changed),
        "by_family": {
            name: {"s82": tally(family_rows, "S82_ids"),
                   "j4": tally(family_rows, "S87_ids")}
            for name, family_rows in by_family.items()
        },
        "hypothesis_for_future_TRAIN_only_research":
            "Investigate conditional second-token copying and EOS under "
            "TRAIN-only compositional perturbations of authorized labels. "
            "Potential frozen S82 pointer bottleneck; no claim of causality "
            "from these aggregate counts and no inspected QA tuning.",
        "candidate_accepted": False,
        "no_LIVE_changes": True,
    }
    if attribution["old"]["full_plus_eos_correct"] != 196:
        raise AssertionError("Original S82 full+EOS TRAIN baseline changed")
    if attribution["j4"]["full_plus_eos_correct"] != 198:
        raise AssertionError("J4 confirmed +2 full+EOS TRAIN outcome changed")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "S87_J4_TRAIN_ERROR_ATTRIBUTION.json").write_text(
        json.dumps(attribution, indent=2, sort_keys=True))
    print(json.dumps({
        "event": "S87_J4_TRAIN_ERROR_ATTRIBUTION_DONE",
        "s82": attribution["old"]["stages"],
        "j4": attribution["j4"]["stages"],
        "first_s82": attribution["old"]["first_correct"],
        "first_j4": attribution["j4"]["first_correct"],
        "second_after_first_s82": attribution["old"]["first_and_second_correct"],
        "second_after_first_j4": attribution["j4"]["first_and_second_correct"],
        "bpp": attribution["physical_bpp"],
        "QA_inspected": False,
        "candidate_accepted": False,
    }), flush=True)


if __name__ == "__main__":
    main()
