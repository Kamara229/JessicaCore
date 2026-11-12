"""CALIBRI S180 research: why S167 template parser abstained.

Read-only join of immutable S179 456 decoder trace and S87 authorized TRAIN
source ID family, AFTER original inference. No model, training, selector,
oracle answer routing, threshold selection, or LIVE/Jessica changes.

S164 parser is intentionally strict: four TRAIN S82 templates. Source
families outside S82_new are OUT OF THE KNOWN TEMPLATE DOMAIN, not bugs.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from calibri2.s180.offline_failure_attribution import analyze

FROZEN_FAMILIES = {"S30": 64, "S54": 64, "history": 8,
                   "S69": 32, "S73": 32, "S82_new": 256}
S164_TEMPLATE_FAMILY = "S82_new"


def analyze_parser_scope(trace: dict, labels: dict) -> dict:
    checked = analyze(trace, labels)
    if checked["full_eos_correct"] != 428 or checked["incorrect_count"] != 28:
        raise AssertionError("S180 immutable accepted S176 baseline drift")
    cases = {r["prompt_sha256"]: r for r in trace["cases"]}
    meta = {}
    family_coverage = Counter()
    for row in labels["all_456_per_identity_results"]:
        sha = row["question_sha256"]
        identity = row["id"]
        family = identity.split("/", 1)[0]
        if (family not in FROZEN_FAMILIES or sha in meta
                or not isinstance(identity, str)):
            raise AssertionError("S180 invalid S87 source family or hash")
        meta[sha] = family
        family_coverage[family] += 1
    if (set(meta) != set(cases) or
        family_coverage != Counter(FROZEN_FAMILIES)):
        raise AssertionError("S180 S87 source family ABI/cohort drift")
    errors = {x["prompt_sha256"]: x
              for x in checked["incorrect_sha_only"]}
    counts = defaultdict(Counter)
    trace_wrong = []
    for sha in sorted(cases):
        source = cases[sha]
        family = meta[sha]
        row = errors.get(sha)
        if row is None:
            group = "correct"
        elif row["first_token_error"]:
            group = "first_wrong"
        elif row["second_token_error"]:
            group = "second_wrong"
        else:
            group = "other_wrong"
        names = source["guard_events"]
        if not isinstance(names, list):
            raise AssertionError("S180 guard events not a list")
        abstain = "S167_owner_parser_abstained" in names
        parsed_family = family == S164_TEMPLATE_FAMILY
        stage = counts[family]
        stage["total"] += 1
        stage[group] += 1
        stage["parser_abstained"] += int(abstain)
        stage[group + "_parser_abstained"] += int(abstain)
        stage["known_S164_template_family"] += int(parsed_family)
        if group != "correct":
            trace_wrong.append({
                "prompt_sha256": sha, "family": family,
                "group": group, "route": source["route"],
                "parser_abstained": abstain,
                "known_S164_template_family": parsed_family,
                "guard_events": names,
            })
    if len(trace_wrong) != 28:
        raise AssertionError("S180 must cover all 28 frozen failures")
    second = [x for x in trace_wrong if x["group"] == "second_wrong"]
    if len(second) != 18 or not all(x["parser_abstained"] for x in second):
        raise AssertionError("S180 frozen 18 token2 parser abstains drift")
    outside = sum(not x["known_S164_template_family"] for x in second)
    in_template = 18 - outside
    return {
        "stage": "S180_S167_PARSER_SCOPE_AFTER_456_OFFLINE",
        "immutable_s179_run": 43323981995,
        "incorrect_total": 28,
        "second_token_misses": 18,
        "second_token_misses_parser_abstained": 18,
        "second_token_misses_outside_known_S164_family": outside,
        "second_token_misses_inside_known_S164_family": in_template,
        "source_family_totals": {f: dict(sorted(counts[f].items()))
                                 for f in sorted(FROZEN_FAMILIES)},
        "incorrect_cases_sha_only": trace_wrong,
        "gold_never_used_in_inference": True,
        "model_accepted": False,
        "no_training_or_modifications": True,
        "scope_explanation": "S164 regex only covers four original TRAIN S82 forms. Outside family is unsupported, NOT automatically parser defect. Inside family requires inspecting source grammar without using labels in selection.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("s179_frozen_trace", type=Path)
    p.add_argument("s87_frozen_labels", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("S180 source-family scope report must be immutable")
    r = analyze_parser_scope(
        json.loads(a.s179_frozen_trace.read_text()),
        json.loads(a.s87_frozen_labels.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8")
    print(json.dumps({
        "event": "S180_S167_PARSER_SCOPE",
        "second_wrong": 18,
        "outside_S164_known_template": r["second_token_misses_outside_known_S164_family"],
        "within_S164_known_template": r["second_token_misses_inside_known_S164_family"],
        "family_counts": {f: s.get("second_wrong", 0)
                          for f,s in r["source_family_totals"].items()},
        "model_accepted": False,
    }), flush=True)

if __name__ == "__main__":
    main()
