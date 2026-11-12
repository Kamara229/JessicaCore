"""CALIBRI S180 posthoc copy-net/guard association (not a causal intervention).

Only frozen S179 inference trace and S87 456 TRAIN gold, joined OFFLINE.
No training, model execution, thresholds, new candidates, or LIVE changes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from calibri2.s180.offline_failure_attribution import analyze

GUARD_PREFIXES = ("S160_", "S161_", "S167_")


def assess(trace_archive: dict, label_archive: dict) -> dict:
    baseline = analyze(trace_archive, label_archive)
    if baseline["full_eos_correct"] != 428 or baseline["incorrect_count"] != 28:
        raise AssertionError("S180 frozen 428/456 parity broken")
    labels = {x["question_sha256"]: x["target_ids"]
              for x in label_archive["all_456_per_identity_results"]}
    rows = {x["prompt_sha256"]: x for x in trace_archive["cases"]}
    if len(rows) != 456 or set(rows) != set(labels):
        raise AssertionError("S180 original prompt cohort coverage broken")
    groups = {"full_correct": Counter(), "first_wrong": Counter(),
              "second_wrong_first_right": Counter(), "other_wrong": Counter()}
    guard_counts = defaultdict(Counter)
    cross = defaultdict(Counter)
    raw_mismatch_sha = []
    copy_override_sha = []
    for sha in sorted(rows):
        row = rows[sha]
        gen, gold = row["full_generated_ids"], labels[sha]
        if gen == gold:
            bucket = "full_correct"
        elif gen[0] != gold[0]:
            bucket = "first_wrong"
        elif len(gen) < 2 or gen[1] != gold[1]:
            bucket = "second_wrong_first_right"
        else:
            bucket = "other_wrong"
        stats = groups[bucket]
        stats["total"] += 1
        route = row["route"]
        stats["route_" + route] += 1
        events = row.get("guard_events")
        if not isinstance(events, list) or any(not isinstance(e, str) for e in events):
            raise AssertionError("S180 malformed guard events")
        eventset = set(events)
        if len(eventset) != len(events):
            raise AssertionError("S180 duplicate guard event")
        for event in eventset:
            guard_counts[bucket][event] += 1
        override = "neural_override_applied" in eventset
        offered = "candidate_present" in eventset
        fastpath = "fastpath_no_autoregressive_step2" in eventset
        fallback = "S160_confidence_fallback" in eventset
        protected = "S161_same_attribute_conflict_parent_fallback" in eventset
        rescued = "S167_confidence_fallback_rescued" in eventset
        parser_abstained = "S167_owner_parser_abstained" in eventset
        for name, active in (
            ("copy_net_candidate", offered),
            ("copy_net_override", override),
            ("parallel_fastpath", fastpath),
            ("low_confidence_fallback", fallback),
            ("same_attribute_conflict", protected),
            ("exact_owner_rescue", rescued),
            ("owner_parser_abstain", parser_abstained),
        ):
            if active:
                stats[name] += 1
                cross[name][bucket] += 1
        if override and not offered:
            raise AssertionError("S180 copy-net override without candidate")
        if route != "S93":
            if (row["conditional_step2"] is not None or offered or override
                    or fastpath or any(e.startswith(GUARD_PREFIXES) for e in eventset)):
                raise AssertionError("S180 non-S93 trace must not fake S93 internals")
        cond = row.get("conditional_step2")
        if fastpath and cond is not None:
            raise AssertionError("S180 parallel fastpath has no conditional AR token2")
        if cond is None:
            stats["no_actual_conditional_second_logits"] += 1
            continue
        if cond.get("score_type") != "actual_ar_decoder_conditional_step2_logits":
            raise AssertionError("S180 surrogate token2 scores forbidden")
        top = cond.get("top5_token_ids_and_logits")
        if not top or not isinstance(top[0][0], int):
            raise AssertionError("S180 malformed raw token2 top1")
        stats["actual_conditional_second_logits"] += 1
        raw_match = len(gen) > 1 and top[0][0] == gen[1]
        if raw_match:
            stats["raw_top1_matches_generated_second"] += 1
        else:
            stats["raw_top1_differs_generated_second"] += 1
            raw_mismatch_sha.append({
                "prompt_sha256": sha, "group": bucket, "copy_override": override,
                "copy_candidate": offered, "route": route})
            if not override:
                # Could be another frozen downstream rule; cannot infer copy-net causality.
                stats["raw_final_disagreement_without_logged_copy_override"] += 1
        if override:
            copy_override_sha.append({
                "prompt_sha256": sha, "group": bucket,
                "raw_top1_differs_from_final": not raw_match})
    if ([groups[x]["total"] for x in
         ("full_correct", "first_wrong", "second_wrong_first_right", "other_wrong")]
            != [428, 9, 18, 1]):
        raise AssertionError("S180 exact immutable error partition drift")
    return {
        "stage": "S180_COPY_NET_GUARD_OBSERVATIONAL_456",
        "source_s179_run_id": 43323981995,
        "trace_cases": 456,
        "group_counts": {
            k: dict(sorted(c.items())) for k, c in groups.items()
        },
        "guard_events_by_group": {
            k: dict(sorted(v.items())) for k, v in guard_counts.items()
        },
        "event_x_group": {
            k: dict(sorted(v.items())) for k, v in cross.items()
        },
        "raw_logit_disagreement_sha_only": raw_mismatch_sha,
        "copy_net_override_sha_only": copy_override_sha,
        "gold_joined_only_after_inference": True,
        "no_counterfactual_ablation": True,
        "no_causal_claim_or_threshold_tuning": True,
        "no_training_or_model_invocation": True,
        "model_accepted": False,
        "notes": [
            "Copy-net candidate and applied override are different frozen decoder flags.",
            "Guard event records do not prove guard caused answer correctness/error.",
            "No observational logits on non-S93 routes or accepted parallel fastpaths.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frozen_s179_trace", type=Path)
    p.add_argument("frozen_s87_train_labels", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("S180 offline audit artifact must be immutable")
    out = assess(json.loads(a.frozen_s179_trace.read_text()),
                 json.loads(a.frozen_s87_train_labels.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "S180_COPY_NET_GUARDS_OBSERVED_OFFLINE",
        "group_counts": {k: v["total"] for k, v in out["group_counts"].items()},
        "event_x_group": out["event_x_group"],
        "raw_mismatch": len(out["raw_logit_disagreement_sha_only"]),
        "copy_override": len(out["copy_net_override_sha_only"]),
        "model_accepted": False,
    }), flush=True)


if __name__ == "__main__":
    main()
