"""Offline audit of S3 TRAIN-only answers; never reads DEV/VALID/QA/TEST.

Usage: python research/calibri2/s0/audit_train_answers.py ACTUAL_CALIBRI2_S3_TRAIN.json
Produces deterministic JSON on stdout. This is a diagnostic, not a training
checkpoint selector. It does not run a model or change weights.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(payload):
    if payload.get("TRAIN_only_no_QA_VALID_checkpoint_selection") is not True:
        raise ValueError("Expected explicitly TRAIN-only S3 artifact")
    stages = list(payload.get("milestones", []))
    final = payload.get("final_reloaded_probe")
    if final:
        stages.append(final)
    report = []
    for stage in stages:
        rows = stage["actual_answers"]
        grouped = defaultdict(dict)
        first = Counter()
        row_details = []
        errors = Counter()
        for row in rows:
            if not {"id", "side", "generated_ids", "gold_ids", "full_with_EOS"} <= row.keys():
                raise ValueError("Missing per-answer TRAIN fields")
            ids = row["generated_ids"]
            gold = row["gold_ids"]
            if not ids or not gold:
                raise ValueError("Empty generated or expected token sequence")
            key = (row["domain"], row["id"])
            if row["side"] not in {"left", "right"} or row["side"] in grouped[key]:
                raise ValueError(f"Duplicate or invalid opposing side for TRAIN pair {key}: {row['side']}")
            if not isinstance(ids, list) or not isinstance(gold, list):
                raise ValueError("Expected token ID lists")
            if bool(row["full_with_EOS"]) and ids != gold:
                raise ValueError(f"Incorrect full_with_EOS flag for TRAIN row {key}")
            grouped[key][row["side"]] = bool(row["full_with_EOS"])
            first[ids[0]] += 1
            row_details.append({
                "domain": row["domain"], "pair_id": row["id"], "side": row["side"],
                "generated_ids": ids, "gold_ids": gold,
                "exact_full_EOS": ids == gold,
                "first_token_correct": ids[0] == gold[0],
                "expected_EOS_at_end": ids[-1] == gold[-1],
            })
            if not row["full_with_EOS"]:
                if gold[-1] not in ids:
                    errors["expected_EOS_token_absent"] += 1
                if len(ids) > 1 and len(set(ids)) == 1:
                    errors["constant_token_repetition"] += 1
                if len(ids) >= 4 and any(ids[i:i+2] == ids[i+2:i+4] for i in range(len(ids)-3)):
                    errors["adjacent_bigram_repetition"] += 1
                if ids[0] != gold[0]:
                    errors["wrong_first_token"] += 1
                else:
                    errors["correct_first_token_but_full_answer_wrong"] += 1
        if len(rows) != 2 * len(grouped) or any(set(s) != {"left", "right"} for s in grouped.values()):
            raise ValueError("Every TRAIN pair must have both contrasting sides")
        pair_counts = Counter(
            "both" if sides["left"] and sides["right"] else
            "neither" if not sides["left"] and not sides["right"] else
            "left_only" if sides["left"] else "right_only"
            for sides in grouped.values()
        )
        report.append({
            "stage": stage.get("stage"), "step": stage.get("step"),
            "full_EOS": sum(bool(r["full_with_EOS"]) for r in rows),
            "total": len(rows), "unique_first_tokens": len(first),
            "opposing_pair_retention": dict(pair_counts),
            "nonexclusive_error_flags": dict(errors),
            "decoded_text_available": False,
            "per_answer_token_audit": row_details,
            "EOS_note": "EOS ID inferred from each TRAIN gold sequence; presence does not prove correct EOS position",
        })
    return {"scope": "TRAIN_ONLY", "not_checkpoint_selection": True,
            "requires_decoded_text_for_semantic_repetition": True,
            "milestones": report}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("train_json", type=Path)
    args = p.parse_args()
    print(json.dumps(audit(json.loads(args.train_json.read_text())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
