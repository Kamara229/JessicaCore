"""CALIBRI v201: read-only fault taxonomy from *already sealed* v200 native QA.

Never perform a new QA inference, change any training values, or use this
public DEV slice to select a compressed source. This is causal *triage*,
not identification of true causality, a new held-out test, or quality parity.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ARMS = ("FP32_existing_135M", "CB186", "GT200")
STATUS = ("machine_correct", "machine_wrong", "ungraded")


def _longest_run(items):
    if not items:
        return 0
    best = run = 1
    for a, b in zip(items, items[1:]):
        run = (run + 1) if a == b else 1
        best = max(best, run)
    return best


def _source_rows(document, arm):
    if arm == "FP32_existing_135M":
        return {int(row["case_id"]): {"verbatim_text": row["verbatim_text"]}
                for row in document["original_actual_135M_FP32_from_previous_sealed_v193"]}
    return {int(row["case_id"]): row["actual_output"]
            for row in document["native_compact_actual_QA"][arm]}


def _grades(document, arm):
    score = document["scores"][arm]
    rows = score["verbatim_answers_and_grades"]
    mapped = {int(r["case_id"]): r for r in rows}
    if len(mapped) != len(rows):
        raise AssertionError("Repeated case grades: " + arm)
    if len(rows) != 16 or score["n"] != 16:
        raise AssertionError("Incorrect frozen v200 DEV size")
    counts = Counter(r["machine_status"] for r in rows)
    if set(counts) - set(STATUS):
        raise AssertionError("Invalid pinned grader label")
    if (counts["machine_correct"] != score["machine_confirmed"]
            or counts["machine_wrong"] != score["machine_recognized_wrong"]
            or counts["ungraded"] != score["ungraded_not_known_wrong"]):
        raise AssertionError("v200 pinned grader aggregate mismatch")
    return mapped


def analyze(document):
    if (not document.get("complete")
            or not document.get("not_new_blind_test_public_dev_only")
            or not document.get("score_changes_not_used_to_tune_the_training_that_just_ran")
            or not document.get("private_v89_TEST_unread")):
        raise AssertionError("Input is not previously sealed public DEV")
    cases = document["cases"]
    ids = [int(c["case_id"]) for c in cases]
    if len(ids) != 16 or len(set(ids)) != 16:
        raise AssertionError("Expected 16 unique sealed public QA cases")
    if set(document["scores"]) != set(ARMS):
        raise AssertionError("Unexpected/absent verified source arm")
    all_grades = {arm: _grades(document, arm) for arm in ARMS}
    rows_by_arm = {arm: _source_rows(document, arm) for arm in ARMS}
    if any(set(rows_by_arm[arm]) != set(ids)
           or set(all_grades[arm]) != set(ids) for arm in ARMS):
        raise AssertionError("One or more genuine responses missing")
    diagnostics = []
    taxonomy = Counter()
    per_domain = defaultdict(Counter)
    for c in cases:
        i = int(c["case_id"])
        arm_rows = {}
        for arm in ARMS:
            out = rows_by_arm[arm][i]
            grade = all_grades[arm][i]
            text = out["verbatim_text"]
            if text != grade["raw"]:
                raise AssertionError("Graded answer does not equal original verbatim response")
            tokens = out.get("token_ids")
            if arm != "FP32_existing_135M":
                if not isinstance(tokens, list):
                    raise AssertionError("Missing actual packed native generated token IDs")
                if any(not isinstance(t, int) for t in tokens):
                    raise AssertionError("Non-integer native token sequence")
                if (out["eos"] and out["cap_hit_without_eos"]):
                    raise AssertionError("Contradictory EOS status")
            arm_rows[arm] = {
                "grade": grade["machine_status"],
                "raw_response": text,
                "candidate": grade["candidate"],
                "recognition_basis": grade["recognition_basis"],
                "token_count": len(tokens) if tokens is not None else None,
                "longest_repeated_token_run": _longest_run(tokens) if tokens is not None else None,
                "adjacent_identical_token_pairs": (
                    sum(a == b for a, b in zip(tokens, tokens[1:]))
                    if tokens is not None else None),
                "eos": out.get("eos"),
                "cap_hit_without_eos": out.get("cap_hit_without_eos"),
            }
        before = arm_rows["CB186"]
        after = arm_rows["GT200"]
        fp32 = arm_rows["FP32_existing_135M"]
        if after["grade"] == "ungraded":
            if after["token_count"] == 0 or not after["raw_response"].strip():
                failure = "blank_or_special_tokens"
            elif after["longest_repeated_token_run"] >= 6:
                failure = "repeated_token_loop"
            elif after["cap_hit_without_eos"]:
                failure = "max_token_cap_unparsed"
            elif after["eos"]:
                failure = "early_eos_or_unparsed_text"
            else:
                failure = "unparsed_answer"
        elif after["grade"] == "machine_wrong":
            failure = "recognized_wrong"
        else:
            failure = "correct"
        if fp32["grade"] == "machine_correct" and after["grade"] != "machine_correct":
            parity = "confirmed_FP32_correct_GT200_not_confirmed"
        elif fp32["grade"] != "machine_correct" and after["grade"] == "machine_correct":
            parity = "GT200_correct_on_FP32_not_confirmed"
        else:
            parity = "same_correctness_confirmation"
        if before["grade"] != "machine_correct" and after["grade"] == "machine_correct":
            delta = "became_confirmed_correct"
        elif before["grade"] == "machine_correct" and after["grade"] != "machine_correct":
            delta = "lost_confirmed_correct"
        else:
            delta = "no_binary_correctness_change"
        taxonomy[failure] += 1
        taxonomy[parity] += 1
        taxonomy[delta] += 1
        per_domain[c["domain"]][failure] += 1
        diagnostics.append({
            "case_id": i, "domain": c["domain"],
            "failure_label_diagnostic_only": failure,
            "FP32_parity_category": parity,
            "CB186_to_GT200_change": delta,
            "original_gold_for_QA_only": c["gold"],
            "arms": arm_rows,
        })
    summary = {
        "complete": True,
        "read_only_reanalysis_of_sealed_v200_artifact": True,
        "not_new_model_test_not_blind_TEST": True,
        "QA_labels_never_used_for_training_or_model_selection": True,
        "source_changed": False,
        "original_unique_parameter_count": 134515008,
        "n": 16,
        "machine_correct": {
            arm: document["scores"][arm]["machine_confirmed"] for arm in ARMS
        },
        "failure_taxonomy": dict(sorted(taxonomy.items())),
        "failure_by_domain": {
            domain: dict(sorted(counts.items()))
            for domain, counts in sorted(per_domain.items())
        },
        "public_DEV_only_no_generalization_claim": True,
        "cases": diagnostics,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("v200_frozen_qa", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    raw = args.v200_frozen_qa.read_bytes()
    doc = json.loads(raw.decode("utf8"))
    report = analyze(doc)
    report["input_artifact_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v201_public_DEV16_read_only_fault_taxonomy.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf8")
    print(json.dumps({
        "stage": "v201_frozen_v200_qa_diagnostic",
        "status": "read_only_complete",
        "n": report["n"], "machine_correct": report["machine_correct"],
        "taxonomy": report["failure_taxonomy"],
        "input_sha256": report["input_artifact_sha256"],
        "no_model_test_or_live_change": True,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
