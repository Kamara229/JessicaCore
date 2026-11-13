"""S191: posthoc attribution of real S190 32 NEW TRAIN CPU J7 failures.

Only consumes immutable S190 post-inference, hash-only audit JSON.
Identifies whether an error is in the first token, second token,
or full-EOS length. Never runs Jessica, neural training, QA/TEST,
and never feeds correctness into the S187/S190 inference router.
The diagnostic cannot infer the exact wrong word from SHA-only metadata.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


EXPECTED_FAMILIES = {
    "owner_A_paraphrase": (8, 3),
    "owner_B_paraphrase": (8, 3),
    "reordered_owner_paraphrase": (8, 3),
    "numeric_new_wording": (8, 2),
}


def diagnostic(report: dict) -> dict:
    if report.get("event") != "S190_REAL_CPU_32_UNSEEN_NEW_TRAIN_PARAPHRASES_STRICT_SOURCE_ABSTAIN":
        raise AssertionError("S191 exact S190 real J7 source report required")
    if (report.get("real_CPU_full_J7_new_TRAIN_all32") is not True
        or report.get("original_real_CPU_FULL_EOS_456") != 456
        or report.get("new128_real_CPU_FULL_EOS") != 128
        or report.get("original_first_full_forgetting") != 0
        or report.get("no_gold_or_SHA_at_inference_or_routing") is not True
        or report.get("all_in_physical_bpp_charging_S188_S189_S190_source", 1) > .7
        or report.get("model_accepted") is not False):
        raise AssertionError("S191 requires completely verified S190 train-only frozen provenance")
    source = report.get("hashed_per_prompt_audit", [])
    if len(source) != 32:
        raise AssertionError("S191 expected exact S190 32 per-prompt posthoc records")
    ids = set()
    score = defaultdict(Counter)
    category_totals = Counter()
    for entry in source:
        group = entry["group"]
        if group not in EXPECTED_FAMILIES:
            raise AssertionError("S191 unexpected new TRAIN group")
        sample = entry["prompt_sha256"]
        if not isinstance(sample, str) or len(sample) != 64 or sample in ids:
            raise AssertionError("S191 invalid/duplicate SHA-only frozen source record")
        ids.add(sample)
        n = entry["generated_tokens"]
        eos = entry["eos_present"]
        first = entry["first_token_correct"]
        full = entry["full_EOS_correct"]
        if type(n) is not int or n <= 0 or type(eos) is not bool or type(first) is not bool or type(full) is not bool:
            raise AssertionError("S191 invalid real CPU frozen output metadata")
        expected = EXPECTED_FAMILIES[group][1]
        if full and (not first or not eos or n != expected):
            raise AssertionError("S191 internal full-answer EOS contradiction")
        label = (
            "full_correct" if full else
            "wrong_first_token" if not first else
            "first_right_but_missing_EOS" if not eos else
            "first_right_wrong_output_length" if n != expected else
            "first_right_wrong_continuation_token"
        )
        score[group][label] += 1
        score[group]["total"] += 1
        score[group]["EOS_present" if eos else "EOS_missing"] += 1
        score[group]["right_first" if first else "wrong_first"] += 1
        score[group][f"output_token_length_{n}"] += 1
        category_totals[label] += 1
    groups = {}
    for key, (count, _) in EXPECTED_FAMILIES.items():
        if score[key]["total"] != count:
            raise AssertionError("S191 incomplete frozen per-group original new TRAIN cases")
        groups[key] = dict(sorted(score[key].items()))
        ref = report["NEW_TRAIN_group_metrics"][key]
        if (ref.get("count") != count or
            ref.get("first_correct") != score[key]["right_first"] or
            ref.get("full_EOS_correct") != score[key]["full_correct"] or
            ref.get("eos_present") != score[key]["EOS_present"]):
            raise AssertionError("S191 aggregated S190 SHA-only results mismatch")
    if (sum(v["right_first"] for v in groups.values()) != report["NEW_TRAIN_first_correct"]
        or sum(v["full_correct"] for v in groups.values()) != report["NEW_TRAIN_full_EOS_correct"]
        or sum(v["EOS_present"] for v in groups.values()) != report["NEW_TRAIN_eos_present"]):
        raise AssertionError("S191 S190 per-family global full/EOS count mismatch")
    return {
        "stage": "S191_POSTHOC_REAL_S190_NEW_TRAIN_FAILURE_MODES_NO_NEURAL",
        "S190_immutable_run": 43340296813,
        "S190_original_456_real_CPU_protected": 456,
        "S190_new128_real_CPU_protected": 128,
        "new_train_probe_count": 32,
        "new_train_full_EOS_correct": report["NEW_TRAIN_full_EOS_correct"],
        "new_train_first_correct": report["NEW_TRAIN_first_correct"],
        "failure_modes": dict(sorted(category_totals.items())),
        "by_family": groups,
        "source_report_bpp": report["all_in_physical_bpp_charging_S188_S189_S190_source"],
        "source_only_train_no_QA_TEST_or_LIVE": True,
        "no_raw_prompt_or_gold_or_target_ids_in_output": True,
        "does_not_explain_wrong_word_identity": True,
        "not_a_new_neural_generalization_result": True,
        "model_accepted": False,
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("S191 requires S190 immutable research-only JSON input and S191 output")
    source, destination = (Path(p) for p in sys.argv[1:])
    if destination.exists():
        raise FileExistsError("S191 report cannot overwrite immutable saved source")
    record = json.loads(source.read_text(encoding="utf-8"))
    result = diagnostic(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
