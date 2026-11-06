"""CALIBRI v224: read-only analyze original sealed v222 TRAIN generation artifact.

No model loads, weight updates, DEV/VALID/TEST grading, Jessica tests or
LIVE/backend/Render/Supabase changes. Repetition is evaluated as adjacent
identical generated token IDs (not grammatical repetition).
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def repeated_positions(token_ids):
    return [
        {"position_1based": index + 1, "token_id": int(token_ids[index]),
         "preceding_position_1based": index}
        for index in range(1, len(token_ids))
        if token_ids[index] == token_ids[index - 1]
    ]


def analyze(data):
    if data.get("completed") is not True or data.get("never_train_or_update_source") is not True:
        raise AssertionError("Not completed immutable v222 TRAIN source comparison")
    expected_hashes = {
        "v206":"5a6d5f07f9e49fa6063dd772d7e2aee53beeb3527d10d5a429715593b68939de",
        "v220":"e7541169b5b2188393859953210f52a38ca24451129e246eebb20cb20466e34d",
    }
    if data.get("source_hashes") != expected_hashes:
        raise AssertionError("Source SHA256 provenance differs from verified v222 archives")
    if data.get("public_DEV16_not_reused") is not True:
        raise AssertionError("v222 DEV isolation not confirmed")
    rows = data["all_48_opposing_TRAIN_rows"]
    if len(rows) != 48 or any(row["split"] != "TRAIN" for row in rows):
        raise AssertionError("Expected exactly 48 original TRAIN generations")
    seen = {(row["pair_id"], row["side"]) for row in rows}
    if len(seen) != 48:
        raise AssertionError("Duplicate or missing original TRAIN case")
    repeats = []
    source_counts = {}
    for label in ("v206", "v220"):
        entries = []
        for row in rows:
            tokens = row[f"{label}_generated_token_ids"]
            positions = repeated_positions(tokens)
            if len(positions) != int(row[f"{label}_repeat_excess"]):
                raise AssertionError("Stored and recomputed adjacent repeat count differ")
            if positions:
                entries.append({
                    "pair_id": row["pair_id"],
                    "side": row["side"],
                    "domain": row["domain"],
                    "positions": positions,
                    "first_divergence_position_1based":row["first_diff_position_1based"],
                    "original_first_12_token_ids":row["v206_generated_token_ids"],
                    "candidate_first_12_token_ids":row["v220_generated_token_ids"],
                    "v206_gold_prefix":row["v206_gold_prefix"],
                    "v220_gold_prefix":row["v220_gold_prefix"],
                })
        source_counts[label] = sum(len(x["positions"]) for x in entries)
        if label == "v220":
            repeats = entries
    if source_counts != {"v206": 0, "v220": 5}:
        raise AssertionError(f"Unexpected v222 precommitted repeat counts: {source_counts}")
    hist = Counter(p["position_1based"] for e in repeats for p in e["positions"])
    result = {
        "completed":True,
        "strictly_frozen_existing_v222_public_TRAIN_only":True,
        "sources":source_counts,
        "v220_repeated_prompts":len(repeats),
        "v220_repeat_position_histogram_1based":dict(sorted(hist.items())),
        "v220_repeat_details":repeats,
        "v220_repeats_on_changed_second_token_prompts":
            sum(row["first_divergence_position_1based"] == 2 for row in repeats),
        "v220_repeats_on_unchanged_second_token_prompts":
            sum(row["first_divergence_position_1based"] != 2 for row in repeats),
        "no_new_models_or_tests_or_training":True,
        "none_of_development_or_validation_or_private_TEST_opened":True,
        "Jessica_tests_ran":False,
        "note":"Repeated adjacent token IDs on TRAIN are not alone proof of DEV16 quality causality.",
    }
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("v222_output",type=Path)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    data=json.loads(a.v222_output.read_text())
    result=analyze(data)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({
        "stage":"v224_read_only_TRAIN_repeat_position_summary",
        "repeated_prompts":result["v220_repeated_prompts"],
        "repeat_positions":result["v220_repeat_position_histogram_1based"],
        "repeats_after_second_token":sum(n for k,n in result["v220_repeat_position_histogram_1based"].items() if int(k)>2),
        "changed_second_token_prompt_overlap":result["v220_repeats_on_changed_second_token_prompts"],
        "no_weight_changes":True
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
