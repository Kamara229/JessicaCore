"""Z39 immutable replay audit of Z38's SPENT synthetic DEV14 output.

Reads ONLY the finished original-S142-43M CI JSON, never checkpoints or
model inputs. Counts paired native-BPE2 corrections, actual cross-prompt
second-token diversity, and failures at BPE3 after a correct BPE1+BPE2.
NOT a new validation set or statistical proof of generalization.
"""
from __future__ import annotations
import argparse
from collections import Counter
from math import comb, log2
import json
from pathlib import Path

STAGE = "CALIBRI_S119Q2I8Z38_REAL43M_NATIVE49152_CENTEREDRMS_ONLY_BPE2_RANK32_84TRAIN14DEV"
DEV_SHA = "308ecdb3dbec7cdff02bfd9333a11355b30f8c7392ac683d74e21fd1aef4ac87"
BASE = "ORIGINAL_NATIVE_GREEDY"
TRUE = "HUMAN_GOLD_BPE2"
WRONG = "WRONG_OFFSET17_LABEL_CONTROL"
MODES = (BASE, TRUE, WRONG)


def require(condition, message):
    if not condition:
        raise ValueError("Z39 fail-closed archived native replay audit: " + message)


def empirical_diversity(ids):
    """Cross-PROMPT empirical BPE2 dispersion, NOT LM predictive entropy."""
    require(len(ids) > 0 and all(type(x) is int for x in ids), "nonempty integer BPE2 IDs")
    counts = Counter(ids)
    entropy = -sum((n / len(ids)) * log2(n / len(ids)) for n in counts.values())
    return {"distinct_ids": len(counts), "empirical_entropy_bits": round(entropy, 6),
            "majority_share": round(max(counts.values()) / len(ids), 6)}


def paired_sign_p(improved, regressed):
    """Exact two-sided discordant-pair sign/McNemar p, descriptive small sample."""
    require(improved >= 0 and regressed >= 0, "nonnegative discordance")
    n = improved + regressed
    return min(1.0, 2.0 * sum(comb(n, k) for k in range(min(improved, regressed) + 1)) / (2**n)) if n else 1.0


def analyze_arm(arm, label):
    rows = arm["NEW_DEV14_per_prompt_first6"]
    require(len(rows) == 14, label + " exactly 14 archived SPENT DEV prompts")
    require(len({row["source_id"] for row in rows}) == 14, label + " distinct source IDs")
    metrics = {m: {"second": [], "first2": 0, "first3": 0, "changed_bpe2": 0,
                   "first2_correct_but_third_wrong": 0} for m in MODES}
    per_prompt = []
    for row in rows:
        gold = row["human_gold_native_first4_POSTHOC"]
        require(isinstance(gold, list) and len(gold) == 4, "posthoc first4 human IDs")
        require(row["DEV_gold_was_never_in_training_or_candidate_selection"] is True,
                "no archived DEV label leakage")
        modes = row["auto_generated_modes_no_teacher_forcing"]
        require(set(modes) == set(MODES), "all three original own-prefix continuations")
        firsts = set()
        observed = {}
        for m in MODES:
            rec = modes[m]
            ids = rec["autonomous_first_six_native_ids"]
            require(2 <= len(ids) <= 6 and all(type(x) is int for x in ids),
                    "original native BPE generated first6")
            firsts.add(ids[0])
            correct2 = ids[:2] == gold[:2]
            correct3 = len(ids) >= 3 and ids[:3] == gold[:3]
            require(rec["first_native_correct"] == (ids[0] == gold[0]) and
                    rec["first_two_native_correct"] == correct2 and
                    rec["first_three_native_correct"] == correct3,
                    "archived correctness flags match physical generated native IDs")
            steps = rec["per_second_to_fourth_real_forward"]
            require(len(steps) == len(ids) - 1, "real forward step count matches sequence")
            for n, step in enumerate(steps, start=2):
                require(step["native_position_1based"] == n and
                        step["chosen_native_id"] == ids[n - 1], "real native step parity")
                if m == BASE or n >= 3:
                    require(step["chosen_native_id"] == step["original_greedy_native_id"],
                            "only second BPE may differ from original native greedy")
                if m != BASE and n == 2:
                    require(step["candidate_original_top64"] is True,
                            "rerank chosen only inside true original top64")
            d = metrics[m]
            d["second"].append(ids[1])
            d["first2"] += int(correct2)
            d["first3"] += int(correct3)
            d["changed_bpe2"] += int(steps[0]["shadow_changed_original_greedy"])
            d["first2_correct_but_third_wrong"] += int(correct2 and not correct3)
            observed[m] = correct2
        require(len(firsts) == 1, "same unmodified own first BPE across three modes")
        per_prompt.append(observed)
    result = {}
    summary = arm["NEW_DEV14_full_autonomous_first6_native"]["matched_autonomous_first6_native_generated"]
    for m in MODES:
        d = metrics[m]
        diversity = empirical_diversity(d.pop("second"))
        require(summary[m]["correct_FIRST_TWO"] == d["first2"] and
                summary[m]["correct_FIRST_THREE"] == d["first3"] and
                summary[m]["distinct_second_native_BPE_ids"] == diversity["distinct_ids"],
                "audited metrics must match frozen Z38 physical summary")
        result[m] = {**d, **diversity}
    for m in (TRUE, WRONG):
        improved = sum(int(p[m] and not p[BASE]) for p in per_prompt)
        regressed = sum(int(p[BASE] and not p[m]) for p in per_prompt)
        result[m]["paired_BPE2_newly_correct"] = improved
        result[m]["paired_BPE2_newly_wrong"] = regressed
        result[m]["exact_discordant_two_sided_p_descriptive"] = paired_sign_p(improved, regressed)
        result[m]["cross_prompt_diversity_ratio_vs_base"] = round(
            result[m]["distinct_ids"] / result[BASE]["distinct_ids"], 6)
    return result


def audit(report):
    require(report["stage"] == STAGE and report["output_gate"] == 0 and
            report["candidate_accepted"] is False, "research-only original S142 evidence")
    require(report["Z37_PRECOMMITTED_14_BRANDNEW_DEV_SHA256"] == DEV_SHA and
            report["Z37_brandNEW_author_DEV14_NOW_SPENT_not_independently_blind"] is True,
            "literal SHA + SPENT internal author DEV14; not reusable to tune")
    for field in ("strict_ONLY_second_native_BPE_candidate_decoding_intervention",
                  "original_S142_43M_GT12_Core_NRA3_NPA1_and_Z10_A_B_unchanged",
                  "no_gold_in_candidate_selection_on_new_DEV",
                  "no_previous_SPENT_DEV_used_in_TRAIN",
                  "no_616J7_EVAL128_GOLD_bpp_CB59_CPU_certification"):
        require(report[field] is True, field)
    require(report["original43M_49k_native_FORWARDS"] == 752,
            "physical 752 decoder forwards pinned to previous success")
    arms = report["actual_original43M_A_B_real_native49k_BPE2_only_rank32"]
    require(len(arms) == 2, "two immutable Z10 A/B true S142 arms")
    result = {label: analyze_arm(arm, label) for label, arm in sorted(arms.items())}
    # Fail closed: the previous experiment is diagnostic ONLY, not a deployed model,
    # regardless of whether a descriptive comparison happens to improve.
    return {"stage": "S119Q2I8Z39_READONLY_SPENT_DEV14_FORENSIC",
            "read_only_archived_source_sha256": DEV_SHA,
            "observed_arms": result,
            "evidence": "archived Z38 CI output only, internal synthetic SPENT14",
            "true_model_entropy_measured": False,
            "new_model_forwards_or_optimizer_steps": 0,
            "validation_or_replication": False,
            "candidate_accepted": False, "output_gate": 0,
            "decision": "REJECT_BPE2_ONLY_AS_SEQUENCE_SOLUTION; PREREG_NEW_DATA_FOR_BPE3_PLUS_ENTROPY_PRESERVATION"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.input.read_text(encoding="utf8"))
    result = audit(source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
