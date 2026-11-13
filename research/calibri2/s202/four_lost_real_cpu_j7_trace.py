"""S202: actual native CPU targeted posthoc transcript for four S201 lost TRAIN IDs.

Never reads a gold answer or question hash to select an inference route. This
instrumentation observes the *unmodified* S201 native 616 J7 research run,
captures outputs from its original retention comparator BEFORE the guarded
S176 rejection, and fails closed if its protected gate would be bypassed.
No QA/TEST, no LIVE, no model acceptance or training.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from calibri2.s158 import targeted_retention_gpu as s158
from calibri2.s201 import real_s200_snapshots_full_j7 as s201

STAGE = "S202_REAL_CPU_J7_FOUR_LOST_TRAIN_TOKEN_TRACES"
TARGET_IDS = (
    "S73/73012/left", "S73/73012/right",
    "S73/73015/left", "S73/73015/right",
)
LOSSES = {
    2: ("S73/73015/left", "S73/73015/right"),
    4: TARGET_IDS,
    8: TARGET_IDS,
}


def diagnostic_row(identity, old, parent_old, candidate_old):
    """Pure analysis of answers already generated, never a selector."""
    sample = old[identity]
    gold = list(sample["expected"])
    parent = list(parent_old[identity])
    candidate = list(candidate_old[identity])
    all_tokens = max(len(gold), len(candidate))
    changed = [i for i in range(all_tokens)
               if (gold[i] if i < len(gold) else None) !=
                  (candidate[i] if i < len(candidate) else None)]
    prompt = sample["question"]
    return {
        "id": identity,
        "question": prompt,
        "question_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "original_parent_generated_ids": parent,
        "candidate_generated_ids": candidate,
        "precommitted_TRAIN_expected_ids_posthoc_only": gold,
        "parent_correct_full_EOS": parent == gold,
        "candidate_correct_full_EOS": candidate == gold,
        "candidate_first_token_correct": bool(candidate and gold and candidate[0] == gold[0]),
        "first_divergent_position": changed[0] if changed else None,
        "all_divergent_positions": changed,
    }


def posthoc_capture(step, out: Path):
    """Capture once S167's *actual* 456+128 run matches immutable failure gate."""
    real_retention = s158.retention
    expected = set(LOSSES[step])
    state = {"matched": 0, "observed": 0}

    def monitored(parent_old, candidate_old, old, parent_new, candidate_new, new):
        stats = real_retention(
            parent_old, candidate_old, old, parent_new, candidate_new, new)
        if all(key in old and key in parent_old and key in candidate_old
               for key in TARGET_IDS):
            state["observed"] += 1
            if (len(old) == 456 and len(new) == 128
                and set(stats["lost_old_full_ids"]) == expected
                and set(stats["lost_old_first_ids"]) == expected
                and not stats["lost_new_full_ids"]
                and not stats["lost_new_first_ids"]):
                rows = [diagnostic_row(k, old, parent_old, candidate_old)
                        for k in TARGET_IDS]
                if (any(not x["parent_correct_full_EOS"] for x in rows)
                    or {x["id"] for x in rows
                        if not x["candidate_correct_full_EOS"]} != expected):
                    raise AssertionError("S202 captured actual J7 tokens contradict protected gate")
                report = {
                    "stage": STAGE,
                    "S200_CPU_gradient_steps": step,
                    "provenance_S201_failed_GitHub_run": 43376544863,
                    "S201_guarded_original_CPU_456": stats["old_full_after"],
                    "S201_guarded_new128_CPU": stats["new_full_after"],
                    "loss_IDs": sorted(expected),
                    "compared_frozen_original_S142_parent_not_S197_accepted": True,
                    "real_native_CPU_predictions_collected_before_gold_scoring": True,
                    "question_or_gold_never_used_for_inference": True,
                    "samples": rows,
                    "no_QA_TEST": True,
                    "no_training_or_model_acceptance": True,
                    "LIVE_unchanged": True,
                    "model_accepted": False,
                }
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"S202_STEP_{step}_FOUR_ORIGINAL_TRAIN_CPU_TRACE.json"
                if path.exists():
                    prev = json.loads(path.read_text())
                    if prev != report:
                        raise AssertionError("S202 inconsistent repeated native CPU trace")
                else:
                    path.write_text(json.dumps(report, indent=2, sort_keys=True))
                state["matched"] += 1
        return stats

    return monitored, state


def main():
    argv = list(sys.argv)
    if len(argv) < 18:
        raise SystemExit("S202 requires S200 step and exact S201 616 native CPU CLI")
    step = int(argv[1])
    if step not in LOSSES:
        raise AssertionError("S202 only original S200 precommitted TRAIN steps 2/4/8")
    out = Path(argv[-1])
    original = s158.retention
    hooked, state = posthoc_capture(step, out)
    s158.retention = hooked
    guard_failed = False
    try:
        # Keep full original guard: do not bypass S176 or pretend a rejected
        # snapshot passes, only capture its *preexisting* CPU predictions.
        s201.main()
    except AssertionError as exc:
        if str(exc) != "S176 rejected: S172 cohort regression or bpp/CPU/protected gate failed":
            raise
        guard_failed = True
    finally:
        s158.retention = original
    if not guard_failed or state["matched"] < 1:
        raise AssertionError("S202 expected guarded S201 failure and actual four-ID CPU trace")
    target = out / f"S202_STEP_{step}_FOUR_ORIGINAL_TRAIN_CPU_TRACE.json"
    if not target.is_file():
        raise AssertionError("S202 output trace absent")
    proof = json.loads(target.read_text())
    if (proof["S201_guarded_original_CPU_456"] != 456-len(LOSSES[step])
        or proof["S201_guarded_new128_CPU"] != 128
        or proof["model_accepted"] is not False):
        raise AssertionError("S202 fail closed on CPU protected mismatch")
    print(json.dumps({
        "event": STAGE, "step": step,
        "lost": proof["loss_IDs"],
        "token_first_divergence": {
            x["id"]: x["first_divergent_position"] for x in proof["samples"]},
        "protected_CPU_J7_REJECTED": True,
        "QA_TEST_or_LIVE": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
