"""S153: analyze S152 immutable TRAIN-only 456+128 loss identities before another GPU trial.

Read *only* the archived completed S152 research artifact. No weights, QA,
Jessica, Render, Supabase, APK, or production systems are touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_OLD = 362
BASE_NEW = 17
WORKERS = {
    "binding_gpu": ("S148", 361),
    "language_gpu": ("S151", 361),
    "completion_cpu": ("S149", 362),
}
LOSS_KEYS = (
    "lost_old_full_ids", "lost_old_first_ids",
    "lost_new_full_ids", "lost_new_first_ids",
)


def inspect(root: Path) -> dict:
    orchestrator = json.loads(
        (root / "S152_ALL_REAL_PARALLEL_SKILLS_AUDIT.json").read_text()
    )
    if (orchestrator.get("all_training_and_physical_audits_finished") is not True
            or orchestrator.get("model_accepted") is not False
            or orchestrator.get("joint_456_128_guard_not_run") is not True):
        raise AssertionError("S153 requires the exact completed, unaccepted S152 trial")
    summaries = {}
    for name, (stage, expected_old) in WORKERS.items():
        audit_path = root / name / (stage + "_REAL_NEURAL_SKILL_456_128_AUDIT.json")
        report = json.loads(audit_path.read_text())
        a = report["full456_and_new128"]
        if (report.get("real_S142_parent_old456") != BASE_OLD
                or report.get("real_S142_parent_new128") != BASE_NEW
                or a["old_full_before"] != BASE_OLD
                or a["new_full_before"] != BASE_NEW
                or a["old_full_after"] != expected_old
                or a["new_full_after"] != BASE_NEW
                or report.get("research_safe_development_candidate") is not False
                or report.get("J7_guarded_skill_rollback_verified") is not True
                or report.get("model_accepted") is not False
                or report.get("independent_heldout_QA_never_run") is not True):
            raise AssertionError(f"S153 immutable S152 {name} score/rollback contract drift")
        for key in LOSS_KEYS:
            if not isinstance(a.get(key), list) or len(a[key]) != len(set(a[key])):
                raise AssertionError(f"S153 {name} has malformed {key}")
        if report["physical"]["all_in_physical_bpp"] > .7:
            raise AssertionError(f"S153 {name} violates physical budget")
        if report["paired_CPU"]["ratio"] < .9:
            raise AssertionError(f"S153 {name} violates speed budget")
        summaries[name] = {
            "old_full": {"before": a["old_full_before"], "after": a["old_full_after"]},
            "new_full": {"before": a["new_full_before"], "after": a["new_full_after"]},
            "old_first_before": a["old_first_before"],
            "new_first_before": a["new_first_before"],
            "new_both_correct_pairs": {
                "before": a["new_pairs_both_before"],
                "after": a["new_pairs_both_after"],
            },
            "protected_losses_by_id": {key: sorted(a[key]) for key in LOSS_KEYS},
            "step_count": len(report["training_steps"]),
            "loss_first": report["training_steps"][0]["loss"],
            "loss_last": report["training_steps"][-1]["loss"],
            "physical_bpp": report["physical"]["all_in_physical_bpp"],
            "cpu_speed_ratio": report["paired_CPU"]["ratio"],
            "rolled_back": True,
        }
    gpu_names = ("binding_gpu", "language_gpu")
    gpu_old_lost = [
        set(summaries[name]["protected_losses_by_id"]["lost_old_full_ids"])
        for name in gpu_names
    ]
    return {
        "stage": "S153_S152_EXACT_RESEARCH_LOSS_IDENTITY_DIAGNOSTIC",
        "immutable_source_S152_run_id": 43211904296,
        "parent_S142_old456": BASE_OLD,
        "parent_S142_new128": BASE_NEW,
        "workers": summaries,
        "gpu_old_full_loss_intersection": sorted(gpu_old_lost[0] & gpu_old_lost[1]),
        "gpu_old_full_loss_union": sorted(gpu_old_lost[0] | gpu_old_lost[1]),
        "all_workers_rejected_J7": True,
        "new_models_accepted": 0,
        "joint_guard_not_run": True,
        "next_research_hypothesis": (
            "Protect exact lost original TRAIN identities with stronger per-step replay; "
            "then explore longer counterfactual S122 training on a single GPU skill "
            "and reject unless +full/EOS with zero loss on 456+128, <=0.7 bpp, CPU >=0.9."
        ),
        "no_QA_no_LIVE_Jessica_Render_Supabase_APK": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("s152_artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = inspect(args.s152_artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("S153_LOSS_DIAGNOSTIC", json.dumps({
        "old_losses": {name: data["protected_losses_by_id"]["lost_old_full_ids"]
                       for name, data in result["workers"].items()},
        "new_losses": {name: data["protected_losses_by_id"]["lost_new_full_ids"]
                       for name, data in result["workers"].items()},
        "common_gpu_loss_ids": result["gpu_old_full_loss_intersection"],
        "new_accepted_skills": 0,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
