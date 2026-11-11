"""S88 456-TRAIN CPU fastpath replay against the immutable S87 negative audit.

No new GPU training, no QA, and no model promotion. Decode the *same*
compressed S87 source + exact serialized sidecar; the sole change is skipping
the verified-zero gate projection/sigmoid in the internal Green Tunnel.
Reference against all 456 saved S87 greedy/EOS token outputs, not only labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath


def main():
    # S87's 6 positional arguments + the immutable S87 negative audit JSON.
    if len(sys.argv) != 8:
        raise SystemExit(
            "usage: guard_zero_gate_fastpath.py original.cb59 s82.safetensors "
            "s87.s61 s87.safetensors s87_manifest.json output_dir "
            "S87_ALL_456_TRAIN_GUARD_REPORT.json")
    prior = json.loads(Path(sys.argv[7]).read_text())
    if (prior.get("stage") !=
            "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT" or
            len(prior.get("all_456_per_identity_results", [])) != 456 or
            prior.get("candidate_accepted") is not False):
        raise AssertionError("Immutable S87 benchmark artifact is not valid")

    original_builder = s87.build_cpu
    patched_layers = []

    def build_optimized_cpu(raw, side, use_internal):
        result = original_builder(raw, side, use_internal)
        if use_internal:
            patched_layers.extend(install_verified_fastpath(result[-1]))
        return result

    s87.build_cpu = build_optimized_cpu
    original_argv = sys.argv
    audit_exit = None
    try:
        sys.argv = original_argv[:7]
        try:
            s87.main()
        except SystemExit as exc:
            # The reference audit writes a full JSON before failing the speed
            # guard. Preserve and compare it before returning a failing exit.
            audit_exit = str(exc)
    finally:
        sys.argv = original_argv
        s87.build_cpu = original_builder

    output = Path(sys.argv[6])
    report = json.loads(
        (output / "S87_ALL_456_TRAIN_GUARD_REPORT.json").read_text())
    if report["source_new_cb59_sha256"] != prior["source_new_cb59_sha256"]:
        raise AssertionError("Fastpath did not replay the identical S87 source")
    if report["all_inclusive_bpp"] != prior["all_inclusive_bpp"]:
        raise AssertionError("Physical bpp changed in an inference-only run")
    saved = {r["id"]: r for r in prior["all_456_per_identity_results"]}
    mismatches = []
    for row in report["all_456_per_identity_results"]:
        old = saved.get(row["id"])
        if (old is None or old["S87_ids"] != row["S87_ids"] or
                old["S82_ids"] != row["S82_ids"] or
                old["target_ids"] != row["target_ids"]):
            mismatches.append(row["id"])
    if len(saved) != 456 or len(report["all_456_per_identity_results"]) != 456:
        raise AssertionError("Incomplete 456-TRAIN comparison")

    final = {
        "stage": "S88_ZERO_GATE_CPU_FASTPATH_456_EXACT_REPLAY",
        "reference_S87_run_id": 42659248083,
        "reference_S87_cpu_ratio": prior["cpu_speed_ratio_48_prompt_median_S82_over_S87"],
        "optimized_cpu_ratio": report["cpu_speed_ratio_48_prompt_median_S82_over_S87"],
        "physical_bpp": report["all_inclusive_bpp"],
        "optimized_internal_layers": patched_layers,
        "exact_S87_token_mismatches": mismatches,
        "lost_full_ids": report["lost_full_ids"],
        "lost_first_ids": report["lost_first_ids"],
        "gained_full_ids": report["gained_full_ids"],
        "all_456_replayed": True,
        "full_safety_guard": report["full_safety_guard"],
        "cpu_speed_90pct_guard": report["cpu_speed_90pct_guard"],
        "QA_embargoed": True,
        "GPU_CPU_parity_not_yet_verified": True,
        "candidate_accepted": False,
    }
    (output / "S88_ZERO_GATE_FASTPATH_456_REPORT.json").write_text(
        json.dumps(final, indent=2))
    print(json.dumps({
        "event": "S88_ZERO_GATE_FASTPATH_456_END",
        "exact_output_mismatches": len(mismatches),
        "lost_full": len(report["lost_full_ids"]),
        "lost_first": len(report["lost_first_ids"]),
        "gained_full": len(report["gained_full_ids"]),
        "optimized_CPU_ratio": final["optimized_cpu_ratio"],
        "physical_bpp": final["physical_bpp"],
        "speed_pass": final["cpu_speed_90pct_guard"],
        "candidate_accepted": False,
    }), flush=True)
    if mismatches:
        raise SystemExit("REJECTED: S88 changes preexisting S87 greedy/EOS tokens")
    if audit_exit:
        raise SystemExit("REJECTED: S88 exact outputs retained, but " + audit_exit)


if __name__ == "__main__":
    main()
