"""S108 Stage A: real physical S101 all-456 TRAIN zero-gain parity on CPU.

No training, no sealed QA, no teacher, no LIVE changes. Reconstruct original
S101 S93 CB61 + S94 bank + S95 router + S98 owner-copy and S101 cached CPU
decoder, then install the NEW shared inter-layer Green Tunnel at all 12
layers. Verify every output token (+EOS) is IDENTICAL at zero gain.

The actual replacement sidecar and minimal deployment metadata are serialized
and counted against the 0.700 bpp hard cap. Diagnostic report is not a
deployment artifact. S101 accepted research artifact remains immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s94.physical_profile_bank import CHANGED, freeze_report, sha
from calibri2.s95.train_and_audit_router import CHOICES, prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    apply_profile, load_frozen_numeric_weights,
)
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s108.dual_donor_interlayer_bridge import DualDonorInterlayerGreenBridge

FIRST_SAMPLE = 48
MAX_BPP = 0.7


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "S98_copy_net", "S98_report", "S95_router", "S95_report",
        "S94_bank", "S94_report", "S93_CB61", "S93_sidecar",
        "S99_full456_report", "S101_full456_report",
        "S87_report", "S91_report", "S93_report",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    s101 = json.loads(args.S101_full456_report.read_text())
    s99 = json.loads(args.S99_full456_report.read_text())
    if (s101.get("stage") != "S101_FROZEN_S99_CPU_KV_CACHE_EXACT_456_TRAIN_ONLY"
            or s101.get("cached_full_EOS_correct") != 353
            or s101.get("new_first_token_correct") != 387
            or s101.get("full_EOS_token_mismatch_count") != 0
            or s99.get("S99_new_correct_full_EOS") != 353
            or s101.get("candidate_accepted") is not False):
        raise AssertionError("S101 immutable 353/387 gold-answer anchor drift")

    for path, expected_sha in (
        (args.S93_CB61, s99["immutable_S95_source_sha256"]),
        (args.S94_bank, s99["S94_weightbank_sha256"]),
        (args.S95_router, s99["S95_neural_profile_selector_sha256"]),
        (args.S98_copy_net, s99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes()) != expected_sha:
            raise AssertionError("Original S101 physical artifact SHA changed")

    _, _, _, bank, router, copy_net = load_frozen_numeric_weights(args)
    side = load_file(str(args.S93_sidecar))
    model, sources, first, later, pointer, head, original_tunnel = s87.build_cpu(
        s87.open_envelope(args.S93_CB61.read_bytes()), side, True
    )
    if install_verified_fastpath(original_tunnel) != ("3", "9"):
        raise AssertionError("S101 original Green Tunnel fastpath changed")
    if len(model.model.layers) != 12:
        raise AssertionError("Original S101 decoder does not have 12 layers")
    added = DualDonorInterlayerGreenBridge(
        original_tunnel, total_layers=12, donor_layers=(3,9),
    ).eval().install(model)

    if (added.connected_layers != tuple(range(12))
            or added.trainable_new_parameters() != 20
            or len(added._handles) + len(original_tunnel._handles) != 12
            or torch.count_nonzero(added.layer_gains).item() != 0):
        raise AssertionError("S108 missing layers, donor reused incorrectly or nonzero gains")

    # Candidate inference payload is the original sidecar + exactly TWENTY new
    # scalar parameters, saved once. Do NOT include diagnostic TRAIN outputs.
    proposed_side = dict(side)
    for key, value in added.physical_sidecar_payload().items():
        if key in proposed_side:
            raise AssertionError("S108 sidecar key collision")
        proposed_side[key] = value
    side_file = args.out / "S108_UNACCEPTED_DUAL_ZERO_GAIN_S93_SIDECAR.safetensors"
    save_file(proposed_side, str(side_file))
    reloaded = load_file(str(side_file))
    if (set(reloaded) != set(proposed_side)
            or any(not torch.equal(reloaded[k], proposed_side[k])
                   for k in proposed_side)):
        raise AssertionError("S108 physical sidecar did not round-trip")

    # Runtime spec is a required PHYSICAL deployment artifact. Its bytes count.
    spec = {
        "stage": "S108_RESEARCH_UNACCEPTED",
        "source": "S101",
        "type": "shared_two_donor_interlayer_existing_Green_Tunnel",
        "donor_layers": [3,9],
        "dedicated_layers": [3, 9],
        "connected_layers": list(added.additional_layers),
        "active_profile": "S93",
        "sidecar_gain_key": "dual_bridge.layer_gains",
    }
    config_file = args.out / "S108_UNACCEPTED_DUAL_BRIDGE_RUNTIME_CONFIG.json"
    config_file.write_text(
        json.dumps(spec, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )
    original_bpp = float(s101["S99_all_inclusive_physical_bpp"])
    delta_sidecar = side_file.stat().st_size - args.S93_sidecar.stat().st_size
    added_physical_bytes = delta_sidecar + config_file.stat().st_size
    bpp = original_bpp + 8 * added_physical_bytes / SOURCE_PARAMS
    if bpp > MAX_BPP or added_physical_bytes < 40:
        raise AssertionError("S108 physical sidecar/config violates 0.7 bpp")
    physical_evidence = {
        "base_bpp": original_bpp,
        "new_all_inclusive_bpp": bpp,
        "S93_sidecar_actual_byte_delta": delta_sidecar,
        "deployment_config_actual_bytes": config_file.stat().st_size,
        "total_added_physical_bytes": added_physical_bytes,
        "proposed_sidecar_sha256": sha(side_file.read_bytes()),
        "bridge_config_sha256": sha(config_file.read_bytes()),
        "physically_below_0_7_bpp": True,
    }

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK = tok
    reports = {name: freeze_report(getattr(args, name + "_report"))
               for name in ("S87", "S91", "S93")}
    questions = {}
    for family, pairs in s87.families(tok).items():
        for pair in pairs:
            for side_name in ("left", "right"):
                key = f'{family}/{pair["pair_id"]}/{side_name}'
                questions[key] = pair[side_name]["question"]
    identities = sorted(questions)
    if len(identities) != 456 or set(identities) != set(reports["S87"]):
        raise AssertionError("S101 456 authorized TRAIN identities changed")
    for ident in identities:
        if sha(questions[ident].encode()) != reports["S87"][ident]["question_sha256"]:
            raise AssertionError("TRAIN prompt drift; stop before evaluation")

    features = torch.as_tensor(
        np.stack([prompt_only_vector(tok, questions[k]) for k in identities]),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        choices = router(features).argmax(dim=1).tolist()
    routes = {ident: CHOICES[int(i)] for ident, i in zip(identities, choices)}
    original_centroids = {
        k: sources[k].centroids.detach().clone() for k in CHANGED
    }
    original_tunnel_weights = {
        k.removeprefix("internal_tunnel."): v for k, v in side.items()
        if k.startswith("internal_tunnel.")
    }

    matching = 0
    exact_correct = 0
    first_correct = 0
    base_correct = 0
    mismatches = []
    base_speeds = []
    new_speeds = []
    by_route = Counter()
    selected = None
    try:
        with torch.inference_mode():
            for i, ident in enumerate(identities):
                route = routes[ident]
                if route != selected:
                    apply_profile(
                        sources, original_tunnel, route, bank,
                        original_centroids, original_tunnel_weights,
                    )
                    selected = route
                question = questions[ident]
                goal = reports["S87"][ident]["target_ids"]
                # Measure ORIGINAL S101 with bridge hooks actually REMOVED,
                # not S101-on-an-S108-model with ten disabled hooks.
                added.uninstall()
                t0 = time.perf_counter()
                baseline, _ = cached_full_cpu_with_neural_copy(
                    model, sources, first, later, head, tok, question,
                    pointer, copy_net,
                )
                elapsed_baseline = time.perf_counter() - t0
                added.install(model)
                with added.profile(route):
                    t0 = time.perf_counter()
                    candidate, _ = cached_full_cpu_with_neural_copy(
                        model, sources, first, later, head, tok, question,
                        pointer, copy_net,
                    )
                    elapsed_candidate = time.perf_counter() - t0

                if i < FIRST_SAMPLE:
                    base_speeds.append(elapsed_baseline)
                    new_speeds.append(elapsed_candidate)
                matching += int(baseline == candidate)
                base_correct += int(baseline == goal)
                exact_correct += int(candidate == goal)
                first_correct += int(bool(candidate) and candidate[0] == goal[0])
                by_route[route] += 1
                if baseline != candidate:
                    mismatches.append(hashlib.sha256(ident.encode()).hexdigest())
    finally:
        added.uninstall()
        original_tunnel.uninstall()

    speed_same_run = statistics.median(base_speeds) / statistics.median(new_speeds)
    report = {
        "stage": "S108_TWO_DONOR_ALL_12_ZERO_GAIN_PHYSICAL_CPU_TRAIN_456",
        "research_only": True,
        "full_456_exact_parity": matching == 456,
        "identities": len(identities),
        "baseline_full_EOS_correct": base_correct,
        "candidate_full_EOS_correct": exact_correct,
        "candidate_first_token_correct": first_correct,
        "mismatched_identity_hashes": mismatches,
        "baseline_vs_12_layer_CPU_speed_ratio": speed_same_run,
        "cpu_overhead_at_most_10_percent": speed_same_run >= 0.9,
        "profiles": dict(by_route),
        "all_12_layers_connected_3_and_9_dedicated": True,
        "two_shared_GT_donors_not_duplicated": True,
        "new_gain_values_all_zero": True,
        "additional_gain_count":20,
        "donor_layers":[3,9],
        "only_authorized_TRAIN_used": True,
        "no_training_performed": True,
        "independent_QA_not_opened": True,
        "candidate_promoted": False,
        "LIVE_Jessica_Render_Supabase_main_APK_untouched": True,
        "physical": physical_evidence,
    }
    result = args.out / "S108_DUAL_DONOR_ZERO_PARITY_FULL456_RESEARCH_REPORT.json"
    result.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "event": "S108_TWO_DONOR_ZERO_GAIN_CPU_EVIDENCE",
        "matched": matching,
        "baseline_correct": base_correct,
        "new_correct": exact_correct,
        "new_first_correct": first_correct,
        "speed_vs_old_S101": speed_same_run,
        "physical_bpp": bpp,
        "ready_for_controlled_gain_training": matching == 456
            and base_correct == 353 and first_correct == 387
            and speed_same_run >= 0.9 and bpp <= 0.7,
    }), flush=True)
    if (matching != 456 or base_correct != 353 or exact_correct != 353
            or first_correct != 387 or speed_same_run < 0.9
            or not physical_evidence["physically_below_0_7_bpp"]):
        raise SystemExit("S108 baseline REJECTED: zero-parity, speed or physical guard")

if __name__ == "__main__":
    main()
