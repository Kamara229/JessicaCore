"""S109: bounded REAL MPS training of TWENTY S108 dual-donor shared interlayer gains.

RESEARCH ONLY: exact immutable S101 353/456 baseline. Train ONLY twenty gain
values; S93 CB59, dedicated 3/9 tunnel, S87/S91 banks, S95 router, S98
copier, S82 controller and tokenizer are FROZEN. One two-step proposal.
Recheck physically loaded S101 cached CPU full+EOS on ALL 456 TRAIN rows.
Never open or train on sealed QA. Fail-closed with full gain/Adam rollback.
No accepted model or live deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import MODEL_SPEC, SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s91 import train_joint_quality_rehearsal as s91
from calibri2.s94.physical_profile_bank import CHANGED, freeze_report, sha
from calibri2.s95.train_and_audit_router import CHOICES, prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    apply_profile, load_frozen_numeric_weights,
)
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s102.hybrid_retention import (
    AtomicProtectedUpdate, strict_train_retention,
)
from calibri2.s108.dual_donor_interlayer_bridge import DualDonorInterlayerGreenBridge
from v176.operators import clear

SEED = 1040104
STEPS = 2
MPS_LR = 0.004
SPEED_SAMPLE = 48
MAX_BPP = 0.7


def cohort(tok, reports, router):
    rows = {}
    for family, pairs in s87.families(tok).items():
        for pair in pairs:
            for side_name in ("left", "right"):
                key = f'{family}/{pair["pair_id"]}/{side_name}'
                row = pair[side_name]
                if sha(row["question"].encode()) != reports["S87"][key]["question_sha256"]:
                    raise AssertionError("TRAIN cohort prompt drift")
                rows[key] = {
                    "id": key,
                    "question": row["question"],
                    "expected": reports["S87"][key]["target_ids"],
                }
    keys = sorted(rows)
    if len(keys) != 456 or set(keys) != set(reports["S87"]):
        raise AssertionError("Exactly 456 immutable authorized TRAIN examples required")
    vectors = torch.as_tensor(
        np.stack([prompt_only_vector(tok, rows[k]["question"]) for k in keys]),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        classes = router(vectors).argmax(dim=1).tolist()
    routes = {k: CHOICES[int(v)] for k, v in zip(keys, classes)}
    return rows, keys, routes


def load_cpu_original(raw, side):
    model, sources, first, later, ptr, head, tunnel = s87.build_cpu(
        raw, side, True,
    )
    if install_verified_fastpath(tunnel) != ("3", "9"):
        raise AssertionError("Original S101 3/9 fastpath ABI changed")
    base_core = {k: sources[k].centroids.detach().clone() for k in CHANGED}
    base_tunnel = {
        k.removeprefix("internal_tunnel."): v for k, v in side.items()
        if k.startswith("internal_tunnel.")
    }
    return model, sources, first, later, ptr, head, tunnel, base_core, base_tunnel


def all_456_cached_cpu(
    state, rows, keys, routes, tok, bank, scorer, *, bridge=None,
):
    model, sources, first, later, ptr, head, tunnel, centroids, original_tunnel = state
    output = {}
    times = []
    selected = None
    for i, key in enumerate(keys):
        profile = routes[key]
        if selected != profile:
            apply_profile(sources, tunnel, profile, bank, centroids, original_tunnel)
            selected = profile
        start = time.perf_counter()
        if bridge is None:
            tokens, _ = cached_full_cpu_with_neural_copy(
                model, sources, first, later, head, tok,
                rows[key]["question"], ptr, scorer,
            )
        else:
            with bridge.profile(profile):
                tokens, _ = cached_full_cpu_with_neural_copy(
                    model, sources, first, later, head, tok,
                    rows[key]["question"], ptr, scorer,
                )
        if i < SPEED_SAMPLE:
            times.append(time.perf_counter() - start)
        output[key] = tokens
    return output, times


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "S98_copy_net", "S98_report", "S95_router", "S95_report",
        "S94_bank", "S94_report", "S93_CB61", "S93_sidecar",
        "S99_full456_report", "S101_full456_report", "S108_parity_report",
        "S87_report", "S91_report", "S93_report",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("REAL Apple MPS GPU with backward is required")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    stage_a = json.loads(a.S108_parity_report.read_text())
    frozen = json.loads(a.S101_full456_report.read_text())
    old = json.loads(a.S99_full456_report.read_text())
    if (stage_a.get("stage") != "S108_TWO_DONOR_ALL_12_ZERO_GAIN_PHYSICAL_CPU_TRAIN_456"
            or stage_a.get("full_456_exact_parity") is not True
            or stage_a.get("candidate_full_EOS_correct") != 353
            or stage_a.get("candidate_first_token_correct") != 387
            or stage_a.get("physical", {}).get("physically_below_0_7_bpp") is not True
            or stage_a.get("cpu_overhead_at_most_10_percent") is not True
            or frozen.get("cached_full_EOS_correct") != 353
            or old.get("S99_new_correct_full_EOS") != 353):
        raise AssertionError("S109 cannot TRAIN before genuine S108 Stage A passes")

    for path, expected_sha in (
        (a.S93_CB61, old["immutable_S95_source_sha256"]),
        (a.S94_bank, old["S94_weightbank_sha256"]),
        (a.S95_router, old["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net, old["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes()) != expected_sha:
            raise AssertionError("S101 provenance SHA mismatch")
    raw = s87.open_envelope(a.S93_CB61.read_bytes())
    side = load_file(str(a.S93_sidecar))
    _, _, _, bank, router, scorer = load_frozen_numeric_weights(a)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK = tok
    reports = {profile: freeze_report(getattr(a, profile + "_report"))
               for profile in ("S87", "S91", "S93")}
    rows, keys, routes = cohort(tok, reports, router)
    expected = {k: rows[k]["expected"] for k in keys}
    cpu = load_cpu_original(raw, side)
    try:
        base, base_speed = all_456_cached_cpu(
            cpu, rows, keys, routes, tok, bank, scorer,
        )
        protected = [k for k in keys if base[k] == expected[k]]
        protected_first = [
            k for k in keys if base[k] and base[k][0] == expected[k][0]
        ]
        if len(protected) != 353 or len(protected_first) != 387:
            raise AssertionError("S101 CPU 353 full/387 first baseline was changed")

        wrong = [k for k in keys if routes[k] == "S93" and k not in protected]
        replay = [k for k in keys if routes[k] == "S93" and k in protected]
        if not wrong or len(replay) < 4:
            raise AssertionError("No valid S93-only TRAIN wrong/protected cohort")

        # Full MPS backward on original compressed 36M source. All
        # weights are frozen: ONLY layer_gains requires gradients.
        student, sources = create_small_paid_model(
            CB59(raw), build_config(MODEL_SPEC)
        )
        student.to("mps").eval().requires_grad_(False)
        inner = IntegratedGreenTunnel(layers=(3, 9), rank=8).to("mps").eval()
        inner.load_state_dict({
            k.removeprefix("internal_tunnel."): v for k, v in side.items()
            if k.startswith("internal_tunnel.")
        }, strict=True)
        inner.requires_grad_(False)
        install_verified_fastpath(inner)
        inner.install(student)
        bridge = DualDonorInterlayerGreenBridge(inner).to("mps").eval().install(student)
        first, later, ptr = s91.frozen_controllers(side)
        head = student.lm_head.source().detach()
        param = bridge.layer_gains
        opt = torch.optim.Adam([param], lr=MPS_LR)
        tx = AtomicProtectedUpdate([param], opt)
        trace = []
        try:
            with bridge.profile("S93"):
                for step in range(STEPS):
                    # TRAIN examples only. Full response+EOS replay.
                    chosen = [
                        rows[wrong[(2 * step + i) % len(wrong)]] for i in range(2)
                    ] + [
                        rows[replay[(4 * step + i) % len(replay)]] for i in range(4)
                    ]
                    opt.zero_grad(set_to_none=True)
                    losses = []
                    for item in chosen:
                        clear(sources)
                        losses.append(s91.policy_loss(
                            student, tok, item, first, later, ptr, head,
                            protected=item["id"] in protected,
                        ))
                    loss = torch.stack(losses).mean()
                    if not torch.isfinite(loss):
                        raise AssertionError("Nonfinite S109 train loss")
                    loss.backward()
                    if (param.grad is None
                            or not torch.isfinite(param.grad).all()
                            or not bool(torch.count_nonzero(param.grad).item())):
                        raise AssertionError("No valid actual MPS gradient in layer gains")
                    grad_nonzero = int(torch.count_nonzero(param.grad).item())
                    grad_norm = float(torch.nn.utils.clip_grad_norm_([param], 1.0))
                    opt.step()
                    with torch.no_grad():
                        param.clamp_(-0.25, 0.25)
                    trace.append({
                        "step": step + 1,
                        "TRAIN_rows": len(chosen),
                        "loss": float(loss.detach()),
                        "nonzero_gain_gradients": grad_nonzero,
                        "gradient_norm": grad_norm,
                        "max_abs_gain": float(param.abs().max()),
                    })
                    print(json.dumps({"event": "S109_MPS_GAIN_ONLY_STEP", **trace[-1]}),
                          flush=True)
                    clear(sources)

            learned = bridge.physical_sidecar_payload()
            # Saved sidecar contains new gains ONCE, plus all original
            # existing S101 tensors unchanged.
            proposed_side = dict(side)
            proposed_side.update(learned)
            proposed_file = a.out / "S109_UNACCEPTED_DUAL_GAIN_S93_SIDECAR.safetensors"
            save_file(proposed_side, str(proposed_file))
            again = load_file(str(proposed_file))
            if (set(again) != set(proposed_side)
                    or any(not torch.equal(again[k], proposed_side[k])
                           for k in proposed_side)):
                raise AssertionError("S109 learned gain physical round-trip failed")
            for key in side:
                if not torch.equal(again[key], side[key]):
                    raise AssertionError("S109 changed frozen original controller")
            config = {
                "stage": "S109_UNACCEPTED_DUAL_DONOR_TWENTY_GAINS",
                "connected_missing_layers": list(bridge.additional_layers),
                "donor_layers": [3, 9], "existing_layers": [3, 9],
                "enable_only_route": "S93",
                "sidecar_key": "dual_bridge.layer_gains",
            }
            config_file = a.out / "S109_UNACCEPTED_RUNTIME_CONFIG.json"
            config_file.write_text(json.dumps(config, sort_keys=True,
                                               separators=(",", ":")))
            added_bytes = (proposed_file.stat().st_size
                           - a.S93_sidecar.stat().st_size
                           + config_file.stat().st_size)
            bpp = float(frozen["S99_all_inclusive_physical_bpp"]) + (
                8.0 * added_bytes / SOURCE_PARAMS
            )
            if not math.isfinite(bpp) or added_bytes < 40:
                raise AssertionError("Invalid S109 physical source/accounting")

            # Physically RELOAD only the new stored gains onto the original
            # frozen CPU S101 model. The S95/S98 routing + KV-cache decode
            # remains the genuine S101 path.
            cpu_bridge = DualDonorInterlayerGreenBridge(
                cpu[6], total_layers=12, donor_layers=(3, 9),
            ).eval()
            cpu_bridge.load_physical_sidecar_payload({
                "dual_bridge.layer_gains": again["dual_bridge.layer_gains"]
            })
            cpu_bridge.install(cpu[0])
            try:
                observed, proposed_speed = all_456_cached_cpu(
                    cpu, rows, keys, routes, tok, bank, scorer,
                    bridge=cpu_bridge,
                )
            finally:
                cpu_bridge.uninstall()
            # Same-host ratio is S108 over S101; S82 ratio should be
            # checked in a separate full canonical native CPU benchmark.
            same_run_ratio = (statistics.median(base_speed)
                              / statistics.median(proposed_speed))
            guard = strict_train_retention(
                expected, base, observed, physical_bpp=bpp,
                cpu_ratio=same_run_ratio, speed_required=True,
            )
            evidence = tx.finalize(guard)
            if not evidence["committed"]:
                if not torch.equal(param.detach(), torch.zeros_like(param)):
                    raise AssertionError("S109 full rollback failed")
            record = {
                "stage": "S109_REAL_MPS_DUAL_DONOR_TWENTY_GAINS_456_CPU_TRANSACTION",
                "real_MPS_steps": STEPS,
                "gradient_trace": trace,
                "trainable_parameter_count": 20,
                "two_preexisting_frozen_Green_Tunnel_donors_reused":True,
                "frozen_core_and_original_S93_tunnel": True,
                "frozen_S87_S91_router_copy": True,
                "original_353_full_387_first_confirmed_on_456": True,
                "baseline_full": len(protected),
                "candidate_full": sum(observed[k] == expected[k] for k in keys),
                "candidate_first": sum(
                    bool(observed[k]) and observed[k][0] == expected[k][0]
                    for k in keys
                ),
                "gains": again["dual_bridge.layer_gains"].tolist(),
                "guard": guard,
                "rollback": evidence,
                "speed_same_host_S101_to_S109": same_run_ratio,
                "same_host_S82_speed_still_pending": True,
                "bpp": bpp,
                "physical_bytes_added": added_bytes,
                "S101_original_CB61_sha256": sha(a.S93_CB61.read_bytes()),
                "S109_sidecar_SHA256": sha(proposed_file.read_bytes()),
                "independent_QA_never_opened": True,
                "research_candidate_eligible_for_further_checks":
                    evidence["committed"] and guard["gained_full"] > 0,
                "model_accepted": False,
                "LIVE_main_Render_Supabase_APK_untouched": True,
            }
            (a.out / "S109_MPS_GAIN_ONLY_456_CPU_GUARD_REPORT.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False)
            )
            print(json.dumps({
                "event": "S109_RESEARCH_TRANSACTION_END",
                "baseline_full": record["baseline_full"],
                "candidate_full": record["candidate_full"],
                "lost_full": guard["lost_full"],
                "lost_first": guard["lost_first"],
                "gained_full": guard["gained_full"],
                "committed": evidence["committed"],
                "bpp": bpp,
                "CPU_ratio_vs_S101": same_run_ratio,
                "model_accepted": False,
            }), flush=True)
        finally:
            bridge.uninstall()
            inner.uninstall()
    finally:
        cpu[6].uninstall()


if __name__ == "__main__":
    main()
