"""S105 REAL Apple MPS read-only gradient interference audit of all-layer GT.

Read-only on frozen S101 compressed 36M CB61 source, S93 Green Tunnel, and
S95/S98 routing. Only original authorized 456 TRAIN questions may be used.
Never opens QA, trains optimizer, serializes new model, or touches LIVE.

Calculate per-example d(full-response policy surrogate)/d(10 shared GT
gains) for S93-route wrong vs protected answers. Analyze conflicts,
project a candidate first-order SAFE direction. Projection is diagnosis,
not proof of exact response preservation. S106 physical full-456 inference
and rollback would be required for any proposed gain update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s91 import train_joint_quality_rehearsal as s91
from calibri2.s94.physical_profile_bank import freeze_report, sha
from calibri2.s99.guard_neural_copy_456_cpu import load_frozen_numeric_weights
from calibri2.s104.gain_only_mps_transaction import cohort
from calibri2.s103.shared_all_layer_green_bridge import SharedInterLayerGreenBridge
from calibri2.s105.protected_gain_direction import project_protected_gain_direction
from v176.operators import clear

MAX_WRONG = 8
MAX_PROTECTED = 12


def balanced_subset(keys, cap):
    """Round-robin the six previously approved TRAIN families, not one family."""
    by_family = defaultdict(list)
    for key in sorted(keys):
        by_family[key.split("/", 1)[0]].append(key)
    output = []
    while len(output) < cap and any(by_family.values()):
        for family in sorted(by_family):
            if len(output) == cap:
                break
            if by_family[family]:
                output.append(by_family[family].pop(0))
    return output


def main():
    parser = argparse.ArgumentParser()
    for key in (
        "S98_copy_net", "S98_report", "S95_router", "S95_report",
        "S94_bank", "S94_report", "S93_CB61", "S93_sidecar",
        "S99_full456_report", "S101_full456_report", "S103_parity_report",
        "S87_report", "S91_report", "S93_report",
    ):
        parser.add_argument(key, type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S105 requires genuine MPS backward")
    torch.manual_seed(1050105)
    torch.set_num_threads(2)

    b101 = json.loads(args.S101_full456_report.read_text())
    b103 = json.loads(args.S103_parity_report.read_text())
    p99 = json.loads(args.S99_full456_report.read_text())
    if (b101.get("cached_full_EOS_correct") != 353
            or b101.get("new_first_token_correct") != 387
            or p99.get("S99_new_correct_full_EOS") != 353
            or b103.get("stage") != "S103_ALL_12_SHARED_GT_ZERO_GAIN_PHYSICAL_CPU_TRAIN_456"
            or b103.get("full_456_exact_parity") is not True
            or b103.get("candidate_full_EOS_correct") != 353
            or b103.get("candidate_first_token_correct") != 387
            or b103.get("physical", {}).get("physically_below_0_7_bpp") is not True):
        raise AssertionError("Frozen S101/S103 guard missing")

    for name, digest in (
        ("S93_CB61", p99["immutable_S95_source_sha256"]),
        ("S94_bank", p99["S94_weightbank_sha256"]),
        ("S95_router", p99["S95_neural_profile_selector_sha256"]),
        ("S98_copy_net", p99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(getattr(args, name).read_bytes()) != digest:
            raise AssertionError("Frozen physical artifact altered: " + name)

    _, _, _, bank, router, scorer = load_frozen_numeric_weights(args)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK = tok
    reports = {p: freeze_report(getattr(args, p+"_report"))
               for p in ("S87", "S91", "S93")}
    rows, ids, routes = cohort(tok, reports, router)
    # Do not derive labels from any exposed QA; the immutable TRAIN report
    # records authoritative S101 goals, only used to divide TRAIN cases.
    old_s101_correct = set()
    # Original S101 full correct ids may be reconstructed from S99 evidence.
    # Compare frozen S103 CPU TRAIN report has only COUNTS, never lookup.
    # Reconstruct *genuine* 456 S101 results to classify the old skills.
    # Reusing a frozen source prediction as training target is forbidden:
    # the split below is ONLY for balancing TRAIN cohort identity protection.
    from calibri2.s104.gain_only_mps_transaction import (
        load_cpu_original, all_456_cached_cpu,
    )
    cpu = load_cpu_original(s87.open_envelope(args.S93_CB61.read_bytes()),
                            load_file(str(args.S93_sidecar)))
    try:
        baseline, _ = all_456_cached_cpu(cpu, rows, ids, routes, tok, bank, scorer)
    finally:
        cpu[6].uninstall()
    old_s101_correct = {
        key for key in ids if baseline[key] == rows[key]["expected"]
    }
    prior_first = sum(bool(baseline[key]) and
                      baseline[key][0] == rows[key]["expected"][0]
                      for key in ids)
    if len(old_s101_correct) != 353 or prior_first != 387:
        raise AssertionError("S105 exact S101 source quality regressed")

    wrong = balanced_subset(
        [key for key in ids if routes[key] == "S93"
         and key not in old_s101_correct], MAX_WRONG)
    protected = balanced_subset(
        [key for key in ids if routes[key] == "S93"
         and key in old_s101_correct], MAX_PROTECTED)
    if len(wrong) != MAX_WRONG or len(protected) != MAX_PROTECTED:
        raise AssertionError("Not enough S93 wrong/protected TRAIN families")

    raw = s87.open_envelope(args.S93_CB61.read_bytes())
    side = load_file(str(args.S93_sidecar))
    student, sources = create_small_paid_model(CB59(raw), build_config(MODEL_SPEC))
    student.to("mps").eval().requires_grad_(False)
    old_gt = IntegratedGreenTunnel(layers=(3, 9), rank=8).to("mps").eval()
    old_gt.load_state_dict({
        key.removeprefix("internal_tunnel."): value for key, value in side.items()
        if key.startswith("internal_tunnel.")
    }, strict=True)
    old_gt.requires_grad_(False)
    install_verified_fastpath(old_gt)
    old_gt.install(student)
    bridge = SharedInterLayerGreenBridge(old_gt).to("mps").eval().install(student)
    first, later, pointer = s91.frozen_controllers(side)
    head = student.lm_head.source().detach()

    gradients = {}
    losses = {}
    try:
        with bridge.profile("S93"):
            for key in wrong + protected:
                clear(sources)
                row = rows[key]
                loss = s91.policy_loss(
                    student, tok, row, first, later, pointer, head,
                    protected=key in old_s101_correct,
                )
                if not torch.isfinite(loss):
                    raise AssertionError("Nonfinite S105 frozen TRAIN surrogate loss")
                (grad,) = torch.autograd.grad(
                    loss, bridge.layer_gains, retain_graph=False,
                )
                if grad is None or not torch.isfinite(grad).all():
                    raise AssertionError("No gradient in S103 shared gain paths")
                gradients[key] = grad.detach().cpu().double()
                losses[key] = float(loss.detach())
                print(json.dumps({
                    "event": "S105_TRAIN_SINGLE_GRADIENT",
                    "family": key.split("/", 1)[0],
                    "protected": key in old_s101_correct,
                    "gradient_norm": float(grad.detach().norm()),
                    "loss": losses[key],
                }), flush=True)
                clear(sources)
    finally:
        bridge.uninstall()
        old_gt.uninstall()

    wrong_matrix = torch.stack([gradients[key] for key in wrong])
    safe_matrix = torch.stack([gradients[key] for key in protected])
    target = wrong_matrix.mean(0)
    projected = project_protected_gain_direction(target, safe_matrix)

    # Individual per-example negative directions may help the new example
    # but damage previously correct examples. Count antagonistic gradient
    # pairs. Never claim projected first-order safety is exact token safety.
    opponent_pairs = sum(
        float(torch.dot(x,y)) < 0 for x in wrong_matrix for y in safe_matrix
    )
    slopes = {
        str(index): {
            "desired_mean_loss_gradient": float(target[position]),
            "protected_mean_loss_gradient": float(safe_matrix[:,position].mean()),
            "projected_update": float(projected.direction[position]),
            "prior_update": float(projected.original_direction[position]),
        }
        for position, index in enumerate(bridge.additional_layers)
    }
    if any(not math.isfinite(v) for data in slopes.values() for v in data.values()):
        raise AssertionError("S105 non-finite per-layer diagnosis")

    report = {
        "stage": "S105_REAL_MPS_GAINS_OLD_VS_NEW_GRADIENT_CONFLICT_TRAIN_ONLY",
        "no_training_or_optimizer_step": True,
        "train_rows_old_456": 456,
        "S101_full_correct": len(old_s101_correct),
        "S101_first_correct": prior_first,
        "S93_wrong_gradient_examples": len(wrong),
        "S93_protected_gradient_examples": len(protected),
        "protect_first_order_only_not_an_inference_guard": True,
        "opposing_gradients_count": opponent_pairs,
        "all_pairs": len(wrong) * len(protected),
        "desired_first_order_improvement": projected.desired_improvement,
        "projected_first_order_improvement": projected.projected_improvement,
        "projected_norm_ratio": projected.norm_ratio,
        "previous_halfspace_violations": projected.original_violations,
        "remaining_halfspace_violations": projected.remaining_violations,
        "iterations": projected.iterations,
        "layers": slopes,
        "previously_inspected_sealed_QA_unused": True,
        "S106_physical_all456_guard_mandatory_before_weight_change": True,
        "model_accepted": False,
        "LIVE_main_Render_Supabase_APK_untouched": True,
    }
    (args.out/"S105_GUARDED_PER_LAYER_GRADIENT_DIAGNOSTIC.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "event": "S105_MPS_GRADIENT_CAUSAL_DIAG_COMPLETE",
        "old_full": report["S101_full_correct"],
        "old_first": report["S101_first_correct"],
        "opposed_pairs": opponent_pairs,
        "projected_gain_norm_ratio": projected.norm_ratio,
        "projected_improvement": projected.projected_improvement,
        "no_new_model": True,
    }), flush=True)


if __name__=="__main__":
    main()
