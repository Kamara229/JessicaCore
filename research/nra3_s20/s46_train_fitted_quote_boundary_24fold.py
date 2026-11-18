"""S46: lexical-identity-free TRAIN-fitted quote-boundary readout for S45.

RESEARCH-ONLY shadow, no Calibri43M/GT/NRA2 forward. The S45 model,
optimizer and 64 additional TRAIN examples are unchanged. A single
nonnegative, learned scalar calibrates its frozen quote logits using the
already-existing RAW Unicode quote-membership feature (column 0 of q).
Only TRAIN-side quote labels fit the scalar. The identical 24 previously
inspected synthetic grouped TRAIN-development folds are diagnostic,
NEVER independent validation or an acceptance gate. The source generator
uses the same quote parser as the input feature, so any improvement is
partly a consistency check, not evidence of grammar generalization.
Do not import or inspect protected S38 examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import model_weights_digest
from research.nra3_s20.s43_new128_multi_event_trainonly_source import (
    build_train128, train_partitions, train_source_sha,
    validate_source as validate_old, PATTERNS,
)
from research.nra3_s20.s43_train128_joint_predicate_source_24fold import (
    batch_raw_plus_train_labels, score_after_frozen_training,
)
from research.nra3_s20.s44_train128_quote_calibrated_source_contrast_24fold import fit_s44
from research.nra3_s20.s45_separately_frozen_new64_scope_curriculum import (
    train64, source_digest, validate_source as validate_new,
)

STAGE = "NRA3_S46_TRAIN_FITTED_RAW_QUOTE_BOUNDARY_CALIBRATION_24FOLD"
S45_RUN = 44252197919
S46_SEED = 461118
CALIBRATION_STEPS = 120
CALIBRATION_LR = .15
CALIBRATION_L2 = .001
S45_BLOB = "3cc3ed61a9798e78dffbbad0692dd6f8789c59b8"


def require(ok, reason):
    if not bool(ok):
        raise ValueError("S46 quote boundary research gate: " + reason)


class QuoteBoundaryReadout(nn.Module):
    """Only 1 scalar fitted; RAW q[...,0] is not a TRAIN label input."""
    def __init__(self):
        super().__init__()
        self.raw_strength = nn.Parameter(torch.tensor(-1., dtype=torch.float32))
        self.output_gate = 0
        self.candidate_accepted = False
        self.activation_allowed = False

    def forward(self, original_logits, raw_quote_column):
        require(original_logits.shape == raw_quote_column.shape,
                "raw quote boundary dimension mismatch")
        require(bool(((raw_quote_column == 0) | (raw_quote_column == 1)).all()),
                "raw punctuation quote bit must be binary")
        strength = F.softplus(self.raw_strength)
        return original_logits + strength * (2. * raw_quote_column - 1.)

    def positive_strength(self):
        return float(F.softplus(self.raw_strength).detach())


def quote_logits_and_labels(frozen_s45, rows):
    pred, q, pair, st, valid, _, _, qgold, _ = batch_raw_plus_train_labels(rows)
    with torch.no_grad():
        frozen_logits = frozen_s45(pred, q, pair, st).quote_logits.detach()
    return frozen_logits, q[..., 0], qgold, valid


def fit_only_train_boundary_scalar(frozen_s45, train_rows, fold):
    require(len(train_rows) in (176, 184), "S45 source/extended train geometry changed")
    original_sha = model_weights_digest(frozen_s45)
    original_logits, boundary, qgold, valid = quote_logits_and_labels(
        frozen_s45, train_rows)
    pos = (qgold > .5) & valid
    neg = (qgold <= .5) & valid
    require(bool(pos.any()) and bool(neg.any()), "missing TRAIN quote classes")
    # The boundary is supplied by the Unicode raw feature parser only.
    # No human-labelled TRAIN token position is used in this inference call.
    torch.manual_seed(S46_SEED + fold)
    calibration = QuoteBoundaryReadout()
    optim = torch.optim.AdamW(calibration.parameters(), lr=CALIBRATION_LR,
                              weight_decay=0.)
    initial = None
    for step in range(CALIBRATION_STEPS):
        optim.zero_grad(set_to_none=True)
        logits = calibration(original_logits, boundary)
        loss = (F.softplus(-logits[pos]).mean()
                + F.softplus(logits[neg]).mean()
                + CALIBRATION_L2 * F.softplus(calibration.raw_strength).square())
        require(bool(torch.isfinite(loss)), "nonfinite quote TRAIN loss")
        loss.backward()
        require(calibration.raw_strength.grad is not None
                and bool(torch.isfinite(calibration.raw_strength.grad)),
                "missing scalar gradient")
        optim.step()
        if step == 0:
            initial = float(loss.detach())
    require(model_weights_digest(frozen_s45) == original_sha,
            "S46 changed original S45 weights")
    calibration.eval()
    return calibration, {"TRAIN_loss_initial": initial,
                         "TRAIN_loss_final": float(loss.detach()),
                         "learned_positive_boundary_strength": calibration.positive_strength(),
                         "TRAIN_steps": CALIBRATION_STEPS, "TRAIN_lr": CALIBRATION_LR}


def counts(predicted, gold, valid):
    tp = int((predicted & gold & valid).sum())
    fp = int((predicted & ~gold & valid).sum())
    fn = int((~predicted & gold & valid).sum())
    return {"TP": tp, "FP": fp, "FN": fn}


def per_fold_held_quotes(frozen_s45, calibration, held):
    old, boundary, qgold, valid = quote_logits_and_labels(frozen_s45, held)
    with torch.no_grad():
        new = calibration(old, boundary)
    original_pred = (old >= 0) & valid
    new_pred = (new >= 0) & valid
    reference = (qgold > .5) & valid
    original = counts(original_pred, reference, valid)
    calibrated = counts(new_pred, reference, valid)
    # Offline forensic only: raw membership of every false positive.
    # These comparisons are strictly after fitting and freezing the scalar.
    outside_fp = int((original_pred & ~reference & (boundary == 0) & valid).sum())
    inside_fp = int((original_pred & ~reference & (boundary == 1) & valid).sum())
    require(original["FP"] == outside_fp + inside_fp, "quote false-positive accounting")
    return original, calibrated, {
        "S45_raw_boundary_0_false_positives": outside_fp,
        "S45_raw_boundary_1_false_positives": inside_fp,
        "same_raw_boundary_as_auto_gold_on_held": bool(
            ((boundary > .5) == reference)[valid].all()),
    }


def micro(records, variant, axis):
    selected = [r[variant] for r in records if r["axis"] == axis]
    tp, fp, fn = (
        sum(m[k] for m in selected) for k in ("TP", "FP", "FN"))
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    require(not args.out.exists(), "no report overwrite permitted")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    old = validate_old()
    extra = validate_new()
    require(old["count"] == 128 and old["grouped_folds"] == 24
            and old["source_sha256"] == train_source_sha()
            and old["S38_reserved_not_imported_or_used"],
            "original NEW128 changed or S38 control accessed")
    require(extra["TRAIN_only_rows"] == 64
            and extra["source_sha256"] == source_digest()
            and extra["no_S38_examples_loaded"],
            "S45 sealed additional64 changed")
    rows = build_train128()
    additional = train64()
    require(len(rows) == 128 and len(additional) == 64,
            "S45 original source geometry mismatch")
    records = []
    for fold, (axis, group, tr, held) in enumerate(train_partitions()):
        require(len(tr) == (112 if axis == "LEXICAL" else 120)
                and len(held) == (16 if axis == "LEXICAL" else 8)
                and not (set(tr) & set(held)), "TRAIN group overlap")
        training = [rows[i] for i in tr] + list(additional)
        held_rows = [rows[i] for i in held]
        require(not ({r.text for r in training} & {r.text for r in held_rows}),
                "synthetic original held text leaked into training")
        frozen_s45, training_diag = fit_s44(
            training, fold, allow_S45_new64=True)
        sha = model_weights_digest(frozen_s45)
        calibration, cal_diag = fit_only_train_boundary_scalar(
            frozen_s45, training, fold)
        old_q, new_q, analysis = per_fold_held_quotes(
            frozen_s45, calibration, held_rows)
        old_all = score_after_frozen_training(frozen_s45, held_rows)
        require(old_all["quote_TP"] == old_q["TP"]
                and old_all["quote_FP"] == old_q["FP"]
                and old_all["quote_FN"] == old_q["FN"],
                "S46 S45 original frozen quote metric mismatch")
        require(model_weights_digest(frozen_s45) == sha,
                "S45 source/link/quote original weights mutated")
        # Important: no recalculation/selection of relation and event heads:
        # they are exactly those of S45; only isolated quote readout differs.
        records.append({
            "fold": fold, "axis": axis, "group_OFFLINE_ONLY": group,
            "group_name_OFFLINE_ONLY": PATTERNS[group]
                if axis == "CONSTRUCTION" else "lexical_family",
            "TRAIN_rows": len(training),
            "original_held_synthetic_TRAIN_dev_rows": len(held_rows),
            "S45_frozen_CPU_weights_SHA256": sha,
            "calibration_train_only": cal_diag,
            "S45_original_quote": old_q, "S46_calibrated_quote": new_q,
            "quote_boundary_diagnostics_OFFLINE_ONLY": analysis,
            "S45_unchanged_event_source": {k: old_all["event_source_" + k]
                                             for k in ("TP", "FP", "FN")},
            "S45_unchanged_REJECTED_links": {k: old_all["REJECTED_link_" + k]
                                             for k in ("TP", "FP", "FN")},
            "source_AND_link_output_unchanged_by_design": True,
            "no_S38_control_ever_loaded_scored": True,
            "output_gate": 0,
        })
        print("S46_TRAIN_FITTED_BOUNDARY_FOLD", json.dumps({
            "fold": fold, "axis": axis, "group": group,
            "S45_quote": old_q, "S46_quote": new_q,
            "outside_FP": analysis["S45_raw_boundary_0_false_positives"],
            "strength": cal_diag["learned_positive_boundary_strength"]},
            sort_keys=True), flush=True)
    summary = {axis: {
        "S45_same_frozen_model_quote": micro(records, "S45_original_quote", axis),
        "S46_train_only_scalar_quote": micro(records, "S46_calibrated_quote", axis)}
        for axis in ("LEXICAL", "CONSTRUCTION")}
    result = {
        "stage": STAGE,
        "S45_previous_SUCCESS_workflow_run": S45_RUN,
        "S45_original_NEW64_frozen_git_blob": S45_BLOB,
        "S43_original128_sha256": old["source_sha256"],
        "S45_extra64_sha256": extra["source_sha256"],
        "experiment": "train a single positive softplus coefficient on raw quote-bit, frozen S45 logits",
        "original_S45_weights_loss_optimizer_120steps_UNCHANGED": True,
        "only_train_held_disjoint_training_of_quote_calibrator": True,
        "no_held_labels_or_group_for_calibration_training": True,
        "same_24_previous_inspected_synthetic_TRAIN_development_folds": records,
        "quote_summary": summary,
        "quote_gold_auto_generated_using_same_quote_parser_as_raw_feature": True,
        "boundary_consistency_does_not_prove_general_Russian_grammar": True,
        "not_independent_validation_NOT_CANDIDATE_ACCEPTANCE": True,
        "S38_adversarial32_not_imported_train_eval_tune": True,
        "no_real_Calibri43M_GT_NRA2_loaded_or_updated": True,
        "J7_616_EVAL32_EVAL128_bpp_CB59_NOT_TESTED": True,
        "LIVE_Render_Supabase_Jessica_APK_unmodified": True,
        "output_gate": 0,
        "candidate_accepted": False,
        "activation_allowed": False,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out / "NRA3_S46_TRAIN_FITTED_QUOTE_BOUNDARY_24FOLD.json"
    out.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False,
                              indent=2) + "\n", encoding="utf-8")
    print("S46_SAME_SYNTHETIC_TRAIN_DEV_QUOTE_SUMMARY",
          json.dumps(summary, sort_keys=True), flush=True)
    print("S46_NO_INDEPENDENT_EVAL_NO_ACTIVE_NRA2_NO_PROMOTION", flush=True)


if __name__ == "__main__":
    main()
