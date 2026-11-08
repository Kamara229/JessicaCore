"""CALIBRI S36: one-step margin-preserving TRAIN experiment.

Research-only experiment. Replay S33 exactly from immutable S26 CB59; no
QA/VALID/DEV, no Jessica chat test, and no LIVE changes. At the SAME S34
first new TRAIN pair/optimizer moments, compare:
 A) unchanged S34 lr=3e-6 (must reproduce S35's lost Yes)
 B) tiny trust-region LR=5e-8 (strict positive Yes-over-No margin)
 C) same LR=3e-6 plus first-order logit-margin correction on 1_left.
Every candidate is individually checked against original six historical
TRAIN exact+EOS replies and new TRAIN loss. No trained weights exported.
S36 can fail; report failures instead of declaring progress.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s34 import continued_protected_color_noun_fp32 as s34
from calibri2.s35.diagnose_exact_s34_step1_1left_logits import one_prompt_logits
from v186.pairs import build_pairs
from v174.run import save

LR_S34 = 3e-6
LR_MICRO = 5e-8
S36_MARGIN_FLOOR = 0.02  # Fixed in advance, not selected using QA.
S36_CORRECTION_PASSES = 4
PAIR_ID = 5000


def margin_first(model, row, yes_id, no_id):
    logits = model(input_ids=row["prompt_ids"][None], use_cache=False).logits[0, -1]
    return logits[yes_id] - logits[no_id]


def describe(model, opt, tok, old, pair, row, protected, initial_loss, label):
    with torch.inference_mode():
        current_loss = float(s34.loss_s34(model, "new", pair, tok))
        out = one_prompt_logits(model, tok, row)
        yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        no_id = tok.encode("No", add_special_tokens=False)[0]
        margin = float(margin_first(model, row, yes_id, no_id))
    kept = s34.exact_old(model, tok, old)
    return {
        "method": label,
        "new_pair_loss": current_loss,
        "new_pair_loss_delta": current_loss - initial_loss,
        "all_original_six_old_TRAIN_preserved": protected.issubset(kept),
        "lost_original_six_ids": sorted(protected - kept),
        "1_left_margin_Yes_minus_No": float(margin),
        "1_left_correct_exact_EOS": out["exact_full_answer_EOS"],
        "1_left_margin_floor_pass": margin >= S36_MARGIN_FLOOR,
        "accepted_by_predeclared_S36_constraints":
            (current_loss + 1e-7 < initial_loss and
             protected.issubset(kept) and
             out["exact_full_answer_EOS"] and margin >= S36_MARGIN_FLOOR),
        "1_left": out,
        "weights_checksum": s30.snapshot_sha(model)
    }


def take_s34_step(model, opt, tok, old, pair, lr):
    for group in opt.param_groups:
        group["lr"] = lr
    return s34.update_with_protected_anchor(
        model, opt, list(model.parameters()), tok, "new", pair, old[:4])


def correct_margin_in_place(model, row, yes_id, no_id, floor):
    """Constrained weight-space correction, no optimizer step or QA labels.

    Locally project a harmful candidate weight vector toward +grad(margin),
    using re-linearization when needed. Measure actual logits afterwards;
    accept only if strict old-six full+EOS and new TRAIN-loss gates pass.
    """
    trace = []
    params = list(model.parameters())
    for index in range(S36_CORRECTION_PASSES):
        model.zero_grad(set_to_none=True)
        margin = margin_first(model, row, yes_id, no_id)
        before = float(margin.detach())
        if before >= floor:
            trace.append({"pass": index, "already_safe": True,
                          "actual_margin_before": before})
            break
        margin.backward()
        grad = [p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p) for p in params]
        norm_sq = float(sum((g.double()*g.double()).sum() for g in grad))
        if not (0 < norm_sq < float("inf")):
            raise AssertionError("Margin gradient invalid")
        multiplier = (floor-before)*1.25/norm_sq
        with torch.no_grad():
            for p, g in zip(params, grad):
                p.add_(g, alpha=multiplier)
        with torch.inference_mode():
            after = float(margin_first(model, row, yes_id, no_id))
        trace.append({
            "pass": index+1,
            "margin_before": before, "margin_after": after,
            "margin_gradient_norm_sq": norm_sq,
            "correction_multiplier": multiplier,
            "correction_succeeded": after >= floor,
        })
        del grad
        if after >= floor:
            break
    model.zero_grad(set_to_none=True)
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(s34.SEED)
    torch.set_num_threads(2)
    raw = args.source.read_bytes()
    if len(raw) != SOURCE_BYTES or hashlib.sha256(raw).hexdigest() != s30.SOURCE_SHA:
        raise AssertionError("Not exact original S26 immutable CB59")
    cb = CB59(raw)
    if cb.meta["original_unique_parameter_count"] != SOURCE_PARAMS:
        raise AssertionError("36M original parameter count changed")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK = tok
    new, old = s30.make_training_pairs(tok), build_pairs(tok, "TRAIN")
    assert len(new) == 32 and len(old) == 24
    row = old[1]["left"]
    assert old[1]["pair_id"] == 1 and row["answer"] == "Yes"
    pair = new[0]
    assert pair["pair_id"] == PAIR_ID
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    assert yes_id != no_id
    model = s30.dense_from_paid(cb)
    protected = s34.exact_old(model, tok, old)
    assert len(protected) == 6 and "1_left" in protected
    print(json.dumps({"event":"S36_S33_REPLAY_START"}),flush=True)
    opt, replay = s34.reproduce_s33(model, tok, new, old, protected)
    if not replay["matches_observed_S33_run_42484161593"]:
        raise AssertionError("S33 reproduction failed")
    base_state = copy.deepcopy(model.state_dict())
    base_opt = copy.deepcopy(opt.state_dict())
    with torch.inference_mode():
        before_loss = float(s34.loss_s34(model, "new", pair, tok))
        base_margin = float(margin_first(model, row, yes_id, no_id))
    assert 0 < base_margin < 0.1, "S35 near-tie was not reproduced"
    baseline = describe(model, opt, tok, old, pair, row, protected,
                        before_loss, "S33_before_S34_one_step")

    # Control A, identically reproduce actual S35 candidate and its lost 1_left.
    control_gradient = take_s34_step(model, opt, tok, old, pair, LR_S34)
    control = describe(model, opt, tok, old, pair, row, protected,
                       before_loss, "S34_original_LR3e-6_control")
    if control["1_left_correct_exact_EOS"] or \
       "1_left" not in control["lost_original_six_ids"]:
        raise AssertionError("S36 control did not reproduce S35 1_left loss")
    print(json.dumps({"event":"S36_CONTROL_S35_REPRODUCED",
                      "margin":control["1_left_margin_Yes_minus_No"],
                      "loss":control["new_pair_loss"]}),flush=True)

    # Candidate B: strict tiny trust-region scaling, same AdamW moments.
    model.load_state_dict(base_state)
    opt.load_state_dict(base_opt)
    micro_gradient = take_s34_step(model, opt, tok, old, pair, LR_MICRO)
    micro = describe(model, opt, tok, old, pair, row, protected,
                     before_loss, "S36_micro_trust_LR5e-8")

    # Candidate C: the ORIGINAL S34 AdamW step, then correction along gradient
    # of historical 1_left first-answer Yes-No margin. This does not assume
    # corrected optimizer moments, and is deliberately one-step diagnostic.
    model.load_state_dict(base_state)
    opt.load_state_dict(base_opt)
    take_s34_step(model, opt, tok, old, pair, LR_S34)
    correction_trace = correct_margin_in_place(
        model, row, yes_id, no_id, S36_MARGIN_FLOOR)
    corrected = describe(model, opt, tok, old, pair, row, protected,
                         before_loss, "S36_full_LR3e-6_margin_correction")
    report = {
        "stage": "CALIBRI_S36_ONE_STEP_MARGIN_GUARD_TRAIN_DIAGNOSTIC",
        "seed_S26_sha256": s30.SOURCE_SHA,
        "seed_physical_bpp": 8*len(raw)/SOURCE_PARAMS,
        "S36_model_dense_FP32_NOT_compressed": True,
        "S33_reproduced": True,
        "S33_accepted_steps": replay["accepted_steps"],
        "fixed_S34_training_pair": PAIR_ID,
        "protected_original_old_ids": sorted(protected),
        "margin_protected_TRAIN_id": "1_left",
        "margin_threshold_fixed_before_trial": S36_MARGIN_FLOOR,
        "S29_immutable_QA_not_accessed": True,
        "S26_previously_inspected_QA_not_accessed": True,
        "all_labels_and_selection_from_TRAIN_only": True,
        "one_step_diagnostic_not_generalization_or_learned_new_model": True,
        "Jessica_chat_test_not_run": True,
        "LIVE_not_modified": True,
        "optimizer_after_manual_projection_not_consistent_for_multistep": True,
        "control_S34_full_step_gradient": control_gradient,
        "tiny_lr_gradient": micro_gradient,
        "margin_repair_iterations": correction_trace,
        "base": baseline,
        "unmodified_S34_candidate": control,
        "micro_trust_candidate": micro,
        "corrected_full_step_candidate": corrected,
        "selection_rule": "new TRAIN loss must improve >=1e-7, original 6 old "
                          "TRAIN exact+EOS must survive, 1_left Yes-No margin >=0.02",
        "next_step": "If a method passes, investigate stability over 20+ "
                     "successive TRAIN-only steps with optimizer-consistent "
                     "projection. Never use S29 QA for method selection.",
    }
    save(args.out/"S36_1LEFT_MARGIN_PRESERVING_STEP_TRAIN_ONLY.json",report)
    print(json.dumps({
        "event":"S36_SINGLE_STEP_RESULTS",
        "control": {"margin":control["1_left_margin_Yes_minus_No"],
                    "correct":control["1_left_correct_exact_EOS"]},
        "micro": {"margin":micro["1_left_margin_Yes_minus_No"],
                  "accepted":micro["accepted_by_predeclared_S36_constraints"],
                  "loss_improved":micro["new_pair_loss"] < before_loss},
        "corrected": {"margin":corrected["1_left_margin_Yes_minus_No"],
                     "accepted":corrected["accepted_by_predeclared_S36_constraints"],
                     "loss_improved":corrected["new_pair_loss"] < before_loss},
        "S29_QA_unread":True, "LIVE_untouched":True,
    },ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
