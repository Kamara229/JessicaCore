"""CALIBRI 2.0 S34: reproducible S33 => protected FP32 color+noun continuation.

This is a TRAIN-only capability/control experiment using the same exact 36M
S26 CB59 physical seed. Reproduce the S33 128-step trajectory, including
AdamW moments, without reading the S33 report or any heldout QA; then continue
80 predeclared FP32 steps. A multi-pair historical TRAIN anchor replaces
S33's single randomly rotating old anchor. Explicit first-color contrast
complements the existing second-noun/anti-EOS margin. Keep the original six
old exact full-answer+EOS IDs after EVERY accepted single-step update.

S29 frozen QA and S26 previously inspected QA must never influence selection.
The recorded 0.6908 bpp is the SEED's physical cost; the newly trained
model below is dense FP32 and is NOT a <=0.7 bpp result. No Jessica tests,
no LIVE / main / Render / Supabase / APK changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s32.microguard_pcgrad_fp32 import old_train_probe, guarded_old_ids
from calibri2.s33 import one_step_lexical_noun_guard_fp32 as s33
from v186.pairs import build_pairs
from v174.run import save

S34_STEPS = 80
S34_LR = (0.00002, 0.000008, 0.000003)
S34_CURRICULUM = ("new", "new", "old", "new")
OLD_ANCHOR_BLEND = 0.40
COLOR_WEIGHT = 4.
NOUN_WEIGHT = 12.
EOS_WEIGHT = 2.
FIRST_COLOR_MARGIN = 1.5
FIRST_COLOR_MARGIN_WEIGHT = .40
NOUN_FOIL_MARGIN = 3.
NOUN_EOS_MARGIN = 2.
SEED = 3030
EXPECTED_S33_ACCEPTED = 30
EXPECTED_S33_NEW = 4
EXPECTED_S33_OLD = 6


def exact_old(model, tok, old):
    """Original six *identities*, not aggregate correctness count."""
    return guarded_old_ids(old_train_probe(model, tok, old))


def s34_color_and_noun_loss(model, pair, tok):
    """Two-sided TRAIN color+noun: first color, distinct second noun, EOS.

    No hardcoded QA labels: both targets and foils come only from the fixed
    S30 TRAIN data. A wrong first token and an early EOS remain observable.
    """
    if pair["kind"] != "new":
        raise AssertionError("Expected a new TRAIN lexical pair")
    l, r = pair["left"], pair["right"]
    if any(len(p["targets"]) != 3 for p in (l, r)):
        raise AssertionError("Expected exact color+noun+EOS tokens")
    if int(l["targets"][0]) != int(r["targets"][0]):
        raise AssertionError("Contrasts must share the first color token")
    if int(l["targets"][1]) == int(r["targets"][1]):
        raise AssertionError("Contrasting nouns may not match")
    parts = []
    for row, foil in ((l, r), (r, l)):
        logits = model(
            input_ids=row["ids"][None], use_cache=False
        ).logits[0][row["positions"]]
        target = row["targets"]
        ce = F.cross_entropy(logits, target, reduction="none")
        # Preserve S33's noun emphasis, now repair first-color failures too.
        weighted = (COLOR_WEIGHT*ce[0] + NOUN_WEIGHT*ce[1] +
                    EOS_WEIGHT*ce[2])/(COLOR_WEIGHT+NOUN_WEIGHT+EOS_WEIGHT)
        noun = logits[1]
        gold_noun = int(target[1])
        wrong_noun = int(foil["targets"][1])
        noun_foil = F.relu(NOUN_FOIL_MARGIN - (
            noun[gold_noun] - noun[wrong_noun]))
        noun_eos = F.relu(NOUN_EOS_MARGIN - (
            noun[gold_noun] - noun[int(tok.eos_token_id)]))
        first = logits[0]
        gold_color = int(target[0])
        # Pick the highest opposing TRAIN prediction without using QA.
        top2 = first.topk(k=2).indices.tolist()
        competitor = top2[1] if top2[0] == gold_color else top2[0]
        color_foil = F.relu(FIRST_COLOR_MARGIN - (
            first[gold_color] - first[competitor]))
        parts.append(weighted + .60*noun_foil + .30*noun_eos +
                     FIRST_COLOR_MARGIN_WEIGHT*color_foil)
    return .5*(parts[0]+parts[1])


def loss_s34(model, kind, pair, tok):
    if kind == "new":
        return s34_color_and_noun_loss(model, pair, tok)
    if kind == "old":
        return s30.train_loss(model, pair, False)
    raise AssertionError("Unknown S34 TRAIN type")


def update_with_protected_anchor(model, opt, params, tok, kind, pair, anchors):
    """Full FP32 step: project on mean 4 old TRAIN anchors, with clipping.

    Compute old anchor gradients incrementally to avoid accumulating four
    computational graphs at once. All eight original old TRAIN prompts are
    included (six protected, two initially wrong).
    """
    opt.zero_grad(set_to_none=True)
    primary = loss_s34(model, kind, pair, tok)
    if not torch.isfinite(primary):
        raise AssertionError("Non-finite S34 TRAIN primary")
    primary_before = float(primary.detach())
    primary.backward()
    g = [p.grad.detach().clone() if p.grad is not None
         else torch.zeros_like(p) for p in params]
    opt.zero_grad(set_to_none=True)
    anchor_before = 0.
    for anchor in anchors:
        old_term = s30.train_loss(model, anchor, False)/len(anchors)
        if not torch.isfinite(old_term):
            raise AssertionError("Non-finite old TRAIN anchor")
        anchor_before += float(old_term.detach())
        old_term.backward()
    h = [p.grad.detach().clone() if p.grad is not None
         else torch.zeros_like(p) for p in params]
    dot = sum((x.double()*y.double()).sum() for x, y in zip(g, h))
    norm = sum((y.double()*y.double()).sum() for y in h)
    if not norm > 0 or not torch.isfinite(norm):
        raise AssertionError("No protected old gradient")
    conflict = bool(dot < 0)
    projection = float(dot/norm) if conflict else 0.
    for p, x, y in zip(params, g, h):
        p.grad = (x - projection*y + OLD_ANCHOR_BLEND*y).detach()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        params, 1.0, error_if_nonfinite=True))
    if not grad_norm > 0:
        raise AssertionError("S34 had no real full FP32 gradient")
    opt.step()
    return {
        "S34_primary_before": primary_before,
        "old_four_pair_anchor_before": anchor_before,
        "gradient_conflict": conflict,
        "gradient_projection": projection,
        "full_FP32_gradient_norm": grad_norm,
    }


def reproduce_s33(model, tok, new, old, protected):
    """Match all 128 S33 TRAIN decisions before any S34 intervention.

    Reuses the actual S33 gradient, loss, AdamW and LR ordering. Does NOT
    read an S33 result file or choose an S33 checkpoint from evaluated QA.
    """
    opt = torch.optim.AdamW(
        model.parameters(), lr=s33.LR_BACKTRACK[0], weight_decay=0.)
    params = list(model.parameters())
    seen = Counter()
    accepted = 0
    failures = []
    for step in range(s33.TOTAL_STEPS):
        kind = s33.SCHEDULE[step % len(s33.SCHEDULE)]
        pos = seen[kind]
        seen[kind] += 1
        pair = new[pos % len(new)] if kind == "new" else old[pos % len(old)]
        anchor = old[(step*7) % len(old)]
        with torch.inference_mode():
            before = float(s33.primary_loss(model, kind, pair, tok))
        weights = copy.deepcopy(model.state_dict())
        moments = copy.deepcopy(opt.state_dict())
        selected = None
        attempts = []
        for lr in s33.LR_BACKTRACK:
            model.load_state_dict(weights)
            opt.load_state_dict(moments)
            for group in opt.param_groups:
                group["lr"] = lr
            info = s33.one_full_dense_fp32_update(
                model, opt, params, tok, kind, pair, anchor)
            with torch.inference_mode():
                after = float(s33.primary_loss(model, kind, pair, tok))
            improved = (after + 1e-7 < before)
            kept = exact_old(model, tok, old) if improved else set()
            passed = improved and protected.issubset(kept)
            attempts.append({
                "LR": lr,
                "reason": "accepted" if passed else (
                    "no_TRAIN_loss_improvement" if not improved
                    else "protected_old_exact_EOS_loss"),
                "lost_original_ids": sorted(protected-kept) if improved else [],
                "loss_before": before,
                "loss_after": after,
                "grad_conflict": info["gradient_conflict"],
            })
            if passed:
                accepted += 1
                selected = lr
                break
        if selected is None:
            model.load_state_dict(weights)
            opt.load_state_dict(moments)
        if step >= 80 or step % 16 == 0:
            failures.append({
                "step": step+1, "kind": kind,
                "accepted_LR": selected,
                "trials": attempts
            })
        if (step+1) % 32 == 0:
            print(json.dumps({
                "event": "S34_REPRODUCING_S33",
                "S33_step": step+1,
                "accepted": accepted
            }), flush=True)
        del weights, moments
    probe = s30.train_probe(model, tok, new, old)
    new_correct = probe["per_kind"]["new_noun_TRAIN"]["full_correct"]
    old_correct = probe["per_kind"]["historical_TRAIN"]["full_correct"]
    compatible = (
        accepted == EXPECTED_S33_ACCEPTED and
        new_correct == EXPECTED_S33_NEW and
        old_correct == EXPECTED_S33_OLD
    )
    return opt, {
        "accepted_steps": accepted,
        "rejected_steps": s33.TOTAL_STEPS-accepted,
        "S33_new_TRAIN_full_EOS": new_correct,
        "S33_old_TRAIN_full_EOS": old_correct,
        "matches_observed_S33_run_42484161593": compatible,
        "S33_decisions_at_end_and_early": failures,
        "S33_final_sha": s30.snapshot_sha(model),
        "S33_full_TRAIN_answers": probe,
        "S33_forced_noun_TRAIN": s33.noun_position_training_probe(
            model, tok, new),
    }


def continue_s34(model, opt, tok, new, old, protected):
    """One predeclared 80-step experiment, no QA, with strict old ID gate."""
    params = list(model.parameters())
    anchors = old[:4]  # Exactly the eight old TRAIN guard question/answers
    seen = Counter()
    results = []
    accepted = 0
    failure_reasons = Counter()
    for step in range(S34_STEPS):
        kind = S34_CURRICULUM[step % len(S34_CURRICULUM)]
        pos = seen[kind]
        seen[kind] += 1
        # Coprime stride covers all 32 fresh TRAIN pairs before repetition.
        pair = (new[(pos*7) % len(new)] if kind == "new"
                else old[(pos*5) % len(old)])
        with torch.inference_mode():
            before = float(loss_s34(model, kind, pair, tok))
        pre_weights = copy.deepcopy(model.state_dict())
        pre_opt = copy.deepcopy(opt.state_dict())
        chosen = None
        trials = []
        for lr in S34_LR:
            model.load_state_dict(pre_weights)
            opt.load_state_dict(pre_opt)
            for group in opt.param_groups:
                group["lr"] = lr
            info = update_with_protected_anchor(
                model, opt, params, tok, kind, pair, anchors)
            with torch.inference_mode():
                after = float(loss_s34(model, kind, pair, tok))
            improved = (after + 1e-7 < before)
            kept = exact_old(model, tok, old) if improved else set()
            valid = improved and protected.issubset(kept)
            reason = ("accepted" if valid else
                      "no_TRAIN_loss_improvement" if not improved
                      else "protected_old_exact_EOS_loss")
            failure_reasons[reason] += 1
            trials.append({
                "LR": lr,
                "decision": reason,
                "TRAIN_primary_before": before,
                "TRAIN_primary_after": after,
                "lost_original_old_TRAIN_ids": sorted(
                    protected - kept) if improved else [],
                **info
            })
            if valid:
                accepted += 1
                chosen = lr
                break
        if chosen is None:
            model.load_state_dict(pre_weights)
            opt.load_state_dict(pre_opt)
        results.append({
            "step": step+1,
            "kind": kind,
            "TRAIN_pair_id": pair["pair_id"],
            "accepted_lr": chosen,
            "rolled_back_model_AND_AdamW": chosen is None,
            "all_candidates": trials
        })
        if step == 0 or (step+1) % 16 == 0:
            print(json.dumps({
                "event": "S34_NEW_TRAIN_PROGRESS",
                "step": step+1, "accepted": accepted,
                "reverted": step+1-accepted
            }), flush=True)
        del pre_weights, pre_opt
    return {
        "step_attempts": S34_STEPS,
        "accepted_steps": accepted,
        "rolled_back_steps": S34_STEPS-accepted,
        "candidate_decision_counts": dict(failure_reasons),
        "all_TRAIN_candidate_trials": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw = args.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != SOURCE_BYTES or digest != s33.SOURCE_SHA:
        raise AssertionError("Not the immutable 0.6908 bpp S26 seed")
    cb = CB59(raw)
    if cb.meta["original_unique_parameter_count"] != SOURCE_PARAMS:
        raise AssertionError("Not the exact S26 36M original parameters")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK = tok
    new = s30.make_training_pairs(tok)
    old = build_pairs(tok, "TRAIN")
    if len(new) != 32 or len(old) != 24:
        raise AssertionError("TRAIN corpus changed")
    model = s30.dense_from_paid(cb)
    initial_parity = s30.double_check_original_logits(cb, model, tok)
    initial_sha = s30.snapshot_sha(model)
    original = s30.train_probe(model, tok, new, old)
    protected = exact_old(model, tok, old)
    if len(protected) != EXPECTED_S33_OLD:
        raise AssertionError("S26 old TRAIN identities changed")

    opt, replay = reproduce_s33(model, tok, new, old, protected)
    # Strict gate: continuation must start from a genuinely reproduced
    # result, not an invented checkpoint or a new, posthoc-best selection.
    save(args.out / "S34_S33_replay_diagnostic.json", replay)
    if not replay["matches_observed_S33_run_42484161593"]:
        raise AssertionError("S33 did not reproduce; no S34 training allowed")
    source_sha_after_replay = replay["S33_final_sha"]
    progress = continue_s34(model, opt, tok, new, old, protected)
    final = s30.train_probe(model, tok, new, old)
    last_old = old_train_probe(model, tok, old)
    if not protected.issubset(guarded_old_ids(last_old)):
        raise AssertionError("S34 lost any original six protected old replies")
    final_sha = s30.snapshot_sha(model)
    rank = s33.noun_position_training_probe(model, tok, new)
    report = {
        "stage": "S34_36M_FP32_PROTECTED_FIRST_COLOR_AND_NOUN",
        "CI_reference_S33": 42484161593,
        "source_S26_sha256": digest,
        "source_S26_physical_bpp": 8*len(raw)/SOURCE_PARAMS,
        "initial_paid_to_dense_max_abs_logit_diff": initial_parity,
        "dense_FP32_not_paid_0_7bpp": True,
        "initial_dense_SHA": initial_sha,
        "reproduced_S33_dense_SHA": source_sha_after_replay,
        "final_S34_dense_SHA": final_sha,
        "old_six_protected_TRAIN_ids": sorted(protected),
        "all_original_six_old_TRAIN_exact_EOS_maintained": True,
        "unseen_S29_QA_unread": True,
        "inspected_S26_QA_unread": True,
        "only_TRAIN_used_in_update_and_selection": True,
        "no_validated_generalization_claim": True,
        "original_S26_TRAIN": original,
        "reproduced_S33_TRAIN": replay["S33_full_TRAIN_answers"],
        "final_S34_TRAIN": final,
        "reproduced_S33_forced_noun_TRAIN":
            replay["S33_forced_noun_TRAIN"],
        "final_S34_forced_noun_TRAIN": rank,
        "S34_training_trials": progress,
        "final_original_old_eight_TRAIN_answers": last_old,
        "physical_compact_weights_not_updated_or_saved": True,
        "checkpoint_format": "dense FP32 weights only; no optimizer",
    }
    save(args.out / "S34_actual_36M_FP32_TRAIN_report.json", report)

    # Unlike S33, retain the actual continued FP32 weights for an
    # independently verifiable next study. Never call this physical CB59.
    torch.save({
        "source_seed_SHA256": digest,
        "S33_reproduced_sha256": source_sha_after_replay,
        "S34_SHA256": final_sha,
        "state_dict": model.state_dict()
    }, args.out / "S34_dense_FP32_ONLY_checkpoint.pt")
    print(json.dumps({
        "event": "S34_FINAL_ACTUAL_TRAIN",
        "S33_reproduced": replay["matches_observed_S33_run_42484161593"],
        "S33_accepted": replay["accepted_steps"],
        "S34_accepted": progress["accepted_steps"],
        "S34_reverted": progress["rolled_back_steps"],
        "S33_new_full": replay["S33_new_TRAIN_full_EOS"],
        "S34_new_full": final["per_kind"]["new_noun_TRAIN"]["full_correct"],
        "S33_old_full": replay["S33_old_TRAIN_full_EOS"],
        "S34_old_full": final["per_kind"]["historical_TRAIN"]["full_correct"],
        "S33_to_S34_weights_changed": final_sha != source_sha_after_replay,
        "S29_heldout_QA_used": False,
        "physical_0_7bpp_achieved_by_new_weights": False
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
