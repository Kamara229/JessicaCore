"""CALIBRI S37: actual S26 36M CB59 compact-weight recovery, TRAIN only.

Unlike S30-S36 this trains ONLY 85 real paid K32x8 centroid dictionaries,
projected by a first-token historical margin guard and an old TRAIN replay
gradient. Every attempted update uses a strict six-old-answer full+EOS
rollback gate; no reliance on teacher forcing alone. All accepted changes
are re-encoded into the SAME exact CB59 physical number of bytes and
compared against a reloaded paid model. No dense FP32 weights exported.

The 5-bit indices are held fixed in this one precommitted compact pilot.
S29 immutable QA, S26 previously inspected QA, and validation/test data are
NEVER accessed or used. No Jessica tests, no LIVE/Render/Supabase/APK changes.
Success here is TRAIN-only; it is not demonstrated FP32 parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.codec import replace_discrete
from v186.pairs import build_pairs
from v174.run import save
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS, SOURCE_BYTES, MODEL_SPEC, PAID_OPERATORS
from calibri2.s20.multiword_curriculum import take_grads
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s34 import continued_protected_color_noun_fp32 as s34

SEED = 3737
STEPS = 12
LR_CANDIDATES = (.25, .08, .025)
MARGIN_FLOOR_CAP = .02
OLD_REPLAY_BLEND = .18
MARGIN_GRAD_BLEND = .28
IMPROVE_FLOOR = 1e-7
FIRST_PAIR_COUNT = 12
assert STEPS == FIRST_PAIR_COUNT


def score_free(model, sources, tok, row, cap=10):
    """Actual greedy full response, including EOS; never prefix-only."""
    clear(sources)
    result = s30.greedy_actual(model, tok, row["question"], max_new=cap)
    gold = tok.encode(row["answer"], add_special_tokens=False) + [tok.eos_token_id]
    return {"gold": gold, "actual": result, "full_correct": result == gold,
            "text": tok.decode(result, skip_special_tokens=True)}


def old_guard(model, sources, tok, old):
    """Evaluate exact six original protected TRAIN identities, not a mere count."""
    rows = {}
    for pair in old[:4]:
        for side in ("left", "right"):
            key = f'{pair["pair_id"]}_{side}'
            rows[key] = score_free(model, sources, tok, pair[side])
    return rows


def preserved_ids(rows):
    return {name for name, row in rows.items() if row["full_correct"]}


def paid_loss(model, sources, tok, pair):
    clear(sources)
    with torch.inference_mode():
        return float(s34.loss_s34(model, "new", pair, tok))


@torch.inference_mode()
def first_margin(model, sources, row, yes_token, no_token):
    clear(sources)
    logits = model(input_ids=row["prompt_ids"][None], use_cache=False).logits[0, -1]
    return float(logits[yes_token] - logits[no_token])


def small_train_probe(model, sources, tok, new, old):
    selected_new = new[::4]  # 8 actual new TRAIN pairs; 16 full/EOS responses
    selected_old = old[:4]   # 4 historical TRAIN pairs; 8 full/EOS responses
    out = {}
    for label, pairs in (("new_TRAIN", selected_new),("historical_TRAIN", selected_old)):
        rows = {}
        for pair in pairs:
            for side in ("left", "right"):
                rows[f'{pair["pair_id"]}_{side}'] = score_free(
                    model, sources, tok, pair[side])
        out[label] = {"fully_correct": sum(x["full_correct"] for x in rows.values()),
                      "total": len(rows), "all_full_answers": rows}
    return out


def normalize(g):
    norm_sq = sum((x.double().square().sum() for x in g))
    norm = float(torch.sqrt(norm_sq))
    if not (np.isfinite(norm) and norm > 1e-11):
        raise AssertionError("Missing/nonfinite compact codebook gradient")
    return [x.float()/norm for x in g], norm


def project_against(vec, guard):
    dot = sum((a.double()*b.double()).sum() for a,b in zip(vec,guard))
    norm_sq = sum((b.double()*b.double()).sum() for b in guard)
    if not (norm_sq > 0 and torch.isfinite(norm_sq)):
        raise AssertionError("Invalid margin/old projection gradient")
    coefficient = float(dot/norm_sq) if dot < 0 else 0.
    if coefficient:
        return [a - coefficient*b for a,b in zip(vec,guard)], coefficient
    return vec, 0.


def margin_loss(model, sources, row, yes_token, no_token):
    # Gradient descent on negative Yes-No difference increases Yes-vs-No.
    clear(sources)
    logits = model(input_ids=row["prompt_ids"][None], use_cache=False).logits[0, -1]
    return logits[no_token] - logits[yes_token]


def train():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw = args.source.read_bytes()
    if (len(raw) != SOURCE_BYTES or
        hashlib.sha256(raw).hexdigest() != s30.SOURCE_SHA):
        raise AssertionError("S37 must start exact sealed S26 CB59 0.6908bpp")
    archive = CB59(raw)
    if archive.meta["original_unique_parameter_count"] != SOURCE_PARAMS:
        raise AssertionError("Unexpected architecture parameter count")
    source_bpp = 8*len(raw)/SOURCE_PARAMS
    if source_bpp > .7:
        raise AssertionError("Source not within complete physical budget")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK = tok
    new, old = s30.make_training_pairs(tok), build_pairs(tok, "TRAIN")
    if len(new)!=32 or len(old)!=24:
        raise AssertionError("Changed predeclared TRAIN samples")
    model, sources = create_small_paid_model(archive, build_config(MODEL_SPEC))
    keys = sorted(sources)
    if len(keys)!=PAID_OPERATORS:
        raise AssertionError("Must train exactly all 85 paid 8D matrix dictionaries")
    centroid_params = [sources[k].centroids for k in keys]
    if sum(x.numel() for x in centroid_params) != PAID_OPERATORS*32*8:
        raise AssertionError("Unexpected paid training parameter count")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != PAID_OPERATORS*32*8:
        raise AssertionError("S37 illegally updates dense weights or unpaid tensors")
    before_train = small_train_probe(model, sources, tok, new, old)
    protected_start = before_train["historical_TRAIN"]["all_full_answers"]
    protected_ids = preserved_ids(protected_start)
    if len(protected_ids) != 6 or "1_left" not in protected_ids:
        raise AssertionError("Sealed S26 six-old TRAIN baseline not reproduced")
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    if yes_id == no_id:
        raise AssertionError("Yes/No cannot share same first token")
    margin_start = first_margin(model, sources, old[1]["left"], yes_id, no_id)
    if margin_start <= 0:
        raise AssertionError("Sealed S26 original protected 1_left is already unstable")
    # Do not demand that a fragile baseline had a margin it never possessed.
    margin_floor = min(MARGIN_FLOOR_CAP, .5*margin_start)
    decisions = []
    accepted = 0
    codebook_cells_updated = set()
    initial_bytes = bytes(raw)
    for step in range(STEPS):
        pair = new[(step*7)%len(new)]
        old_anchor = old[(step*5)%4]  # Historical TRAIN only.
        original = paid_loss(model, sources, tok, pair)
        old_anchor_obj, old_grad, _ = take_grads(
            model, sources, keys, lambda: s30.train_loss(model, old_anchor, False))
        objective_start, main_grad, _ = take_grads(
            model, sources, keys, lambda: s34.loss_s34(model, "new", pair, tok))
        margin_obj, margin_grad, _ = take_grads(
            model, sources, keys,
            lambda: margin_loss(model, sources, old[1]["left"], yes_id, no_id))
        if abs(original-objective_start)>2e-5:
            raise AssertionError("Unexpected S37 TRAIN primary loss mismatch")
        primary, gn = normalize(main_grad)
        old_vector, old_gn = normalize(old_grad)
        margin_vector, margin_gn = normalize(margin_grad)
        direction, old_coef = project_against(primary, old_vector)
        direction, margin_coef = project_against(direction, margin_vector)
        direction = [x + OLD_REPLAY_BLEND*y + MARGIN_GRAD_BLEND*z
                     for x,y,z in zip(direction,old_vector,margin_vector)]
        direction, final_norm = normalize(direction)
        # Snapshot only 21,760 paid FP16 dictionary cells; NOT dense 36M weights.
        snapshot = [p.detach().clone() for p in centroid_params]
        tested = []
        selected = None
        for lr in LR_CANDIDATES:
            with torch.no_grad():
                for p, backup, delta in zip(centroid_params, snapshot, direction):
                    p.copy_((backup-lr*delta).clamp(-16,16).half().float())
            actual = paid_loss(model, sources, tok, pair)
            improved = bool(actual + IMPROVE_FLOOR < original)
            retained = None
            current_margin = None
            passed = False
            if improved:
                current_margin = first_margin(
                    model, sources, old[1]["left"], yes_id, no_id)
                if current_margin >= margin_floor:
                    candidate_guard = old_guard(model, sources, tok, old)
                    retained = sorted(preserved_ids(candidate_guard))
                    passed = protected_ids.issubset(set(retained))
            tested.append({"lr": lr, "new_TRAIN_loss": actual,
                           "new_TRAIN_improved": improved,
                           "1_left_Yes_minus_No_margin": current_margin,
                           "retained_original_old_TRAIN_ids": retained,
                           "accepted": passed})
            if passed:
                selected = lr
                accepted += 1
                with torch.no_grad():
                    for key,p,b in zip(keys,centroid_params,snapshot):
                        if not torch.equal(p,b):
                            codebook_cells_updated.add(key)
                break
        if selected is None:
            with torch.no_grad():
                for p, backup in zip(centroid_params, snapshot):
                    p.copy_(backup)
        clear(sources)
        decisions.append({
            "step":step+1, "new_pair_TRAIN_id":pair["pair_id"],
            "old_TRAIN_anchor_id":old_anchor["pair_id"],
            "old_TRAIN_anchor_loss":old_anchor_obj,
            "1_left_negative_margin_loss_before":margin_obj,
            "primary_grad_norm":gn,
            "old_grad_norm":old_gn,
            "margin_grad_norm":margin_gn,
            "old_projection_coefficient":old_coef,
            "margin_projection_coefficient":margin_coef,
            "combined_gradient_norm":final_norm,
            "selected_lr":selected, "rolled_back":selected is None,
            "trials":tested
        })
        print(json.dumps({"event":"S37_COMPRESSED_TRAIN_PROGRESS",
                          "step":step+1,"accepted":accepted,
                          "new_pair":pair["pair_id"],"selected_lr":selected,
                          "compact_only":True},ensure_ascii=False),flush=True)
    final_probe = small_train_probe(model, sources, tok, new, old)
    final_old = final_probe["historical_TRAIN"]["all_full_answers"]
    if not protected_ids.issubset(preserved_ids(final_old)):
        raise AssertionError("Final physically paid S37 lost old TRAIN skill")
    margin_end = first_margin(model, sources, old[1]["left"], yes_id, no_id)
    if margin_end < margin_floor-1e-4:
        raise AssertionError("Final 1_left token logit margin regressed")
    output_path = args.out/"CALIBRI2_S37_36M_PAID_TRAIN.cb59"
    physical = None
    reload_check = None
    if accepted:
        packed, physical = replace_discrete(archive, sources, strict=False)
        if len(packed)!=len(initial_bytes) or 8*len(packed)/SOURCE_PARAMS>.7:
            raise AssertionError("S37 physical budget exceeded")
        if hashlib.sha256(packed).hexdigest()==s30.SOURCE_SHA:
            raise AssertionError("Model accepted steps but unchanged packed bytes")
        if physical["actually_changed_FP16_codebook_cells"]<=0:
            raise AssertionError("S37 accepted updates have no persistent paid cells")
        packed_archive = CB59(packed)
        reloaded, re_sources = create_small_paid_model(
            packed_archive, build_config(MODEL_SPEC))
        reload_probe = small_train_probe(reloaded, re_sources, tok, new, old)
        exact_reload = reload_probe==final_probe
        if not exact_reload:
            raise AssertionError("S37 reloaded compact archive changes actual full answers")
        reload_margin = first_margin(reloaded, re_sources,
                                     old[1]["left"],yes_id,no_id)
        if abs(reload_margin - margin_end)>0.003:
            raise AssertionError("S37 reload changes Yes-No margin")
        output_path.write_bytes(packed)
        reload_check = {"full_sequences_equal":exact_reload,
                        "margin_before":margin_end,
                        "margin_after":reload_margin,
                        "new_SHA256":hashlib.sha256(packed).hexdigest()}
    report={
        "stage":"S37_36M_PHYSICALLY_PAID_CB59_FIRST_PROTECTED_RECOVERY",
        "S26_immutable_source_SHA256":s30.SOURCE_SHA,
        "S26_source_bytes":len(raw),
        "S26_physical_bpp":source_bpp,
        "original_unique_parameters":SOURCE_PARAMS,
        "trainable_FP16_codebook_cells":PAID_OPERATORS*32*8,
        "trainable_5bit_indices_this_pilot":0,
        "all_original_5bit_indices_unchanged":True,
        "S29_heldout_QA_never_read":True,
        "S26_inspected_QA_never_read":True,
        "only_TRAIN_questions":True,
        "not_proof_of_unseen_generalization_or_FP32_parity":True,
        "Jessica_tests_not_run":True,
        "LIVE_unmodified":True,
        "steps_attempted":STEPS,
        "accepted_steps":accepted,
        "rejected_steps":STEPS-accepted,
        "margin_floor_1_left":margin_floor,
        "before_1_left_margin":margin_start,
        "after_1_left_margin":margin_end,
        "original_old_six_protected_TRAIN_ids":sorted(protected_ids),
        "before_greedy_TRAIN":before_train,
        "after_greedy_TRAIN":final_probe,
        "all_candidate_steps_and_rejections":decisions,
        "paid_operators_with_updated_centroids":len(codebook_cells_updated),
        "changed_operators":sorted(codebook_cells_updated),
        "physical_result":physical,
        "physical_reload":reload_check,
        "physical_checkpoint_created":output_path.exists(),
        "if_no_update_did_not_claim_a_new_compact_model":True,
        "no_green_tunnel_sidecar_or_FP32_weight_export":True,
    }
    save(args.out/"S37_ACTUAL_PHYSICAL_CB59_TRAIN_ONLY.json",report)
    print(json.dumps({"event":"S37_PHYSICAL_FINISHED",
                      "accepted_steps":accepted,
                      "new_TRAIN_full_before":before_train["new_TRAIN"]["fully_correct"],
                      "new_TRAIN_full_after":final_probe["new_TRAIN"]["fully_correct"],
                      "old_TRAIN_full_before":before_train["historical_TRAIN"]["fully_correct"],
                      "old_TRAIN_full_after":final_probe["historical_TRAIN"]["fully_correct"],
                      "physical_bpp":source_bpp,
                      "physical_checkpoint_created":output_path.exists(),
                      "heldout_QA_untouched":True},ensure_ascii=False),flush=True)


if __name__=="__main__":
    train()
