"""S35: ONE exact S34 TRAIN update; diagnose why old 1_left (Yes, EOS) fails.

- Replay S33 from the exact S26 0.6908bpp CB59 source with optimizer moments.
- Use the SAME S34 first TRAIN pair, gradients, anchors and AdamW state.
- Take ONLY the formerly rejected smallest backtrack LR (3e-6).
- Compare free generation AND teacher-forced first-token/EOS logits, before/after.
- Research-only FP32 probe, do not save candidate weights. No S29 QA.
- Jessica/LIVE/Render/Supabase/APK are not involved.
"""
from __future__ import annotations

import argparse
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
from v186.pairs import build_pairs
from v174.run import save

LR = 0.000003
PROTECTED_ID = "1_left"


@torch.inference_mode()
def one_prompt_logits(model, tok, row):
    """Inspect both causal positions, not just teacher-forced noun rankings."""
    gold = tok.encode(row["answer"], add_special_tokens=False)
    if gold != [10539] or tok.eos_token_id != 2:
        raise AssertionError("Historical 1_left tokenized Yes/EOS changed")
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": row["question"]}],
        tokenize=False, add_generation_prompt=True
    )
    prompt_ids = torch.tensor(
        tok.encode(prompt, add_special_tokens=False),
        dtype=torch.long
    )[None]
    result = []
    context = prompt_ids
    for target in [gold[0], tok.eos_token_id]:
        logits = model(input_ids=context, use_cache=False).logits[0, -1].float()
        high = torch.topk(logits, k=5)
        selected = int(logits.argmax())
        margin = float(logits[target] - logits[selected])
        competitors = [
            {"token": int(t), "text": tok.decode([int(t)]),
             "logit": float(v)}
            for t, v in zip(high.indices, high.values)
        ]
        result.append({
            "gold_token": int(target),
            "gold_text": tok.decode([int(target)]),
            "greedy_token": selected,
            "greedy_text": tok.decode([selected]),
            "gold_logit": float(logits[target]),
            "greedy_logit": float(logits[selected]),
            "gold_minus_greedy_margin": margin,
            "eos_logit": float(logits[int(tok.eos_token_id)]),
            "yes_logit": float(logits[int(gold[0])]),
            "top5": competitors,
        })
        context = torch.cat((
            context, torch.tensor([[int(target)]], dtype=torch.long)
        ), dim=1)
    free = s30.greedy_actual(model, tok, row["question"], max_new=10)
    return {
        "expected_full_tokens_EOS": gold + [tok.eos_token_id],
        "actual_free_generated_tokens": free,
        "actual_free_text": tok.decode(free, skip_special_tokens=True),
        "exact_full_answer_EOS": free == gold + [tok.eos_token_id],
        "positions": {"first_answer_token": result[0],
                      "EOS_after_forced_gold_Yes": result[1]}
    }


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
        raise AssertionError("S35 must begin with exact immutable S26 CB59 archive")
    paid = CB59(raw)
    if paid.meta["original_unique_parameter_count"] != SOURCE_PARAMS:
        raise AssertionError("S35 seed must contain same 36M unique parameters")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK = tok
    new = s30.make_training_pairs(tok)
    old = build_pairs(tok, "TRAIN")
    if len(new) != 32 or len(old) != 24:
        raise AssertionError("S35 changed declared TRAIN corpus")
    model = s30.dense_from_paid(paid)
    protected = s34.exact_old(model, tok, old)
    if len(protected) != 6 or PROTECTED_ID not in protected:
        raise AssertionError("S35 original protected identities changed")

    print(json.dumps({"event":"S35_S33_REPLAY_START"}),flush=True)
    opt, replay = s34.reproduce_s33(model, tok, new, old, protected)
    if not replay["matches_observed_S33_run_42484161593"]:
        raise AssertionError("S35 cannot analyze unmatched S33")
    if replay["accepted_steps"] != 30:
        raise AssertionError("S35 cannot analyze incorrect S33 replay")
    assert PROTECTED_ID in s34.exact_old(model, tok, old)
    train_row = old[1]["left"]
    if old[1]["pair_id"] != 1:
        raise AssertionError("1_left historical TRAIN pair ID changed")
    before = one_prompt_logits(model, tok, train_row)
    before_guard = s34.exact_old(model, tok, old)
    before_sha = s30.snapshot_sha(model)
    pair = new[0]
    if pair["pair_id"] != 5000:
        raise AssertionError("S34 first new TRAIN pair changed")
    with torch.inference_mode():
        primary_before = float(s34.loss_s34(model, "new", pair, tok))
    for group in opt.param_groups:
        group["lr"] = LR
    grad_info = s34.update_with_protected_anchor(
        model, opt, list(model.parameters()), tok,
        "new", pair, old[:4]
    )
    with torch.inference_mode():
        primary_after = float(s34.loss_s34(model, "new", pair, tok))
    after = one_prompt_logits(model, tok, train_row)
    after_guard = s34.exact_old(model, tok, old)
    after_sha = s30.snapshot_sha(model)
    result = {
        "stage": "S35_DIAGNOSE_EXACT_S34_STEP1_LR3E-6_1LEFT",
        "source_S26_SHA256": s30.SOURCE_SHA,
        "starting_physical_bpp_is_NOT_current_FP32_model_bpp":
            8*len(raw)/SOURCE_PARAMS,
        "S33_reproduced_with_exact_optimizer_moments": True,
        "S33_accepted": replay["accepted_steps"],
        "S29_heldout_QA_used": False,
        "S26_inspected_QA_used": False,
        "only_historical_TRAIN_probed": True,
        "S34_update_kind": "new",
        "S34_update_pair_id": pair["pair_id"],
        "S34_LR": LR,
        "S34_model_is_dense_FP32": True,
        "rejected_candidate_weights_not_saved": True,
        "protected_before": sorted(before_guard),
        "protected_after": sorted(after_guard),
        "lost_protected_ids": sorted(before_guard - after_guard),
        "1_left_before": before,
        "1_left_after": after,
        "S34_first_pair_loss_before": primary_before,
        "S34_first_pair_loss_after": primary_after,
        "S34_gradient": grad_info,
        "weights_SHA_before": before_sha,
        "weights_SHA_after": after_sha,
        "weights_actually_changed": before_sha != after_sha,
        "previous_S34_step1_smallest_LR_already_rejected_for_1_left": True,
        "one_step_is_diagnostic_not_training_progress": True,
        "LIVE_unchanged": True,
        "Jessica_tests_not_run": True,
    }
    if not (primary_after < primary_before):
        raise AssertionError("S35 did not reproduce S34 step1 TRAIN loss improvement")
    if after_sha == before_sha:
        raise AssertionError("No full-FP32 update was actually performed")
    save(args.out/"S35_EXACT_S34_STEP1_LR3E-6_1LEFT_TOKEN_LOGITS.json",result)
    print(json.dumps({
        "event": "S35_ONE_STEP_DIAGNOSTIC",
        "S33_reproduced": True,
        "S34_pair": pair["pair_id"],
        "LR": LR,
        "1_left_was_correct": before["exact_full_answer_EOS"],
        "1_left_is_correct_after": after["exact_full_answer_EOS"],
        "before_first_greedy": before["positions"]["first_answer_token"]["greedy_token"],
        "after_first_greedy": after["positions"]["first_answer_token"]["greedy_token"],
        "before_EOS_after_gold": before["positions"]["EOS_after_forced_gold_Yes"]["greedy_token"],
        "after_EOS_after_gold": after["positions"]["EOS_after_forced_gold_Yes"]["greedy_token"],
        "lost_protected_ids": result["lost_protected_ids"],
        "weights_saved": False,
        "S29_QA_used": False
    },ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
