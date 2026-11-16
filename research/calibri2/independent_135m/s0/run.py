#!/usr/bin/env python3
"""S0 actual original trained FP32 -> new physically paid CB59 (135M) pilot.

All question/answer examples here are TRAIN diagnostics, NOT held-out EVAL.
Green Tunnel is codebook-only and J7-like rollback uses immutable TRAIN
full sequences, not oracle TEST. Parallel paths are a batched algorithm,
not a claim of wall-clock speedup or superior correctness.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from direct_codebook_v59 import CB59, encode
from v176.operators import clear
from v185.codec import replace_discrete
from v196.recover import build_trainable_from_paid_source

SEED = 13520261116
NEW_TOKENS = 4
# All prompts are TRAIN/DIAGNOSTIC; not independent QA and NEVER used as EVAL.
TRAIN_PROMPTS = (
    "Answer with one word: What color is fresh snow?",
    "Finish in one word: Water turns into ice when it ...",
    "Answer with one digit: How much is 2 plus 2?",
    "Answer with one word: What is the opposite of hot?",
    "Answer with one word: What do bees make?",
    "Answer with one word: What is the opposite of up?",
    "Complete briefly: The sun rises in the ...",
    "Answer with one word: What color is a ripe banana?",
)
FIT_IDS = (0, 1, 2, 3)
GUARD_IDS = (4, 5, 6, 7)


def emit(stage, **kwargs):
    print(json.dumps({"stage": stage, **kwargs}, ensure_ascii=False), flush=True)


def save(path, data):
    Path(path).write_text(json.dumps(
        data, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n", encoding="utf-8")


def prompt_ids(tokenizer, text):
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True
    )
    seq = tokenizer.encode(chat, add_special_tokens=False)
    if not 2 <= len(seq) <= 160:
        raise AssertionError("TRAIN context length unexpected")
    return torch.tensor([seq], dtype=torch.long)


@torch.no_grad()
def greedy(model, source, tokenizer, context, max_new=NEW_TOKENS):
    tokens = []
    ids = context.clone()
    for _ in range(max_new):
        if source is not None:
            clear(source)
        logits = model(input_ids=ids, use_cache=False).logits[0, -1]
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite logits in actual model")
        next_id = int(logits.argmax())
        if next_id == tokenizer.eos_token_id:
            break
        tokens.append(next_id)
        ids = torch.cat((ids, ids.new_tensor([[next_id]])), dim=1)
    if source is not None:
        clear(source)
    return {"ids": tokens, "text": tokenizer.decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )}


@torch.no_grad()
def batched_paths(model, source, tokenizer, context, n=3):
    """Compute N independent continuations in a single real BATCH forward.

    Path0 is greedy. Paths 1..N-1 are separate fixed-seed samples; consensus
    ties resolve to path0. This is not FP32/teacher/reranker assisted.
    """
    if n < 2:
        raise ValueError("At least two paths requested")
    generators = [torch.Generator().manual_seed(SEED + i*1009)
                  for i in range(n)]
    rows = context.expand(n, -1).clone()
    done = [False] * n
    outputs = [[] for _ in range(n)]
    t0 = time.perf_counter()
    for _ in range(NEW_TOKENS):
        if source is not None:
            clear(source)
        logits = model(input_ids=rows, use_cache=False).logits[:, -1, :]
        new = []
        for i in range(n):
            if done[i]:
                token = tokenizer.eos_token_id
            elif i == 0:
                token = int(logits[i].argmax())
            else:
                probs = torch.softmax(logits[i].float()/0.8, dim=-1)
                token = int(torch.multinomial(probs, 1,
                                            generator=generators[i]))
            if token == tokenizer.eos_token_id:
                done[i] = True
            elif not done[i]:
                outputs[i].append(token)
            new.append(token)
        rows = torch.cat((rows, rows.new_tensor(new)[:, None]), dim=1)
        if all(done):
            break
    if source is not None:
        clear(source)
    texts = [tokenizer.decode(x, skip_special_tokens=True,
             clean_up_tokenization_spaces=False) for x in outputs]
    votes = Counter(t.strip().casefold() for t in texts)
    best = max(votes.values())
    winners = {x for x, count in votes.items() if count == best}
    selected = next((i for i, v in enumerate(texts)
                     if v.strip().casefold() in winners), 0)
    # In equal-vote ties path0 wins explicitly (no gold labels).
    if len(winners) > 1:
        selected = 0
    return {"generated": texts, "chosen_path": selected,
            "chosen_text": texts[selected],
            "consensus_count": votes[texts[selected].strip().casefold()],
            "elapsed_seconds": round(time.perf_counter()-t0, 4),
            "selection_uses_gold_labels": False,
            "batched_paths": n,
            "not_native_C_or_wall_clock_speedup_claim": True}


def snapshot(source, optimizer):
    return {name: s.centroids.detach().clone() for name, s in source.items()}, \
        copy.deepcopy(optimizer.state_dict())


def restore(source, optimizer, snap):
    values, optim = snap
    with torch.no_grad():
        for name, s in source.items():
            s.centroids.copy_(values[name])
    optimizer.load_state_dict(optim)
    clear(source)


def score_agreement(student, source, tok, contexts, teacher_answers, ids):
    observations = {}
    for idx in ids:
        got = greedy(student, source, tok, contexts[idx])
        observations[idx] = {
            "generated_text": got["text"],
            "generated_ids": got["ids"],
            "same_as_fp32": got["ids"] == teacher_answers[idx]["ids"]
        }
    return observations


def train_paid_gt(student, source, tokenizer, contexts, teachers,
                  baseline, out, steps):
    # Original source is frozen; train ONLY previously paid centroids.
    optim = torch.optim.Adam([s.centroids for s in source.values()], lr=5e-5)
    all_orig = {k: s.centroids.detach().clone() for k, s in source.items()}
    initial_guard = {i: baseline[i]["ids"] for i in GUARD_IDS}
    accepted = 0
    history = []
    for iteration in range(steps):
        idx = FIT_IDS[iteration % len(FIT_IDS)]
        teacher_target = teachers[idx]["ids"][:NEW_TOKENS]
        if not teacher_target:
            teacher_target = [tokenizer.eos_token_id]
        seq = torch.cat((contexts[idx],
                         contexts[idx].new_tensor([teacher_target])), dim=1)
        frame = snapshot(source, optim)
        clear(source)
        optim.zero_grad(set_to_none=True)
        try:
            logits = student(input_ids=seq[:, :-1], use_cache=False).logits
            left = contexts[idx].shape[1] - 1
            aligned = logits[:, left:left+len(teacher_target), :]
            if aligned.shape[1] != len(teacher_target):
                raise AssertionError("Autoregressive target offset incorrect")
            ce = F.cross_entropy(
                aligned.reshape(-1, aligned.shape[-1]),
                seq[:, contexts[idx].shape[1]:].reshape(-1))
            anchor = torch.stack([
                (s.centroids - all_orig[name]).square().mean()
                for name, s in source.items()
            ]).mean()
            loss = ce + 0.03 * anchor
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite original TRAIN loss")
            loss.backward()
            active = sum(s.centroids.grad is not None
                         for s in source.values())
            if active != 211:
                raise AssertionError(
                    f"Green Tunnel missed original paid codebooks: {active}/211"
                )
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [s.centroids for s in source.values()], 1.0,
                error_if_nonfinite=True))
            optim.step()
            with torch.no_grad():
                for s in source.values():
                    s.centroids.clamp_(-16.0, 16.0)
        finally:
            clear(source)
        current_guard = score_agreement(
            student, source, tokenizer, contexts, teachers, GUARD_IDS
        )
        lost = [idx for idx in GUARD_IDS
                if current_guard[idx]["generated_ids"] != initial_guard[idx]]
        candidate_fit = greedy(student, source, tokenizer, contexts[idx])
        baseline_fit_match = baseline[idx]["ids"] == teachers[idx]["ids"]
        candidate_fit_match = candidate_fit["ids"] == teachers[idx]["ids"]
        # Fail closed: checkpoint may be kept for a strict full-answer TRAIN
        # improvement, never for a lower CE alone; any guard loss rolls back.
        keep = (not lost and candidate_fit_match and
                not baseline_fit_match)
        if keep:
            accepted += 1
        else:
            restore(source, optim, frame)
        result = {
            "TRAIN_iteration": iteration+1,
            "fit_prompt_index": idx,
            "paid_source_gradient_count": active,
            "loss_diagnostic": round(float(loss.detach()), 6),
            "gradient_norm": grad_norm,
            "frozen_full_guard_indices": list(GUARD_IDS),
            "lost_original_guard_outputs": lost,
            "fit_before_FP32_exact": baseline_fit_match,
            "fit_after_FP32_exact": candidate_fit_match,
            "accepted_only_on_STRICT_TRAIN_full_answer_gain": keep,
            "optimizer_and_all_211_centroids_rolled_back": not keep,
        }
        history.append(result)
        emit("S0_green_tunnel_J7_transaction", **result)
    return accepted, history


def run(outdir, steps):
    if not 0 <= steps <= 2:
        raise AssertionError("S0 fixed max two paid TRAIN steps")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    contexts = [prompt_ids(tok, x) for x in TRAIN_PROMPTS]
    precommit = {
        "run_kind": "TRAIN diagnostic engineering pilot, NOT held-out EVAL",
        "model": MODEL, "model_sha": MODEL_SHA,
        "seed": SEED, "maximum_steps": steps,
        "fit_indices": FIT_IDS, "guard_indices": GUARD_IDS,
        "queries": TRAIN_PROMPTS, "maximum_output_tokens": NEW_TOKENS,
        "bpp_ceiling": 0.7, "only_paid_211_codebooks": True,
        "no_FP32_reranker_at_inference": True,
        "no_Jessica_or_live": True
    }
    save(out/"S0_PRECOMMITTED_TRAIN_ONLY.json", precommit)
    emit("S0_FP32_load_start", model=MODEL, revision=MODEL_SHA)
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_SHA, torch_dtype=torch.float32
    ).cpu().eval()
    count = sum(p.numel() for p in teacher.parameters())
    if count != EXPECTED_UNIQUE or any(p.dtype != torch.float32
                                      for p in teacher.parameters()):
        raise AssertionError("Not original 135M FP32 checkpoint")
    teacher_answers = [greedy(teacher, None, tok, c) for c in contexts]
    emit("S0_FP32_reference_complete", answers=len(teacher_answers))
    compressed, meta, stats = encode(teacher.state_dict())
    del teacher
    gc.collect()
    archive = CB59(compressed)
    unique = int(meta["original_unique_parameter_count"])
    matrices = sum(1 for x in meta["tensor_descriptors"].values()
                   if x["dtype"] == "CB59_K32V8")
    if unique != EXPECTED_UNIQUE or matrices != 211:
        raise AssertionError("Original 135M matrix/parameter audit failed")
    cap_bytes = (unique*7)//80
    if len(compressed) > cap_bytes:
        raise AssertionError("Actual source exceeds 0.700 physical bpp")
    if archive.handles["model.embed_tokens.weight"] is not archive.handles["lm_head.weight"]:
        raise AssertionError("Original tied embedding/head double-counted")
    (out/"S0_NEW_DIRECT_FP32_TO_CB59_135M.cb59").write_bytes(compressed)
    seal = {"bytes": len(compressed),
            "full_bpp": 8*len(compressed)/unique,
            "byte_cap": cap_bytes,
            "original_FP32_unique_parameters": unique,
            "full_original_2D_matrices": matrices,
            "new_source_SHA256": hashlib.sha256(compressed).hexdigest(),
            "same_hf_sha": MODEL_SHA, "codec_summary": stats,
            "archive_is_physically_stored": True}
    save(out/"S0_NEW_CB59_PHYSICAL_SEAL.json", seal)
    emit("S0_CB59_new_direct_full_physical_source", **{
        k: seal[k] for k in ("bytes", "full_bpp", "full_original_2D_matrices",
                             "new_source_SHA256")})
    student, source = build_trainable_from_paid_source(archive)
    student.cpu().eval()
    baseline = [greedy(student, source, tok, c) for c in contexts]
    baseline_agreement = sum(
        baseline[i]["ids"] == teacher_answers[i]["ids"]
        for i in range(len(contexts)))
    initial_paths = batched_paths(student, source, tok, contexts[0], n=3)
    accepted, history = train_paid_gt(
        student, source, tok, contexts, teacher_answers,
        baseline, out, steps
    )
    after = [greedy(student, source, tok, c) for c in contexts]
    after_paths = batched_paths(student, source, tok, contexts[0], n=3)
    packed, rewritten = replace_discrete(archive, source, strict=False)
    if len(packed) != len(compressed) or rewritten["complete_bpp"] > .7:
        raise AssertionError("J7 candidate has unpaid or changed physical size")
    if accepted:
        (out/"S0_ACCEPTED_TRAIN_ONLY_GT_GUARDED_135M.cb59").write_bytes(packed)
    final_agreement = sum(
        after[i]["ids"] == teacher_answers[i]["ids"]
        for i in range(len(contexts)))
    report = {
        "pilot_completed": True,
        "independent_heldout_EVAL_passed": False,
        "FP32_quality_parity_proven": False,
        "native_C_speed_measured": False,
        "direct_new_CB59": seal,
        "teacher_TRAIN_generated_answers": teacher_answers,
        "new_direct_CB59_TRAIN_generated_answers": baseline,
        "after_train_TRAIN_generated_answers": after,
        "baseline_FP32_exact_TRAIN_agreement": baseline_agreement,
        "after_FP32_exact_TRAIN_agreement": final_agreement,
        "all_TRAIN_only": True,
        "protected TRAIN indices": GUARD_IDS,
        "green_tunnel_transactions_accepted": accepted,
        "green_tunnel_transactions_attempted": steps,
        "transaction_log": history,
        "GT_export_full_source_shasum": rewritten["sha256"],
        "GT_paid_repack_report": rewritten,
        "parallel_paths_direct_CB59": initial_paths,
        "parallel_paths_GT_J7": after_paths,
        "parallel_uses_only_CB59_candidate_and_no_teacher_online": True,
        "parallel_is_real_batched_candidates_not_measured_native_parallel_speed": True,
        "no_merge_or_live": True
    }
    save(out/"S0_ACTUAL_ENGINEERING_PILOT_REPORT.json", report)
    emit("S0_final_TRAIN_DIAGNOSTICS_NOT_EVAL",
         baseline_exact=baseline_agreement, after_exact=final_agreement,
         GT_steps_accepted=accepted, bpp=seal["full_bpp"],
         new_direct_SHA=seal["new_source_SHA256"],
         final_repacked_SHA=rewritten["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.steps)
