#!/usr/bin/env python3
"""S1: Four actual TRAIN-only cycles on independently compressed FP32 135M.

S1 reads the S0 CB59 source ONLY. Physically paid FP16 centroids + 5bit
group-index proposals; no FP32 model/teacher in the training process.
Every cycle is J7-like transactional with whole model and optimizer rollback.
Do not interpret TRAIN scores as heldout language/FP32 parity.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.codec import replace_discrete
from v196.recover import build_trainable_from_paid_source

SEED = 135_2026_1116
EXPECTED_INPUT_SHA = "50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
EXPECTED_BYTES = 11_229_303
CYCLES = 4
STEPS_PER_CYCLE = 8
LR = 0.0002
MAX_NEW_TOKENS = 8
MAX_REPLAY_CE_INCREASE = 0.25
FIT = (
    ("Answer with one word only. What color is fresh snow?", "white"),
    ("Answer with one word only. What is the opposite of hot?", "cold"),
    ("Answer with one word only. What color is a ripe banana?", "yellow"),
    ("Answer with one digit only. What is 2 plus 2?", "4"),
    ("Answer with one word only. What is the opposite of up?", "down"),
    ("Answer with one word only. What do bees make?", "honey"),
    ("Answer with one word only. In what direction does the sun rise?", "east"),
    ("Answer with one digit only. How many sides does a triangle have?", "3"),
)
REPLAY = (
    ("Answer with one word only. What is the opposite of yes?", "no"),
    ("Answer with one word only. What is the opposite of black?", "white"),
    ("Answer with one word only. What color is most grass?", "green"),
    ("Answer with one word only. What sound does a cat make?", "meow"),
)
ALL_TRAIN = FIT + REPLAY


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n", encoding="utf-8")


def make_contexts(tok):
    result = []
    if tok.eos_token_id is None:
        raise AssertionError("Expected original EOS token")
    for question, answer in ALL_TRAIN:
        prompt = tok.apply_chat_template([{
            "role": "user", "content": question
        }], tokenize=False, add_generation_prompt=True)
        ctx = tok.encode(prompt, add_special_tokens=False)
        target = tok.encode(" " + answer, add_special_tokens=False)
        # The token before EOS must unambiguously express the requested
        # literal answer: no gold token is ever taken from EVAL/VALID.
        if not (1 <= len(target) <= 6 and 4 <= len(ctx) <= 160):
            raise AssertionError("Incorrect fixed TRAIN target/context geometry")
        target.append(tok.eos_token_id)
        result.append({
            "question": question, "gold_answer": answer,
            "input": torch.tensor([ctx], dtype=torch.long),
            "target": torch.tensor([target], dtype=torch.long),
            "split": "TRAIN_FIT" if len(result) < len(FIT) else "TRAIN_REPLAY",
        })
    return result


def token_ce(student, source, row):
    clear(source)
    ids = torch.cat((row["input"], row["target"]), dim=1)
    prefix_length = row["input"].shape[1]
    pred = student(input_ids=ids[:, :-1], use_cache=False).logits
    aligned = pred[:, prefix_length-1:, :]
    target = ids[:, prefix_length:]
    if aligned.shape[:2] != target.shape:
        raise AssertionError("Token alignment error including EOS")
    loss = F.cross_entropy(aligned.reshape(-1, aligned.shape[-1]),
                           target.reshape(-1))
    if not torch.isfinite(loss):
        raise AssertionError("TRAIN target loss became nonfinite")
    return loss


@torch.inference_mode()
def generate(student, source, tok, row):
    ids = row["input"].clone()
    generated = []
    eos = False
    for _ in range(MAX_NEW_TOKENS):
        clear(source)
        logits = student(input_ids=ids, use_cache=False).logits[0, -1]
        if not torch.isfinite(logits).all():
            raise AssertionError("Nonfinite actual output logits")
        tid = int(logits.argmax())
        if tid == tok.eos_token_id:
            eos = True
            break
        generated.append(tid)
        ids = torch.cat((ids, ids.new_tensor([[tid]])), dim=1)
    clear(source)
    value = tok.decode(generated, skip_special_tokens=True,
                       clean_up_tokenization_spaces=False)
    literal = value.strip().casefold()
    return {"output": value, "token_ids": generated, "EOS_seen": eos,
            "correct_full_literal": literal == row["gold_answer"].casefold(),
            "first_gold_token_match": bool(generated and
                generated[0] == int(row["target"][0, 0])),
            "gold_answer": row["gold_answer"]}


@torch.inference_mode()
def score(student, source, tok, rows):
    generated = []
    nll = []
    for i, row in enumerate(rows):
        raw = token_ce(student, source, row)
        nll.append(float(raw.detach().cpu()))
        clear(source)
        got = generate(student, source, tok, row)
        generated.append({"id": i, "split": row["split"], **got})
    correct = [i for i, x in enumerate(generated) if x["correct_full_literal"]]
    return {
        "fit_train_CE": sum(nll[:len(FIT)]) / len(FIT),
        "replay_train_CE": sum(nll[len(FIT):]) / len(REPLAY),
        "all_train_exact": len(correct),
        "fit_train_exact": sum(x["correct_full_literal"]
                               for x in generated[:len(FIT)]),
        "replay_train_exact": sum(x["correct_full_literal"]
                                  for x in generated[len(FIT):]),
        "first_gold_token_matches": sum(x["first_gold_token_match"]
                                        for x in generated),
        "EOS_seen": sum(x["EOS_seen"] for x in generated),
        "correct_indices": correct,
        "full_generated_answers": generated,
    }


def snapshot(source, optimizer):
    # Full 211-matrix / all-index transactional snapshot; snapshots exist
    # in TRAIN RAM only, never in the exported compressed model.
    return {
        "centroids": {name: x.centroids.detach().clone()
                      for name, x in source.items()},
        "codes": {name: x.codes.detach().clone()
                  for name, x in source.items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }


def rollback(source, optimizer, state):
    with torch.no_grad():
        for name, x in source.items():
            x.centroids.copy_(state["centroids"][name])
            x.codes.copy_(state["codes"][name])
    optimizer.load_state_dict(state["optimizer"])
    clear(source)


def choose_operators(source):
    keys = list(source)
    def group(text):
        found = sorted((k for k in keys if k.endswith(text)),
                       key=lambda k: int(k.split(".")[2]))
        if len(found) < 10:
            raise AssertionError("Missing original source operator " + text)
        return found
    q = group(".self_attn.q_proj.weight")
    o = group(".self_attn.o_proj.weight")
    down = group(".mlp.down_proj.weight")
    return (q[0], down[len(down)//2], q[-1], o[len(o)//2])


def sample_gradient_index_proposal(src, dense_weight_gradient, cycle, max_swaps=64):
    # Local first-order TRAIN gradient: choose a different existing 5bit
    # code for at most 64 groups out of predetermined 4096 sampled.
    # No original dense FP32 W, new inference weights or extra code bits.
    m, n = src.shape
    n_per_row = n//8
    total = m*n_per_row
    count = min(4096, total)
    positions = (np.arange(count, dtype=np.int64)
                 * total // count + (cycle+1)*919) % total
    positions = np.unique(positions)
    rows = positions // n_per_row
    groups = positions % n_per_row
    cols = 8*groups[:, None] + np.arange(8)[None, :]
    grad = dense_weight_gradient[rows[:, None], cols].astype(np.float32)
    centers = src.centroids.detach().numpy().astype(np.float32)
    scales = src.base_scales.detach().numpy()[rows]
    score = (grad*scales[:, None]) @ centers.T
    old = src.codes.detach().numpy()[positions].astype(np.int64)
    best = np.argmin(score, axis=1)
    gain = score[np.arange(len(old)), old] - score[
        np.arange(len(old)), best]
    good = np.flatnonzero((best != old) & (gain > 1e-9))
    if len(good) == 0:
        return {"proposed": 0, "ids": [], "old": [], "new": []}
    order = good[np.argsort(-gain[good], kind="stable")]
    choice = order[:max_swaps]
    return {"proposed": len(choice),
            "ids": positions[choice].tolist(),
            "old": old[choice].tolist(),
            "new": best[choice].tolist(),
            "gradient_ranked_candidate_groups": len(good)}


def apply_index_proposal(src, proposal, new=True):
    if not proposal["proposed"]:
        return
    ids = torch.tensor(proposal["ids"], dtype=torch.long)
    value = torch.tensor(proposal["new" if new else "old"],
                         dtype=torch.long)
    with torch.no_grad():
        src.codes[ids] = value


@torch.inference_mode()
def index_probe(student, source, row):
    return float(token_ce(student, source, row))


def codebook_changes(source, baseline):
    changed_fp16 = sum(int(torch.count_nonzero(
        x.centroids.detach().half() != baseline[name].half()))
        for name, x in source.items())
    return changed_fp16


def run(infile, outdir):
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xffffffff)
    torch.set_num_threads(2)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    binary = Path(infile).read_bytes()
    input_sha = hashlib.sha256(binary).hexdigest()
    if input_sha != EXPECTED_INPUT_SHA or len(binary) != EXPECTED_BYTES:
        raise AssertionError("Not the immutable independently recoded original 135M S0")
    archive = CB59(binary)
    expected_keys = archive.meta["tensor_descriptors"]
    n_unique = int(archive.meta["original_unique_parameter_count"])
    n_matrices = sum(int(d["dtype"] == "CB59_K32V8")
                     for d in expected_keys.values())
    if n_unique != EXPECTED_UNIQUE or n_matrices != 211:
        raise AssertionError("Source pretrained 135M geometry mismatch")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    data = make_contexts(tok)
    manifest = {
        "protocol": "4 cycles x 8 real autograd steps / all 211 paid",
        "original_source_sha256": input_sha,
        "original_source_bytes": len(binary),
        "original_unique_parameters": n_unique,
        "source_bpp": 8*len(binary)/n_unique,
        "CYCLES": CYCLES, "STEPS_PER_CYCLE": STEPS_PER_CYCLE,
        "steps": CYCLES*STEPS_PER_CYCLE, "learning_rate": LR,
        "fit": list(FIT), "replay": list(REPLAY),
        "fully_supervised_train_explicit_gold_and_EOS": True,
        "training_samples_used_for_checkpoint_selection": "TRAIN ONLY",
        "old_v231_v232_valid_eval_reused": False,
        "target_not_pretrained_teacher_output": True,
        "no_teacher_FP32_weights_loaded_during_S1": True,
        "no_Jessica_LIVE_main_Render_Supabase_APK": True,
        "not_claiming_generalization": True,
    }
    save(out/"S1_PRECOMMITTED_TRAIN_PROTOCOL.json", manifest)
    student, source = build_trainable_from_paid_source(archive)
    student.cpu().eval()
    if len(source) != 211:
        raise AssertionError("Missing original 211 codebooks")
    base_centroids = {name: s.centroids.detach().clone()
                      for name, s in source.items()}
    operators = choose_operators(source)
    optimizer = torch.optim.Adam((s.centroids for s in source.values()), lr=LR)
    initial = score(student, source, tok, data)
    emit("S1_initial_actual_TRAIN_only", fit_CE=initial["fit_train_CE"],
         replay_CE=initial["replay_train_CE"],
         full_correct=initial["all_train_exact"],
         first_gold=initial["first_gold_token_matches"])
    best = initial
    # Only TRAIN selection. No hidden validation checkpoint tuning.
    cycles = []
    for cycle in range(CYCLES):
        snap = snapshot(source, optimizer)
        last_grad = None
        target_name = operators[cycle]
        target_operator = source[target_name]
        history = []
        start = time.perf_counter()
        for k in range(STEPS_PER_CYCLE):
            train_id = k
            idx = len(FIT) + ((k//2+cycle)%len(REPLAY))
            optimizer.zero_grad(set_to_none=True)
            recorded = []
            def hook(_module, _args, reconstructed):
                if not any(reconstructed is seen for seen in recorded):
                    reconstructed.retain_grad()
                    recorded.append(reconstructed)
            h = target_operator.register_forward_hook(hook)
            try:
                fit_ce = token_ce(student, source, data[train_id])
                fit_ce.backward()
                if len(recorded) != 1 or recorded[0].grad is None:
                    raise AssertionError("Real through-model operator gradient absent")
                last_grad = recorded[0].grad.detach().numpy().copy()
            finally:
                h.remove()
                clear(source)
            # Every other step explicitly rehearses old TRAIN knowledge.
            if k % 2 == 0:
                replay_ce = token_ce(student, source, data[idx])
                (0.35*replay_ce).backward()
                val_replay = float(replay_ce.detach())
                clear(source)
            else:
                val_replay = None
            anchor = torch.stack([
                (s.centroids-base_centroids[name]).square().mean()
                for name, s in source.items()]).mean()
            (0.01*anchor).backward()
            active = sum(s.centroids.grad is not None
                         for s in source.values())
            if active != 211:
                raise AssertionError("Full 211-codebook Green Tunnel grad check failed")
            norm = float(torch.nn.utils.clip_grad_norm_(
                [s.centroids for s in source.values()], 1.0,
                error_if_nonfinite=True))
            optimizer.step()
            with torch.no_grad():
                for s in source.values():
                    s.centroids.clamp_(-16.0, 16.0)
            clear(source)
            row = {"cycle": cycle+1, "step": k+1,
                   "fit_idx": train_id, "replay_idx": idx if k%2==0 else None,
                   "fit_train_teacher_forcing_CE": float(fit_ce.detach()),
                   "replay_train_CE": val_replay, "gradients_paid": active,
                   "grad_norm_before_clip": norm}
            history.append(row)
        if last_grad is None:
            raise AssertionError("No actual local code-swap gradient")
        # Candidate discrete code swaps with a real, same-TRAIN loss gate.
        pair = data[STEPS_PER_CYCLE-1]
        protected = data[len(FIT)+cycle%len(REPLAY)]
        before_idx_fit = index_probe(student, source, pair)
        before_idx_replay = index_probe(student, source, protected)
        proposal = sample_gradient_index_proposal(
            target_operator, last_grad, cycle)
        apply_index_proposal(target_operator, proposal, new=True)
        clear(source)
        after_idx_fit = index_probe(student, source, pair)
        after_idx_replay = index_probe(student, source, protected)
        code_ok = bool(proposal["proposed"] and
                       after_idx_fit+1e-7 < before_idx_fit and
                       after_idx_replay <= before_idx_replay+0.01)
        if not code_ok:
            apply_index_proposal(target_operator, proposal, new=False)
            clear(source)
        candidate = score(student, source, tok, data)
        old_correct = set(best["correct_indices"])
        lost = sorted(old_correct-set(candidate["correct_indices"]))
        keep = bool(
            candidate["fit_train_CE"] + 1e-4 < best["fit_train_CE"]
            and candidate["replay_train_CE"] <=
              best["replay_train_CE"] + MAX_REPLAY_CE_INCREASE
            and not lost
        )
        diagnostics = {
            "cycle": cycle+1,
            "eight_actual_autograd_steps": len(history),
            "all_211_original_paid_codebooks_updated_per_step": True,
            "updated_operator_index_candidate": target_name,
            "sampled_index_swaps_proposed": proposal["proposed"],
            "5bit_code_indices_accepted": proposal["proposed"] if code_ok else 0,
            "index_fit_CE_before": before_idx_fit,
            "index_fit_CE_after": after_idx_fit,
            "index_replay_CE_before": before_idx_replay,
            "index_replay_CE_after": after_idx_replay,
            "fit_CE_before": best["fit_train_CE"],
            "fit_CE_candidate": candidate["fit_train_CE"],
            "guard_replay_CE_before": best["replay_train_CE"],
            "guard_replay_CE_candidate": candidate["replay_train_CE"],
            "full_correct_candidate": candidate["all_train_exact"],
            "first_gold_token_matches_candidate":
                candidate["first_gold_token_matches"],
            "previous_correct_full_answers_lost": lost,
            "J7_guard_checkpoint_accepted": keep,
            "physically_paid": True,
            "training_not_heldout_EVAL": True,
            "wall_seconds": round(time.perf_counter()-start, 2),
            "step_diagnostics": history,
        }
        if keep:
            best = candidate
            candidate_bytes, seal = replace_discrete(archive, source,
                                                       strict=False)
            if (len(candidate_bytes) != len(binary) or
                seal["complete_bpp"] > .700 or
                seal["all_2d_original_matrices"] != 211):
                raise AssertionError("Unpaid candidate escaped source-only budget")
            fname = f"S1_ACCEPTED_CYCLE{cycle+1}_135M_PAID.cb59"
            (out/fname).write_bytes(candidate_bytes)
            diagnostics["accepted_physical_source_sha256"] = seal["sha256"]
            diagnostics["changed_paid_fp16_dictionary_cells_vs_start"] = \
                codebook_changes(source, base_centroids)
        else:
            rollback(source, optimizer, snap)
            if score(student, source, tok, data)["full_generated_answers"] \
                    != best["full_generated_answers"]:
                raise AssertionError("J7 rollback does not restore exact all TRAIN outputs")
        cycles.append(diagnostics)
        save(out/"S1_ALL_FOUR_CYCLE_DIAGNOSTICS_IN_PROGRESS.json", cycles)
        emit("S1_cycle_actual", cycle=cycle+1,
             accepted=keep, fit_CE=round(best["fit_train_CE"], 5),
             replay_CE=round(best["replay_train_CE"], 5),
             full_correct=best["all_train_exact"],
             accepted_swaps=diagnostics["5bit_code_indices_accepted"],
             lost_correct=lost)
    final_bytes, final_seal = replace_discrete(archive, source, strict=False)
    if len(final_bytes) != EXPECTED_BYTES or final_seal["complete_bpp"] > 0.7:
        raise AssertionError("Training violated immutable physical budget")
    (out/"S1_FINAL_135M_4CYCLES_PAID.cb59").write_bytes(final_bytes)
    reload = CB59(final_bytes)
    if reload.hash() != final_seal["sha256"]:
        raise AssertionError("Corrupt physically saved final compressed model")
    report = {
        "stage": "S1_four_real_training_cycles",
        "completed": True,
        "real_optimizer_steps_attempted": CYCLES*STEPS_PER_CYCLE,
        "true_all_211_codebook_gradient_checks": CYCLES*STEPS_PER_CYCLE,
        "accepted_cycles": sum(c["J7_guard_checkpoint_accepted"]
                                for c in cycles),
        "baseline": initial,
        "final": best,
        "cycles": cycles,
        "source_input_SHA": input_sha,
        "physically_paid_final_source_sha256": final_seal["sha256"],
        "final_physical_source_bytes": len(final_bytes),
        "complete_final_bpp": final_seal["complete_bpp"],
        "total_fp16_centroid_cells_changed": codebook_changes(source,
                                                              base_centroids),
        "conclusion": "Only TRAIN-only diagnostic; not FP32 or heldout parity",
        "new_independent_EVAL_ran": False,
        "native_C_benchmark_ran": False,
        "JESSICA_tests_ran": False,
        "production_modified": False,
    }
    save(out/"S1_REAL_FOUR_CYCLE_FINAL_REPORT.json", report)
    emit("S1_final_real_four_cycle_training",
         accepted_cycles=report["accepted_cycles"],
         baseline_full_correct=initial["all_train_exact"],
         final_full_correct=best["all_train_exact"],
         baseline_fit_CE=round(initial["fit_train_CE"], 4),
         final_fit_CE=round(best["fit_train_CE"], 4),
         baseline_guard_CE=round(initial["replay_train_CE"], 4),
         final_guard_CE=round(best["replay_train_CE"], 4),
         total_changed_centroid_cells=report["total_fp16_centroid_cells_changed"],
         physical_bpp=final_seal["complete_bpp"],
         final_SHA=final_seal["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("S0_original_CB59", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    run(a.S0_original_CB59, a.output)
