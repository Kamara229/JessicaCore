"""CALIBRI S160: first REAL S142 shared-prefill parallel-parameter neural head trial.

Only original S87 456 and S122 128 TRAIN; no holdout/QA. Freeze all real S142
43M weights, existing route bank and Green Tunnel. Cache genuine S142 prefill
hidden states, train a small joint owner/span neural head on Apple MPS, then
physically persist FP16 and reload for actual CPU cached full+EOS inference.
A single prefill generates both independent slot scores simultaneously.
Confidence-gated early exit or EXACT original S101 decoder fallback using
the SAME prefill/cache. Never run N copies of 43M inference.
J7: reject if ANY previously correct full/first answer is lost; physical
bpp <= .7; paired native CPU speed >= .9; require genuine new full/pair gain.
No MAIN, LIVE Jessica, Render, Supabase or APK changes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s91 import train_joint_quality_rehearsal as s91
from calibri2.s94.physical_profile_bank import freeze_report, sha
from calibri2.s95.train_and_audit_router import prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import apply_profile, load_frozen_numeric_weights
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s98.train_owner_anchored_copy import candidates
from calibri2.s115.cached_phase_gate import frozen_cpu_phase_scope
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import cohort, load_cpu_original
from calibri2.s122.independent_binding_train_curriculum import materialize_train_only
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import PREFIX, load_compact_q2
from calibri2.s127.q2_prefill_gate_control import prefill_only_tunnel
from calibri2.s132.progressive_core import ProgressiveRouteCore, IDS_KEY, WEIGHTS_KEY
from calibri2.s144.train_one_continued_j7 import frozen_routes, score_full_and_pairs
from calibri2.s158 import targeted_retention_gpu as s158
from calibri2.s159.parallel_slot_binding import ParallelSlotBindingHead, source_span_nll
from v176.operators import clear

SEED = 1600160
TRAIN_STEPS = 120
BATCH = 12
LR = 0.002
SLOT_COUNT = 2
SLOT_RANK = 16
MIN_CONFIDENCE = 0.995
MIN_MARGIN = 3.5
MAX_BPP = 0.7
MIN_CPU_RATIO = 0.9
CPU_SPEED_COUNT = 16


def prompt_tokens(tok, question):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    return tok.encode(prompt, add_special_tokens=False)


def collect_real_prefills(cpu, old, new, old_ids, new_ids, routes, tok,
                          bank, bridge, full, expansion):
    """Single frozen S142 prefill per TRAIN identity, no gold used in forward."""
    model, sources, *_ = cpu
    rows = {}
    chosen_profile = None
    counts = Counter()
    for key in (*old_ids, *new_ids):
        row = old.get(key) or new[key]
        route = routes[key]
        if route != "S93":
            counts["non_S93_parent_only"] += 1
            continue
        if route != chosen_profile:
            apply_profile(sources, cpu[6], route, bank, cpu[7], cpu[8])
            chosen_profile = route
        ids = prompt_tokens(tok, row["question"])
        gold = row["expected"]
        if len(gold) != SLOT_COUNT + 1 or gold[-1] != tok.eos_token_id:
            counts["not_two_token_EOS"] += 1
            continue
        starts = [p for p in range(len(ids) - SLOT_COUNT + 1)
                  if ids[p:p + SLOT_COUNT] == gold[:SLOT_COUNT]]
        if not starts:
            counts["no_exact_contiguous_source_target"] += 1
            continue
        features = torch.as_tensor(
            prompt_only_vector(tok, row["question"]), dtype=torch.float32)
        with torch.inference_mode(), expansion.prefill_only(model), prefill_only_tunnel(model, full):
            with frozen_cpu_phase_scope(model, bridge, route, features, phase="prefill_only"):
                clear(sources)
                output = model(input_ids=torch.tensor([ids], dtype=torch.long),
                               past_key_values=None, use_cache=False, output_hidden_states=True)
                states = output.hidden_states[-1][0].detach().float().contiguous()
                del output
                clear(sources)
        if states.shape != (len(ids), 320):
            raise AssertionError("real frozen S142 hidden-state dimension drift")
        rows[key] = {
            "states": states,
            "ids": torch.tensor(ids, dtype=torch.long),
            "gold": torch.tensor(gold[:2], dtype=torch.long),
            "answer": row.get("answer"),
            "target_span_count": len(starts),
            "group": "new" if key in new else "old",
            "question": row["question"],
        }
        counts["matched_new" if key in new else "matched_old"] += 1
    print(json.dumps({"event": "S160_REAL_S142_SOURCE_SPAN_FEASIBILITY",
                      "counts": dict(counts), "available_for_neural_training": len(rows)}),
          flush=True)
    return rows, dict(counts)


def packed_batch(rows, keys, device):
    maxlen = max(len(rows[k]["ids"]) for k in keys)
    states = torch.zeros(len(keys), maxlen, 320, dtype=torch.float32)
    ids = torch.zeros(len(keys), maxlen, dtype=torch.long)
    masks = torch.zeros(len(keys), maxlen, dtype=torch.bool)
    labels = torch.zeros(len(keys), 2, dtype=torch.long)
    for i, key in enumerate(keys):
        row = rows[key]
        size = len(row["ids"])
        states[i, :size] = row["states"]
        ids[i, :size] = row["ids"]
        masks[i, :size] = True
        labels[i] = row["gold"]
    return (states.to(device), ids.to(device), masks.to(device), labels.to(device))


def train_neural_slot_head(rows):
    """All labels are precommitted ORIGINAL TRAIN; no QA and no parent answer key."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("Real MPS/GPU is required; CPU imitation disallowed")
    old = sorted(k for k in rows if rows[k]["group"] == "old")
    new = sorted(k for k in rows if rows[k]["group"] == "new")
    if len(old) < 50 or len(new) < 16:
        raise AssertionError("S160 source span matching too sparse for real neural training")
    head = ParallelSlotBindingHead(320, slots=2, rank=SLOT_RANK).to("mps")
    initial = copy.deepcopy(head.state_dict())
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    old_adam = copy.deepcopy(optimizer.state_dict())
    trace = []
    for step in range(TRAIN_STEPS):
        # Deterministic precommitted rotating replay from both ORIGINAL TRAIN
        # cohorts; never choose rows based on validation results.
        half = BATCH // 2
        keys = ([old[(step * half + j) % len(old)] for j in range(half)] +
                [new[(step * half + j) % len(new)] for j in range(half)])
        states, ids, mask, labels = packed_batch(rows, keys, "mps")
        optimizer.zero_grad(set_to_none=True)
        out = head(states, ids, mask)
        loss = source_span_nll(out, ids, labels, mask)
        if not torch.isfinite(loss):
            raise AssertionError("S160 real MPS joint span loss is nonfinite")
        loss.backward()
        grad = head.query.weight.grad
        if grad is None or not float(grad.abs().sum()) > 0:
            raise AssertionError("S160 real neural slot query has no gradient")
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step in (0, 15, 31, 63, 95, TRAIN_STEPS - 1):
            torch.mps.synchronize()
            record = {"step": step + 1, "loss": float(loss.detach()),
                      "sampled_original_TRAIN_ids": keys}
            trace.append(record)
            print(json.dumps({"event": "S160_REAL_S142_SHARED_PREFILL_MPS_TRAIN", **record}),
                  flush=True)
    return head, optimizer, initial, old_adam, trace


def select_parallel(head, hidden, ids):
    """Shared prefill and one vectorized joint span pass, original route S93."""
    with torch.inference_mode():
        result = head(hidden.unsqueeze(0).float(), ids)
        admitted = (
            int(result["candidate_count"].item()) >= 2 and
            float(result["confidence"].item()) >= MIN_CONFIDENCE and
            float(result["margin"].item()) >= MIN_MARGIN
        )
        return result["slot_ids"][0].tolist() if admitted else None


def cached_full_cpu_parallel(
    model, sources, first, later, lm_head, tok, question, pointer,
    copy_net, parallel_head,
):
    """Exact S101 full+EOS decoder except for one optional step0 early exit.

    Reuses step0 hidden states + original KV cache. Fallback follows S101
    token-selection logic with no repeated prefill. Only admitted candidates
    bypass the second and subsequent 43M decoding steps.
    """
    from calibri2.s99.guard_neural_copy_456_cpu import CONFIDENCE_THRESHOLD
    ids = prompt_tokens(tok, question)
    prompt_input = torch.tensor([ids], dtype=torch.long)
    kept = s82.candidate_positions(tok, ids) if pointer is not None else []
    source_token_ids = torch.tensor([ids[i] for i in kept], dtype=torch.long)
    generated = []
    cache = None
    source_keys = None
    eligible = False
    applied = False
    with torch.inference_mode():
        for step in range(s82.s56.MAX_NEW):
            inputs = prompt_input if step == 0 else torch.tensor([[generated[-1]]], dtype=torch.long)
            clear(sources)
            result = model(input_ids=inputs, past_key_values=cache,
                           use_cache=True, output_hidden_states=True)
            cache = result.past_key_values
            if cache is None:
                raise AssertionError("S160 original S101 cached autoregressive forward failed")
            hidden = result.hidden_states[-1][0]
            if step == 0:
                proposal = select_parallel(parallel_head, hidden, prompt_input)
                if proposal is not None:
                    del result, cache
                    clear(sources)
                    return proposal + [tok.eos_token_id], {"parallel": True}
            if step == 0 and pointer is not None and kept:
                source_keys = hidden[torch.tensor(kept, dtype=torch.long)].detach().clone()
            h = hidden[-1]
            logits = result.logits[0, -1] + F.linear(
                first(h) if step == 0 else later(h), lm_head)
            if step == 1 and pointer is not None and kept:
                if int(logits.argmax()) != tok.eos_token_id:
                    bonus = pointer.full_vocab_bonus(h, source_keys, source_token_ids)
                    logits = logits + torch.zeros_like(logits).index_add(
                        0, source_token_ids, bonus)
            chosen = int(logits.argmax())
            if step == 1 and copy_net is not None and generated:
                pair = candidates(tok, question, generated[0])
                if len(pair) == 2 and pair[0][0] != pair[1][0]:
                    eligible = True
                    vectors = torch.as_tensor(
                        np.stack([x[1] for x in pair]), dtype=torch.float32)
                    score = copy_net(vectors)
                    probs = torch.softmax(score, dim=0)
                    winner = int(score.argmax())
                    if float(probs[winner]) >= CONFIDENCE_THRESHOLD:
                        desired = int(pair[winner][0])
                        if desired != chosen and desired != tok.eos_token_id:
                            chosen = desired
                            applied = True
            generated.append(chosen)
            del result
            clear(sources)
            if chosen == tok.eos_token_id:
                break
    del cache, source_keys
    return generated, {"parallel": False, "candidate_present": eligible,
                       "neural_override_applied": applied}


def audit_cpu_outputs(cpu, old, new, old_ids, new_ids, routes, tok, bank,
                      copy_net, bridge, full, expansion, parent_old,
                      parent_new, head):
    model, sources, first, later, pointer, lm_head, *_ = cpu
    predicted_old, predicted_new = {}, {}
    chosen_route = None
    stats = Counter()
    sample_records = []
    for key in (*old_ids, *new_ids):
        row = old.get(key) or new[key]
        route = routes[key]
        if route != chosen_route:
            apply_profile(sources, cpu[6], route, bank, cpu[7], cpu[8])
            chosen_route = route
        if route != "S93":
            reply = (parent_old if key in old else parent_new)[key]
            used = False
        else:
            features = torch.as_tensor(prompt_only_vector(tok, row["question"]),
                                       dtype=torch.float32)
            with torch.inference_mode(), expansion.prefill_only(model), prefill_only_tunnel(model, full):
                with frozen_cpu_phase_scope(model, bridge, route, features, phase="prefill_only"):
                    reply, note = cached_full_cpu_parallel(
                        model, sources, first, later, lm_head, tok,
                        row["question"], pointer, copy_net, head)
            used = note["parallel"]
        prediction = predicted_old if key in old else predicted_new
        prediction[key] = reply
        stats["parallel_accepted" if used else "parent_fallback"] += 1
        gold = row["expected"]
        parent = (parent_old if key in old else parent_new)[key]
        if reply == gold:
            stats["correct_after"] += 1
        if used:
            stats["parallel_correct" if reply == gold else "parallel_wrong"] += 1
        if (used or (parent != gold and reply == gold)) and len(sample_records) < 48:
            sample_records.append({
                "id": key,
                "question": row["question"],
                "expected_ids": gold,
                "parent_generated_ids": parent,
                "candidate_generated_ids": reply,
                "candidate_correct": reply == gold,
                "parent_correct": parent == gold,
                "fastpath": used,
            })
    return predicted_old, predicted_new, dict(stats), sample_records


def paired_cpu_ratio(cpu, old, old_ids, routes, tok, bank, copy_net,
                     bridge, full, expansion, parent_old, head):
    model, sources, first, later, pointer, lm_head, *_ = cpu
    base_times, trial_times = [], []
    chosen = None
    for n, key in enumerate(old_ids[:CPU_SPEED_COUNT]):
        row = old[key]
        route = routes[key]
        if route != chosen:
            apply_profile(sources, cpu[6], route, bank, cpu[7], cpu[8])
            chosen = route
        feat = torch.as_tensor(prompt_only_vector(tok, row["question"]),
                               dtype=torch.float32) if route == "S93" else None
        def go(with_head):
            with torch.inference_mode(), expansion.prefill_only(model), prefill_only_tunnel(model, full):
                with frozen_cpu_phase_scope(model, bridge, route, feat,
                                            phase="prefill_only"):
                    started = time.perf_counter()
                    if with_head and route == "S93":
                        answer, _ = cached_full_cpu_parallel(
                            model, sources, first, later, lm_head, tok,
                            row["question"], pointer, copy_net, head)
                    else:
                        answer, _ = cached_full_cpu_with_neural_copy(
                            model, sources, first, later, lm_head, tok,
                            row["question"], pointer, copy_net)
                    elapsed = time.perf_counter() - started
            return answer, elapsed
        if n % 2:
            trial = go(True)
            baseline = go(False)
        else:
            baseline = go(False)
            trial = go(True)
        if baseline[0] != parent_old[key]:
            raise AssertionError("S160 physical CPU speed parent decoding drift")
        (base_times if baseline[1] > 0 else []).append(baseline[1])
        trial_times.append(trial[1])
    return {
        "ratio": statistics.median(base_times) / statistics.median(trial_times),
        "baseline_median_s": statistics.median(base_times),
        "trial_median_s": statistics.median(trial_times),
        "paired_samples": len(base_times),
        "same_cpu": True,
    }


def main():
    a = s158.args_from_cli()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    guard, side, q2, inherited = s158.check_sources(a)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK = tok
    novel, novel_digest, _ = materialize_train_only(tok)
    expected_digest = json.loads(a.S122_curriculum_report.read_text())[
        "new_TRAIN_question_hashset_sha256"]
    if novel_digest != expected_digest or len(novel) != 128:
        raise AssertionError("Original S122 TRAIN corpus drift")
    new = {row["id"]: row for row in novel}
    new_ids = sorted(new)
    _, _, _, bank, router, copy_net = load_frozen_numeric_weights(a)
    new_routes = frozen_routes(tok, router, new)
    reports = {name: freeze_report(getattr(a, name + "_report"))
               for name in ("S87", "S91", "S93")}
    old, old_ids, old_routes = cohort(tok, reports, router)
    routes = {**old_routes, **new_routes}
    if len(old_ids) != 456 or len(new_ids) != 128:
        raise AssertionError("S160 exact original 456+128 TRAIN required")
    raw = s87.open_envelope(a.S93_CB61.read_bytes())
    cpu = load_cpu_original(raw, side)
    bridge = full = expansion = None
    try:
        bridge = PromptConditionedDualTunnel(cpu[6]).eval().install(cpu[0])
        bridge.coefficients.requires_grad_(False)
        full = FullLayerStateTunnel(width=320, depth=12, rank=4).eval()
        load_compact_q2(full, q2)
        full.install(cpu[0])
        expansion = ProgressiveRouteCore(rank=896).eval().bind_original_route_bridge(
            bridge).install(cpu[0])
        expansion.load_active_fp16({IDS_KEY: inherited[IDS_KEY],
                                    WEIGHTS_KEY: inherited[WEIGHTS_KEY]})
        original_hash = sha(a.S142_parent_sidecar.read_bytes())
        parent_old, parent_new = s158.run_outputs(
            cpu, old, new, old_ids, new_ids, old_routes, new_routes,
            tok, bank, copy_net, bridge, full, expansion)
        if (sum(parent_old[k] == old[k]["expected"] for k in old) != 362 or
            sum(parent_new[k] == new[k]["expected"] for k in new) != 17):
            raise AssertionError("S160 immutable S142 full+EOS 456+128 baseline drift")
        features, matching = collect_real_prefills(
            cpu, old, new, old_ids, new_ids, routes, tok, bank, bridge, full, expansion)
        trained, opt, initial, initial_adam, trace = train_neural_slot_head(features)
        physical = a.out / "S160_REAL_PARALLEL_SLOT_FP16.safetensors"
        save_file({k: v.detach().cpu().half().contiguous()
                   for k, v in trained.state_dict().items()}, str(physical))
        reloaded = load_file(str(physical))
        if set(reloaded) != set(trained.state_dict()):
            raise AssertionError("S160 physical FP16 tensor keys drift")
        for k, v in reloaded.items():
            if v.dtype != torch.float16 or not torch.isfinite(v).all():
                raise AssertionError("S160 non-finite or wrong precision FP16")
        physical_head = ParallelSlotBindingHead(320, slots=2, rank=SLOT_RANK).eval()
        physical_head.load_state_dict({k: v.float() for k, v in reloaded.items()}, strict=True)
        # Physical head is gated by fixed thresholds and used only for S93.
        pred_old, pred_new, activations, examples = audit_cpu_outputs(
            cpu, old, new, old_ids, new_ids, routes, tok, bank, copy_net,
            bridge, full, expansion, parent_old, parent_new, physical_head)
        retention = s158.retention(parent_old, pred_old, old, parent_new, pred_new, new)
        strict = all(not retention[key] for key in (
            "lost_old_full_ids", "lost_old_first_ids",
            "lost_new_full_ids", "lost_new_first_ids"))
        gain = (
            retention["old_full_after"] > retention["old_full_before"] or
            retention["new_full_after"] > retention["new_full_before"] or
            retention["new_pairs_both_after"] > retention["new_pairs_both_before"])
        runtime = a.out / "S160_RUNTIME.json"
        runtime.write_text(json.dumps({
            "stage": "S160_REAL_JOINT_SLOT_CAUSAL_S142",
            "model_accepted": False, "parent_sha": original_hash,
            "slots": SLOT_COUNT, "rank": SLOT_RANK,
            "confidence_threshold": MIN_CONFIDENCE, "margin_threshold": MIN_MARGIN,
            "route": "S93", "default_enabled": False, "trained_on": "original TRAIN ONLY",
            "no_QA_no_LIVE": True,
        }, sort_keys=True))
        params = int(guard.get("architectural_parameter_count", 43_261_760))
        if params < 43_261_760:
            params = 43_261_760
        new_bytes = physical.stat().st_size + runtime.stat().st_size
        bpp = (guard["physical_bpp"] * params + 8 * new_bytes) / (
            params + sum(p.numel() for p in physical_head.parameters()))
        cpu_ratio = paired_cpu_ratio(
            cpu, old, old_ids, routes, tok, bank, copy_net, bridge, full,
            expansion, parent_old, physical_head)
        permitted = strict and gain and bpp <= MAX_BPP and cpu_ratio["ratio"] >= MIN_CPU_RATIO
        if not permitted:
            trained.load_state_dict(initial, strict=True)
            opt.load_state_dict(initial_adam)
            if not all(torch.equal(trained.state_dict()[k], v)
                       for k, v in initial.items()):
                raise AssertionError("S160 J7 head rollback weights differ")
            if opt.state_dict() != initial_adam:
                raise AssertionError("S160 J7 optimizer rollback differs")
        if sha(a.S142_parent_sidecar.read_bytes()) != original_hash:
            raise AssertionError("S160 immutable parent physical checkpoint changed")
        # Store 48 factual TRAIN examples only; no sealed QA or invented answers.
        report = {
            "stage": "CALIBRI_S160_FIRST_REAL_43M_PARALLEL_SLOT_MPS_EXPERIMENT",
            "S142_physical_parent_sha256": original_hash,
            "source_matching_original_TRAIN": matching,
            "mps_train_steps": TRAIN_STEPS,
            "mps_train_trace": trace,
            "physical_FP16_sha256": sha(physical.read_bytes()),
            "physical_all_in_bpp": bpp,
            "paired_CPU_speed": cpu_ratio,
            "retention_456_128": retention,
            "candidate_activation": activations,
            "actual_train_examples": examples,
            "strict_no_forgetting": strict,
            "actual_train_gain": gain,
            "safe_research_candidate": permitted,
            "J7_rollback_verified": not permitted,
            "model_accepted": False,
            "original_S142_unchanged": True,
            "embargoed_QA_never_run": True,
            "no_MAIN_LIVE_Jessica_Render_Supabase_APK_changes": True,
        }
        (a.out / "S160_REAL_PARALLEL_456_128_J7_AUDIT.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        print(json.dumps({
            "event": "S160_FIRST_REAL_PARALLEL_43M_456_128_J7_AUDIT",
            "old_before": retention["old_full_before"],
            "old_after": retention["old_full_after"],
            "new_before": retention["new_full_before"],
            "new_after": retention["new_full_after"],
            "lost_old": len(retention["lost_old_full_ids"]),
            "lost_new": len(retention["lost_new_full_ids"]),
            "parallel_accepted": activations.get("parallel_accepted", 0),
            "parallel_correct": activations.get("parallel_correct", 0),
            "all_in_bpp": bpp,
            "CPU_ratio": cpu_ratio["ratio"],
            "safe_candidate": permitted, "rolled_back": not permitted,
        }), flush=True)
    finally:
        if expansion is not None:
            expansion.uninstall()
        if full is not None:
            full.uninstall()
        if bridge is not None:
            bridge.uninstall()
        if cpu is not None:
            cpu[6].uninstall()


if __name__ == "__main__":
    main()
