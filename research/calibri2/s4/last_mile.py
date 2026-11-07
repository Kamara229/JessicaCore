"""CALIBRI 2.0 S4: bounded last-mile multi-token training on 22M paid CB59.
Research only. Exactly one predeclared continuation from the latest S3 physical
archive. Training sees TRAIN pairs only, never QA/VALID/TEST or FP32 weights.
All 43 compact dictionaries remain paid, no dense inference sidecar.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.optim import propose_codes, apply_codes
from v185.codec import replace_discrete
from v186.pairs import build_pairs, tokenize_one
from v200.rollout import probe_greedy_train
from calibri2.s0.bootstrap import CANDIDATES, build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.diverse_task_replay import make_new_pairs
from v174.run import save

START_SHA = "b7ea64d44b6553c3475377973f6cb5ddd2376be871e065e3d77123bf923d1e5c"
START_BYTES = 1932195
PARAMS = 22368320
SEED = 2044
STEPS = 2048
LR = 0.003
WEIGHT_LATE = 3.0
MAX_SWAPS = 512

LEFT_A = ("amber", "quiet", "small", "golden", "gentle", "purple", "yellow", "wooden")
RIGHT_A = ("bright", "silver", "open", "winter", "fresh", "distant", "brave", "smooth")
LEFT_B = ("harbor", "planet", "thread", "lake", "station", "window", "river", "bridge")
RIGHT_B = ("mountain", "beacon", "tunnel", "castle", "pillow", "market", "garden", "forest")
SENTENCE_PAIRS = (
 ("What does a lock do?", "Locks keep doors closed.", "What does a key do?", "Keys open locks."),
 ("Why do people drink water?", "Water helps the body.", "Why do plants need rain?", "Rain gives plants water."),
 ("What does a bus carry?", "Buses carry passengers.", "What does a truck carry?", "Trucks carry cargo."),
 ("What can a pencil do?", "Pencils can write words.", "What can an eraser do?", "Erasers remove marks."),
 ("Why do people cook food?", "Cooking makes food ready.", "Why do people use blankets?", "Blankets keep people warm."),
 ("What does a window provide?", "Windows let in light.", "What does a roof provide?", "Roofs keep rain out."),
 ("Where do fish live?", "Fish live in water.", "Where do birds build nests?", "Birds build nests in trees."),
 ("What helps a boat move?", "Moving water pushes boats.", "What helps a plane fly?", "Its wings produce lift."),
)

def deterministic_cpu():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def new_pairs(tok):
    raw = []
    def add(domain, left, right):
        assert left[0] != right[0] and left[1] != right[1]
        raw.append((domain, left, right))
    for i in range(32):
        word_l = LEFT_A[i % 8] + " " + LEFT_B[(i // 8 + i) % 8]
        word_r = RIGHT_A[i % 8] + " " + RIGHT_B[(i // 8 + i) % 8]
        context = f"Training card PACK{i:03d} carries the two-word code"
        if i % 2:
            left = (f"{context} '{word_l}'. Copy both words of the code and stop.", word_l)
            right = (f"{context} '{word_r}'. Copy both words of the code and stop.", word_r)
        else:
            left = (f"Quote exactly the two words '{word_l}' from memory card PACK{i:03d}.", word_l)
            right = (f"Quote exactly the two words '{word_r}' from memory card PACK{i:03d}.", word_r)
        add("s4_two_word_copy", left, right)
    for i in range(8):
        l = f"green {LEFT_B[i]} light"
        r = f"red {RIGHT_B[i]} rain"
        add("s4_three_word_copy",
            (f"Read the three-word phrase '{l}' on TRAIN card {i}. Repeat all three words.", l),
            (f"Read the three-word phrase '{r}' on TRAIN card {i}. Repeat all three words.", r))
    for i in range(8):
        owner = f"NaraTrain{i:03d}"
        other = f"TebaTrain{i:03d}"
        add("s4_two_word_binding",
            (f"At the school, {owner} has the red folder and {other} has the blue folder. Give {owner}'s object using exactly two words.", "red folder"),
            (f"At the school, {owner} has the blue folder and {other} has the red folder. Give {owner}'s object using exactly two words.", "blue folder"))
    for q1, a1, q2, a2 in SENTENCE_PAIRS:
        add("s4_short_sentence", (q1 + " Respond in one short sentence.", a1),
            (q2 + " Respond in one short sentence.", a2))
    assert len(raw) == 56
    out = []
    for i, (domain, a, b) in enumerate(raw, start=88):
        l, r = tokenize_one(tok, *a), tokenize_one(tok, *b)
        if l["first_expected_id"] == r["first_expected_id"]:
            raise AssertionError(f"No opposing first word on TRAIN pair {i}")
        out.append({"split": "TRAIN", "pair_id": i, "domain": domain, "left": l, "right": r})
    return out

def weighted_pair_loss(model, pair):
    selected = []
    expected = []
    losses = []
    late = ("two_word" in pair["domain"] or "three_word" in pair["domain"] or "sentence" in pair["domain"])
    for side in ("left", "right"):
        sample = pair[side]
        ids = sample["ids"][None]
        pos = sample["positions"]
        target = sample["targets"]
        logits = model(input_ids=ids, use_cache=False).logits[0][pos]
        ce = F.cross_entropy(logits, target, reduction="none")
        weights = torch.ones_like(ce)
        if late and len(weights) > 2:
            weights[1:-1] = WEIGHT_LATE
            weights[-1] = 2.0
        losses.append(torch.sum(ce * weights) / torch.sum(weights))
        selected.append(logits[0])
        expected.append(int(target[0]))
    if expected[0] == expected[1]:
        raise AssertionError("Opposing TRAIN first tokens collapsed into same label")
    gap1 = selected[0][expected[0]] - selected[0][expected[1]]
    gap2 = selected[1][expected[1]] - selected[1][expected[0]]
    margin = (F.relu(1.0 - gap1) + F.relu(1.0 - gap2)) / 2.0
    return (losses[0] + losses[1]) / 2.0 + 0.15 * margin

def training_probe(model, sources, pairs):
    predictions = probe_greedy_train(model, sources, pairs, max_new_tokens=14)
    domains = defaultdict(lambda: {"correct": 0, "total": 0})
    rows = []
    for p in pairs:
        for side in ("left", "right"):
            found = predictions[p["pair_id"]]["sides"][side]
            domain = domains[p["domain"]]
            good = bool(found["exact_with_EOS"])
            domain["correct"] += int(good)
            domain["total"] += 1
            rows.append({"pair_id": p["pair_id"], "domain": p["domain"],
                         "side": side, "gold": p[side]["answer"],
                         "predicted_ids": found["generated_tokens"],
                         "gold_ids": found["gold_tokens_with_EOS"],
                         "correct_full_EOS": good})
    return {"domains": dict(domains), "correct": sum(x["correct"] for x in domains.values()),
            "total": len(rows), "rows": rows}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    deterministic_cpu()
    raw = a.source.read_bytes()
    if len(raw) != START_BYTES or hashlib.sha256(raw).hexdigest() != START_SHA:
        raise AssertionError("S4 only starts from exact precommitted S3 22M archive")
    cb = CB59(raw)
    assert cb.meta["original_unique_parameter_count"] == PARAMS
    cfg = build_config(next(s for s in CANDIDATES if s[0] == "small_320x6"))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    old = build_pairs(tok, "TRAIN")
    s3 = make_new_pairs(tok)
    s4 = new_pairs(tok)
    assert [len(old), len(s3), len(s4)] == [24, 64, 56]
    original = old + s3
    old_color_numeric = [q for q in s3 if q["domain"] not in ("s3_two_word_training", "s3_sentence_training")]
    old_multi = [q for q in s3 if q["domain"] in ("s3_two_word_training", "s3_sentence_training")]
    new_copy = [q for q in s4 if q["domain"] in ("s4_two_word_copy", "s4_three_word_copy")]
    new_other = [q for q in s4 if q["domain"] not in ("s4_two_word_copy", "s4_three_word_copy")]
    pools = (old, old_color_numeric, old_multi, new_copy, new_other)
    seen = [0] * len(pools)
    model, sources = create_small_paid_model(cb, cfg)
    optimizer = torch.optim.Adam([w.centroids for w in sources.values()], lr=LR)
    keys = sorted(sources)
    train_log = []
    print(json.dumps({"event": "S4_PRECOMMITTED_START", "steps": STEPS, "input_sha": START_SHA,
                      "n_s3_pairs": len(original), "n_s4_pairs": len(s4), "paid_ops": len(keys),
                      "sampling": "20% old, 10% S3 basic, 20% S3 multi, 40% S4 copy, 10% S4 other",
                      "late_token_CE_weight": WEIGHT_LATE}), flush=True)
    for step in range(STEPS):
        j = step % 10
        pool_idx = 0 if j < 2 else 1 if j == 2 else 2 if j < 5 else 3 if j < 9 else 4
        pool = pools[pool_idx]
        pair = pool[seen[pool_idx] % len(pool)]
        seen[pool_idx] += 1
        key = keys[step % len(keys)]
        source = sources[key]
        captured = []
        def hook(_module, _inputs, mat):
            mat.retain_grad()
            captured.append(mat)
        h = source.register_forward_hook(hook)
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        try:
            loss = weighted_pair_loss(model, pair)
            before = float(loss.detach())
            loss.backward()
        finally:
            h.remove()
        if not captured or captured[-1].grad is None:
            raise AssertionError("S4 no gradient through targeted paid operator")
        gradient = captured[-1].grad.detach().cpu().numpy().copy()
        if sum(w.centroids.grad is not None for w in sources.values()) != 43:
            raise AssertionError("S4 absent all-layer codebook gradients")
        norm = float(torch.nn.utils.clip_grad_norm_(
            [w.centroids for w in sources.values()], 1., error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for w in sources.values():
                w.centroids.clamp_(-16, 16)
        clear(sources)
        changed = 0
        if (step + 1) % 16 == 0:
            with torch.inference_mode():
                baseline = float(weighted_pair_loss(model, pair))
                matrix = source().detach().cpu().numpy().copy()
            proposal = propose_codes(source, gradient, matrix)
            if proposal["proposed_groups"] > MAX_SWAPS:
                raise AssertionError("Unexpected change of 5-bit index budget")
            if proposal["proposed_groups"]:
                with torch.no_grad():
                    apply_codes(source, proposal, True)
                clear(sources)
                with torch.inference_mode():
                    improved = float(weighted_pair_loss(model, pair))
                if np.isfinite(improved) and improved + 1e-7 < baseline:
                    changed = proposal["proposed_groups"]
                else:
                    with torch.no_grad():
                        apply_codes(source, proposal, False)
        clear(sources)
        train_log.append({"step": step+1, "pair_id": pair["pair_id"],
                          "domain": pair["domain"], "loss": before,
                          "grad": norm, "accepted_codes": changed})
        if (step + 1) % 256 == 0:
            print(json.dumps({"event": "S4_TRAIN_PROGRESS", "steps": step+1,
                              "latest_domain": pair["domain"], "latest_loss": before,
                              "accepted_swaps_so_far": sum(z["accepted_codes"] for z in train_log)}),
                  flush=True)
    packed, info = replace_discrete(cb, sources, strict=False)
    path = a.output / "CALIBRI2_S4_22M_2048steps_paid.cb59"
    path.write_bytes(packed)
    sha = hashlib.sha256(packed).hexdigest()
    if len(packed) != START_BYTES or info["complete_bpp"] > 0.700:
        raise AssertionError("S4 invalid physical budget or forbidden sidecar")
    reloaded, reloaded_sources = create_small_paid_model(CB59(packed), cfg)
    sample = old[:4] + [p for p in s3 if "two_word" in p["domain"]][:4] + s4[:8]
    in_memory = training_probe(model, sources, sample)
    from_disk = training_probe(reloaded, reloaded_sources, sample)
    if [x["predicted_ids"] for x in in_memory["rows"]] != [x["predicted_ids"] for x in from_disk["rows"]]:
        raise AssertionError("S4 paid archive changed predictions upon native reload")
    save(a.output / "ACTUAL_CALIBRI2_S4_TRAIN.json", {
        "experiment": "S4 22M last-mile multi-token balanced TRAIN only",
        "source_s3_sha256": START_SHA, "result_sha256": sha,
        "physical_bytes": len(packed), "physical_bpp": info["complete_bpp"],
        "unique_parameters": PARAMS, "steps": STEPS, "torch_threads": 1,
        "deterministic_cpu_algorithms": True, "all_43_operators_trained": True,
        "zero_FP32_or_public_QA_trainer_inputs": True,
        "no_checkpoint_selection_on_heldout": True,
        "old_new_pool_counts": seen, "final_train_only_probe": from_disk,
        "training_history": train_log})
    print(json.dumps({"event": "S4_FINISHED_SEALED",
                      "sha256": sha, "bytes": len(packed), "bpp": info["complete_bpp"],
                      "sample_train_exact": from_disk["correct"],
                      "sample_train_total": from_disk["total"],
                      "accepted_code_indices": sum(z["accepted_codes"] for z in train_log)}), flush=True)

if __name__ == "__main__":
    main()
