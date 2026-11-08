"""CALIBRI 2.0 S10: READ-ONLY TRAIN-gradient conflict across 85 paid codebooks.

Hypothesis: simultaneous learning of color binding, Yes/No numeric predicates,
phrase copying and next-token language causes destructive interference in
the shared paid K32xV8 dictionaries. The test does NOT infer a cause from S9's
frozen QA. It computes task gradients on TRAIN only at immutable S7 and S8.

Definitions: for fixed representative TRAIN examples compute the arithmetic
mean of true paid-codebook gradients separately for each objective. On each
snapshot, cosine < 0 means an infinitesimal gradient step on task B would
increase task A loss to first order, without changing weights. Dot products
of these local gradients are not proof of catastrophic forgetting or overall
generalization performance. No optimizer, no weight writes, no QA examples,
no validation/test or teacher FP32. Exactly the same inputs for both models.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v174.run import save
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import (
    SOURCE_SHA, SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC,
    PAID_OPERATORS, make_broad_train, freeze_language_corpus, weighted_pair_loss
)

SEALED_S8_SHA = "386054d9fe2abfa5f43df5f9ae7292629fade5a2e527e64b46325d5201d4c8f8"
SEED = 2085
TRAIN_EXAMPLES = 3
TRAIN_WINDOW = 48
TASKS = ("color_binding", "numeric_comparison", "phrase_copy", "real_text")
TASK_PAIRS = (
    ("color_binding", "numeric_comparison"),
    ("color_binding", "phrase_copy"),
    ("color_binding", "real_text"),
    ("numeric_comparison", "phrase_copy"),
    ("numeric_comparison", "real_text"),
    ("phrase_copy", "real_text"),
)


def check_snapshot(path: Path, expected_sha: str):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha or len(raw) != SOURCE_BYTES:
        raise AssertionError("Input CB59 snapshot SHA/physical byte count not sealed")
    archive = CB59(raw)
    if archive.meta["original_unique_parameter_count"] != SOURCE_PARAMS:
        raise AssertionError("Not the same 36M unique-parameter architecture")
    paid = [d for d in archive.meta["tensor_descriptors"].values()
            if d["dtype"] == "CB59_K32V8"]
    if len(paid) != PAID_OPERATORS:
        raise AssertionError("Not exactly 85 paid 5-bit matrix dictionaries")
    if 8 * len(raw) / SOURCE_PARAMS > .700:
        raise AssertionError("Physically paid file exceeded 0.7 bpp")
    return archive


def task_examples(tok):
    all_train = make_broad_train(tok)
    by_domain = {}
    for item in all_train:
        by_domain.setdefault(item["domain"], []).append(item)
    ids, corpus_evidence = freeze_language_corpus(tok)
    selected = {
        "color_binding": by_domain["train_binding"][:TRAIN_EXAMPLES],
        "numeric_comparison": by_domain["train_numeric"][:TRAIN_EXAMPLES],
        "phrase_copy": by_domain["train_copy_two"][:TRAIN_EXAMPLES],
        "real_text": [ids[i*TRAIN_WINDOW:(i+1)*TRAIN_WINDOW]
                      for i in range(TRAIN_EXAMPLES)],
    }
    if any(len(selected[k]) != TRAIN_EXAMPLES for k in TASKS):
        raise AssertionError("Not exactly three frozen TRAIN examples per task")
    prompts = {}
    for task, rows in selected.items():
        if task == "real_text":
            prompts[task] = hashlib.sha256(torch.cat(rows).numpy().tobytes()).hexdigest()
        else:
            pair_text = [(p["left"]["question"], p["left"]["answer"],
                          p["right"]["question"], p["right"]["answer"]) for p in rows]
            prompts[task] = hashlib.sha256(
                json.dumps(pair_text, sort_keys=True).encode()).hexdigest()
    return selected, prompts, corpus_evidence


def gradients_for_one_task(model, sources, sorted_keys, examples, language):
    gradients = {name: torch.zeros_like(sources[name].centroids.detach(),
                                        dtype=torch.float64).reshape(-1)
                 for name in sorted_keys}
    losses = []
    for example in examples:
        model.zero_grad(set_to_none=True)
        clear(sources)
        if language:
            logits = model(input_ids=example[None], use_cache=False).logits[0]
            loss = F.cross_entropy(logits[:-1], example[1:])
        else:
            loss = weighted_pair_loss(model, example)
        if not torch.isfinite(loss).all():
            raise AssertionError("TRAIN loss not finite")
        losses.append(float(loss.detach()))
        loss.backward()
        for name in sorted_keys:
            tensor = sources[name].centroids.grad
            if tensor is None:
                raise AssertionError(f"Paid TRAIN dictionary missing autograd: {name}")
            if not torch.isfinite(tensor).all():
                raise AssertionError("TRAIN gradient had NaN/Inf")
            gradients[name] += tensor.detach().double().reshape(-1)
        clear(sources)
    return {k: v / len(examples) for k, v in gradients.items()}, losses


def pairwise_evidence(gradients, keys):
    report = []
    for one, two in TASK_PAIRS:
        x, y = gradients[one], gradients[two]
        numerator = denominator1 = denominator2 = 0.
        shared_nonzero = conflict = 0
        total_opposite_energy = 0.
        operators = []
        for key in keys:
            gx, gy = x[key], y[key]
            dot = float(torch.dot(gx, gy))
            nx = float(torch.linalg.vector_norm(gx))
            ny = float(torch.linalg.vector_norm(gy))
            if nx > 1e-15 and ny > 1e-15:
                shared_nonzero += 1
                cos = dot / (nx * ny)
                if dot < 0:
                    conflict += 1
                    total_opposite_energy += -dot
            else:
                cos = None
            numerator += dot
            denominator1 += nx**2
            denominator2 += ny**2
            operators.append({"matrix":key,"dot":dot,
                              "cosine":cos,"norm_A":nx,"norm_B":ny,
                              "opposes_first_order":bool(dot < 0)})
        if denominator1 <= 0 or denominator2 <= 0:
            raise AssertionError(f"Task {one} or {two} has zero total gradient")
        global_cos = numerator / (denominator1**.5 * denominator2**.5)
        operators.sort(key=lambda row: row["dot"])
        report.append({
            "task_A":one,"task_B":two,"global_grad_dot":numerator,
            "global_cosine":global_cos,"first_order_conflict":numerator < 0,
            "shared_nonzero_paid_matrices":shared_nonzero,
            "opposing_paid_matrices":conflict,
            "paid_matrices_total":PAID_OPERATORS,
            "opposing_mat_fraction_of_nonzero":
               conflict / shared_nonzero if shared_nonzero else None,
            "sum_negative_dot_magnitudes":total_opposite_energy,
            "all_85_paid_matrix_gradients":operators
        })
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("s7_cb59",type=Path)
    p.add_argument("s8_cb59",type=Path)
    p.add_argument("results_dir",type=Path)
    args = p.parse_args()
    args.results_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    inputs, input_shas, corpus_source = task_examples(tok)
    cfg = build_config(MODEL_SPEC)
    saved = {}
    for label, path, sha in (
        ("S7_BEFORE_DIVERSE_TRAIN", args.s7_cb59, SOURCE_SHA),
        ("S8_AFTER_DIVERSE_TRAIN", args.s8_cb59, SEALED_S8_SHA),
    ):
        archive = check_snapshot(path, sha)
        model, sources = create_small_paid_model(archive,cfg)
        model.eval()
        keys = sorted(sources)
        results = {}
        all_grads = {}
        for name in TASKS:
            raw_grads, train_losses = gradients_for_one_task(
                model,sources,keys,inputs[name],name=="real_text"
            )
            all_grads[name] = raw_grads
            results[name] = {
               "training_loss_each_example":train_losses,
               "mean_training_loss":sum(train_losses)/len(train_losses),
               "total_paid_grad_L2":float(np.sqrt(sum(
                    float(torch.dot(v,v)) for v in raw_grads.values())))
            }
            print(json.dumps({"step":"S10_TRAIN_ONLY_GRADIENT",
                 "snapshot":label,"task":name,
                 "loss":results[name]["mean_training_loss"],
                 "total_grad_L2":results[name]["total_paid_grad_L2"]}),flush=True)
        saved[label] = {
          "physical_model_sha256":sha,
          "physical_file_bytes":len(path.read_bytes()),
          "physical_bpp":8*len(path.read_bytes())/SOURCE_PARAMS,
          "TRAIN_only_task_losses_and_gradient_norms":results,
          "paired_gradient_conflicts":pairwise_evidence(all_grads,keys),
        }
        print(json.dumps({"event":"S10_SNAPSHOT_DONE","label":label,
            "pairwise_global_cosines":{
              f"{x['task_A']}/{x['task_B']}":x["global_cosine"]
              for x in saved[label]["paired_gradient_conflicts"]}}),flush=True)
        del model, sources, all_grads
        gc.collect()
    result = {
      "stage":"CALIBRI2_S10_READONLY_TRAIN_GRADIENT_INTERFERENCE_DIAGNOSIS",
      "original_S7_sha256":SOURCE_SHA,
      "actual_S8_after_sha256":SEALED_S8_SHA,
      "unique_original_parameter_count":SOURCE_PARAMS,
      "paid_K32V8_matrices":PAID_OPERATORS,
      "representative_examples_per_task":TRAIN_EXAMPLES,
      "preselected_exact_TRAIN_input_hashes":input_shas,
      "external_corpus_used_for_gradient_evaluation":corpus_source,
      "snapshots":saved,
      "no_optimizer_no_model_changes":True,
      "no_QA_DEV_VALID_TEST_read":True,
      "no_pretrained_FP32_weights_or_teacher_used":True,
      "NOT_proof_of_causality_or_quality":True,
      "interpretation":
        "Negative global dot of TRAIN gradients implies antagonistic local "
        "first-order loss change under a gradient step, not proof of the "
        "observed heldout regression. Positive or zero dot means this "
        "sampling does not establish destructive interference.",
      "no_production_or_Jessica_changes":True,
    }
    save(args.results_dir/"ACTUAL_CALIBRI2_S10_S7_S8_TRAIN_GRADIENT_CONFLICT.json",result)
    print(json.dumps({"event":"S10_READONLY_TRAIN_GRADIENT_DIAGNOSTIC_DONE",
      "source_bpp":saved["S7_BEFORE_DIVERSE_TRAIN"]["physical_bpp"],
      "after_bpp":saved["S8_AFTER_DIVERSE_TRAIN"]["physical_bpp"],
      "task_global_cosines_before":{
        f"{z['task_A']}/{z['task_B']}":z["global_cosine"]
        for z in saved["S7_BEFORE_DIVERSE_TRAIN"]["paired_gradient_conflicts"]},
      "task_global_cosines_after":{
        f"{z['task_A']}/{z['task_B']}":z["global_cosine"]
        for z in saved["S8_AFTER_DIVERSE_TRAIN"]["paired_gradient_conflicts"]}
    }),flush=True)


if __name__=="__main__":
    main()
