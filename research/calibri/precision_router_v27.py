#!/usr/bin/env python3
"""CALIBRI v27: diagnose cross-host BF16 Switch variance using a FP32 compute control.

FP32 uses the *exact same originally loaded BF16 checkpoint weight values*,
converted to FP32 for arithmetic. This is NOT an independently trained FP32
teacher, a compression benchmark, PPL measurement, or full-logit-diff dump.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import platform
import time

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from moe_allocation_v24 import (
    MODEL, DATASET, EXPERTS, PINNED_SWITCH_SHA,
    token_data, load_new_corpus, make_windows,
    identify_next_router, evaluation,
)
from moe_replay_v25 import fingerprint, hash_bytes


CORPORA = ("WikiText2_validation", "TinyStories_validation")
MODES = ("bf16_mkldnn", "fp32_no_mkldnn")


def trace(model, windows, next_router, probes=24):
    """Return sampled token logits and exact downstream dispatch masks, never weights."""
    with torch.inference_mode():
        logits, masks = evaluation(model, windows, next_router)
    if len(logits) != len(masks) or not logits:
        raise AssertionError("router coverage differs from model outputs")
    vocab = int(logits[0].shape[-1])
    positions = torch.linspace(0, vocab - 1, steps=probes).round().long()
    packed = []
    top1 = []
    router = []
    norms = []
    for values, mask in zip(logits, masks):
        if values.shape[-1] != vocab or values.ndim != 3:
            raise AssertionError("unexpected output logit geometry")
        if mask.shape[-1] != 8:
            raise AssertionError("expected eight pretrained Switch experts")
        rows = values.reshape(-1, vocab)
        packed.extend(rows.index_select(-1, positions).float().tolist())
        top1.extend(rows.argmax(-1).tolist())
        norms.extend(rows.float().square().sum(-1).sqrt().tolist())
        router.extend((mask.reshape(-1, 8) > 0).int().tolist())
    if len(router) != len(top1) or len(packed) != len(top1):
        raise AssertionError("logit/token router alignment changed")
    return {
        "sampled_logit_positions": positions.tolist(),
        "sampled_token_logits": packed,
        "per_token_full_logit_l2": norms,
        "full_vocab_top1_token_id": top1,
        "exact_eight_expert_mask_per_token": router,
        "full_logit_sha256": hash_bytes(b"".join(
            tensor.contiguous().numpy().tobytes() for tensor in logits
        )),
        "router_mask_sha256": hash_bytes(b"".join(
            mask.contiguous().numpy().tobytes() for mask in masks
        )),
        "tokens_evaluated": len(top1),
        "vocab": vocab,
    }


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(27)
    torch.use_deterministic_algorithms(True)
    api = HfApi()
    revision = api.model_info(MODEL).sha
    if revision != PINNED_SWITCH_SHA:
        raise RuntimeError("pinned original Switch pretrained checkpoint changed")
    wiki_rev = api.dataset_info(DATASET).sha
    tiny_rev = api.dataset_info("roneneldan/TinyStories").sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=PINNED_SWITCH_SHA)
    model, loading = SwitchTransformersForConditionalGeneration.from_pretrained(
        MODEL, revision=PINNED_SWITCH_SHA,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    model.eval()
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise AssertionError("incomplete checkpoint loading: " + key)
    for _n, module in model.named_modules():
        if _n.endswith(".mlp.router"):
            module.jitter_noise = 0.0
    downstream = identify_next_router(model)
    if not downstream:
        raise AssertionError("pretrained next decoder router not found")
    wiki = token_data(tokenizer, wiki_rev, "validation", args.tokens)
    tiny, _ = load_new_corpus(tokenizer, args.tokens, tiny_rev)
    windows = {
        CORPORA[0]: list(make_windows(wiki, args.encoder_len, args.decoder_len, tokenizer.pad_token_id)),
        CORPORA[1]: list(make_windows(tiny, args.encoder_len, args.decoder_len, tokenizer.pad_token_id)),
    }
    if not all(windows.values()):
        raise AssertionError("evaluation has no valid real corpus windows")
    key_weights = {
        n: fingerprint(model.get_submodule(n).weight)
        for n in ("decoder.block.1.layer.2.mlp.router.classifier",
                  f"{EXPERTS}.expert_0.wi", f"{EXPERTS}.expert_0.wo")
    }
    report = {
        "experiment": "CALIBRI_v27_pretrained_Switch_BF16_vs_FP32_CPU_crosshost",
        "source_model": MODEL, "model_revision": revision,
        "dataset_revisions": {"WikiText2": wiki_rev, "TinyStories": tiny_rev},
        "source_token_sha256": {
            CORPORA[0]: hash_bytes(wiki.numpy().tobytes()),
            CORPORA[1]: hash_bytes(tiny.numpy().tobytes()),
        },
        "original_BF16_weight_sha256": key_weights,
        "torch_version": str(torch.__version__),
        "transformers_version": "4.46.3",
        "cpu_arch": platform.machine(),
        "cpu_brand": next((line.split(":", 1)[1].strip()
            for line in open("/proc/cpuinfo", encoding="utf-8")
            if line.lower().startswith("model name")), platform.processor()),
        "torch_cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "router_jitter_forced_zero": True,
        "checkpoint_loading_missing_unexpected_mismatched": [0, 0, 0],
        "modes": {},
        "limitation": "FP32 converted from BF16-loaded pretrained weights; only 24 fixed vocab probes plus full-vocab top1, norms and hashes; no full-logit absolute maximum; not a PPL/throughput claim",
    }

    try:
        torch.backends.mkldnn.enabled = True
        if next(model.parameters()).dtype != torch.bfloat16:
            raise AssertionError("BF16 control had wrong precision")
        report["modes"][MODES[0]] = {}
        for corpus, win in windows.items():
            t0 = time.monotonic()
            report["modes"][MODES[0]][corpus] = trace(model, win, downstream)
            report["modes"][MODES[0]][corpus]["elapsed_seconds_diagnostic"] = time.monotonic() - t0

        # Control keeps original BF16 information, changing *compute precision*.
        model.float()
        model.eval()
        torch.backends.mkldnn.enabled = False
        if next(model.parameters()).dtype != torch.float32:
            raise AssertionError("FP32 control had wrong precision")
        report["modes"][MODES[1]] = {}
        for corpus, win in windows.items():
            t0 = time.monotonic()
            report["modes"][MODES[1]][corpus] = trace(model, win, downstream)
            report["modes"][MODES[1]][corpus]["elapsed_seconds_diagnostic"] = time.monotonic() - t0
    finally:
        torch.backends.mkldnn.enabled = True

    # Distinguish within-host arithmetic mode changes from cross-host drift.
    mode_delta = {}
    for corpus in CORPORA:
        bf = report["modes"][MODES[0]][corpus]
        fp = report["modes"][MODES[1]][corpus]
        a = np.asarray(bf["sampled_token_logits"], dtype=np.float64)
        b = np.asarray(fp["sampled_token_logits"], dtype=np.float64)
        if a.shape != b.shape:
            raise AssertionError("precision modes generated incompatible logits")
        mode_delta[corpus] = {
            "probed_logits_max_abs_difference": float(np.max(np.abs(a - b))),
            "probed_logits_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "next_router_token_mask_exact_agreement": float(np.mean(
                np.all(np.asarray(bf["exact_eight_expert_mask_per_token"]) ==
                       np.asarray(fp["exact_eight_expert_mask_per_token"]), axis=-1))),
            "full_vocab_top1_agreement": float(np.mean(
                np.asarray(bf["full_vocab_top1_token_id"]) ==
                np.asarray(fp["full_vocab_top1_token_id"]))),
        }
    report["within_host_precision_control"] = mode_delta
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"experiment": report["experiment"],
                     "cpu": report["cpu_brand"], "mode_deltas": mode_delta},
                    ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/calibri_v27_precision.json")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--encoder-len", type=int, default=64)
    parser.add_argument("--decoder-len", type=int, default=32)
    run(parser.parse_args())
