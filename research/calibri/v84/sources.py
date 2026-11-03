"""Frozen real-model TRAIN/VALID capture, separated from TEST and optimizer."""
from __future__ import annotations
import hashlib
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM
from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import get_tokens, WINDOW

TARGETS = (
    "model.layers.11.mlp.down_proj",
    "model.layers.28.mlp.down_proj",
    "model.layers.29.mlp.down_proj",
)
DOMAIN_WINDOWS = {
    "wiki_fit": 2,
    "news_fit": 2,
    "wiki_valid": 1,
    "news_valid": 1,
}


def original_teacher():
    return AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_SHA, torch_dtype=torch.float32,
        attn_implementation="eager").eval()


def capture_original(teacher, target_paths=TARGETS):
    """Replay genuine original teacher activations; no CB59 training leaks."""
    if len(set(target_paths)) != len(target_paths):
        raise AssertionError("duplicate original operator targets")
    samples = {}
    for domain, count in DOMAIN_WINDOWS.items():
        if not (domain.endswith("_fit") or domain.endswith("_valid")):
            raise AssertionError("unseen TEST data must not be used here")
        tokens, provenance = get_tokens(domain)
        if provenance.get("split") != "train":
            raise AssertionError("expected original TRAIN split for fit/valid")
        windows = []
        for k in range(count):
            x = tokens[k * WINDOW:(k + 1) * WINDOW].clone()
            windows.append({"tokens": x, "sha256": hashlib.sha256(
                x.numpy().tobytes()).hexdigest()})
        out = {"source": provenance, "windows": windows, "probes": {
            name: [] for name in target_paths}}
        listeners = []
        try:
            for name in target_paths:
                module = teacher.get_submodule(name)
                if not isinstance(module, torch.nn.Linear):
                    raise AssertionError("teacher linear missing: " + name)

                def hook(_layer, inputs, result, key=name):
                    if len(inputs) != 1:
                        raise AssertionError("unexpected teacher module inputs")
                    out["probes"][key].append((
                        inputs[0].detach().clone().reshape(-1, inputs[0].shape[-1]),
                        result.detach().clone().reshape(-1, result.shape[-1])))

                listeners.append(module.register_forward_hook(hook))
            with torch.inference_mode():
                for sample in windows:
                    a = teacher(input_ids=sample["tokens"][None],use_cache=False)
                    sample["teacher_logits"] = a.logits[0].detach().clone()
            for key in target_paths:
                if len(out["probes"][key]) != count:
                    raise AssertionError("missed teacher projection/window")
        finally:
            for handle in listeners:
                handle.remove()
        samples[domain] = out
        print({"stage": "v84_teacher_source_original", "domain": domain,
               "windows": count, "targets": len(target_paths)}, flush=True)
    return samples


def paired_inputs(samples, domains: tuple[str, ...], path: str):
    all_x, all_y = [], []
    for domain in domains:
        rows = samples[domain]["probes"][path]
        for x,y in rows:
            all_x.append(x)
            all_y.append(y)
    if not all_x:
        raise AssertionError("no paired teacher projector data")
    return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)
