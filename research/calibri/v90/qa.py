"""Matched public BoolQ TRAIN/VALID observations; last-logit constrained Yes/No."""
from __future__ import annotations
import hashlib
import json
import math
from datasets import load_dataset
from huggingface_hub import HfApi
import torch
from .provenance import BOOLQ_DATASET, SPLITS, check_slices

def build_cases(tok):
    check_slices()
    revision = HfApi().dataset_info(BOOLQ_DATASET).sha
    answer_ids = (tok.encode(" Yes", add_special_tokens=False),
                  tok.encode(" No", add_special_tokens=False))
    if any(len(ids) != 1 for ids in answer_ids):
        raise AssertionError("Yes/No scoring must use genuine one-token completions")
    cases, sources = {}, {}
    for split, (lo, hi) in SPLITS.items():
        # NEVER use BoolQ validation or TEST; TRAIN_VALID is a distinct part of train.
        rows = load_dataset(BOOLQ_DATASET, split=f"train[{lo}:{hi}]", revision=revision)
        items = []
        for i, row in enumerate(rows):
            snippet = tok.decode(tok.encode(str(row["passage"]), add_special_tokens=False)[:256],
                                 skip_special_tokens=True)
            prompt = f"Context: {snippet}\nQuestion: {row['question']}\nAnswer:"
            ids = tok.encode(prompt, add_special_tokens=False)
            if not 1 < len(ids) <= 600:
                raise AssertionError("unexpected BoolQ diagnostic prefix length")
            gold = 0 if bool(row["answer"]) else 1
            digest = hashlib.sha256(torch.tensor(ids, dtype=torch.int64).numpy().astype("<i8").tobytes()).hexdigest()
            items.append({"record_index": lo+i, "tokens": ids, "gold": gold, "prompt_SHA": digest})
        if len(items) != hi-lo:
            raise AssertionError("incomplete original TRAIN sample")
        cases[split] = items
        sources[split] = {"dataset": BOOLQ_DATASET, "revision": revision, "split": "train",
                          "record_range": [lo, hi], "count": len(items),
                          "input_fingerprint": hashlib.sha256(json.dumps(
                              [(x["record_index"], x["gold"], x["prompt_SHA"]) for x in items]
                          ).encode()).hexdigest()}
    return cases, sources, [i[0] for i in answer_ids]

@torch.inference_mode()
def score_cases(model, cases, answer_tokens):
    yes, no = answer_tokens
    all_rows = {}
    for split, items in cases.items():
        rows = []
        for item in items:
            ids = torch.tensor(item["tokens"], dtype=torch.long)[None, :]
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            margin = float((logits[yes]-logits[no]).item())
            if not math.isfinite(margin):
                raise AssertionError("nonfinite original model Boolean margin")
            pred = 0 if margin >= 0 else 1
            rows.append({"record_index": item["record_index"], "gold": item["gold"],
                         "prompt_SHA": item["prompt_SHA"], "pred": pred,
                         "correct": int(pred == item["gold"]),
                         "yes_no_logit_margin": margin,
                         "yes_conditional_probability": 1/(1+math.exp(-max(-80, min(80, margin)))),
                         "prediction_confidence": 1/(1+math.exp(-min(80, abs(margin))))})
        all_rows[split] = rows
    return all_rows
