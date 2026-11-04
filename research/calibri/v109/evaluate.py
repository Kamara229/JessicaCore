"""Validation facade for real first turns, labeled answers and matched fixed-work CPU timing."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v86.provenance import ORIGINAL_CB59_SHA
from v94.generation import generate_one
from v108.generation import build_chat
from .protocol import PROMPTS, QA_SPANS, TIMING
from .tasks import cases, score


def packed(name, path, out, expected_sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"unexpected {name} archive")
    folder = out / name
    folder.mkdir(exist_ok=True)
    model, _, _ = make_model("baseline", SimpleNamespace(original=path, folder=folder))
    return model


def assess(model, tok):
    answers = {}
    for suite, prompts in PROMPTS.items():
        turns = []
        for _, question, reference in prompts:
            result = generate_one(model, tok, build_chat(tok, question), stop_at_eos=True)
            turns.append({"question": question, "reference": reference,
                          "prompt_sha256": result["prompt_SHA"], "ids": result["ids"],
                          "text": result["text"], "eos": result["eos_generated"]})
        answers[suite] = turns
    return {"chat": answers,
            "forced_choice": {name: score(model, *cases(tok, name)) for name in QA_SPANS}}


def paired_speed(models, tok):
    # Identical fixed-length work including prefill. Warm up both models and
    # alternate AB/BA across repetitions on *each* individual prompt.
    order = tuple(models)
    prompts = [build_chat(tok, question) for question in TIMING]
    for name in order:
        generate_one(models[name], tok, prompts[0])
    runs = []
    for index, prompt in enumerate(prompts):
        for repeat in range(4):
            pair = {}
            for name in (order if (index + repeat) % 2 == 0 else order[::-1]):
                row = generate_one(models[name], tok, prompt)
                pair[name] = {"seconds": row["elapsed_seconds"], "tokens": len(row["ids"]),
                              "prefill_seconds": row["prefill_seconds"], "decode_seconds": row["decode_seconds"]}
            runs.append({"prompt_index": index, "repeat": repeat, "models": pair})
    rates = {name: sum(r["models"][name]["tokens"] for r in runs)
                    / sum(r["models"][name]["seconds"] for r in runs) for name in order}
    decode = {name: sum(r["models"][name]["tokens"] - 1 for r in runs)
                     / sum(r["models"][name]["decode_seconds"] for r in runs) for name in order}
    return {"tokens_per_second": rates, "decode_only_tokens_per_second": decode,
            "candidate_ratio": rates["v109"] / rates["CB59"],
            "runs": runs}


def evaluate(mode, original, out):
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    if tok.eos_token_id != 2:
        raise ValueError("unexpected end-of-turn ID for pinned SmolLM2")
    build = json.loads((out / "build.json").read_text())
    if mode == "FP32":
        checkpoint = Path(hf_hub_download(MODEL, "model.safetensors", revision=MODEL_SHA))
        with checkpoint.open("rb") as stream:
            source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if source_sha != "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c":
            raise ValueError("unexpected pretrained checkpoint")
        result = assess(original_teacher(), tok)
    else:
        paths = {"CB59": (original, ORIGINAL_CB59_SHA), "v109": (out / "v109.cb59", build["sha256"])}
        names = tuple(paths) if mode == "speed" else (mode,)
        models = {n: packed(n, paths[n][0], out, paths[n][1]) for n in names}
        source_sha = {n: paths[n][1] for n in names} if mode == "speed" else paths[mode][1]
        result = paired_speed(models, tok) if mode == "speed" else assess(models[mode], tok)
    result.update({"model_revision": MODEL_SHA, "mode": mode, "source_sha256": source_sha})
    (out / f"{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("FP32", "CB59", "v109", "speed"))
    parser.add_argument("original", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    evaluate(args.mode, args.original, args.out)
