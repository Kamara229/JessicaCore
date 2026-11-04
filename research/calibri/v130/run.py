"""CALIBRI v130 diagnostic facade: no fitting, oracle, promotion, or Jessica tests.

Real SmolLM2 teacher versus the same pinned CB59 and v119 paid host. A
fixed 8.23MB Green is checked both on its prompt and on a wrong paired prompt.
Only a diagnostic set is used; the established independent TEST is untouched.
"""
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import argparse
import gc
import hashlib
import json

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v94.generation import generate_one
from v108.generation import build_chat
from v117.protocol import TOTAL_UNIQUE
from v128.protocol import read_v119
from v129.operators import installed
from v129.run import load_green
from .capture import observations, read_snapshot, compare
from .metrics import grade, likelihood
from .protocol import cases, shuffled_questions


@torch.inference_mode()
def evaluate(model, tok, rows, *, bridge=None, policy=None, shuffle=None):
    result, states = [], {}
    with observations(model) as latest:
        for row in rows:
            question = row["question"]
            if bridge is not None:
                controller_prompt = shuffle[row["id"]] if shuffle is not None else question
                bridge.control(policy, controller_prompt)
            prompt = build_chat(tok, question)
            ids = tok.encode(prompt, add_special_tokens=False)
            if not 1 < len(ids) < 160:
                raise ValueError("diagnostic prompt unexpectedly too long")
            model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
            states[row["id"]] = read_snapshot(latest)
            gold = likelihood(model, tok, ids, row["gold"])
            foil = likelihood(model, tok, ids, row["foil"])
            generated = generate_one(model, tok, prompt, stop_at_eos=True)
            report = grade(generated["text"], row["gold"], generated["eos_generated"])
            result.append({
                **row,
                "first_word_matches": report["first_word_matches"],
                "exact_text": report["exact_text"],
                "exact_with_eos": report["exact_text_with_eos"],
                "generated_text": generated["text"][:500],
                "generated_ids": generated["ids"],
                "gold_lexical_logp": gold["best_logp"],
                "foil_lexical_logp": foil["best_logp"],
                "lexical_margin": gold["best_logp"] - foil["best_logp"],
                "prefers_gold_under_teacher_forcing": gold["best_logp"] > foil["best_logp"],
                "gold_token_counts": gold["token_counts"],
                "foil_token_counts": foil["token_counts"],
                "eos_generated": generated["eos_generated"],
            })
    return result, states


def summarize(items):
    index = {r["id"]: r for r in items}
    if len(index) != len(items):
        raise ValueError("duplicate diagnostic cases")
    joint, prefix_joint = 0, 0
    for item in items:
        if item["orientation"] != "original":
            continue
        other = index[item["pair"] + ":" + item["focus"] + ":swapped"]
        joint += int(item["prefers_gold_under_teacher_forcing"] and
                     other["prefers_gold_under_teacher_forcing"])
        prefix_joint += int(item["first_word_matches"] and other["first_word_matches"])
    return {
        "questions": len(items),
        "first_word_matches": sum(r["first_word_matches"] for r in items),
        "exact_text": sum(r["exact_text"] for r in items),
        "exact_with_eos": sum(r["exact_with_eos"] for r in items),
        "positive_lexical_margin": sum(r["prefers_gold_under_teacher_forcing"] for r in items),
        "both_swapped_halves_lexical": joint,
        "both_swapped_halves_first_word": prefix_joint,
        "num_swap_tests": len(items)//2,
        "average_gold_minus_foil_logp": sum(r["lexical_margin"] for r in items)/len(items),
        "all_text_requires_human_semantic_review": True,
    }


def run(host: Path, green: Path, output: Path):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    fixed = read_v119(host)       # SHA covers the COMPLETE 11.73MB file
    policy = load_green(green)    # SHA covers the frozen 8.23MB file
    rows = cases()
    shuffle = shuffled_questions(rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)

    teacher = original_teacher()
    teacher.eval()
    teacher_rows, teacher_states = evaluate(teacher, tokenizer, rows)
    del teacher
    gc.collect()

    reports = {"FP32": {"summary": summarize(teacher_rows), "rows": teacher_rows}}
    with TemporaryDirectory(dir=output) as folder:
        temp = Path(folder)
        original = temp / "original.cb59"
        original.write_bytes(fixed.base)
        model, metadata, _ = make_model("baseline", SimpleNamespace(
            original=original, folder=temp))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("packed core accidentally converted to dense weights")
        model.eval()
        for mode in ("CB59", "v119_neutral", "v119_green", "v119_green_shuffled"):
            ctx = (nullcontext(None) if mode == "CB59"
                   else installed(model, fixed, False))
            with ctx as bridge:
                policy_for_mode = (policy if mode in ("v119_green", "v119_green_shuffled")
                                   else None)
                generated, states = evaluate(
                    model, tokenizer, rows, bridge=bridge, policy=policy_for_mode,
                    shuffle=shuffle if mode == "v119_green_shuffled" else None)
            reports[mode] = {
                "summary": summarize(generated), "rows": generated,
                "activation_diagnostics_vs_FP32": compare(teacher_states, states, rows),
            }

    report = {
        "version": "v130", "purpose": "prompt-grounded information retention under swapped relations",
        "model": MODEL, "model_revision": MODEL_SHA,
        "source_v119_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
        "source_bpp": 8 * len(host.read_bytes()) / TOTAL_UNIQUE,
        "frozen_green_sha256": hashlib.sha256(green.read_bytes()).hexdigest(),
        "green_file_bytes": green.stat().st_size,
        "no_new_weights_training_or_archives": True,
        "no_official_TEST_used": True,
        "not_a_blind_benchmark": True,
        "student_eager_attention_matches_FP32": True,
        "scoring_caveat": "lexical teacher forcing is NOT free-answer factual correctness",
        "controller_permutation": "fixed 7-case derangement; no labels in Green input",
        "modes": reports,
        "not_promoted_into_main_Render_Supabase_APK_or_LIVE_Jessica": True,
    }
    path = output / "v130_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "v130_paired_information_probe", "summary": {
        k: v["summary"] for k, v in reports.items()},
        "output": str(path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", type=Path)
    parser.add_argument("green", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.host, args.green, args.out)
