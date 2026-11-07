"""New S4 frozen 160 objective + 16 free speech evaluations.
Read-only before and after a ONE-SHOT S4 TRAIN run, no QA checkpoint selection.
Compositional word combinations are held out of all S4 TRAIN pairs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import CANDIDATES, build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.fresh_heldout import generate
from calibri2.s3.diverse_task_replay import make_new_pairs
from calibri2.s4.last_mile import new_pairs, LEFT_A, LEFT_B, RIGHT_A, RIGHT_B, PARAMS, START_BYTES
from v186.pairs import build_pairs
from v174.run import save

OPEN = (
 "Explain why a pencil is useful.",
 "Explain how a train can carry people.",
 "How do birds use their wings?",
 "Describe one reason to read a book.",
 "What happens when ice becomes water?",
 "Write a simple sentence about a colorful garden.",
 "Explain what a bridge connects.",
 "Why does a person need a map?",
 "Write a sentence about a green forest.",
 "What can a light in a room help people do?",
 "Why is a lock used on a door?",
 "What happens when someone turns a key?",
 "Write one sentence about a quiet lake.",
 "Explain why a coat helps in winter.",
 "What does a bus do?",
 "Describe a bright window.",
)

def cases():
    out = []
    def add(domain, question, gold):
        out.append({"id": domain + f"_{len(out):03d}",
                    "domain": domain, "question": question, "gold": gold})
    for i in range(64):
        j = i % 32
        k = j % 8
        g = j // 8
        if i < 32:
            phrase = LEFT_A[k] + " " + LEFT_B[(g + k + 4) % 8]
        else:
            phrase = RIGHT_A[k] + " " + RIGHT_B[(g + k + 4) % 8]
        # All held-out pairs of words differ from TRAIN pairings.
        q = (f"In a fictional note for EvalReader{i:03d}, the two words are '{phrase}'. "
             "What are the two words? Write them exactly.")
        add("two_word_copy", q, phrase)
    for i in range(32):
        red = (i % 2 == 0)
        color = "red" if red else "blue"
        other = "blue" if red else "red"
        q = (f"At the theater, PeraEval{i:03d} is carrying a {color} folder and "
             f"BeraEval{i:03d} is carrying a {other} folder. "
             f"What two words describe PeraEval{i:03d}'s object?")
        add("two_word_binding", q, color + " folder")
    for i in range(32):
        j = i % 8
        phrase = ("green " + LEFT_B[j] + " water" if i % 2 == 0 else
                  "red " + RIGHT_B[j] + " stone")
        q = (f"A letter for SelaEval{i:03d} shows the three-word expression '{phrase}'. "
             "Repeat the complete expression exactly.")
        add("three_word_copy", q, phrase)
    for i in range(32):
        a = 9231 + i * 37
        b = a + 5 + i % 13
        yes = i % 2 == 0
        q = (f"Consider two numbers: {b if yes else a} and {a if yes else b}. "
             "Is the first number greater than the second? Answer Yes or No.")
        add("numeric_relation", q, "Yes" if yes else "No")
    assert len(out) == 160 and len(OPEN) == 16
    assert len(set(x["question"] for x in out)) == 160
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("physical", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--label", required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    raw = a.physical.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    cb = CB59(raw)
    if len(raw) != START_BYTES or cb.meta["original_unique_parameter_count"] != PARAMS or 8*len(raw)/PARAMS > .700:
        raise AssertionError("S4 QA requires full physical 22M <=.700 bpp archive")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    train = build_pairs(tok, "TRAIN") + make_new_pairs(tok) + new_pairs(tok)
    training_questions = {x[s]["question"] for x in train for s in ("left", "right")}
    objective = cases()
    assert not training_questions.intersection(x["question"] for x in objective)
    expected_gold = {x[s]["answer"] for x in new_pairs(tok) for s in ("left", "right")}
    observed_pairs = {x["gold"] for x in objective if x["domain"] == "two_word_copy"}
    assert not observed_pairs.intersection(expected_gold), "S4 QA copied exact S4 TRAIN phrases"
    manifest_sha = hashlib.sha256(json.dumps([objective, OPEN], sort_keys=True).encode()).hexdigest()
    cfg = build_config(next(z for z in CANDIDATES if z[0] == "small_320x6"))
    model, sources = create_small_paid_model(cb, cfg)
    model.eval()
    correct = Counter()
    total = Counter()
    first = Counter()
    initial_tokens = Counter()
    answers = []
    for i, x in enumerate(objective):
        gold_ids = tok.encode(x["gold"], add_special_tokens=False) + [tok.eos_token_id]
        if not 2 <= len(gold_ids) <= 14:
            raise AssertionError("Not a valid full generation target")
        response = generate(model, sources, tok, x["question"], 14)
        exact = response == gold_ids
        correct[x["domain"]] += int(exact)
        total[x["domain"]] += 1
        first[x["domain"]] += int(bool(response) and response[0] == gold_ids[0])
        if response: initial_tokens[response[0]] += 1
        answers.append({**x, "actual": tok.decode(response, skip_special_tokens=True),
                        "actual_ids": response, "gold_ids": gold_ids,
                        "strict_full_EOS_correct": exact})
        if (i+1) % 32 == 0:
            print(json.dumps({"event": "S4_FROZEN_160_PROGRESS",
                              "label": a.label, "completed": i+1,
                              "correct": sum(correct.values())}), flush=True)
    open_answers = []
    for i, question in enumerate(OPEN):
        response = generate(model, sources, tok, question, 24)
        open_answers.append({"id": i, "question": question,
                             "answer": tok.decode(response, skip_special_tokens=True),
                             "tokens": response})
    by = [{"domain": key, "correct": correct[key], "first": first[key],
           "total": total[key]} for key in sorted(total)]
    result = {
        "stage": "S4_INDEPENDENT_GENERALIZATION", "label": a.label, "model_sha": sha,
        "suite_sha": manifest_sha, "physical_bpp": 8*len(raw)/PARAMS,
        "overall_strict_full_EOS": sum(correct.values()),
        "denominator": len(objective), "by_domain": by,
        "first_token_diversity": len(initial_tokens),
        "actual_160_answers": answers, "open_16_actual_answers": open_answers,
        "not_FP32_comparison": True, "QA_never_used_for_training": True,
        "holdout_not_used_for_checkpoint_selection": True,
        "not_a_proof_of_freeform_language_understanding": True}
    save(a.output/"ACTUAL_CALIBRI2_S4_FRESH_GENERALIZATION.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in
                      ("actual_160_answers", "open_16_actual_answers")}), flush=True)

if __name__ == "__main__":
    main()
