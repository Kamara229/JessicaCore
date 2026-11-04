"""Replay the 27 historical v108 token streams, without rerunning or tuning the model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from v108.generation import assistant_turn, build_chat
from v108.spec import ALL_PROMPTS


def audit(path, tok):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "dfa3f3b20abbd827d8fdc494d8dfcf09545c6681d0f5c38b3a540adc2a82dd1d":
        raise ValueError("expected the original JSON from Actions 41750616156")
    report = json.loads(raw)
    rows = report["all_actual_readable_generated_answers_for_independent_review"]
    by_model = {}
    for row, (_, question, gold) in zip(rows, ALL_PROMPTS, strict=True):
        prompt_hash = hashlib.sha256(build_chat(tok, question).encode()).hexdigest()
        for name, sample in row["source_locked_models_actual_32_token_outputs"].items():
            ids = sample["actual_32_generated_token_ids"]
            if sample["actual_chat_prompt_SHA256"] != prompt_hash or len(ids) != 32:
                raise ValueError("historical prompt or fixed-work length mismatch")
            if tok.decode(ids, skip_special_tokens=True) != sample["actual_decoded_model_output"]:
                raise ValueError("historical token and text evidence disagree")
            turn = assistant_turn(tok, ids)
            by_model.setdefault(name, []).append({"question": question, "gold": gold,
                                                  "ids": ids, "turn": turn})
    counts = {}
    for name, samples in by_model.items():
        tokens = Counter(t for sample in samples for t in sample["ids"])
        counts[name] = {"dominant_id_and_count": tokens.most_common(1)[0],
                        "total_positions": tokens.total(),
                        "turns_ending_in_eos": sum(x["turn"]["ended_by_eos"] for x in samples),
                        "tokens_after_turns": sum(x["turn"]["tokens_after_eos"] for x in samples)}
    return {"counts": counts, "turns_for_review": by_model}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    print(json.dumps(audit(args.source, tok), ensure_ascii=False, indent=2))
