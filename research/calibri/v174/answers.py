"""Free generation with EOS/cap evidence; lexical clues are not semantic passes."""
import hashlib
import time
from collections import Counter
import torch
from v169.grade import detect
from .protocol import MAX_NEW_TOKENS


@torch.inference_mode()
def generate(model, tokenizer, question):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    if question not in prompt or tokenizer.eos_token_id is None:
        raise ValueError("Question/template/EOS invalid")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not 1 < len(ids) < 256:
        raise ValueError("Unexpected prompt length")
    x = torch.tensor([ids], dtype=torch.long)
    emitted, cache = [], None
    start = time.perf_counter()
    for step in range(MAX_NEW_TOKENS):
        output = model(input_ids=x, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite generation logits")
        x = logits.argmax(dim=-1, keepdim=True)
        emitted.append(int(x.item()))
        if step == 0:
            prefill = time.perf_counter() - start
        if emitted[-1] == tokenizer.eos_token_id:
            break
    elapsed = time.perf_counter() - start
    ended = emitted[-1] == tokenizer.eos_token_id
    return {
        "verbatim_text": tokenizer.decode(emitted, skip_special_tokens=True),
        "token_ids": emitted, "eos": ended,
        "cap_hit_without_eos": not ended and len(emitted) == MAX_NEW_TOKENS,
        "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(),
        "diagnostic_dense_wall_seconds": elapsed, "prefill_seconds": prefill,
        "distinct_token_fraction": len(set(emitted)) / len(emitted),
    }


def annotate(case, answer):
    status, extracted = detect({**case, **answer})
    # Reuse the conservative prior extractor; gold occurrence stays ungraded.
    correct = (str(extracted).casefold() == str(case["gold"]).casefold()
               if status == "parsed" else None)
    return {**answer, "question_id": case["question_id"], "domain": case["domain"],
            "question": case["question"], "gold_post_generation": case["gold"],
            "kind": case["kind"], "extraction_status": status,
            "extracted_answer": extracted, "machine_correct": correct,
            "semantic_review_required": correct is None}


def summary(rows):
    parsed = [r for r in rows if r["machine_correct"] is not None]
    return {
        "questions": len(rows), "machine_parsed": len(parsed),
        "machine_correct": sum(r["machine_correct"] for r in parsed),
        "ungraded": len(rows) - len(parsed),
        "lexical_gold_only": sum(r["extraction_status"] == "candidate_gold_present" for r in rows),
        "cap_hit_without_eos": sum(r["cap_hit_without_eos"] for r in rows),
        "mean_token_distinctness": sum(r["distinct_token_fraction"] for r in rows) / len(rows),
        "extraction_status": dict(Counter(r["extraction_status"] for r in rows)),
    }
