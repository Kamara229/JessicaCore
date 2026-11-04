"""Real greedy responses on identical original-model chat-template prefixes.

Record exact decoded generations and conservative exact-label prefix
presence only: lexical form is NOT an adequate semantic correctness judge.
"""
from __future__ import annotations
import re
from hashlib import sha256
from v94.generation import generate_one
from v96.grade import string_breakdown
from .qa import chat

def matched_generation(model,tok,message,truth):
    content=chat(tok,message)
    original=generate_one(model,tok,content)
    response=original["text"]
    cleaned=response.strip().lower()
    match=re.match(r"^(?:the\s+answer\s+is\s+)?([a-z]+|\d+)\b",cleaned)
    observed=match.group(1) if match else None
    expected=truth.lower()
    return {"actual_decoded_text":response,
        "raw_generation_ids":original["ids"],
        "correct_answer_expected":truth,
        "first_value_token":observed,
        "exact_label_prefix_matches_truth":observed==expected,
        "prompt_sha256":sha256(message.encode()).hexdigest(),
        "chat_template_rendered_prompt_sha256":original["prompt_SHA"],
        "visible_lexical":string_breakdown(response,original["ids"]),
        "actual_full_model_cpu_KV_cache_tokens_per_second":
            original["full_model_token_steps"]/max(original["elapsed_seconds"],1e-10),
        "semantic_correctness_requires_human_review":True}
