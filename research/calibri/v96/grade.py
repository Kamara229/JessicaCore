"""Deterministic exact-answer grading plus observable repetition flags.

No teacher probability loss, no randomness-as-quality proxy. Open-ended
response quality stays UNVERIFIED until judged by an independent rubric.
"""
from __future__ import annotations
import re

def normalized_front(text):
    s=(text or "").strip().replace("\ufffd","�")
    s=re.sub(r"^(?:answer\s*:\s*|the answer is\s+|it is\s+)","",s,flags=re.I)
    return s.strip(" \t\r\n:;,.!?*#\"'([{<").lower()

def grade_answer(response,valid):
    out=normalized_front(response)
    # Require answer in the first words, never count a random later substring.
    correct=False
    matched=None
    for term in valid:
        key=term.lower()
        if re.match(r"^"+re.escape(key)+r"(?=$|[^a-z0-9])",out):
            correct=True;matched=term;break
    return {"correct":correct,"matching_reference":matched,
            "answer_reference":list(valid),"answer_first_80_chars":(response or "")[:80]}

def string_breakdown(text,tokens):
    low=(text or "").lower()
    alnum=[ch for ch in low if ch.isalnum()]
    words=re.findall(r"[a-z]+",low)
    repeated_trigrams=0
    if len(words)>=3:
        tri=[tuple(words[i:i+3]) for i in range(len(words)-2)]
        repeated_trigrams=len(tri)-len(set(tri))
    most=max((tokens.count(x) for x in set(tokens)),default=0)/max(1,len(tokens))
    return {"replacement_chars":low.count("�"),
            "visible_alpha_numeric_fraction":len(alnum)/max(1,len(low)),
            "word_trigram_repetition_count":repeated_trigrams,
            "most_common_token_share":most,
            "unique_token_fraction":len(set(tokens))/max(1,len(tokens)),
            "repeat_over_75pct":most>.75,
            "only_one_distinct_token":len(set(tokens))<=1,
            "requires_independent_semantic_review":True}

def paired_behavior(before,after,original_fp32):
    if not (len(before)==len(after)==len(original_fp32)):
        raise AssertionError("different canary sets")
    for a,b,c in zip(before,after,original_fp32):
        if not a["prompt_SHA"]==b["prompt_SHA"]==c["prompt_SHA"]:
            raise AssertionError("paired prompts differ")
    sums={"CB59":sum(row["graded"]["correct"] for row in before),
          "inplace_rewrite":sum(row["graded"]["correct"] for row in after),
          "FP32":sum(row["graded"]["correct"] for row in original_fp32)}
    return {"ground_truth_correct_counts":sums,"questions":len(before),
            "rewrite_correct_answer_gain":sums["inplace_rewrite"]-sums["CB59"],
            "original_FP32_answer_quality_preserved":sums["inplace_rewrite"]>=sums["FP32"],
            "no_claim_that_unique_tokens_mean_coherent_text":True}
