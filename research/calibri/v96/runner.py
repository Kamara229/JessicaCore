"""Exact same fresh canary prompts on full original FP32 and archived CB59.
Full greedy inference with real KV-cache, no changing archived weights.
"""
from __future__ import annotations
import json
from v94.generation import generate_one
from .spec import CANARIES,OPEN_ENDED
from .grade import grade_answer,string_breakdown

def run_canaries(model,tok,model_name):
    output=[]
    for category,prompt,valid in CANARIES:
        item=generate_one(model,tok,prompt)
        output.append({"model":model_name,"category":category,
            "prompt_SHA":item["prompt_SHA"],
            "actual_generated_text":item["text"],
            "generated_token_ids":item["ids"],
            "correctness_reference_grading":grade_answer(item["text"],valid),
            "graded":grade_answer(item["text"],valid),
            "lexical_quality_observations":string_breakdown(item["text"],item["ids"]),
            "duration_seconds":item["elapsed_seconds"]})
    return output

def run_open_ended(model,tok,model_name):
    output=[]
    for category,prompt in OPEN_ENDED:
        item=generate_one(model,tok,prompt)
        output.append({"model":model_name,"category":category,
            "prompt_SHA":item["prompt_SHA"],
            "actual_generated_text":item["text"],
            "generated_token_ids":item["ids"],
            "lexical_quality_observations":string_breakdown(item["text"],item["ids"]),
            "human_factual_coherence_review":"NOT_YET_PERFORMED",
            "duration_seconds":item["elapsed_seconds"]})
    return output

def quality_stats(canaries,open_ended):
    rows=canaries+open_ended
    return {"correct_short_answers":sum(int(x["graded"]["correct"]) for x in canaries),
        "short_answer_questions":len(canaries),
        "nonliteral_open_ended_prompts":len(open_ended),
        "repeat_over_75pct":sum(int(x["lexical_quality_observations"]["repeat_over_75pct"]) for x in rows),
        "mean_distinct_token_ratio":sum(x["lexical_quality_observations"]["unique_token_fraction"] for x in rows)/len(rows),
        "full_model_CPU_total_elapsed_seconds":sum(x["duration_seconds"] for x in rows),
        "total_generated_tokens":sum(len(x["generated_token_ids"]) for x in rows),
        "human_semantic_quality_evaluated":False}
