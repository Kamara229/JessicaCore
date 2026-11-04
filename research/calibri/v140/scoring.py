"""Transparent native end-to-end diagnostics. EOS fixes alone are not language."""
from statistics import mean
from v130.scoring import summarize


def summarize_native(arms):
    expected=("original_CB59","paid_CB140_6bit_tied")
    if tuple(arms)!=expected:
        raise ValueError("incorrect native AB control ordering")
    ids=[(r["pair"],r["side"],r["question"]) for r in arms[expected[0]]]
    if len(ids)!=16 or len(set(ids))!=16:
        raise ValueError("unmatched diagnostic corpus")
    by={}
    for name,rows in arms.items():
        if [(r["pair"],r["side"],r["question"]) for r in rows]!=ids:
            raise ValueError("matched prompt deviation")
        sec=sum(r["elapsed_seconds"] for r in rows)
        tokens=sum(r["generated_tokens"] for r in rows)
        by[name]={
            "first_teacher_KL_mean":mean(r["teacher_to_student_first_token_KL"] for r in rows),
            "first_FP32_argmax_agreements":sum(r["first_token_agrees_with_FP32"] for r in rows),
            "first_EOS":sum(r["eos_first"] for r in rows),
            "gold_first_token_rank_top8_PROXY":sum(r["gold_first_token_top8_PROXY"] for r in rows),
            "whole_reply_observations_NOT_semantic_accuracy":summarize(rows),
            "CPU_full_model_generation_seconds_total":sec,
            "actual_generated_tokens":tokens,
            "CPU_full_model_generated_tokens_per_sec":tokens/max(sec,1e-9),
        }
    return {"by_arm":by,
            "relative_CPU_tokens_per_second":by[expected[1]]["CPU_full_model_generated_tokens_per_sec"]/max(by[expected[0]]["CPU_full_model_generated_tokens_per_sec"],1e-9),
            "quality_limit":"One-word targets are proxies. No independent TEST, SciQ, BoolQ or PPL. Speed includes prefill and may depend on token count and EOS; compare with caution."}
