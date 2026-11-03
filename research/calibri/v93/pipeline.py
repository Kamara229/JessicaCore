"""v93 functional-quality first: replay frozen v92 observations without KL/NLL.

The v92 candidate archive remains research-only, and v89 independent TEST
is neither opened nor used for training/selection.
"""
from __future__ import annotations
import json
from pathlib import Path
from .evidence import read,extract
from .answers import compare_answers
from .policy import assess

def run(report_path:Path,out:Path):
    data,origin=read(report_path)
    measurements=extract(data)
    answers=compare_answers(measurements["answer_rows"])
    decision=assess(qa=answers,generation=measurements["generation"],
        physical_bytes=measurements["archive_bytes"],
        native_speed_ratio=measurements["native_block_speed_ratio"],
        native_speed_is_full_model=measurements["native_speed_is_full_model"],
        generation_prompts=measurements["generation_prompts"])
    result={
        "experiment":"CALIBRI_v93_quality_first_reinterpret_v92_no_KL_NLL",
        "source_evidence":origin,
        "physically_complete_rewritten_source_SHA":measurements["new_source_SHA"],
        "actual_paired_correct_answer_quality":answers,
        "greedy_generation_diversity_and_repetition":measurements["generation"],
        "greedy_prompts":measurements["generation_prompts"],
        "source_bytes":measurements["archive_bytes"],
        "native_block_only_speed_ratio":measurements["native_block_speed_ratio"],
        "final_outcome_first_decision":decision,
        "sample_limitations":"12 paired BoolQ questions, 3 short greedy prompts, "
              "and only a native single-block microbenchmark: quality neither "
              "proved nor generalizable. Semantics, response accuracy over "
              "long answers, and full model speed still unmeasured.",
        "no_frozen_independent_TEST_access_or_new_weights":True,
        "no_Jessica_automatic_live_deploy_or_merge":True}
    out.mkdir(parents=True,exist_ok=True)
    dest=out/"v93_outcome_first_v92_candidate_review.json"
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v93_quality_first_comparison_completed",
        "baseline_correct":answers["original"]["correct"],
        "candidate_correct":answers["candidate"]["correct"],
        "questions":answers["original"]["count"],
        "baseline_diversity":measurements["generation"]["baseline"]["mean_distinct_ratio"],
        "candidate_diversity":measurements["generation"]["candidate"]["mean_distinct_ratio"],
        "status":decision["status"],"KL_or_NLL_not_measured_or_used":True,
        "working_Jessica_untouched":True},ensure_ascii=False),flush=True)
    return result
