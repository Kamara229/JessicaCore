"""Complete FROZEN original trained model independent TEST, no training."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
import repro_qa_calibration_v78 as evaluation
from v84.sources import original_teacher, TARGETS
from v85.physical import CB85
from v85.native_ops import insert_native_adapters,restore_native_adapters

from .provenance import freeze_audit,activate_new_test_slices
from .stats import paired_agreement


def assess_real_model(kind,model,tokenizer,folder,physical):
    with torch.inference_mode():
        model(input_ids=torch.tensor([[1,2,3,4]]),use_cache=False)
        language=evaluation.metric_language(model,kind,tokenizer,folder)
        qa=evaluation.QA_teacher_forced_likelihood(model,kind,tokenizer)
        generation=evaluation.real_generation(model,kind,tokenizer)
    return {"source_actual_physical":physical,
            "real_full_model_language":language,
            "original_heldout_QA":qa,
            "real_KV_cached_greedy_generation":generation}


def report(audit,results):
    baseline,corrected=results["baseline"],results["adaptive"]
    qa={}
    for task in ("sciq","boolq"):
        x=baseline["original_heldout_QA"][task]
        y=corrected["original_heldout_QA"][task]
        if (x["source"]["revision"]!=y["source"]["revision"] or
            len(x["items"])!=len(y["items"]) or any(
                u["record_index"]!=v["record_index"] or
                u["gold"]!=v["gold"] or u["prompt_SHA"]!=v["prompt_SHA"]
                for u,v in zip(x["items"],y["items"]))):
            raise AssertionError("candidate compared on nonmatching original QA TEST")
        qa[task]={
            "official_source_split_and_range":x["source"],
            "original_teacher_correct":results["teacher"]["original_heldout_QA"][task]["correct_count"],
            "original_native_CB59_correct":x["correct_count"],
            "conditional_native_CB85_correct":y["correct_count"],
            "paired_statistics":paired_agreement(
                [z["correct"] for z in x["items"]],
                [z["correct"] for z in y["items"]])}
    lang={}
    for domain in ("wiki_test","news_test"):
        a=baseline["real_full_model_language"][domain]
        b=corrected["real_full_model_language"][domain]
        t=results["teacher"]["real_full_model_language"][domain]
        if not (a["source"]["input_SHA"]==b["source"]["input_SHA"]==
                t["source"]["input_SHA"]):
            raise AssertionError("candidate/teacher TEST original token streams differ")
        def mean(rows,key):
            return float(np.mean([x[key] for x in rows]))
        lang[domain]={
            "original_source":a["source"],
            "independent_scored_next_tokens":a["scored_next_tokens"],
            "original_CB59_KL":mean(a["data"],"teacher_student_KL"),
            "physical_CB85_KL":mean(b["data"],"teacher_student_KL"),
            "CB85_minus_CB59_KL":mean(b["data"],"teacher_student_KL")-
                                 mean(a["data"],"teacher_student_KL"),
            "original_CB59_NLL":mean(a["data"],"heldout_token_NLL"),
            "physical_CB85_NLL":mean(b["data"],"heldout_token_NLL"),
            "CB85_minus_CB59_NLL":mean(b["data"],"heldout_token_NLL")-
                                  mean(a["data"],"heldout_token_NLL"),
            "base_CPU_prefill_seconds":a["CPU_prefill_seconds"],
            "adaptive_CPU_prefill_seconds":b["CPU_prefill_seconds"],
        }
    generation={
        kind:{key:res["real_KV_cached_greedy_generation"][key]
              for key in ("genuine_tokens_after_prefill",
                          "tokens_per_second_one_cpu",
                          "mean_distinct_token_ratio",
                          "mean_top_token_share")}
        for kind,res in results.items()}
    checks={
        "SciQ_nonregress_vs_original_packed_CB59":
            qa["sciq"]["conditional_native_CB85_correct"]>=
            qa["sciq"]["original_native_CB59_correct"],
        "BoolQ_nonregress_vs_original_packed_CB59":
            qa["boolq"]["conditional_native_CB85_correct"]>=
            qa["boolq"]["original_native_CB59_correct"],
        "new_Wiki_TEST_full_vocab_KL_improves":
            lang["wiki_test"]["CB85_minus_CB59_KL"]<0,
        "new_News_TEST_full_vocab_KL_improves":
            lang["news_test"]["CB85_minus_CB59_KL"]<0,
        "new_Wiki_TEST_next_token_NLL_nonregress":
            lang["wiki_test"]["CB85_minus_CB59_NLL"]<=0,
        "new_News_TEST_next_token_NLL_nonregress":
            lang["news_test"]["CB85_minus_CB59_NLL"]<=0,
        "genuine_new_greedy_distinct_ratio_ge_0p25":
            generation["adaptive"]["mean_distinct_token_ratio"]>=0.25,
        "actual_model_physical_source_bpw_le_0p700":
            audit["new_candidate_architecture_source_complete"]["all_in_bpw"]<=0.700,
    }
    return {
        "experiment":"CALIBRI_v86_frozen_native_input_adaptive_independent_TEST",
        "freeze_before_TEST":audit,
        "original_real_trained_model":MODEL,
        "original_pinned_revision":MODEL_SHA,
        "QA":qa,
        "independent_new_language_TEST":lang,
        "true_KV_cached_greedy_generation":generation,
        "predeclared_research_checks":checks,
        "all_independent_TEST_checks_pass":all(checks.values()),
        "important_caveat":"Single 135M model; very small short-answer custom SciQ/BoolQ slices, "
          "truncated context, 8x64 tokens per language domain, only two greedy prompts; "
          "no confidence to claim general assistant intelligence. No testing "
          "of Jessica, no source training or TEST-guided source selection.",
        "live_Jessica_or_main_modified":False,
    }


def run(original_path:Path,adaptive_path:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    raw=original_path.read_bytes()
    candidate=adaptive_path.read_bytes()
    audit=freeze_audit(raw,candidate)
    (out_dir/"v86_frozen_source_and_new_test_preregistration.json").write_text(
        json.dumps(audit,indent=2)+"\n")
    activate_new_test_slices()
    model=original_teacher()
    native,info,_=make_model("baseline",SimpleNamespace(
        original=original_path,folder=out_dir))
    if info["zero_dense_pretrained_2D_hot_weights"]!=True:
        raise AssertionError("dense original weights leaked into native model")
    if info["complete_physical_SHA256"]!=audit["new_candidate_architecture_source_complete"][
             "source_SHA256"] and info["complete_physical_SHA256"]!=
                CB85(candidate).original.hash():
        raise AssertionError("whole original pretrained CB59 source differs")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    results={}
    results["teacher"]=assess_real_model("teacher",model,tokenizer,out_dir,{
        "real_original_pretrained_FP32":True})
    results["baseline"]=assess_real_model("baseline",native,tokenizer,out_dir,info)
    cb85=CB85(candidate)
    old=insert_native_adapters(native,cb85,TARGETS)
    try:
        results["adaptive"]=assess_real_model("adaptive",native,tokenizer,out_dir,
                                               cb85.physical_stats())
    finally:
        restore_native_adapters(old)
    data=report(audit,results)
    path=out_dir/"v86_TRUE_independent_pretrained_answer_KL_greedy_TEST.json"
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "stage":"v86_REAL_independent_frozen_adaptive_source_TEST_complete",
        "physical_bpw":audit["new_candidate_architecture_source_complete"]["all_in_bpw"],
        "qa_counts":{k:{name:v["original_heldout_QA"][k]["correct_count"]
                           for name,v in results.items()}
                     for k in ("sciq","boolq")},
        "TEST_KL_change":{k:v["CB85_minus_CB59_KL"] for k,v in
                          data["independent_new_language_TEST"].items()},
        "greedy_distinct":data["true_KV_cached_greedy_generation"],
        "research_checks_pass":data["all_independent_TEST_checks_pass"],
        "Jessica_untouched":True,
    },ensure_ascii=False),flush=True)
    return data
