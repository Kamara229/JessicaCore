"""One immutable original-pretrained 135M, *NEW* official independent TEST.

No source training, tuning, validation-based selection, or TEST-guided
adaptive changes. Original CB59 and v85 CB85 remain unmodified.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import TARGETS,original_teacher
from v85.physical import CB85
from v85.native_ops import insert_native_adapters,restore_native_adapters
from v88.physical import unpack, audit
from v88.tensor import TensorPackedHead
from v86.pipeline import assess_real_model
from .provenance import freeze_before_test,activate_unseen_test
from .stats import paired_agreement,paired_kl

def extract(results,physical):
    comparisons={}
    for task in ("sciq","boolq"):
        a=results["CB59"]["original_heldout_QA"][task]
        b=results["CB85"]["original_heldout_QA"][task]
        c=results["CB88"]["original_heldout_QA"][task]
        indices=[(row["record_index"],row["gold"],row["prompt_SHA"])
                 for row in a["items"]]
        for other in (b,c):
            if [(row["record_index"],row["gold"],row["prompt_SHA"])
                for row in other["items"]]!=indices:
                raise AssertionError("original unseen QA inputs changed between models")
        comparisons[task]={
            "original_source":a["source"],
            "teacher_correct":results["teacher"]["original_heldout_QA"][task]["correct_count"],
            "original_native_CB59_correct":a["correct_count"],
            "frozen_adaptive_CB85_correct":b["correct_count"],
            "frozen_4D_CB88_correct":c["correct_count"],
            "CB88_vs_CB59_McNemar_bootstrap":paired_agreement(
                [int(row["correct"]) for row in a["items"]],
                [int(row["correct"]) for row in c["items"]]),
            "CB88_vs_CB85_McNemar_bootstrap":paired_agreement(
                [int(row["correct"]) for row in b["items"]],
                [int(row["correct"]) for row in c["items"]])}
    language={}
    for domain in ("wiki_test","news_test"):
        teacher=results["teacher"]["real_full_model_language"][domain]
        raw={k:results[k]["real_full_model_language"][domain]
             for k in ("CB59","CB85","CB88")}
        if any(m["source"]["input_SHA"]!=teacher["source"]["input_SHA"]
               for m in raw.values()):
            raise AssertionError("new official language TEST differs between models")
        def v(model,key):
            return float(np.mean([w[key] for w in raw[model]["data"]]))
        language[domain]={
            "original_source":teacher["source"],
            "official_next_token_positions":raw["CB59"]["scored_next_tokens"],
            "CB59_KL":v("CB59","teacher_student_KL"),
            "CB85_KL":v("CB85","teacher_student_KL"),
            "CB88_KL":v("CB88","teacher_student_KL"),
            "CB59_NLL":v("CB59","heldout_token_NLL"),
            "CB85_NLL":v("CB85","heldout_token_NLL"),
            "CB88_NLL":v("CB88","heldout_token_NLL"),
            "CB88_minus_CB59_KL_paired_bootstrap":paired_kl(
                raw["CB59"]["data"],raw["CB88"]["data"],"teacher_student_KL"),
            "CB88_minus_CB85_KL_paired_bootstrap":paired_kl(
                raw["CB85"]["data"],raw["CB88"]["data"],"teacher_student_KL"),
            "CB88_minus_CB59_NLL_paired_bootstrap":paired_kl(
                raw["CB59"]["data"],raw["CB88"]["data"],"heldout_token_NLL"),
            "CB59_prefill_CPU_seconds":raw["CB59"]["CPU_prefill_seconds"],
            "CB88_prefill_CPU_seconds":raw["CB88"]["CPU_prefill_seconds"]}
    generation={k:r["real_KV_cached_greedy_generation"] for k,r in results.items()}
    checks={
        "source_fully_paid_le_0p700bpw":
            physical["all_in_bits_per_original_parameter"]<=.7,
        "SciQ_no_regression_vs_both_compressed_models":
            comparisons["sciq"]["frozen_4D_CB88_correct"]>=max(
                comparisons["sciq"]["original_native_CB59_correct"],
                comparisons["sciq"]["frozen_adaptive_CB85_correct"]),
        "BoolQ_no_regression_vs_both_compressed_models":
            comparisons["boolq"]["frozen_4D_CB88_correct"]>=max(
                comparisons["boolq"]["original_native_CB59_correct"],
                comparisons["boolq"]["frozen_adaptive_CB85_correct"]),
        "Wiki_both_KL_baselines_improve":
            language["wiki_test"]["CB88_KL"]<min(
                language["wiki_test"]["CB59_KL"],
                language["wiki_test"]["CB85_KL"]),
        "News_both_KL_baselines_improve":
            language["news_test"]["CB88_KL"]<min(
                language["news_test"]["CB59_KL"],
                language["news_test"]["CB85_KL"]),
        "Wiki_new_true_token_NLL_nonregress":
            language["wiki_test"]["CB88_NLL"]<=min(
                language["wiki_test"]["CB59_NLL"],
                language["wiki_test"]["CB85_NLL"]),
        "News_new_true_token_NLL_nonregress":
            language["news_test"]["CB88_NLL"]<=min(
                language["news_test"]["CB59_NLL"],
                language["news_test"]["CB85_NLL"]),
        "real_greedy_distinct_ratio_at_least_0p25":
            generation["CB88"]["mean_distinct_token_ratio"]>=.25,
        "exploratory_CPU_decode_at_least_90pct_of_CB59":
            generation["CB88"]["tokens_per_second_one_cpu"]>=
              .90*generation["CB59"]["tokens_per_second_one_cpu"],
    }
    return {"new_untouched_independent_QA":comparisons,
            "new_unseen_official_language_TEST":language,
            "real_4_prompt_CPU_KV_cache_greedy":generation,
            "predeclared_scientific_checks":checks,
            "all_scientific_checks_pass":all(checks.values()),
            "caveat":"Small original 135M model and only 48 SciQ/BoolQ rows, "
              "16x64 language contexts per corpus and four 24-token prompts. "
              "QA, diversity and same-host speed are exploratory. "
              "Software CI SUCCESS never means scientific success."}

def run(original_path:Path,cb85_path:Path,cb88_path:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    original=original_path.read_bytes()
    cb85_blob=cb85_path.read_bytes()
    cb88_blob=cb88_path.read_bytes()
    prereg=freeze_before_test(original,cb85_blob,cb88_blob)
    (out/"v89_PRETEST_frozen_sources_and_disjoint_inputs.json").write_text(
        json.dumps(prereg,indent=2)+"\n")
    activate_unseen_test()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=original_teacher()
    student,info,_=make_model("baseline",SimpleNamespace(
        original=original_path,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("pretrained original dense source in packed inference")
    results={}
    results["teacher"]=assess_real_model(
        "teacher",teacher,tokenizer,out,{"real_original_FP32":True})
    results["CB59"]=assess_real_model("baseline",student,tokenizer,out,info)
    adaptive=CB85(cb85_blob)
    old=insert_native_adapters(student,adaptive,TARGETS)
    try:
        results["CB85"]=assess_real_model("adaptive",student,tokenizer,out,
                                         adaptive.physical_stats())
    finally:
        restore_native_adapters(old)
    _,A,B,bias=unpack(cb88_blob)
    original_head=student.lm_head
    student.lm_head=TensorPackedHead(original_head,A,B,bias)
    try:
        results["CB88"]=assess_real_model("CB88",student,tokenizer,out,
                                         audit(cb88_blob))
    finally:
        student.lm_head=original_head
    payload=extract(results,audit(cb88_blob))
    report={
        "experiment":"CALIBRI_v89_frozen_TRUE_independent_original_pretrained_4D_neuroweights_TEST",
        "model":MODEL,"pinned_original_pretrained_revision":MODEL_SHA,
        "frozen_provenance_before_ANY_TEST":prereg,
        **payload,
        "no_training_or_weights_source_selection_on_TEST":True,
        "original_CB59_CB85_CB88_unchanged":True,
        "Jessica_main_Render_Supabase_APK_untouched":True}
    (out/"v89_TRUE_original_unseen_4D_neuroweights_QA_KL_NLL_greedy_speed_TEST.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":"v89_original_real_pretrained_4D_independent_TEST_complete",
        "QA":{t:{k:results[k]["original_heldout_QA"][t]["correct_count"]
                 for k in results} for t in ("sciq","boolq")},
        "language_KL":{d:{k:report["new_unseen_official_language_TEST"][d][k+"_KL"]
                    for k in ("CB59","CB85","CB88")}
                    for d in ("wiki_test","news_test")},
        "greedy_distinct":{k:v["mean_distinct_token_ratio"]
                          for k,v in payload["real_4_prompt_CPU_KV_cache_greedy"].items()},
        "real_tokens_per_sec":{k:v["tokens_per_second_one_cpu"]
                          for k,v in payload["real_4_prompt_CPU_KV_cache_greedy"].items()},
        "scientific_pass":payload["all_scientific_checks_pass"],
        "all_negative_results_reported":True
    },ensure_ascii=False),flush=True)
    return report
