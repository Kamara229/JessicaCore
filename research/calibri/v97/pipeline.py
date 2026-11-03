"""v97 one-time causal localization: original pretrained oracle swaps on frozen CB59.

An FP32 oracle swap is deliberately *not* physically compressed nor speed
comparable. Its only valid interpretation: which native block is a plausible
source of lost labeled accuracy/meaningful language. No candidates are
selected, no KL/NLL, no previous official independent TEST reopened.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from .spec import TRIALS
from .oracle import temporary_oracle_modules
from .behavior import original_boolq,evaluate_model,paired_checks

def run(original:Path,locked:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    lock=json.loads(locked.read_text())
    if lock.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04":
        raise AssertionError("frozen source baseline contract is required")
    source=original.read_bytes()
    pinned=lock["frozen_reference_sources"]["CB59"]
    if (len(source)!=pinned["complete_archive_bytes"] or
        hashlib.sha256(source).hexdigest()!=pinned["sha256"] or
        8*len(source)/EXPECTED_UNIQUE>.700):
        raise AssertionError("original native source altered or exceeds frozen budget")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases,provenance,answers=original_boolq(tok)
    packed,info,_=make_model("baseline",SimpleNamespace(original=original,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("not a real native compressed source model")
    fp32=original_teacher()
    results={}
    with torch.inference_mode():
        results["ORIGINAL_FP32"]=evaluate_model(fp32,tok,cases,answers)
        results["CB59_NATIVE_FROZEN"]=evaluate_model(packed,tok,cases,answers)
        for name,paths in TRIALS:
            with temporary_oracle_modules(packed,fp32,paths):
                results[name]=evaluate_model(packed,tok,cases,answers)
            # A failed restoration is a serious safety failure.
            for module_name in paths:
                module=packed.get_submodule(module_name)
                if not hasattr(module,"handle"):
                    raise AssertionError("native source remained replaced after diagnosis")
            print(json.dumps({"stage":"causal_FP32_diagnostic_only",
              "oracle":name,"boolq_correct":results[name]["boolq"]["summary"]["correct"],
              "reference_first_sentence":results[name]["first_sentence_answer_reference_present"],
              "repeated_prompts":results[name]["repeated_over_75pct"]}),flush=True)
    paired_checks(results)
    baseline=results["CB59_NATIVE_FROZEN"]
    teacher=results["ORIGINAL_FP32"]
    summary={}
    for name,paths in TRIALS:
        scored=results[name]
        summary[name]={
            "temporarily_oracled_FP32_modules":list(paths),
            "correct_answers_vs_ground_truth":scored["boolq"]["summary"]["correct"],
            "gain_in_correct_answers_vs_packed_CB59":
                scored["boolq"]["summary"]["correct"]-baseline["boolq"]["summary"]["correct"],
            "first_sentence_answer_reference_signal":scored["first_sentence_answer_reference_present"],
            "change_in_first_sentence_reference_vs_CB59":
                scored["first_sentence_answer_reference_present"]-
                baseline["first_sentence_answer_reference_present"],
            "repetition_alarms":scored["repeated_over_75pct"],
            "permanent_extra_storage_bytes":None,
            "physically_deployable_at_0p667839_bpp":False,
            "native_full_CPU_speed_comparison_permitted":False,
            "NOT_an_automatic_model_selection":True}
    result={"experiment":"CALIBRI_v97_frozen_native_causal_FP32_oracle_generation_probes",
       "original_public_train_only":provenance,
       "original_pretrained_revision":MODEL_SHA,
       "physical_complete_native_CB59_bytes":len(source),
       "all_in_bits_per_original_param":8*len(source)/EXPECTED_UNIQUE,
       "original_frozen_CB59_SHA256":hashlib.sha256(source).hexdigest(),
       "results":results,
       "hypothesis_localization_not_validated_quality_gain":summary,
       "original_FP32_correct_answers":teacher["boolq"]["summary"]["correct"],
       "native_CB59_correct_answers":baseline["boolq"]["summary"]["correct"],
       "physical_weight_rewrite_performed":False,
       "FP32_oracle_only_for_causal_diagnosis_NOT_release":True,
       "KL_NLL_or_teacher_probability_agreement_used":False,
       "v89_independent_TEST_not_loaded":True,
       "no_live_Jessica_main_Render_Supabase_APK_changes":True,
       "limitations":"Only 16 original public BoolQ TRAIN questions and 4 synthetic prompts "
          "per variant; output-reference signal is not semantics or factual accuracy. "
          "FP32 module substitution is unpaid and cannot be a physical compression solution. "
          "No scientific performance conclusion without new independent heldout data."}
    target=out/"v97_CB59_causal_FP32_oracles_ANSWER_AND_TEXT_ONLY.json"
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"v97_completed_causal_diagnostics",
        "FP32_BoolQ":teacher["boolq"]["summary"]["correct"],
        "CB59_BoolQ":baseline["boolq"]["summary"]["correct"],
        "oracle_changes":{k:v["gain_in_correct_answers_vs_packed_CB59"] for k,v in summary.items()},
        "frozen_source_bytes":len(source),
        "old_independent_TEST_untouched":True}),flush=True)
    return result
