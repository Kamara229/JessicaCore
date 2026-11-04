"""Fresh actual C-native and original FP32 chat generation against fixed v106 alarm.

The hypotheses and prompts were committed BEFORE this model inference.
No trained weights change, no v89 original official TEST, and no
generation outputs used to change source or alarm thresholds.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from .spec import ALL_PROMPTS,GOLD_PROMPTS,OPEN_PROMPTS,V102_SOURCE_SHA256
from .generation import observed_answer

def locked_sources(frozen:Path,candidate:Path,metadata:Path,contract:Path):
    base=frozen.read_bytes()
    changed=candidate.read_bytes()
    info=json.loads(metadata.read_text())
    lock=json.loads(contract.read_text())
    pin=lock["frozen_reference_sources"]["CB59"]
    sha=sha256(base).hexdigest()
    recode_sha=sha256(changed).hexdigest()
    if (lock.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04"
        or sha!=pin["sha256"]
        or len(base)!=pin["complete_archive_bytes"]
        or recode_sha!=V102_SOURCE_SHA256
        or recode_sha!=info["new_separate_source_sha256"]
        or info["original_immutable_reference_sha256"]!=sha
        or MODEL_SHA!=lock["original_model_revision"]
        or len(base)!=len(changed)
        or 8*len(changed)/EXPECTED_UNIQUE>.700):
        raise AssertionError("original physical model lock / v102 locked experiment violated")
    old=CB59(base)
    new=CB59(changed)
    for model in (old,new):
        if model.handles["lm_head.weight"] is not model.handles["model.embed_tokens.weight"]:
            raise AssertionError("original real model tied output/embedding source failed")
    return {"original_frozen_CB59_sha256":sha,
        "actual_paid_recode_v102_sha256":recode_sha,
        "complete_bytes_each":len(base),
        "all_in_bits_per_original_parameter":8*len(changed)/EXPECTED_UNIQUE}

def run(frozen:Path,candidate:Path,metadata:Path,contract:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    provenance=locked_sources(frozen,candidate,metadata,contract)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    baseline_dir=out/"baseline_runtime"
    recode_dir=out/"recode_runtime"
    baseline_dir.mkdir(exist_ok=True)
    recode_dir.mkdir(exist_ok=True)
    cb,cb_meta,_=make_model("baseline",SimpleNamespace(original=frozen,folder=baseline_dir))
    v102,v102_meta,_=make_model("baseline",SimpleNamespace(original=candidate,folder=recode_dir))
    teacher=original_teacher()
    for model in (cb,v102):
        if (model.get_input_embeddings().handle is not model.get_output_embeddings().handle
            or any(p.ndim>=2 for p in model.parameters())):
            raise AssertionError("original compressed runtime retains dense trained source")
    if not cb_meta["zero_dense_pretrained_2D_hot_weights"] or not v102_meta["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("original packed model C code was not loaded")
    models=(("original_FP32_pretrained",teacher),("original_CB59_native",cb),
            ("fully_paid_v102_CB59_native",v102))
    results=[]
    for index,(domain,message,gold) in enumerate(ALL_PROMPTS):
        # Alternate ordering for partially reducing temperature/caching bias.
        ordered=models if index%2==0 else tuple(reversed(models))
        outputs={}
        with torch.inference_mode():
            for name,model in ordered:
                outputs[name]=observed_answer(model,tok,domain,message,gold)
        if len({z["actual_chat_prompt_SHA256"] for z in outputs.values()})!=1:
            raise AssertionError("original pretrained and physical model used different prompts")
        results.append({"domain":domain,
            "prompt_source_SHA256":sha256(message.encode("utf8")).hexdigest(),
            "is_objective_gold_canary":bool(gold),
            "answers_only_for_human_review":list(gold),
            "source_locked_models_actual_32_token_outputs":outputs})
        print(json.dumps({"event":"v108_original_chat_new_canary",
           "domain":domain,"index":index,
           "morphology_alarm_by_model":{n:x["v106_frozen_unchanged_morphology"]["exploratory_fivegram_AND_zlib_anomaly_alarm"]
             for n,x in outputs.items()},
           "strict_prefix_gold_by_model":{n:(x["v96_first_response_prefix_gold_signal"] or {}).get("correct")
             for n,x in outputs.items()}},ensure_ascii=False),flush=True)
    summary={}
    for name,_ in models:
        rows=[entry["source_locked_models_actual_32_token_outputs"][name] for entry in results]
        summary[name]={
            "actual_generated_32_token_responses":len(rows),
            "v106_original_threshold_alarm_count":sum(
                int(x["v106_frozen_unchanged_morphology"]["exploratory_fivegram_AND_zlib_anomaly_alarm"])
                for x in rows),
            "synthetic_gold_direct_prefix_hits_not_accuracy":sum(
                int((x["v96_first_response_prefix_gold_signal"] or {}).get("correct",False))
                for x in rows),
            "synthetic_gold_first_sentence_presence_not_accuracy":sum(
                int((x["v96_first_sentence_gold_presence_not_semantic_proof"] or {}).get(
                    "reference_present_in_first_answer_sentence",False))
                for x in rows),
            "fake_role_lines_in_generated_text":sum(int(
                x["v106_frozen_unchanged_morphology"]["simulated_role_line_present"])
                for x in rows),
            "role_lines_within_real_first_assistant_turn":sum(
                x["role_line_within_first_assistant_turn"] for x in rows),
            "fixed_work_tokens_after_first_eos":sum(
                x["first_assistant_turn"]["tokens_after_eos"] for x in rows),
            "actual_generated_tokens_per_second_including_prefill":
                (32*len(rows))/max(1e-10,sum(
                    x["full_model_real_CPU_KV_cache_elapsed_seconds"] for x in rows)),
            "fluency_label_and_full_semantic_accuracy_pending_human_review":True,
        }
    report={
        "experiment":"CALIBRI_v108_preregistered_unseen_original_chat_generation_and_fixed_v106_alarm",
        "original_complete_immutable_model_provenance":provenance,
        "genuinely_fresh_synthetic_gold_canary_count":len(GOLD_PROMPTS),
        "genuinely_fresh_synthetic_open_ended_prompts":len(OPEN_PROMPTS),
        "genuinely_actual_native_CPU_KV_cache_32_token_texts":len(results)*len(models),
        "original_model_revision":MODEL_SHA,
        "original_model_chat_template_used":True,
        "original_v106_signal_features_and_thresholds_unchanged":True,
        "no_new_classifier_fitting_or_tuning":True,
        "per_model_summary_no_semantic_accuracy_claim":summary,
        "all_actual_readable_generated_answers_for_independent_review":results,
        "original_FP32_outputs_can_be_factually_wrong":True,
        "predeclared_gold_not_teacher_as_truth":True,
        "all_in_compressed_archive_rate_maximum_bpp":0.7,
        "no_new_candidate_or_model_weight_selection":True,
        "official_independent_v89_TEST_not_opened":True,
        "original_model_CB59_and_v102_sources_unchanged":True,
        "no_Jessica_chat_test_attempted":True,
        "live_Jessica_main_Render_Supabase_APK_unchanged":True,
        "interpretation":"This is a prospective original-chat inference sample, but "
            "anomalous-word morphology and answer presence are not independent "
            "blinded human semantic labels; do not treat any alarm counts as "
            "scientific sensitivity/specificity until separate reviews."}
    target=out/"v108_original_chat_new_answer_generation_frozen_detector.json"
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"event":"v108_finished_new_original_chat_real_native_generation",
        "flagged_counts":{name:s["v106_original_threshold_alarm_count"]
                         for name,s in summary.items()},
        "full_model_cpu_rates":{name:round(s["actual_generated_tokens_per_second_including_prefill"],3)
                                for name,s in summary.items()},
        "original_TEST_unopened":True},ensure_ascii=False),flush=True)
    return report
