"""Original trained tied head same-byte codebook reconstruction trial.

All improvements (if any) from full actual pretrained original source only.
Selection never touches official TEST or any generated/VALID text.
A built candidate is NOT promoted to production on CI PASS.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v94.generation import generate_one
from v96.grade import string_breakdown
from v99.patch import capture_one
from .spec import TEST_PROMPTS,GENERATE_PROMPTS
from .recode import recode_source
from .metrics import original_head_weight_rmse,compare_vocab_logits

def original_logits_and_hidden(model,ids):
    with torch.inference_mode(),capture_one(model,"model.norm") as snapshots:
        logit=model(input_ids=ids,use_cache=False).logits[0,-1].detach().float()
    if len(snapshots)!=1:raise AssertionError("missing pinned original hidden state")
    return logit,snapshots[0][0,-1].float()

def sample(model,tok,prompt):
    saved=generate_one(model,tok,prompt)
    return {"actual_text":saved["text"],
        "generated_token_ids":saved["ids"],
        "actual_CPU_KV_cache_generated_tokens":saved["full_model_token_steps"],
        "elapsed_seconds":saved["elapsed_seconds"],
        "lexical":string_breakdown(saved["text"],saved["ids"]),
        "meaningfulness_requires_human_review":True}

def run(original:Path,locked:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    raw=original.read_bytes()
    lock=json.loads(locked.read_text())
    control=lock["frozen_reference_sources"]["CB59"]
    if (lock.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04"
        or hashlib.sha256(raw).hexdigest()!=control["sha256"]
        or len(raw)!=control["complete_archive_bytes"]
        or MODEL_SHA!=lock["original_model_revision"]):
        raise AssertionError("original reference source is not bytewise frozen")
    original_archive=CB59(raw)
    teacher=original_teacher()
    candidate_bytes,details=recode_source(raw,teacher.lm_head.weight)
    if len(candidate_bytes)!=len(raw) or 8*len(candidate_bytes)/EXPECTED_UNIQUE>.700:
        raise AssertionError("new re-encoding exceeds entire physical budget")
    compressed_file=out/"CALIBRI_v102_reencoded_whole_original_TIED_HEAD.cb59"
    compressed_file.write_bytes(candidate_bytes)
    candidate_archive=CB59(candidate_bytes)
    original_source_rmse=original_head_weight_rmse(original_archive.handles["lm_head.weight"],
                                                   teacher.lm_head.weight)
    selected_source_rmse=original_head_weight_rmse(candidate_archive.handles["lm_head.weight"],
                                                   teacher.lm_head.weight)
    if abs(selected_source_rmse-min(item["original_tied_head_relative_RMSE"]
            for item in details["all_source_only_alternatives"]))>0.00005:
        raise AssertionError("actual encoded head RMSE differs from source-only model selection")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    baseroot=out/"baseline_runtime";baseroot.mkdir()
    candroot=out/"candidate_runtime";candroot.mkdir()
    reference,baseline_info,_=make_model("baseline",SimpleNamespace(original=original,folder=baseroot))
    native,candidate_info,_=make_model("baseline",SimpleNamespace(original=compressed_file,folder=candroot))
    if not baseline_info["zero_dense_pretrained_2D_hot_weights"] or not candidate_info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("real native packed kernels required")
    if (native.lm_head.handle is not native.get_input_embeddings().handle
        or reference.lm_head.handle is not reference.get_input_embeddings().handle):
        raise AssertionError("original single codebook output-input tie was broken")
    metrics=[]
    for domain,prompt in TEST_PROMPTS:
        enc=tok.encode(prompt,add_special_tokens=False)
        ids=torch.tensor(enc,dtype=torch.long)[None,:]
        teacher_logits,correct_hidden=original_logits_and_hidden(teacher,ids)
        with torch.inference_mode():
            original_reference=reference(input_ids=ids,use_cache=False).logits[0,-1].detach().float()
            new_reference=native(input_ids=ids,use_cache=False).logits[0,-1].detach().float()
            # Isolate head codebook's actual information loss with same *original* pretrained vector.
            old_head=reference.lm_head(correct_hidden[None,None,:])[0,-1]
            new_head=native.lm_head(correct_hidden[None,None,:])[0,-1]
        record={
            "domain":domain,
            "prompt_SHA256":hashlib.sha256(prompt.encode()).hexdigest(),
            "true_native_full_model_CB59":compare_vocab_logits(teacher_logits,original_reference,tok),
            "true_native_full_model_reencoded":compare_vocab_logits(teacher_logits,new_reference,tok),
            "codec_only_original_CB59_on_true_FP32_hidden":compare_vocab_logits(teacher_logits,old_head,tok),
            "codec_only_reencoded_on_true_FP32_hidden":compare_vocab_logits(teacher_logits,new_head,tok),
        }
        metrics.append(record)
        print(json.dumps({"stage":"v102_source_recode_same_prefix_native_vocabulary",
            "domain":domain,
            "original_full_top10":record["true_native_full_model_CB59"]["reference_top10_overlap"],
            "reencoded_full_top10":record["true_native_full_model_reencoded"]["reference_top10_overlap"],
            "original_correct_hidden_top10":record["codec_only_original_CB59_on_true_FP32_hidden"]["reference_top10_overlap"],
            "reencoded_correct_hidden_top10":record["codec_only_reencoded_on_true_FP32_hidden"]["reference_top10_overlap"]},ensure_ascii=False),flush=True)
    # Paired independent generated text is an OBSERVATION, never selection.
    generation=[]
    for i,(domain,prompt) in enumerate(GENERATE_PROMPTS):
        models=[("original_CB59",reference),("whole_head_reencoded",native)]
        if i%2:models.reverse()
        observed={name:sample(model,tok,prompt) for name,model in models}
        observed["original_FP32"]=sample(teacher,tok,prompt)
        generation.append({"domain":domain,
            "prompt_SHA256":hashlib.sha256(prompt.encode()).hexdigest(),
            "actual_greedy_text":observed})
    def tokens_per_second(kind):
        return sum(x["actual_greedy_text"][kind]["actual_CPU_KV_cache_generated_tokens"] for x in generation)/max(
               sum(x["actual_greedy_text"][kind]["elapsed_seconds"] for x in generation),1e-10)
    old_speed=tokens_per_second("original_CB59")
    new_speed=tokens_per_second("whole_head_reencoded")
    speed_ratio=new_speed/old_speed
    result={
        "experiment":"CALIBRI_v102_same_byte_whole_tied_head_CB59_codebook_refit",
        "original_immutable_reference_sha256":hashlib.sha256(raw).hexdigest(),
        "new_separate_source_sha256":hashlib.sha256(candidate_bytes).hexdigest(),
        "original_complete_archive_bytes":len(raw),
        "reencoded_complete_archive_bytes":len(candidate_bytes),
        "whole_model_all_in_bpp":8*len(candidate_bytes)/EXPECTED_UNIQUE,
        "original_model_revision":MODEL_SHA,
        "old_original_tied_head_weight_RMSE":original_source_rmse,
        "new_tied_head_weight_RMSE":selected_source_rmse,
        "source_relative_weight_error_reduction":1-selected_source_rmse/original_source_rmse,
        "source_only_recode_details":details,
        "new_synthetic_READ_ONLY_codebook_and_full_model":metrics,
        "matched_paired_32_token_native_KV_cache_generation":generation,
        "original_native_generated_tokens_per_s":old_speed,
        "candidate_native_generated_tokens_per_s":new_speed,
        "candidate_vs_reference_CPU_speed_ratio":speed_ratio,
        "CPU_fallback_at_least_90pct":speed_ratio>=.9,
        "new_archive_released_or_selected_for_main":False,
        "scientific_semantic_quality_pass_unverified":True,
        "all_other_original_pretrained_tensors_unchanged":True,
        "original_reference_archive_preserved":True,
        "old_official_independent_TEST_unopened":True,
        "Live_Jessica_main_Render_Supabase_APK_unmodified":True,
        "note":"Optimizing source-weight L2 does not guarantee actual language improvement. "
          "Actual test text is printed to artifact for human review, and source-rate/CPU "
          "GEMV are measured on actual separately encoded model. No scientific PASS "
          "without genuine semantic accuracy and independent heldout test."}
    dest=out/"v102_head_refit_same_byte_source_actual_text_speed_and_quality.json"
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"v102_completed_same_byte_head_source_reencoding",
        "original_weight_RMSE":original_source_rmse,
        "candidate_weight_RMSE":selected_source_rmse,
        "runtime_speed_ratio":speed_ratio,
        "complete_bytes":len(candidate_bytes),
        "semantic_quality_not_proven":True,
        "official_TEST_untouched":True},ensure_ascii=False),flush=True)
    return result
