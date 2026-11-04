"""Frozen v100 read-only 2x2 original teacher residual/head diagnostic."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v98.metrics import logits_topk
from v99.patch import native_logits
from .factors import factor_logits
from .generation import greedy
from .spec import PROBES,FACTORS,GENERATE_DOMAINS

def compare_exact_teacher(teacher_logits,variant_logits,tok):
    exact=logits_topk(teacher_logits,tok)
    observed=logits_topk(variant_logits,tok)
    overlap=len(set(exact["top10_token_ids"]) & set(observed["top10_token_ids"]))
    return {"teacher":exact,"student_with_unpaid_oracle":observed,
        "teacher_top10_overlap":overlap,
        "teacher_top1_matched":exact["top10_token_ids"][0]==observed["top10_token_ids"][0],
        "top10_is_not_semantic_correctness":True}

def run(original:Path,locked:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    lock=json.loads(locked.read_text())
    raw=original.read_bytes()
    expected=lock["frozen_reference_sources"]["CB59"]
    sha=hashlib.sha256(raw).hexdigest()
    bpw=8*len(raw)/EXPECTED_UNIQUE
    if (lock.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04"
        or sha!=expected["sha256"]
        or len(raw)!=expected["complete_archive_bytes"]
        or MODEL_SHA!=lock["original_model_revision"] or bpw>.700):
        raise AssertionError("immutable original native source contract failed")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    student,info,_=make_model("baseline",SimpleNamespace(original=original,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("not a legitimate original native source")
    teacher=original_teacher()
    observations=[]
    with torch.inference_mode():
        for kind,prompt in PROBES:
            prompt_tokens=tok.encode(prompt,add_special_tokens=False)
            if not (1<len(prompt_tokens)<=96):
                raise AssertionError("unexpected diagnostic context geometry")
            ids=torch.tensor(prompt_tokens,dtype=torch.long)[None,:]
            reference=native_logits(teacher,ids)
            variants={}
            for factor in FACTORS:
                native=factor_logits(teacher,student,ids,factor)
                outcomes=compare_exact_teacher(reference,native,tok)
                if kind in GENERATE_DOMAINS:
                    outcomes["actual_greedy_12_token_output"]=greedy(
                        student,tok,prompt,teacher=teacher,variant=factor)
                variants[factor]=outcomes
                print(json.dumps({"event":"v100_2x2_causal_oracle",
                  "prompt_domain":kind,"factor":factor,
                  "top10_overlap":outcomes["teacher_top10_overlap"],
                  "top1_match":outcomes["teacher_top1_matched"],
                  "unpaid_original_FP32_head":factor in ("head_only","norm_and_head"),
                  "unpaid_original_FP32_reference_residual":factor in ("norm_only","norm_and_head")}),flush=True)
            independent_teacher_gen=greedy(teacher,tok,prompt) if kind in GENERATE_DOMAINS else None
            observations.append({
                "prompt_domain":kind,
                "prompt_SHA256":hashlib.sha256(prompt.encode()).hexdigest(),
                "same_token_prefix_length":len(prompt_tokens),
                "exact_original_FP32_generated_text":independent_teacher_gen,
                "factorial_interventions":variants})
    summary={}
    for factor in FACTORS:
        values=[item["factorial_interventions"][factor] for item in observations]
        summary[factor]={
            "total_top10_overlap_across_four_contexts":sum(
                v["teacher_top10_overlap"] for v in values),
            "original_FP32_top1_matches_out_of_four":sum(int(
                v["teacher_top1_matched"]) for v in values),
            "original_reference_required_per_token":factor in ("norm_only","norm_and_head"),
            "unpaid_FP32_dense_head_required":factor in ("head_only","norm_and_head"),
            "physically_deployable_under_0p7_bpp":False if factor!="native" else None}
    output={"experiment":"CALIBRI_v100_double_fault_hidden_reference_and_output_head_2x2",
        "frozen_native_CB59_SHA256":sha,
        "frozen_native_CB59_total_bytes":len(raw),
        "native_CB59_all_in_bits_per_original_parameter":bpw,
        "original_model_revision":MODEL_SHA,
        "new_synthetic_only_same_prefix_contexts":observations,
        "factorial_summary":summary,
        "read_only_causal_upper_bound_NOT_compressed_candidate":True,
        "real_trained_weights_unchanged_and_no_weights_saved":True,
        "independent_v89_TEST_not_accessed":True,
        "Jessica_main_Render_Supabase_APK_live_not_modified":True,
        "interpretation_warning":"Double oracle requires FP32 full head and exact original "
            "reference activation EVERY output token: even if successful "
            "it is not same-byte quality restoration or an eligible speed benchmark."}
    dest=out/"v100_original_CB59_hidden_head_2x2_full_decoded_text.json"
    dest.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"event":"v100_complete_diagnostic_only",
        "factorial_top10_overlap":{k:v["total_top10_overlap_across_four_contexts"]
                    for k,v in summary.items()},
        "fixed_complete_source_bytes":len(raw),
        "old_TEST_untouched":True}),flush=True)
    return output
