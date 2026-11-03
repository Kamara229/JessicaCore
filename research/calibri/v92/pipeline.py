"""CALIBRI v92 proof-of-concept self-repair loop: detect -> rank -> rewrite -> validate.

ZERO live Jessica mutation. Uses original TRAIN-only v91 artifact and disjoint
real pretrained BoolQ train slices. Rewrites one complete 5-bit packed block
in place, preserving whole model native ABI and byte count. Never retunes on
the frozen independent official TEST.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v90.provenance import verify_sources
from v90.qa import build_cases
from v90.generation import generate_diagnostic
from .evidence import ingest,triage
from .policy import require_research_only,decide_candidate
from .features import capture_energy
from .encode import reencode_8d
from .archive import inplace_replace
from .validate import compare_frozen_whole_models,microbenchmark_one_operator,swap_one_block

def priority_target(report):
    causal=report["single_module_FP32_oracle_rescue_not_paid_source"]
    options=[x for x in causal["ranked_single_module_rescues"]
             if x["module"].startswith("model.layers.") and
             x["mean_KL_gain_from_exact_FP32_one_module"]>0]
    if not options:
        raise AssertionError("do not rewrite a module without any TRAIN causal evidence")
    return max(options,key=lambda x:x["mean_KL_gain_from_exact_FP32_one_module"])["module"]

def run(report_path:Path,original_path:Path,cb88_path:Path,out:Path):
    require_research_only("sandbox_rewrite")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    report,provenance=ingest(report_path)
    incidents=triage(report)
    target=priority_target(report)
    source=original_path.read_bytes()
    frozen=cb88_path.read_bytes()
    pinned=verify_sources(source,frozen)
    if provenance["source_SHA256"]!=pinned["CB59_SHA"] or provenance["correction_SHA256"]!=pinned["CB88_SHA"]:
        raise AssertionError("diagnosis and original trained sources have different immutable SHAs")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases,origin,answer_tokens=build_cases(tokenizer)
    if origin!=report["public_train_dataset_sources"]:
        raise AssertionError("fresh public TRAIN provenance changed; refuse silent source drift")
    teacher=original_teacher()
    original_model,info,_=make_model("baseline",SimpleNamespace(original=original_path,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("cannot rewrite if original native source was materialized dense")
    old_archive=CB59(source)
    if target+".weight" not in old_archive.handles:
        raise AssertionError("top causal target has no native 5bit source block")
    train_valid,features=capture_energy(teacher,target,cases)
    teacher_weight=teacher.get_submodule(target).weight.detach().cpu().float().numpy()
    trial=reencode_8d(teacher_weight,old_archive.handles[target+".weight"],
                      train_valid["TRAIN"],train_valid["TRAIN_VALID"],iterations=4)
    rebuilt,change=inplace_replace(source,target+".weight",trial)
    if len(rebuilt)!=len(source):raise AssertionError("no new storage may be allocated")
    metrics=compare_frozen_whole_models(original_model,teacher,cases,answer_tokens,source,rebuilt,target)
    timing=microbenchmark_one_operator(original_model,source,rebuilt,target)
    base_greedy=generate_diagnostic(original_model,tokenizer)
    parent,leaf,old,new=swap_one_block(original_model,source,rebuilt,target)
    try:
        setattr(parent,leaf,new)
        new_greedy=generate_diagnostic(original_model,tokenizer)
    finally:setattr(parent,leaf,old)
    decision=decide_candidate(source_bytes=len(rebuilt),
        valid_teacher_KL_gain=metrics["teacher_KL_gain"],
        valid_QA_regression=metrics["VALID_QA_count_change"],
        valid_greedy_ratio=new_greedy["mean_distinct_ratio"],
        valid_speed_ratio=timing["candidate_speed_ratio"],
        full_model_speed_proven=False,true_language_NLL_proven=False)
    decision["research_archive_keep_for_analysis"]=(bool(trial["change_accepted_by_LOCAL_metric"])
        and metrics["teacher_KL_gain"]>0 and metrics["VALID_QA_count_change"]>=0)
    decision["must_not_merge_into_any_production_release"]=True
    decision["VALID_sample_size_too_small_for_general_claims"]=True
    # A new physical source is experimental; CB88 sidecar includes original SHA,
    # so it CANNOT be silently reused with this rewritten base.
    payload={"experiment":"CALIBRI_v92_automated_detect_triage_rewrite_validate_RESEARCH_ONLY",
        "input_artifact_provenance":provenance,"source_verification":pinned,
        "immutable_original_model_revision":MODEL_SHA,"public_TRAIN_sources":origin,
        "automatic_findings":incidents,"selected_causal_module":target,
        "original_FP32_teacher_train_activation_source":features,
        "entire_5bit_block_rewrite":{k:v for k,v in trial.items()
            if k not in ("indices","centroids","row_scales")},
        "physical_candidate":change,"same_size_original_native_operator":True,
        "paired_new_train_VALID_quality":metrics,"paired_native_block_microbenchmark":timing,
        "greedy_on_same_fresh_diagnostic_prompts": {"CB59":base_greedy,"rewritten":new_greedy},
        "automatic_safe_decision":decision,
        "CB88_sidecar_must_be_retrained_or_omitted_for_rewritten_CB59":True,
        "original_TEST_not_used_for_tuning":True,
        "Jessica_or_main_Render_Supabase_APK_untouched":True,
        "notes":"One causal candidate from two short v91 TRAIN probes; 12+12 activation "
            "TRAIN/VALID and 12 BoolQ VALID answer prompts. This is a safe automatic "
            "prototype, NOT a proof of a quality-preserving autonomous self-modifying LLM."}
    (out/"v92_research_quality_doctor_report.json").write_text(
        json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    (out/"v92_rewritten_single_block_RESEARCH_ONLY.cb59").write_bytes(rebuilt)
    print(json.dumps({"stage":"automated_quality_doctor_trial_complete",
        "target":target,"codebook_local_VALID_gain":trial["validation_gain"],
        "real_model_teacher_KL_gain":metrics["teacher_KL_gain"],
        "real_model_QA_VALID_difference":metrics["VALID_QA_count_change"],
        "original_greedy_distinct":base_greedy["mean_distinct_ratio"],
        "rewritten_greedy_distinct":new_greedy["mean_distinct_ratio"],
        "original_and_new_bytes":len(source),
        "decision":decision["status"],"live_untouched":True},ensure_ascii=False),flush=True)
    return payload
