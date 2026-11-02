#!/usr/bin/env python3
"""CALIBRI v59 independently verifies real 135M sub0.7 rate & direct kernel.

CI PASS indicates physical target and packed full-model forward, NOT
semantic quality or speed (explicitly deferred by user).
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path


def check(report,host,errors):
    if report["experiment"]!="CALIBRI_v59_first_rate_native_direct_K32V8_135M":
        errors.append(host+" experiment scope mismatch")
    if report["trained_model"]!="HuggingFaceTB/SmolLM2-135M-Instruct":
        errors.append(host+" not a real instruction-tuned open checkpoint")
    src=report["physical_codec_and_source"]
    runtime=report["native_compressed_resident_hot_runtime"]
    teacher=report["teacher_separate_process_source"]
    params=src["actual_unique_original_parameter_count"]
    bytecount=src["physical_file"]["physical_bytes"]
    if params!=134515008:
        errors.append(host+" wrong unique tied source pretrained parameter count")
    if len(src["physical_file"]["sha256"])!=64 or bytecount<=0:
        errors.append(host+" missing real physical SHA/checkpoint bytes")
    if bytecount>math.floor(0.7*params/8) or report["full_real_checkpoint_actual_bpw"]>0.7:
        errors.append(host+" failed actual <=0.7 bits/unique model param ceiling")
    if src["actual_full_source_rate_below_0_7"] is not True:
        errors.append(host+" stored full model physical format failed its rate gate")
    if (runtime["all_2d_original_source_dense_parameters_in_hot_runtime"]!=0
        or runtime["no_original_pretrained_teacher_loaded_in_this_process"] is not True
        or runtime["runtime_uses_native_C_LUT_GEMV_no_dense_weight_dequantization"] is not True
        or runtime["packed_actual_in_RAM_checkpoint_bytes"]!=bytecount):
        errors.append(host+" no genuine direct kernel execution / dense source retained")
    if (runtime["checksum_source"]!=src["physical_file"]["sha256"] or
        runtime["full_model_forward_is_finite"] is not True or
        runtime["autoregressive_KV_cache_forward_completed"] is not True):
        errors.append(host+" compressed full transformer cannot do real token decoding")
    mod=runtime["full_Llama_from_config_has_only_compressed_linear_and_embedding"]
    if mod["compressed_real_2D_module_count"]<60 or (
        not mod["tied_embed_and_output_codebook_SHARED_IN_RAM"]):
        errors.append(host+" one or more of genuine original model matrices remains dense")
    q=report["real_8token_model_and_next_KV_decode_logits"]
    if (not q["finite_full_model_forward_and_KV_decode"] or
        not all(math.isfinite(q[key]) for key in (
            "full_vocab_logits_relative_RMSE",
            "original_distribution_KL_vs_packed"))):
        errors.append(host+" catastrophic numerical NaN/Inf compressed model")
    elif (abs(float(q["original_distribution_KL_vs_packed"]))>1e5 or
          q["full_vocab_logits_relative_RMSE"]>1e5):
        errors.append(host+" unbounded source output reconstruction")
    seconds=report["paired_cpu_single_sequence_runtime_ratio_NOT_optimized"]
    if not math.isfinite(seconds) or seconds<=0:
        errors.append(host+" nonmeasurable compressed model KV compute")
    if not 0<src["weight_source_global_relative_RMSE"]<10:
        errors.append(host+" erroneous source-weight reconstruction bound")
    return {
        "real_complete_checkpoint_bytes":bytecount,
        "unique_pretrained_source_params":params,
        "physical_bpw":report["full_real_checkpoint_actual_bpw"],
        "entire_full_CB59_SHA256":src["physical_file"]["sha256"],
        "runtime_all_original_dense_2D_parameter_count":0,
        "native_direct_packed_linear_embedding_full_KV":True,
        "original_teacher_vs_packed_next_token_agree":q["top1_agree"],
        "full_vocab_normalized_logit_RMSE":q["full_vocab_logits_relative_RMSE"],
        "teacher_to_packed_output_KL":q["original_distribution_KL_vs_packed"],
        "native_C_batch1_CPU_time_ratio_UNOPTIMIZED":seconds,
        "quality_warning":report["quality_degradation_significant_possible"],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True)
    p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    errors=[]
    measured={}
    for host,source in (("ubuntu22",a),("ubuntu24",b)):
        measured[host]=check(source,host,errors)
    for field in ("trained_model","original_unique_source_parameters"):
        if a[field]!=b[field]:
            errors.append("two independent CPU hosts did not use same pretrained source: "+field)
    h1=a["physical_codec_and_source"]["source_model_revision"]
    h2=b["physical_codec_and_source"]["source_model_revision"]
    if h1!=h2:
        errors.append("source model revision not pinned identically")
    same_file=(measured["ubuntu22"]["entire_full_CB59_SHA256"]==
               measured["ubuntu24"]["entire_full_CB59_SHA256"])
    if not same_file:
        errors.append("physical CB59 checksum not bitwise reproducible across independent CPUs")
    if measured["ubuntu22"]["real_complete_checkpoint_bytes"]!=measured[
            "ubuntu24"]["real_complete_checkpoint_bytes"]:
        errors.append("physical source byte count differs between hosts")
    stage_complete=a["first_stage_physical_rate_and_native_inference_PASS"] and (
        b["first_stage_physical_rate_and_native_inference_PASS"])
    if not stage_complete:
        errors.append("first stage actual rate+direct inference did not complete")
    result={
        "PASS":not errors,"errors":errors,
        "new_CB59_sub07_target_real_checkpoint_and_direct_kernel":stage_complete,
        "crosshost_physical_intact_source_identical":same_file,
        "host_measurements":measured,
        "real_experimental_limitations":[
            "Dense 135M model quality under 0.7bpw may be catastrophically degraded.",
            "This verifier does not claim restored semantics, speed parity with FP32, or production Jessica readiness.",
            "Current native CPU codebook LUT matvec is a working unoptimized reference implementation.",
            "Unlike previous CALIBRI <=6.5bpw, direct 5-bit compressed source indices never expand into full dense pretrained hot RAM."
        ]}
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if errors:raise SystemExit(1)


if __name__=="__main__":
    main()
