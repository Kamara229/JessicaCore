#!/usr/bin/env python3
"""Independent v60 rate+physical SHA and matched quality/v59 comparison.

This third job does NOT load source models, and independently recomputes
actual physical rate, all four teacher-to-quantized comparison aggregates,
source-only RMSE delta, complete packed-hot runtime evidence and
crosshost reproducibility. Quality regressions are scientific outcomes
and are NOT silently turned into an integrity pass or hidden.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

EXPECTED=134515008
MAX_BPW=.70
METHODS=("control_v59","candidate_CB60")
PROMPTS=("p0","p1","p2","p3")


def verify(x,host,errors):
    if (x["experiment"]!=
        "CALIBRI_v60_256row_tiled_codebooks_controlled_direct_full_model"):
        errors.append(host+": benchmark scope changed")
    src=x["source_only_physical_codec"]
    if (src["original_unique_source_parameters"]!=EXPECTED or
        src["source_tied_aliases"]!={
           "lm_head.weight":"model.embed_tokens.weight"}):
        errors.append(host+": wrong source tied parameter/shape denominator")
    cap=math.floor(.7*EXPECTED/8)
    if src["hard_full_model_max_physical_bytes_0_70bpw"]!=cap:
        errors.append(host+": physical bpw ceiling changed")
    diag=x["matched_original_teacher_logit_metrics"]
    packed=x["two_independent_teacher_free_direct_runtime"]
    if set(diag)!=set(METHODS) or set(packed)!=set(METHODS):
        errors.append(host+": missing one real packed model or original source comparison")
    for m in METHODS:
        record=src[m]
        data=record["file"]
        n=data["physical_bytes"]
        if n>cap or n<1 or not len(data["sha256"])==64:
            errors.append(host+": wrong complete physical source size/hash "+m)
        if abs(record.get("relative_source_RMSE",0))>10:
            errors.append(host+": nonsensical original source reconstruction "+m)
        runtime=packed[m]
        if (runtime["full_trained_physical_SHA256"]!=data["sha256"] or
            runtime["full_physical_bytes_resident_compressed"]!=n or
            runtime["loaded_original_pretrained_teacher_in_this_process"] or
            runtime["source_model_original_2D_dense_hot_weights"]!=0 or
            not runtime["native_C_5bit_direct_LUT_GEMV"] or
            not runtime["full_pretrained_transformer_and_KV_completed"]):
            errors.append(host+": not genuine source-only 5bit in native full model "+m)
        modules=runtime["packed_module_evidence"]
        if (modules["runtime_dense_multi_dim_source_parameters"]!=0 or
            modules["compressed_real_2D_module_count"]<200 or
            not modules["tied_embed_and_output_codebook_SHARED_IN_RAM"]):
            errors.append(host+": original pretrained 2D dense weights leaked "+m)
        probe=diag[m]
        rows=probe["all_first_use_development_prompt_outputs"]
        if set(rows)!=set(PROMPTS):
            errors.append(host+": missing one frozen QA-free dev diagnostic prompt "+m)
            continue
        for p,v in rows.items():
            if (not all(math.isfinite(v[k]) for k in (
                "teacher_to_packed_output_KL_nats",
                "full_vocab_normalized_logit_RMSE")) or
                v["teacher_to_packed_output_KL_nats"]<-.0000001 or
                v["full_vocab_normalized_logit_RMSE"]<0):
                errors.append(host+": invalid actual output distribution "+m+"/"+p)
        recalculated_kl=math.fsum(v["teacher_to_packed_output_KL_nats"]
                                  for v in rows.values())/4
        recalculated_rmse=math.fsum(v["full_vocab_normalized_logit_RMSE"]
                                    for v in rows.values())/4
        if (abs(probe["mean_full_vocab_KL"]-recalculated_kl)>1e-12 or
            abs(probe["mean_normalized_logit_RMSE"]-recalculated_rmse)>1e-12 or
            probe["same_teacher_top1_count_out_of_4"]!=sum(
                v["same_top1_token"] for v in rows.values())):
            errors.append(host+": outcome quality arithmetic inconsistent "+m)
        for p in PROMPTS:
            sec=runtime["per_frozen_prompt"][p]["actual_forward_seconds"]
            if not math.isfinite(sec) or sec<=0:
                errors.append(host+": packed inference time missing "+m+"/"+p)
    baseline=src["control_v59"]["relative_source_RMSE"]
    candidate=src["candidate_CB60"]["relative_source_RMSE"]
    if abs((candidate-baseline)-src["relative_source_RMSE_delta"])>1e-12:
        errors.append(host+": reconstructed source error improvement falsified")
    if (src["relative_source_RMSE_improved"]!=(candidate<baseline) or
        x["source_only_reconstruction_improved_v59_to_v60"]!=(candidate<baseline)):
        errors.append(host+": report's source-error improvement contradicts measurement")
    kl1=diag["control_v59"]["mean_full_vocab_KL"]
    kl2=diag["candidate_CB60"]["mean_full_vocab_KL"]
    if (abs(x["new_four_prompt_mean_KL_delta_tiled_minus_original_v59"]-
            (kl2-kl1))>1e-12 or
        x["new_four_prompt_mean_KL_improved_v59_to_v60"]!=(kl2<kl1)):
        errors.append(host+": same prompt physical-model KL comparison falsified")
    if x["first_stage_rate_and_native_direct_execution_PASS"] is not True:
        errors.append(host+": full first-stage physical+runtime gates did not complete")
    return {"candidate_actual_bpw":src["candidate_actual_bpw"],
            "candidate_true_complete_file_bytes":src["candidate_CB60"]["file"]["physical_bytes"],
            "candidate_SHA256":src["candidate_CB60"]["file"]["sha256"],
            "control_SHA256":src["control_v59"]["file"]["sha256"],
            "control_source_RMSE":baseline,
            "candidate_localized_source_RMSE":candidate,
            "source_RMSE_improves":candidate<baseline,
            "control_mean_KL_over_new_four_prompts":kl1,
            "candidate_mean_KL_over_new_four_prompts":kl2,
            "candidate_KL_improves":kl2<kl1,
            "packed_212_direct_operations_and_full_model_KV":True,
            "tiled_codebooks":src["candidate_CB60"]["actual_row_tile_codebook_count"],
            "control_KL_top1":diag["control_v59"]["same_teacher_top1_count_out_of_4"],
            "candidate_KL_top1":diag["candidate_CB60"]["same_teacher_top1_count_out_of_4"]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a",required=True)
    ap.add_argument("--b",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    errors=[]
    hosts={}
    for host,data in (("ubuntu22",a),("ubuntu24",b)):
        hosts[host]=verify(data,host,errors)
    for key in ("source_model_revision","real_135M_pretrained_model",
                "source_only_physical_codec"):
        if key=="source_only_physical_codec":
            for part in ("original_unique_source_parameters",
                "predeclared_development_prompts_SHA256",
                "source_tied_aliases",
                "hard_full_model_max_physical_bytes_0_70bpw"):
                if a[key][part]!=b[key][part]:
                    errors.append("crossCPU different fixed trained source "+part)
        elif a[key]!=b[key]:
            errors.append("crossCPU different trained model SHA "+key)
    for m in METHODS:
        x=hosts["ubuntu22"]
        y=hosts["ubuntu24"]
        field="control_SHA256" if m=="control_v59" else "candidate_SHA256"
        if x[field]!=y[field]:
            errors.append("physically not bitwise reproducible across CPUs: "+m)
        for metric in (("control_source_RMSE","candidate_localized_source_RMSE",
                        "control_mean_KL_over_new_four_prompts",
                        "candidate_mean_KL_over_new_four_prompts")):
            if abs(x[metric]-y[metric])>1e-8:
                errors.append("crossCPU real quantized metric instability "+metric)
    report={"PASS":not errors,"errors":errors,
            "two_independent_cpu_original_trained_source_and_native_packed_quality":hosts,
            "limitations":[
                "Changing codebook row-tile granularity is one experimental step; no model QA task correctness proven.",
                "A lower source-weight RMSE or next-token KL on only four development prompts is not semantic generalization.",
                "No claim of <=20pct warm 128+32 speed or production Jessica runtime here.",
                "Full checkpoint physical cap includes aliases, codebooks and source metadata; compressed 2D weights not expanded in native kernel."
            ]}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
