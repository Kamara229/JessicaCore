#!/usr/bin/env python3
"""CB61 independent full-trained-model physical rate/direct exec/syndrome verifier."""
from __future__ import annotations
import argparse,math,json
from pathlib import Path

PARAMETERS=134515008
METHODS=("base_CB59","CB61_activation_error_syndrome")
SAMPLES=tuple("p"+str(x) for x in range(8))


def audit(one,host,errors):
    p=one["real_complete_both_source_checkpoints"]
    runtime=one["teacher_free_native_packed_model_evidence"]
    result=one["eight_predeclared_new_output_diagnostics"]
    teacher=one["teacher_process_source"]
    limit=math.floor(.7*PARAMETERS/8)
    src=p["base_original_full_CB59_source"]
    fixed=p["candidate_CB61_full_single_crc_sha_file"]
    if p["unique_pretrained_source_parameter_count"]!=PARAMETERS:
        errors.append(host+" incorrect unique real source denominator")
    if p["hard_0_70bpw_checkpoint_max_bytes"]!=limit:
        errors.append(host+" user hard maximum storage budget changed")
    if src["bytes"]!=11229303 or len(src["sha256"])!=64:
        errors.append(host+" pretrained original source CB59 changed")
    if not (src["bytes"]<fixed["physical_bytes"]<=limit and
            0<fixed["selected_columns"] and
            len(fixed["sha256"])==64 and
            fixed["residual_FP16_column_payload_bytes"]>0 and
            fixed["actual_manifest_bytes"]>0):
        errors.append(host+" incomplete syndrome or physical 0.70bpw budget failed")
    if (abs(fixed["actual_effective_bpw"]-
            8*fixed["physical_bytes"]/PARAMETERS)>1e-12 or
        fixed["parent_CB59_bytes"]!=src["bytes"]):
        errors.append(host+" syndrome rate/accounting misrepresented")
    if (p["teacher_TRAIN_input_activation_squared_moments_source"][
           "trained_module_input_activation_columns"]<100 or
        p["teacher_TRAIN_input_activation_squared_moments_source"][
           "no_heldout_task_data_used"] is not True):
        errors.append(host+" missing genuine TRAIN teacher activation signal")
    if set(result)!=set(METHODS) or set(runtime)!=set(METHODS):
        errors.append(host+" no complete physical experiment for both variants")
    for method in METHODS:
        r=result[method]
        bound=runtime[method]
        expected=src if method=="base_CB59" else fixed
        expected_sha=expected["sha256"]
        expected_bytes=expected["bytes"] if method=="base_CB59" else expected["physical_bytes"]
        if (bound["physical_file_SHA256"]!=expected_sha or
            bound["physical_file_bytes"]!=expected_bytes or
            not bound["pretrained_teacher_was_NOT_loaded_this_process"] or
            bound["hot_dense_pretrained_original_2D_weight_count"]!=0 or
            not bound["full_real_pretrained_model_direct_and_KV_completed"] or
            not bound["native_base_C_5bit_GEMV_plus_optional_1D_error_syndromes"]):
            errors.append(host+" cannot demonstrate teacher-free native direct runtime "+method)
        mods=bound["all_212_packed_source_modules"]
        if (mods["compressed_real_2D_module_count"]<200 or
            mods["runtime_dense_multi_dim_source_parameters"]!=0 or
            not mods["tied_embed_and_output_codebook_SHARED_IN_RAM"]):
            errors.append(host+" dense trained W or tied alias leaked "+method)
        if method==METHODS[1] and (
            mods["number_of_actual_one_column_residual_hard_syndromes_applied"]<
            fixed["selected_columns"]):
            errors.append(host+" not all explicit paid residual columns executed")
        m=r["individual_eight_new_dev_prompt_diagnostics"]
        if set(m)!=set(SAMPLES):
            errors.append(host+" incomplete eight fixed first-use diagnostics "+method)
            continue
        mean_kl=math.fsum(v["full_vocab_teacher_to_compressed_KL_nats"] for v in m.values())/8
        mean_rmse=math.fsum(v["normalized_logit_RMSE"] for v in m.values())/8
        if (not math.isfinite(mean_kl) or mean_kl<0 or
            abs(mean_kl-r["mean_held_out_from_calibration_KL"])>1e-10 or
            abs(mean_rmse-r["mean_normalized_output_logit_RMSE"])>1e-10 or
            r["matched_teacher_next_top1_count"]!=sum(
                v["same_teacher_top1_token"] for v in m.values())):
            errors.append(host+" independent logit KL/count arithmetic fails "+method)
        for prompt in SAMPLES:
            t=bound["actual_direct_inference_timing_per_fresh_dev_prompt"][prompt]
            if not math.isfinite(t) or t<=0:
                errors.append(host+" missing real measured source direct model runtime")
    real_delta=result[METHODS[1]]["mean_held_out_from_calibration_KL"]-(
        result[METHODS[0]]["mean_held_out_from_calibration_KL"])
    if (abs(real_delta-one["CB61_minus_CB59_mean_original_output_KL"])>1e-10 or
        one["CB61_reduces_mean_original_output_KL"]!=(real_delta<0)):
        errors.append(host+" misleading quality gain")
    if not one["strict_full_model_rate_and_direct_inference_PASS"]:
        errors.append(host+" first-stage factual rate/native execution invalid")
    return {
        "actual_CB61_complete_physical_bytes":fixed["physical_bytes"],
        "actual_CB61_unique_bpw":fixed["actual_effective_bpw"],
        "actual_syndrome_columns":fixed["selected_columns"],
        "corrected_source_sha256":fixed["sha256"],
        "parent_base_sha256":src["sha256"],
        "eight_original_vs_base_mean_KL":result[METHODS[0]]["mean_held_out_from_calibration_KL"],
        "eight_original_vs_syndrome_mean_KL":result[METHODS[1]]["mean_held_out_from_calibration_KL"],
        "syndrome_effect_on_quality_better":real_delta<0,
        "original_teacher_top1_base":result[METHODS[0]]["matched_teacher_next_top1_count"],
        "original_teacher_top1_CB61":result[METHODS[1]]["matched_teacher_next_top1_count"],
        "real_5bit_native_and_no_dense_pretrained_2D":True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True)
    p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    errors=[]
    measurements={}
    for name,data in (("ubuntu22",a),("ubuntu24",b)):
        measurements[name]=audit(data,name,errors)
    for key in ("real_pretrained_model","pinned_revision"):
        if a[key]!=b[key]:
            errors.append("different original real trained model source "+key)
    pa=a["real_complete_both_source_checkpoints"]
    pb=b["real_complete_both_source_checkpoints"]
    if (pa["teacher_TRAIN_input_activation_squared_moments_source"]!=
        pb["teacher_TRAIN_input_activation_squared_moments_source"]):
        errors.append("different two CPU TRAIN calibration source/hash")
    same_parent=(measurements["ubuntu22"]["parent_base_sha256"]==
                 measurements["ubuntu24"]["parent_base_sha256"])
    same_corrected=(measurements["ubuntu22"]["corrected_source_sha256"]==
                    measurements["ubuntu24"]["corrected_source_sha256"])
    if not same_parent:errors.append("crossCPU CB59 base file not bitwise identical")
    if not same_corrected:errors.append("crossCPU CB61 syndrome correction not bitwise identical")
    for metric in ("eight_original_vs_base_mean_KL",
                   "eight_original_vs_syndrome_mean_KL"):
        if abs(measurements["ubuntu22"][metric]-measurements["ubuntu24"][metric])>1e-6:
            errors.append("crossCPU original next-token output KL differs materially "+metric)
    if measurements["ubuntu22"]["actual_syndrome_columns"]!=measurements[
        "ubuntu24"]["actual_syndrome_columns"]:
        errors.append("different paid residual selection count on independent CPUs")
    result={"PASS":not errors,"errors":errors,
        "original_0_7bpw_trained_pretrained_model_and_fec_analogy":measurements,
        "both_actual_entire_CB61_files_bit_identical":same_corrected,
        "original_CB59_parent_bit_identical":same_parent,
        "warnings":[
            "This experiment learns only a TRAIN activation-guided approximation to an explicitly stored quantization residual; it is not Shannon packet-erasure correction.",
            "Improvement in 8 development prompt logit KL is NOT enough to prove nonfatal semantic loss, utility, or quality preservation.",
            "Only direct unoptimized CPU native kernels; no 20% speed requirement verified in this pass.",
            "No pretrained original W dense matrix constructed by packed student runtime."
        ]}
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if errors:raise SystemExit(1)


if __name__=="__main__":
    main()
