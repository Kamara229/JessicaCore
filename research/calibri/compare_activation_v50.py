#!/usr/bin/env python3
"""Independent CPU reproducibility proof for CALIBRI v50.

Sources and physically saved CB46/CR47 binaries MUST match bit for bit
across CPUs. Quality gains and <=20%-speed are measured and reported
as science, not accidentally confused with integrity.
"""
import argparse
import json
import math
from pathlib import Path

METHODS=("base_int6_pre","sse_balanced_12pct","activation_matched_MLP")
CORPORA=("Yelp_test_448_512","OpenBookQA_test_256_384",
         "QNLI_validation_256_384")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True)
    p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    errors=[]
    fixed_keys=(
        "experiment","real_pretrained_model","source_revision",
        "source_checkpoint_fp32_SHA256","source_all_parameter_count",
        "wiki_revision","ag_news_revision","external_revision_by_domain",
        "WikiText_TRAIN_A_SHA256","WikiText_TRAIN_B_SHA256",
        "AGNews_TRAIN_256_384_SHA256","external_untouched_test_SHA256"
    )
    for key in fixed_keys:
        if a[key]!=b[key]:
            errors.append("original trained full model/token source differs: "+key)
    names=("ubuntu22","ubuntu24")
    result={}
    for host,report in zip(names,(a,b)):
        if report["base_INT6_complete_source"]["bytes"]!=10803661:
            errors.append(host+": original pretrained fully physical INT6 bytes changed")
        proof=report["deterministic_row_budget"]
        if set(proof["plans"])!=set(METHODS[1:]):
            errors.append(host+": incomplete precision allocation candidates")
        patches=proof["plans"]
        for method in METHODS[1:]:
            item=patches[method]
            if not (0<item["bytes"] and item["selected_real_rows"]>0):
                errors.append(host+": invalid physical residual row manifest "+method)
            if (item["bytes"]+report["base_INT6_complete_source"]["bytes"]+
                proof["reserved_other_correction_bytes"]>
                proof["max_all_model_plus_all_correction_bytes"]):
                errors.append(host+": violation of complete model 6.50bpw ceiling "+method)
        selected=report["one_shot_train_only_gate"]["chosen"]
        if selected not in METHODS:
            errors.append(host+": unauthorized one-shot TRAIN gate candidate")
        if set(report["all_candidates_untouched_test_PPL"])!=set(METHODS):
            errors.append(host+": missing heldout quality controls")
        if set(report["original_teacher_untouched_test_PPL"])!=set(CORPORA):
            errors.append(host+": missing original pretrained heldout PPL")
        for name in METHODS:
            if set(report["all_candidates_untouched_test_PPL"][name])!=set(CORPORA):
                errors.append(host+": missing corpora for "+name)
            if any(not math.isfinite(r["ppl"]) or r["ppl"]<=0
                   for r in report["all_candidates_untouched_test_PPL"][name].values()):
                errors.append(host+": nonfinite PPL "+name)
        speed=report["actual_full_model_warm_speed"]
        if not math.isfinite(speed["chosen_divided_by_teacher"]) or speed[
            "chosen_divided_by_teacher"]<=0:
            errors.append(host+": invalid warm speed ratio")
        for variant in ("original_teacher","chosen"):
            for measure in ("prefill_ms","decode32_ms","total_prefill_and_decode32_ms"):
                timed=speed[variant][measure]["raw"]
                if len(timed)!=7 or any(not math.isfinite(v) or v<=0 for v in timed):
                    errors.append(host+": missing 7 actual end-to-end latency repeats")
        result[host]={
            "TRAIN_only_selected":selected,
            "actual_selected_bpw":report["selected_actual_bpw"],
            "warm_selected_div_teacher":speed["chosen_divided_by_teacher"],
            "speed_penalty_le20pct":speed["passes_plus20pct_user_limit"],
            "heldout_quality_selected_minus_original_pre":{
                name:report["all_candidates_untouched_test_PPL"][selected][name]["ppl"]-
                     report["all_candidates_untouched_test_PPL"]["base_int6_pre"][name]["ppl"]
                for name in CORPORA},
            "heldout_quality_selected_minus_teacher":{
                name:report["all_candidates_untouched_test_PPL"][selected][name]["ppl"]-
                     report["original_teacher_untouched_test_PPL"][name]["ppl"]
                for name in CORPORA},
            "physically_stored_residual_sidecar_bytes":{
                m:patches[m]["bytes"] for m in METHODS[1:]}
        }
    # v50 is EXACT matched physical bytes and per-tensor precision quotas;
    # differences between scores must not be conflated with storage budget.
    for host,report in zip(("ubuntu22","ubuntu24"),(a,b)):
        variants=report["deterministic_row_budget"]["plans"]
        baseline=variants["sse_balanced_12pct"]
        adaptive=variants["activation_matched_MLP"]
        if baseline["bytes"]!=adaptive["bytes"]:
            errors.append(host+": activation-aware correction physical bytes differ from frozen v49 control")
        if baseline["selected_real_rows"]!=adaptive["selected_real_rows"]:
            errors.append(host+": changed number of row repairs")
        if baseline["module_counts"]!=adaptive["module_counts"]:
            errors.append(host+": changed per-layer precision allocation quotas")
    if a["base_INT6_complete_source"]["sha256"]!=b[
        "base_INT6_complete_source"]["sha256"]:
        errors.append("full pretrained compressed INT6 CB46 source is NOT bitwise reproducible")
    crosshost_patch={}
    for method in METHODS[1:]:
        x=a["deterministic_row_budget"]["plans"][method]
        y=b["deterministic_row_budget"]["plans"][method]
        crosshost_patch[method]={"identical_bytes":x["bytes"]==y["bytes"],
                                 "identical_SHA256":x["sha256"]==y["sha256"],
                                 "same_row_counts":x["selected_real_rows"]==y["selected_real_rows"],
                                 "same_module_allocation":x["module_counts"]==y["module_counts"]}
        if not all(crosshost_patch[method].values()):
            errors.append("deterministic correction not bit-identical across independent CPUs: "+method)
    if result["ubuntu22"]["TRAIN_only_selected"]!=result["ubuntu24"]["TRAIN_only_selected"]:
        errors.append("identical TRAIN-only accept/reject decision failed")
    final={
        "physically_deterministic_reconstruction_integrity_PASS":not errors,
        "errors":errors,
        "complete_source_bytes":a["base_INT6_complete_source"]["bytes"],
        "correction_crosshost_bitwise_SHA":crosshost_patch,
        "both_real_CPU_outcomes":result,
        "limits":[
            "Negative speed or PPL means the quality-speed target failed, not a physical file-integrity failure.",
            "Full pretrained model is INT6 on disk, but each running CPU model has full reconstructed FP32 hot cache.",
            "External PPL only on three 2048-token unseen record slices of public datasets, not production Jessica task quality.",
            "No Jessica ordinary chat, APK, backend deployment, Supabase or Render modifications."
        ]}
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(final,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"PASS":not errors,"errors":errors,"physical":crosshost_patch,
                      "outcomes":result},ensure_ascii=False),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
