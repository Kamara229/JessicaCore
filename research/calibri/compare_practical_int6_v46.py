#!/usr/bin/env python3
"""Independent physical source, rate, quality and speed comparison for v46.

A negative science/SLA result is an honest valid experimental outcome,
whereas corruption, missing source or invalid measurement is a CI failure.
"""
import argparse, json, math
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a",required=True)
    ap.add_argument("--b",required=True)
    ap.add_argument("--output",required=True)
    x=ap.parse_args()
    a=json.loads(Path(x.a).read_text())
    b=json.loads(Path(x.b).read_text())
    errs=[]
    immutable=("experiment","source_model","model_revision","wiki_revision",
               "external_revision","original_source_state_SHA256",
               "all_parameter_count","two_dimensional_matrix_coefficients",
               "full_source_FP32_state_theoretical_bytes",
               "train_A_first2048_SHA256","train_B_disjoint2048_SHA256",
               "eval_token_SHA256","SLA")
    for key in immutable:
        if a[key]!=b[key]:
            errs.append("immutable source/train/test contract mismatch: "+key)
    packed={}
    for kind in ("raw","pre","post"):
        left=a["physical_checkpoint_files"][kind]
        right=b["physical_checkpoint_files"][kind]
        if left["physical_bytes"]!=right["physical_bytes"]:
            errs.append(kind+" physical length differs across CPU hosts")
        packed[kind]=left["physical_bytes"]
        if kind in ("raw","pre") and left["SHA256"]!=right["SHA256"]:
            errs.append(kind+" physically quantized full model differs across CPU hosts")
        if kind=="post":
            if left["total_checkpoint_plus_sidecar_bytes"]!=right["total_checkpoint_plus_sidecar_bytes"]:
                errs.append("true corrected full model file sizes differ across hosts")
            if left["all_parameter_effective_bpw"]>a["SLA"]["max_effective_disk_bpw_incl_real_metadata"]:
                errs.append("post corrected size violated 6.5 physical bpw")
        else:
            if left["effective_bpw"]>a["SLA"]["max_effective_disk_bpw_incl_real_metadata"]:
                errs.append(kind+" violated 6.5 physical bpw")
    variants=("original_teacher","full_INT6_raw","full_INT6_PRE","full_INT6_PRE_POST")
    domains=("WikiText2_DEV_known","ARC_Challenge_test_disjoint","Yelp_test_192_256_disjoint")
    measurements={}
    outcomes={}
    for variant in variants:
        measurements[variant]={}
        for host,item in (("ubuntu22",a),("ubuntu24",b)):
            v=item["variant_results"][variant]
            if set(v["ppl"])!=set(domains):
                errs.append(host+":"+variant+" missing held-out corpus")
            warm=v["warm_timing"]
            if not all(len(warm[key]["raw"])==7 and
                       all(math.isfinite(vv) and vv>0 for vv in warm[key]["raw"])
                       for key in ("prefill_ms","decode32_ms","total_prefill_and_decode32_ms")):
                errs.append(host+":"+variant+" malformed warm timing")
            if any(not math.isfinite(val["ppl"]) or val["ppl"]<=0
                   for val in v["ppl"].values()):
                errs.append(host+":"+variant+" invalid PPL")
            if variant!="original_teacher":
                if v["cold_loader"]["compressed_disk_bytes"]!=a["physical_checkpoint_files"][
                    "raw" if variant=="full_INT6_raw" else "pre"]["physical_bytes"]:
                    errs.append(host+":"+variant+" not actually loaded from compressed full state")
            measurements[variant][host]={
                "PPL":{d:v["ppl"][d]["ppl"] for d in domains},
                "prefill_ms":warm["prefill_ms"]["median"],
                "autoregressive_decode32_ms":warm["decode32_ms"]["median"],
                "total_latency_ms":warm["total_prefill_and_decode32_ms"]["median"],
                "ratio_to_original":v.get("SLA_warm_latency_20pct",{}).get(
                    "total_prefill_and_decode32_ms",{}).get("ratio_vs_uncompressed_teacher")}
        if variant!="original_teacher":
            outcomes[variant]={
                h:{"speed_20pct_pass":d["variant_results"][variant][
                    "SLA_warm_latency_20pct"]["total_prefill_and_decode32_ms"]["pass"],
                   "quality_5pct_both_new_pass":all(q["pass"] for q in d[
                    "variant_results"][variant]["SLA_quality_5pct_each_new_corpus"].values()),
                   "combined_pass":d["variant_results"][variant]["overall_practical_stage_gate"]}
                for h,d in (("ubuntu22",a),("ubuntu24",b))}
            outcomes[variant]["both_hosts_practical_gate"]=all(
                outcomes[variant][h]["combined_pass"] for h in ("ubuntu22","ubuntu24"))
    verdict={
        "verified_source_data_and_byte_integrity":not errs,
        "errors":errs,
        "original_full_state_parameters":a["all_parameter_count"],
        "physical_checkpoint_bytes":packed,
        "total_physical_PRE_POST_bytes":a["physical_checkpoint_files"]["post"][
            "total_checkpoint_plus_sidecar_bytes"],
        "full_model_real_warm_latency_and_PPL_each_host":measurements,
        "engineering_quality_and_20pct_speed_gate":outcomes,
        "limitations":[
            "Speed-gate PASS is reported only if both hosts satisfy ratio <=1.2 AND held-out PPL <=1.05x original on both samples.",
            "Unsuccessful quality or speed targets are not checksum failures and MUST remain visible.",
            "Warm runtime uses expanded FP32 matrices (speed-priority tier); full-model compressed-in-RAM inference is NOT established.",
            "Cold fresh from_config decode time is recorded without misleading download-dependent source cold comparison.",
            "Only 2 short fixed external samples, not production Jessica task accuracy."
        ]}
    dest=Path(x.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(verdict,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"integrity_pass":not errs,"errors":errs,
                      "real_physical_source_bytes":packed,
                      "speed_quality_outcomes":outcomes,
                      "measured":measurements},ensure_ascii=False),flush=True)
    if errs:
        raise SystemExit(1)


if __name__=="__main__":
    main()
