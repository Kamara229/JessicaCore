#!/usr/bin/env python3
"""Independent v47 source integrity + exact physical row-SSE matched-control check.

Speed and quality regressions are real experimental outcomes, not CI errors.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

VARIANTS=("teacher","INT6_PRE_only","INT6_PRE_POST_affine",
          "INT6_gradient_recovery","INT6_SSE_module_matched",
          "INT6_SSE_global","INT6_gradient_recovery_POST_affine")
DOMAINS=("WikiText2_DEV_known","OpenBookQA_test_new",
         "QNLI_validation_new","Yelp_test_256_320_disjoint")
SIDECARS=("gradient_global","sse_module_size_matched","sse_global")


def run(args):
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    errors=[]
    immutables=("experiment","original_model","source_revision",
        "source_full_state_SHA256","train_dataset","wiki_revision",
        "external_revisions","trainA_SHA256","trainB_SHA256",
        "validation_test_SHA256","actual_pretrained_state_parameter_count",
        "true_max_total_physical_bytes","engineering_gate")
    for key in immutables:
        if a[key]!=b[key]:
            errors.append("original source or frozen evaluation SHA differs: "+key)
    for name,item in (("ubuntu22",a),("ubuntu24",b)):
        base=item["base_complete_CB46_checkpoint"]
        if base["bytes"]!=10803661:
            errors.append(name+" original pretrained complete INT6 bytes changed")
        if set(item["variants"])!=set(VARIANTS):
            errors.append(name+" missing prespecified full-model variant")
        parts=item["sparse_C47_correction_sidecars"]
        if set(parts)!=set(SIDECARS):
            errors.append(name+" missing physical sparse sidecars")
        grad=parts["gradient_global"]
        matched=parts["sse_module_size_matched"]
        if grad["bytes"]!=matched["bytes"]:
            errors.append(name+" MSE matched control differs in real physical bytes")
        if grad["selected_module_counts"]!=matched["selected_module_counts"]:
            errors.append(name+" MSE matched control differs in precision per module")
        for mode,patch in parts.items():
            if not (patch["selected_rows"]>0 and
                    patch["bytes"]+base["bytes"]+item["post_affine_sidecars"][
                        "gradient"]["bytes"]<=item["true_max_total_physical_bytes"]):
                errors.append(name+":"+mode+" violates all-in physical rate cap")
            if not 6.0<patch["total_bpw"]<6.5:
                errors.append(name+":"+mode+" invalid physical all-model bit rate")
        if item["post_affine_sidecars"]["gradient"]["bytes"]!=15480:
            errors.append(name+" changed frozen output-affine trained correction size")
        for variant in VARIANTS:
            row=(item["variants"][variant] if variant!="teacher"
                 else {"results":item["variants"][variant]})
            stats=row["results"]
            if set(stats["PPL"])!=set(DOMAINS):
                errors.append(name+":"+variant+" missing domain PPL")
            for domain,p in stats["PPL"].items():
                if not math.isfinite(p["ppl"]) or p["ppl"]<=0:
                    errors.append(name+":"+variant+" nonfinite evaluation PPL")
            for key in ("prefill_ms","decode32_ms","total_prefill_and_decode32_ms"):
                times=stats["latency"][key]["raw"]
                if len(times)!=7 or any(not math.isfinite(t) or t<=0 for t in times):
                    errors.append(name+":"+variant+" invalid 7 measured full-model timings")
            if variant!="teacher":
                load=row["checkpoint_loader"]
                if load["compressed_disk_bytes"]!=base["bytes"]:
                    errors.append(name+":"+variant+" not restored from physical full INT6 source")
    # Pretrained complete binary is unchanged by row-gradient optimization
    if a["base_complete_CB46_checkpoint"]["sha256"]!=b[
        "base_complete_CB46_checkpoint"]["sha256"]:
        errors.append("actual pretrained physical complete model SHA mismatch across CPUs")
    conclusions={}
    for variant in VARIANTS[1:]:
        conclusions[variant]={}
        for host,item in (("ubuntu22",a),("ubuntu24",b)):
            measure=item["variants"][variant]
            base=item["variants"]["INT6_PRE_only"]
            diff={d:measure["results"]["PPL"][d]["ppl"]-
                   base["results"]["PPL"][d]["ppl"] for d in DOMAINS}
            conclusions[variant][host]={
                "PPL_minus_frozen_PRE":diff,
                "real_full_model_warm_128plus32_latency_ratio":measure[
                    "latency_ratio_to_teacher"]["total_prefill_and_decode32_ms"],
                "speed_20pct_pass":measure["speed_20pct_pass"],
                "PPL_5pct_provisional_all_3_new_pass":measure["quality_5pct_all_new_pass"],
                "all_3_new_PPL_better_than_PRE":all(diff[d]<0 for d in DOMAINS[1:])}
        conclusions[variant]["speed_both_hosts"]=all(conclusions[variant][h][
            "speed_20pct_pass"] for h in ("ubuntu22","ubuntu24"))
        conclusions[variant]["quality_both_hosts"]=all(conclusions[variant][h][
            "PPL_5pct_provisional_all_3_new_pass"] for h in ("ubuntu22","ubuntu24"))
    verdict={
        "physical_integrity_and_measurement_PASS":not errors,
        "issues":errors,
        "full_source_state_SHA256":a["source_full_state_SHA256"],
        "real_pretrained_complete_INT6_CB46_bytes":a[
            "base_complete_CB46_checkpoint"]["bytes"],
        "sidecar_bytes_ubuntu22":{k:v["bytes"] for k,v in a[
            "sparse_C47_correction_sidecars"].items()},
        "sidecar_bytes_ubuntu24":{k:v["bytes"] for k,v in b[
            "sparse_C47_correction_sidecars"].items()},
        "crosshost_sidecar_SHA_same":{
            k:a["sparse_C47_correction_sidecars"][k]["sha256"]==
              b["sparse_C47_correction_sidecars"][k]["sha256"] for k in SIDECARS},
        "outcomes":conclusions,
        "conclusion_limits":[
            "A speed PASS applies to dense-hot-cache reconstruction after cold load, not compressed RAM or native packed GEMM.",
            "Exact-size matched control has same physical byte count and module counts as trained-gradient correction.",
            "A negative PPL is a scientific result and cannot be hidden by green CRC/integrity CI.",
            "Post-test retuning forbidden for all 3 external slices; study only small fixed 2048-token PPL excerpts, not Jessica accuracy."
        ]}
    p=Path(args.output)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(verdict,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"strict_integrity_PASS":not errors,"errors":errors,
        "bytes":verdict["sidecar_bytes_ubuntu22"],
        "outcomes":conclusions},ensure_ascii=False),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True)
    p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    run(p.parse_args())
