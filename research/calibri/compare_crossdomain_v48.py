#!/usr/bin/env python3
"""CALIBRI v48 cross-CPU source, physical, independent TRAIN gate verifier."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

MODES=("base_INT6_PRE","gradient_global","sse_module_size_matched")
CORPORA=("Yelp_new_test_320_384","ARCChallenge_new_validation_128_256")


def run():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a",required=True)
    ap.add_argument("--b",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    errors=[]
    for key in ("experiment","model","model_revision","wiki_revision",
                "AGNews_training_revision","AGNews_train_slice",
                "external_test_revisions","original_pretrained_full_state_SHA256",
                "train_A_token_sha256","train_B_token_sha256",
                "gate_AGNews_token_sha256","external_TEST_token_sha256",
                "source_full_model_coefficient_count",
                "max_total_all_coefficient_disk_bytes"):
        if a[key]!=b[key]:
            errors.append("cross-host model/source/train/test mismatch: "+key)
    per_host={}
    for host,d in (("ubuntu22",a),("ubuntu24",b)):
        bstate=d["original_frozen_CB46_complete_INT6_file"]
        if bstate["bytes"]!=10803661:
            errors.append(host+" changed original full pretrained physical source")
        patches=d["physical_CR47_correction_files"]
        if set(patches)!={"gradient_global","sse_module_size_matched"}:
            errors.append(host+" missing exact-byte physical controls")
        if patches["gradient_global"]["bytes"]!=patches["sse_module_size_matched"]["bytes"]:
            errors.append(host+" physical candidate sidecars not matched")
        selected=d["selection_one_shot"]["decision"]
        if selected not in MODES:
            errors.append(host+" gate selected unregistered method")
        for tag,source in patches.items():
            if source["rows"]<=0 or source["all_model_physical_bytes"]+15480>d[
                    "max_total_all_coefficient_disk_bytes"]:
                errors.append(host+":"+tag+" physical byte ceiling exceeded")
        total=d["selected_real_disk_bytes"]
        if total>d["max_total_all_coefficient_disk_bytes"]:
            errors.append(host+" chosen correction exceeded physical byte budget")
        if set(d["untouched_external_test_variants"])!=set(MODES):
            errors.append(host+" missing heldout variant results")
        if set(d["untouched_external_test_original_teacher"])!=set(CORPORA):
            errors.append(host+" missing heldout teacher results")
        for tag in MODES:
            if set(d["untouched_external_test_variants"][tag])!=set(CORPORA):
                errors.append(host+":"+tag+" missing new test PPL")
            for row in d["untouched_external_test_variants"][tag].values():
                if not math.isfinite(row["ppl"]) or row["ppl"]<=0:
                    errors.append(host+":"+tag+" invalid external PPL")
        t=d["true_warm_full_model_speed"]
        if not math.isfinite(t["chosen_div_teacher"]) or t["chosen_div_teacher"]<=0:
            errors.append(host+" invalid end-to-end full-model speed ratio")
        for variant in ("teacher","chosen"):
            for key in ("prefill_ms","decode32_ms","total_prefill_and_decode32_ms"):
                samples=t[variant][key]["raw"]
                if len(samples)!=7 or any(not math.isfinite(s) or s<=0 for s in samples):
                    errors.append(host+" missing 7 actual warm end-to-end timings")
        per_host[host]={
            "selected_by_train_only_gate":selected,
            "actual_selected_bpw":d["selected_real_disk_bpw"],
            "latency_ratio_to_original_teacher":t["chosen_div_teacher"],
            "speed_at_most_20pct":t["chosen_div_teacher"]<=1.20,
            "external_selected_vs_baseline_PPL":{
                name:d["untouched_external_test_variants"][selected][name]["ppl"]-
                     d["untouched_external_test_variants"]["base_INT6_PRE"][name]["ppl"]
                for name in CORPORA},
            "external_selected_vs_teacher_PPL":{
                name:d["untouched_external_test_variants"][selected][name]["ppl"]-
                     d["untouched_external_test_original_teacher"][name]["ppl"]
                for name in CORPORA}
        }
    if a["original_frozen_CB46_complete_INT6_file"]["sha256"]!=b[
            "original_frozen_CB46_complete_INT6_file"]["sha256"]:
        errors.append("original physically stored pretrained CB46 SHA not reproducible")
    cross_sidecar_sha={
        k:a["physical_CR47_correction_files"][k]["sha256"]==
          b["physical_CR47_correction_files"][k]["sha256"]
        for k in ("gradient_global","sse_module_size_matched")}
    if per_host["ubuntu22"]["selected_by_train_only_gate"]!=per_host[
        "ubuntu24"]["selected_by_train_only_gate"]:
        errors.append("TRAIN-only gate decision disagreed across independent CPU hosts")
    result={
        "physical_model_and_protocol_integrity_PASS":not errors,
        "errors":errors,"per_host":per_host,
        "full_CB46_real_bytes":a["original_frozen_CB46_complete_INT6_file"]["bytes"],
        "matched_CR47_correction_bytes":{h:d["physical_CR47_correction_files"][
            "sse_module_size_matched"]["bytes"] for h,d in (("ubuntu22",a),("ubuntu24",b))},
        "crosshost_sidecar_SHA_equal":cross_sidecar_sha,
        "notes":[
            "A TRAIN-only selected method may regress on new heldout corpora: this is scientific evidence, not CI failure.",
            "Speed tier hot-expands all 2D compressed weights to FP32 when loading model once; no compressed RAM claimed.",
            "Training-only gate uses previous WikiText TRAIN-B and first-use AGNews TRAIN; two 2048-token out-of-family disjoint tests are exploratory, not task accuracy.",
            "No production Jessica chat test is requested; no main/Render/Supabase/APK changes."]
    }
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"PASS":not errors,"errors":errors,"host_outcomes":per_host,
                      "sidecar_same_SHA":cross_sidecar_sha},ensure_ascii=False),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    run()
