#!/usr/bin/env python3
"""CALIBRI v39 joint-full-model-TRAIN-NLL cross-host quality/physical verifier.

Never reject negative PPL as a software failure. TRAIN-only sensitivity
float diagnostics may vary at the last digits between CPU models; exact
model source, token, row assignments, file bytes and physical SHA do not.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

VARIANTS=("local_3p5bpw_baseline","local_3p5bpw_calibrated",
          "global_train_greedy_uncalibrated","global_train_greedy_calibrated",
          "INT4_overbudget_reference","joint_TRAIN_NLL_uncalibrated",
          "joint_TRAIN_NLL_calibrated")
CONFIGURATIONS=("source_fp32_original",)+VARIANTS
CORPORA=("WikiText2_development","NEW_SCIQ_validation")
INVARIANTS=("experiment","model","model_sha","datasets","token_sha256",
            "token_counts","source_MLP_coefficient_count",
            "source_MLP_FP16_theoretical_bytes","frozen_global_limit_bytes")


def audit(name,record):
    bad=[]
    files=record.get("actual_physical_bundles",{})
    matrices=record.get("per_matrix",{})
    if set(files)!=set(VARIANTS) or len(matrices)!=12:
        bad.append(name+".missing_physical_files_or_12_modules")
        return bad
    total=record.get("source_MLP_coefficient_count",0)
    limit=record.get("frozen_global_limit_bytes",0)
    if total!=786432 or limit!=343840:
        bad.append(name+".changed_predefined_storage_budget")
    for variant in VARIANTS:
        d=files[variant]
        size=d.get("actual_disk_bytes",0)
        if (size<=0 or not d.get("sha256") or d.get("matrix_count")!=12
                or d.get("effective_bits_per_MLP_weight")!=8*size/total):
            bad.append(name+"."+variant+".bad_physical_metadata")
    local=files["local_3p5bpw_baseline"]["actual_disk_bytes"]
    local_cal=files["local_3p5bpw_calibrated"]["actual_disk_bytes"]
    greedy=files["global_train_greedy_uncalibrated"]["actual_disk_bytes"]
    greedy_cal=files["global_train_greedy_calibrated"]["actual_disk_bytes"]
    int4=files["INT4_overbudget_reference"]["actual_disk_bytes"]
    joint=files["joint_TRAIN_NLL_uncalibrated"]["actual_disk_bytes"]
    joint_cal=files["joint_TRAIN_NLL_calibrated"]["actual_disk_bytes"]
    if not (local==local_cal==joint==joint_cal==343840 and
            0<greedy==greedy_cal<=343840<int4):
        bad.append(name+".global_vs_local_equal_physical_budget_contract")
    for key,d in matrices.items():
        dims=d.get("shape",[])
        if (dims not in ([128,512],[512,128])
                or d.get("local_bytes",0)<=0 or d.get("local_bytes",28673)>28672
                or d.get("local_calibrated_train_SSE_ratio",1.)>1.000001
                or d.get("global_calibrated_train_SSE_ratio",1.)>1.000001
                or d.get("global_bytes",0)<1
                or d.get("global_4bit_row_count",-1)>dims[0]
                or d.get("global_4bit_row_count",-1)<0
                or not d.get("source_SHA")):
            bad.append(name+"."+key+".source_shape_or_train_calibration")
    quality=record.get("full_model_PPL_with_all12_MLP_restored",{})
    if set(quality)!=set(CONFIGURATIONS):
        bad.append(name+".missing_PPL_variant")
    else:
        for variant,d in quality.items():
            if set(d)!=set(CORPORA):
                bad.append(name+"."+variant+".missing_evaluation")
            elif any((not math.isfinite(v.get("ppl",float("nan")))
                      or v["ppl"]<=0 or v.get("tokens",0)<=0) for v in d.values()):
                bad.append(name+"."+variant+".nonfinite_PPL")
    train_meta=record.get("train_only_global_allocation",{})
    if (train_meta.get("global_fixed_max_bytes")!=343840
        or train_meta.get("greedy_used_bytes")!=greedy):
        bad.append(name+".training_selection_budget_mismatch")
    bounds=train_meta.get("bounded_precisions_by_module",{})
    if set(bounds)!=set(matrices) or train_meta.get("floor_fraction")!=0.75 or train_meta.get("ceiling_fraction")!=1.25:
        bad.append(name+".precision_floor_ceil_manifest_missing")
    else:
        for module,limit in bounds.items():
            actual=matrices[module].get("global_4bit_row_count")
            local=matrices[module].get("local_4bit_row_count")
            if (actual!=limit.get("final_4bit_rows")
                or local!=limit.get("local_4bit_rows")
                or not (limit.get("minimum_4bit_rows",9999)<=actual<=limit.get("maximum_4bit_rows",-1))
                or not (0<limit.get("minimum_4bit_rows",0)<=local<=limit.get("maximum_4bit_rows",-1))):
                bad.append(name+"."+module+".protected_row_floors_violated")
    joint_meta=record.get("train_only_joint_model_NLL_search",{})
    if (joint_meta.get("preregistered_max_rounds")!=2 or
            joint_meta.get("preregistered_max_probes_per_round")!=20 or
            joint_meta.get("actual_physical_binary_bytes")!=343840 or
            not joint_meta.get("no_development_or_test_quality_used_in_selection",False) or
            joint_meta.get("selected_joint_train_NLL",float("inf")) >
                 joint_meta.get("original_baseline_joint_train_NLL",float("-inf"))+1e-8):
        bad.append(name+".joint_train_selection_or_byte_contract")
    protected=joint_meta.get("precision_row_bounds",{})
    if set(protected)!=set(matrices):
        bad.append(name+".joint_row_bounds_missing")
    else:
        for mod,bnd in protected.items():
            chosen=matrices[mod].get("joint_selected_4bit_row_count")
            if (chosen!=bnd.get("selected_4bit") or
                    bnd.get("local_4bit")!=matrices[mod].get("local_4bit_row_count") or
                    not bnd.get("floor_4bit",9999)<=chosen<=bnd.get("ceiling_4bit",-1)):
                bad.append(name+"."+mod+".joint_row_bound_violated")
    return bad


def run():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True);p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    aa=json.loads(Path(args.a).read_text(encoding="utf-8"))
    bb=json.loads(Path(args.b).read_text(encoding="utf-8"))
    issues=[k for k in INVARIANTS if aa.get(k)!=bb.get(k)]
    issues+=audit("ubuntu_22_04",aa)+audit("ubuntu_24_04",bb)
    am=aa.get("per_matrix",{})
    bm=bb.get("per_matrix",{})
    if set(am)!=set(bm):
        issues.append("module_names_differ")
    else:
        for n in am:
            for key in ("shape","source_SHA","local_bytes","global_bytes",
                        "global_4bit_row_count","local_4bit_row_count","joint_selected_4bit_row_count"):
                if am[n].get(key)!=bm[n].get(key):
                    issues.append(n+"."+key+".crosshost_mismatch")
    f1=aa.get("actual_physical_bundles",{})
    f2=bb.get("actual_physical_bundles",{})
    same_physical=set(f1)==set(f2) and all(
        f1[k].get("sha256")==f2[k].get("sha256") and
        f1[k].get("actual_disk_bytes")==f2[k].get("actual_disk_bytes")
        for k in f1)
    if not same_physical:
        issues.append("physical_bundle_SHA_differs_across_hosts")
    q1=aa.get("full_model_PPL_with_all12_MLP_restored",{})
    q2=bb.get("full_model_PPL_with_all12_MLP_restored",{})
    same_ppl=q1==q2
    report={
        "experiment":"CALIBRI_v39_jnt_model_train_NLL_real_MLP_crosshost_new_SciQ",
        "source_and_physical_integrity_pass":not issues,
        "issues":issues,
        "physical_bundle_SHA_and_bytes_equal_between_hosts":same_physical,
        "full_model_PPL_equal_between_hosts":same_ppl,
        "first_use_SciQ_validation":aa.get("datasets",{}).get("fresh_SciQ",{}),
        "results":{},
        "limitations":[
            "Model-level row-bit optimization is accepted by joint TRAIN NLL across all 12 calibrated modules; no validation PPL in training.",
            "SciQ science-question validation is first-use; single 2048-token subset is exploratory only.",
            "Only 12 MLP weight modules are packed, not the deployable original model.",
            "All PPLs evaluated from restored dense weights, no packed native speed.",
        ],
    }
    for name,d in (("ubuntu_22_04",aa),("ubuntu_24_04",bb)):
        report["results"][name]={
            "physical_bytes":{k:v["actual_disk_bytes"] for k,v in
                              d["actual_physical_bundles"].items()},
            "full_model_PPL":{k:{corpus:score["ppl"] for corpus,score in value.items()}
                              for k,value in d["full_model_PPL_with_all12_MLP_restored"].items()},
            "quality_delta_global_minus_local_CALIBRATED":
                d["quality_delta_global_minus_local_CALIBRATED"],
            "quality_delta_joint_minus_local_CALIBRATED":
                d["quality_delta_joint_minus_local_CALIBRATED"],
            "joint_train_NLL_before_after":{
                "before":d["train_only_joint_model_NLL_search"]["original_baseline_joint_train_NLL"],
                "after":d["train_only_joint_model_NLL_search"]["selected_joint_train_NLL"],
                "rounds":d["train_only_joint_model_NLL_search"]["rounds_completed"],
            },
            "global_allocation_by_module":{
                k:v["global_4bit_row_count"] for k,v in d["per_matrix"].items()},
            "train_only_sensitivity_multipliers":
                d["train_only_global_allocation"]["train_only_clipped_layer_multipliers"],
        }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "integrity_pass":not issues,
        "physical_same_SHA":same_physical,
        "PPL_same":same_ppl,
        "issues":issues,
        "results":report["results"],
    },ensure_ascii=False),flush=True)
    if issues:
        raise SystemExit(2)


if __name__=="__main__":
    run()
