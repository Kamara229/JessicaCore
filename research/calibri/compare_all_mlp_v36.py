#!/usr/bin/env python3
"""Verify both true 12-matrix MLP byte bundles and all-corpus PPL across hosts.

Fail on missing source, token, shape, corrupted physical artifacts or
violated strict same-byte cap. Never classify worse held-out perplexity
as a software failure or conceal such negative measurements.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CHECK_KEYS=(
    "experiment","model","model_revision","datasets","token_sha256",
    "tokens","source_total_parameters","source_MLP_projection_weights",
    "source_MLP_FP16_reference_bytes_not_saved",
    "physical_global_MLP_3p5bpw_byte_budget_with_bundle",
    "per_source_MLP_module","per_row_bits_identical_in_calibrated",
    "calibration",
)
LABELS=("uncalibrated_3p5bpw_all12","calibrated_3p5bpw_all12",
        "INT4_above_budget_all12")
CORPORA=("WikiText2_dev_validation","AG_NEWS_NEW_test")
VARIANTS=("original_source_model",)+LABELS

def summarize(record,name):
    errors=[]
    blobs=record.get("physical_bundled_MLPs",{})
    module=record.get("per_source_MLP_module",{})
    if len(module)!=12 or set(blobs)!=set(LABELS):
        errors.append(name+".missing_module_or_physical_variant")
    total=record.get("source_MLP_projection_weights")
    cap=record.get("physical_global_MLP_3p5bpw_byte_budget_with_bundle")
    if total!=786432 or cap!=344064:
        errors.append(name+".incorrect_global_coefficient_budget")
    if cap and blobs:
        first=blobs.get(LABELS[0],{}).get("actual_disk_bytes")
        second=blobs.get(LABELS[1],{}).get("actual_disk_bytes")
        high=blobs.get(LABELS[2],{}).get("actual_disk_bytes")
        if not (isinstance(first,int) and isinstance(second,int)
                and isinstance(high,int) and first==second and
                0<first<=cap<high):
            errors.append(name+".physical_not_same_rate_or_highrate_not_overbudget")
        for key in LABELS:
            data=blobs.get(key,{})
            if (data.get("matrix_count")!=12 or not data.get("sha256")
                or data.get("effective_bits_per_MLP_weight_including_bundling")!=
                   8*data.get("actual_disk_bytes",0)/total):
                errors.append(name+"."+key+".binary_metadata_inconsistent")
    for key,stats in module.items():
        if (stats.get("shape") not in ([128,512],[512,128])
                or stats.get("rowwise_3p5bpw_bytes")!=stats.get("calibrated_same_byte_size")
                or stats.get("rowwise_3p5bpw_bytes",0)>28672
                or stats.get("bitwidth_rows",{}).get("INT3",0)+
                   stats.get("bitwidth_rows",{}).get("INT4",0)!=stats.get("shape",[0])[0]):
            errors.append(name+"."+key+".module_shape_bytes_or_row_bit_count")
    quality=record.get("full_model_dense_restored_PPL",{})
    if set(quality)!=set(VARIANTS):
        errors.append(name+".missing_PPL_variants")
    else:
        for variant,per_corpus in quality.items():
            if set(per_corpus)!=set(CORPORA):
                errors.append(name+"."+variant+".missing_corpus")
            for item in per_corpus.values():
                if not (item.get("ppl",0)>0 and item.get("tokens",0)>0):
                    errors.append(name+"."+variant+".invalid_PPL")
    return errors,{
        "MLP_merged_actual_bytes":{k:v["actual_disk_bytes"] for k,v in blobs.items()},
        "MLP_merged_SHA256":{k:v["sha256"] for k,v in blobs.items()},
        "MLP_actual_bpw":{k:v["effective_bits_per_MLP_weight_including_bundling"]
                          for k,v in blobs.items()},
        "source_MLP_FP16_reference_bytes_not_saved":record.get("source_MLP_FP16_reference_bytes_not_saved"),
        "PPL":{name:{c:score["ppl"] for c,score in corpora.items()}
               for name,corpora in quality.items()},
        "module_train_SSE_ratios":{n:v["train_output_SSE_calibrated_divided_by_baseline"]
                                   for n,v in module.items()},
        "relative_PPL_change":record.get("relative_quality"),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--first",required=True)
    p.add_argument("--second",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    first=json.loads(Path(args.first).read_text(encoding="utf-8"))
    second=json.loads(Path(args.second).read_text(encoding="utf-8"))
    mismatch=[k for k in CHECK_KEYS if first.get(k)!=second.get(k)]
    a_err,a=summarize(first,"ubuntu_22_04")
    b_err,b=summarize(second,"ubuntu_24_04")
    mismatch+=a_err+b_err
    same_sha=(a["MLP_merged_SHA256"]==b["MLP_merged_SHA256"])
    if not same_sha:
        mismatch.append("bundled_physical_sha256_differs_across_hosts")
    same_ppl=a["PPL"]==b["PPL"]
    report={
        "experiment":"CALIBRI_v36_all12_physical_Pythia_AG_NEWS_independent_CPU",
        "source_and_physical_integrity_pass":not mismatch,
        "errors":sorted(set(mismatch)),
        "all_three_full_MLP_bundle_SHA256_same_between_hosts":same_sha,
        "all_four_model_variant_PPL_exactly_same_between_hosts":same_ppl,
        "actual_results":{"ubuntu_22_04":a,"ubuntu_24_04":b},
        "limits":[
            "AG News is one new predeclared cross-domain TEST; not a guarantee of future generalization.",
            "Only 12 MLP modules physically serialized; rest of model absent from bundle.",
            "Full model PPL uses restored dense weights from true compressed physical bytes.",
            "No native compressed inference, throughput or QMoE 0.8-bpw result.",
        ],
    }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "source_and_physical_integrity_pass":not mismatch,
        "same_bytes_SHA":same_sha,
        "same_PPL":same_ppl,
        "errors":mismatch,
        "results":{k:{
            "physical_bytes":v["MLP_merged_actual_bytes"],
            "actual_bpw":v["MLP_actual_bpw"],
            "PPL":v["PPL"],
        } for k,v in report["actual_results"].items()},
    },ensure_ascii=False),flush=True)
    if mismatch:
        raise SystemExit(2)


if __name__=="__main__":
    main()
