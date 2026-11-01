#!/usr/bin/env python3
"""Independent integrity check of physical bytes and pretrained PPL at 3.5bpw.

Reject source changes and binary disagreement, not negative quality outcomes.
"""
import argparse
import json
from pathlib import Path

FROZEN=("experiment","model","model_revision","corpus","corpus_revision",
        "source_trained_projection","projection_shape","source_weight_sha256",
        "token_sha256","training_tokens","validation_tokens","evaluation_context",
        "material_new_byte_ceiling","exact_byte_cap_bits_per_weight_including_overhead",
        "selection_rule","selected_3p5bpw_train_only")
EXPECTED={
    "int3_rowwise_low_rate",
    "rowwise_1x512_mixed3_4",
    "tiled_16x32_mixed3_4",
    "tiled_8x64_mixed3_4",
}


def check(name,result):
    issues=[]
    if set(result.get("physical_packed_formats",{}))!=EXPECTED:
        issues.append(name+".candidate_list_changed")
    if result.get("material_new_byte_ceiling")!=28672:
        issues.append(name+".byte_cap_changed")
    if result.get("exact_byte_cap_bits_per_weight_including_overhead")!=3.5:
        issues.append(name+".bpw_cap_changed")
    variants=result.get("physical_packed_formats",{})
    for variant,item in variants.items():
        if (item.get("physical_file_bytes",0)<=0 or
            item.get("physical_file_bytes",28673)>28672 or
            item.get("actual_3bit_tile_count",0)+item.get("actual_4bit_tile_count",0)!=128 or
            not item.get("physical_file_sha256")):
            issues.append(name+"."+variant+".invalid_binary")
    if variants.get("int3_rowwise_low_rate",{}).get("actual_4bit_tile_count")!=0:
        issues.append(name+".int3_ref_not_all3")
    if result.get("selected_3p5bpw_train_only") not in EXPECTED-{"int3_rowwise_low_rate"}:
        issues.append(name+".no_train_selected_3p5bpw_candidate")
    scores=result.get("single_projection_substituted_full_model_PPL",{})
    if set(scores)!=EXPECTED|{"int4_over_budget_reference"}:
        issues.append(name+".quality_measurements_incomplete")
    if not (result.get("int4_over_budget_reference_bytes",0)>28672):
        issues.append(name+".reference_int4_not_over_budget")
    return issues


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--first",required=True)
    p.add_argument("--second",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    records={
        "ubuntu_22_04":json.loads(Path(a.first).read_text(encoding="utf-8")),
        "ubuntu_24_04":json.loads(Path(a.second).read_text(encoding="utf-8")),
    }
    x,y=records.values()
    errors=[key for key in FROZEN if x.get(key)!=y.get(key)]
    for name,doc in records.items():
        errors+=check(name,doc)
    same_binary={
        key:x["physical_packed_formats"][key]["physical_file_sha256"]==
            y["physical_packed_formats"][key]["physical_file_sha256"]
        for key in EXPECTED
    }
    for key,passed in same_binary.items():
        if not passed:errors.append(key+".crosshost_physical_sha256_mismatch")
    results={}
    for name,doc in records.items():
        results[name]={
            "source_PPL":doc["source_PPL_validation"]["ppl"],
            "int4_overbudget_reference_bytes":doc["int4_over_budget_reference_bytes"],
            "int4_overbudget_reference_PPL":doc["single_projection_substituted_full_model_PPL"]["int4_over_budget_reference"]["ppl"],
            "cap_3p5bpw_bytes":doc["material_new_byte_ceiling"],
            "actual_bytes":{k:v["physical_file_bytes"]
                            for k,v in doc["physical_packed_formats"].items()},
            "actual_bpw":{k:v["bits_per_weight_including_all_overhead"]
                          for k,v in doc["physical_packed_formats"].items()},
            "quality_PPL":{k:v["ppl"]
                           for k,v in doc["single_projection_substituted_full_model_PPL"].items()},
            "selected_train_only":doc["selected_3p5bpw_train_only"],
        }
    quality_same=(
        results["ubuntu_22_04"]["quality_PPL"]==
        results["ubuntu_24_04"]["quality_PPL"])
    report={
        "experiment":"CALIBRI_v34_paired_CPU_strict_3p5bpw_reproducibility",
        "integrity_pass":not errors,
        "integrity_errors":errors,
        "physical_bytes_identical_between_hosts":all(same_binary.values()),
        "physical_binary_SHA256_equal_by_variant":same_binary,
        "PPL_exact_equal_between_hosts":quality_same,
        "results":results,
        "limitations":"One Pythia-14m MLP matrix, not compressed full model; validation reused across research; PPL on decoded dense W, no packed native speed; INT4 is over 3.5bpw budget.",
    }
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "integrity_pass":not errors,
        "PPL_exact_equal":quality_same,
        "physical_SHA_equal":all(same_binary.values()),
        "results":results,
        "integrity_errors":errors,
    },ensure_ascii=False),flush=True)
    if errors:raise SystemExit(2)


if __name__=="__main__":
    main()
