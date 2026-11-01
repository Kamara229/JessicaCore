#!/usr/bin/env python3
"""CALIBRI v40 physical source and true 256/1024 TRAIN experiment comparator.

Only immutable checkpoint and physical hashes must match across hosts.
CPU-sensitive TRAIN NLL proxies are documented but not required to be
bitwise equal if selected physical bitstreams still match.
Negative PPL changes are science, NOT CI failure.
"""
import argparse
import json
import math
from pathlib import Path

TRAIN_SIZES=(256,1024)
PHYSICAL_NAMES=tuple(
    f"train{n}_{kind}" for n in TRAIN_SIZES for kind in
    ("local_uncalibrated","local_calibrated","joint_calibrated")
)+("INT4_overbudget_reference",)
OUTPUT_KEYS=("original_source_teacher",)+PHYSICAL_NAMES
CORPORA=("WikiText2_development","NEW_IMDB_test")
STRICT_SOURCE=(
    "experiment","pinned_source_model","pinned_model_revision",
    "original_MLP_weights","source_MLP_weights_SHA256",
    "training_WikiText_dataset_revision","new_IMDB_dataset_revision",
    "corpus_test","corpus_test_reviews","train_only_full_1024_token_SHA256",
    "dev_WikiText_token_SHA256","new_IMDB_test_token_SHA256",
    "frozen_context","physical_bytes_cap",
    "new_corpus_first_loaded_after_training_physical_lock",
)

def audit(who,doc):
    errors=[]
    if any(k not in doc for k in STRICT_SOURCE):
        errors.append(who+".missing_source_metadata")
    if (doc.get("original_MLP_weights")!=786432 or
            len(doc.get("source_MLP_weights_SHA256",{}))!=12 or
            doc.get("physical_bytes_cap")!=343840 or
            not doc.get("new_corpus_first_loaded_after_training_physical_lock")):
        errors.append(who+".changed_source_or_test_blinding")
    physical=doc.get("physical_MLP_files",{})
    if set(physical)!=set(PHYSICAL_NAMES):
        errors.append(who+".wrong_number_of_physical_versions")
    else:
        for name,p in physical.items():
            b=p.get("disk_bytes")
            if (not isinstance(b,int) or b<=0 or
                    p.get("actual_number_of_source_MLP_matrices")!=12 or
                    not p.get("sha256") or
                    p.get("bits_per_MLP_parameter")!=8*b/786432):
                errors.append(who+"."+name+".invalid_binary")
            if name!="INT4_overbudget_reference" and b!=343840:
                errors.append(who+"."+name+".not_equal_physical_bytes")
            if name=="INT4_overbudget_reference" and b<=343840:
                errors.append(who+".overbudget_INT4_reference_not_labeled")
    training=doc.get("training_choices",{})
    if set(training)!={"train256","train1024"}:
        errors.append(who+".training_arm_missing")
    else:
        for n in TRAIN_SIZES:
            arm=training["train"+str(n)]
            meta=arm.get("selection",{})
            if (arm.get("token_count")!=n or not arm.get("token_sha256") or
                    meta.get("number_of_module_matrices")!=12 or
                    meta.get("actual_physical_binary_bytes")!=343840 or
                    meta.get("preregistered_max_rounds")!=2 or
                    meta.get("preregistered_max_probes_per_round")!=20 or
                    not meta.get("no_development_or_test_quality_used_in_selection") or
                    not math.isfinite(arm.get("train_NLL_joint_selected",float("nan"))) or
                    arm.get("train_NLL_joint_selected",float("inf")) >
                        arm.get("train_NLL_reference",float("-inf"))+1e-8):
                errors.append(who+f".train{n}.bad_predeclared_joint_search")
            if (arm.get("source_local_binary_sha256")!=
                    physical[f"train{n}_local_uncalibrated"]["sha256"] or
                    arm.get("selected_joint_binary_sha256")!=
                    physical[f"train{n}_joint_calibrated"]["sha256"]):
                errors.append(who+f".train{n}.source_binary_sha_mismatch")
    quality=doc.get("full_model_restored_MLP_PPL",{})
    if set(quality)!=set(OUTPUT_KEYS):
        errors.append(who+".missing_PPL_arm")
    else:
        for name,c in quality.items():
            if set(c)!=set(CORPORA) or any(
                    not math.isfinite(score.get("ppl",float("nan")))
                    or score.get("ppl",0)<=0 or score.get("tokens",0)<=0
                    for score in c.values()):
                errors.append(who+"."+name+".missing_nonfinite_PPL")
    env=doc.get("environment",{})
    if (env.get("selected_ATen_CPU_capability")!="AVX2" or
            env.get("oneDNN_ISA_max_requested")!="AVX2" or
            env.get("threads")!=2 or not env.get("deterministic")):
        errors.append(who+".CPU_numeric_control")
    return errors

def summary(doc):
    return {
        "physical_bytes":{k:v["disk_bytes"] for k,v in doc["physical_MLP_files"].items()},
        "physical_SHA":{k:v["sha256"] for k,v in doc["physical_MLP_files"].items()},
        "TRAIN_NLL":{k:{
            "before":v["train_NLL_reference"],
            "after":v["train_NLL_joint_selected"],
            "gain":v["train_NLL_improvement"],
            "accepted_rounds":v["selection"]["rounds_completed"],
        } for k,v in doc["training_choices"].items()},
        "PPL":{k:{corpus:metric["ppl"] for corpus,metric in val.items()}
               for k,val in doc["full_model_restored_MLP_PPL"].items()},
        "delta_joint_vs_local_calibrated":doc["primary_comparisons"],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True)
    p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text(encoding="utf-8"))
    b=json.loads(Path(args.b).read_text(encoding="utf-8"))
    issues=[key+".source_differs" for key in STRICT_SOURCE if a.get(key)!=b.get(key)]
    issues+=audit("ubuntu22",a)
    issues+=audit("ubuntu24",b)
    lhs=summary(a);rhs=summary(b)
    same_file=(lhs["physical_SHA"]==rhs["physical_SHA"] and
               lhs["physical_bytes"]==rhs["physical_bytes"])
    if not same_file:issues.append("cross_host_physical_bytes_or_SHA_diff")
    same_ppl=lhs["PPL"]==rhs["PPL"]
    out={
        "experiment":"CALIBRI_v40_joint_train_1024_Pythia_12_MLP_IMDB_crosshost",
        "immutable_source_and_real_physical_integrity_pass":not issues,
        "issues":issues,
        "all_physical_SHA_exact_equal":same_file,
        "all_PPL_exact_equal":same_ppl,
        "results":{"ubuntu22":lhs,"ubuntu24":rhs},
        "limits":"all 12 pretrained Pythia MLPs physically compressed, not whole model; validation uses dense-decoded W; IMDb 2048 first-test tokens are only one new exploratory probe; no native packed speed",
    }
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"integrity":not issues,"same_real_sha":same_file,
                      "same_ppl":same_ppl,"issues":issues,
                      "results":out["results"]},ensure_ascii=False),flush=True)
    if issues:raise SystemExit(2)

if __name__=="__main__":main()
