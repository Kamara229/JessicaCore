#!/usr/bin/env python3
"""Compare independent default and requested AVX2 BF16 Switch CPU forwards.

No claim that DNNL_MAX_CPU_ISA governs ALL PyTorch compute kernels.
A cross-host AMD-vs-Intel comparison is only established if hardware differs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from compare_determinism_v28 import compare

CORPORA=("WikiText2_validation","TinyStories_validation")
SOURCE_KEYS=(
    "experiment","source_model","model_revision","dataset_revisions",
    "source_token_sha256","source_key_weight_SHA256","torch_version",
    "transformers_version","cpu_arch","torch_threads",
    "torch_deterministic_algorithms","model_compute_dtype",
    "mkldnn_enabled","router_jitter_forced_zero",
)

def run(args):
    labels=("ubuntu_22_04","ubuntu_24_04")
    files={
        "ubuntu_22_04":{"default":Path(args.a_default),"avx2":Path(args.a_cap)},
        "ubuntu_24_04":{"default":Path(args.b_default),"avx2":Path(args.b_cap)},
    }
    data={label:{mode:json.loads(p.read_text(encoding="utf-8"))
          for mode,p in filemap.items()} for label,filemap in files.items()}
    baseline=data[labels[0]]["default"]
    mismatches=[]
    all_items=[(label,mode,doc) for label,records in data.items()
               for mode,doc in records.items()]
    for label,mode,doc in all_items:
        for key in SOURCE_KEYS:
            if doc.get(key)!=baseline.get(key):
                mismatches.append(f"{label}.{mode}.{key}")
        expected="DEFAULT" if mode=="default" else "AVX2"
        if doc.get("requested_dnnl_max_cpu_isa")!=expected or doc.get("requested_onednn_max_cpu_isa")!=expected:
            mismatches.append(f"{label}.{mode}.request_mode")
        if not doc.get("cpu_supports",{}).get("avx2",False):
            mismatches.append(f"{label}.{mode}.cpu_has_no_avx2")
        for corpus in CORPORA:
            out=doc.get("outputs",{}).get(corpus)
            if not isinstance(out,dict):
                mismatches.append(f"{label}.{mode}.{corpus}.missing_output")
            elif out.get("tokens_evaluated",0)<=0 or not out.get("full_logit_sha256"):
                mismatches.append(f"{label}.{mode}.{corpus}.empty_or_invalid")
    for label in labels:
        default=data[label]["default"]
        capped=data[label]["avx2"]
        if default["cpu_brand"]!=capped["cpu_brand"] or default["cpu_supports"]!=capped["cpu_supports"]:
            mismatches.append(f"{label}.not_same_cpu_between_separate_processes")

    metrics={"within_host_default_vs_avx2":{},"cross_host_default":{},
             "cross_host_avx2":{}}
    for label in labels:
        metrics["within_host_default_vs_avx2"][label]={}
        for corpus in CORPORA:
            try:
                left=data[label]["default"]["outputs"][corpus]
                right=data[label]["avx2"]["outputs"][corpus]
                if left["sampled_logit_positions"]!=right["sampled_logit_positions"]:
                    raise ValueError("probe ids differ")
                metrics["within_host_default_vs_avx2"][label][corpus]=compare(left,right)
            except (KeyError,ValueError,TypeError) as e:
                mismatches.append(f"{label}.{corpus}.invalid_pair:{type(e).__name__}")
    for mode,key in (("default","cross_host_default"),("avx2","cross_host_avx2")):
        for corpus in CORPORA:
            try:
                left=data[labels[0]][mode]["outputs"][corpus]
                right=data[labels[1]][mode]["outputs"][corpus]
                if left["sampled_logit_positions"]!=right["sampled_logit_positions"]:
                    raise ValueError("probe ids differ")
                metrics[key][corpus]=compare(left,right)
            except (KeyError,ValueError,TypeError) as e:
                mismatches.append(f"{key}.{corpus}.invalid_pair:{type(e).__name__}")
    cpu_a=data[labels[0]]["default"]["cpu_brand"]
    cpu_b=data[labels[1]]["default"]["cpu_brand"]
    heterogeneous=cpu_a!=cpu_b or data[labels[0]]["default"]["cpu_supports"]!=data[labels[1]]["default"]["cpu_supports"]
    report={
        "experiment":"CALIBRI_v29_requested_AVX2_cap_CPU_diagnosis",
        "source_integrity_pass":not mismatches,
        "validation_errors":sorted(set(mismatches)),
        "host_cpu":{"ubuntu_22_04":cpu_a,"ubuntu_24_04":cpu_b},
        "host_cpu_flags":{label:data[label]["default"]["cpu_supports"] for label in labels},
        "heterogeneous_CPU_hardware_observed":heterogeneous,
        "measured_AVX2_effective_ISA_verified":False,
        "results":metrics,
        "source_sha256":{f"{label}_{mode}":hashlib.sha256(path.read_bytes()).hexdigest()
             for label,modes in files.items() for mode,path in modes.items()},
        "conclusion_limits":"Requested AVX2 CPU ceiling via DNNL_MAX_CPU_ISA/ONEDNN_MAX_CPU_ISA in a fresh process; these environment variables do not prove effective kernels or cover every PyTorch operation. Different CPUs not guaranteed on hosted runners. No full vocab max delta, PPL, physical compression, or speed benchmark.",
    }
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    preview={category:{host:{corpus:{
        "hash_equal":m["full_logits_exact_sha256_equal"],
        "max_abs_probed":m["sampled_logits_max_abs_delta"],
        "router_token_agreement":m["next_router_exact_per_token_agreement"]}
        for corpus,m in group.items()} for host,group in per.items()}
        if category=="within_host_default_vs_avx2" else
        {corpus:{"hash_equal":m["full_logits_exact_sha256_equal"],
                 "max_abs_probed":m["sampled_logits_max_abs_delta"],
                 "router_token_agreement":m["next_router_exact_per_token_agreement"]}
         for corpus,m in per.items()}
        for category,per in metrics.items()}
    print(json.dumps({"source_integrity_pass":report["source_integrity_pass"],
                      "heterogeneous_cpu":heterogeneous,"metrics":preview,
                      "validation_errors":report["validation_errors"]},
                     ensure_ascii=False),flush=True)
    if mismatches:
        raise SystemExit(2)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("a-default","a-cap","b-default","b-cap","output"):
        p.add_argument("--"+name,required=True)
    run(p.parse_args())
