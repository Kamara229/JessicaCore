#!/usr/bin/env python3
"""Paired-host one-factor ATen CPU dispatch diagnostic on real Switch BF16.

Scientific distinction:
- ATEN_CPU_CAPABILITY controls selected PyTorch CPU DispatchStub capability.
- oneDNN verbose only confirms listed oneDNN primitives; no output in a
  capped process DOES NOT certify the exact lower-level ATen/BLAS ISA.
- Exact full logits hashes and next-expert masks are actual model outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from compare_determinism_v28 import compare
from compare_dispatch_v30 import one_dnn_trace

CORPORA = ("WikiText2_validation", "TinyStories_validation")
HOSTS = ("ubuntu_22_04", "ubuntu_24_04")
MODES = ("default", "avx2")
INVARIANTS = (
    "source_model", "model_revision", "dataset_revisions",
    "source_token_sha256", "source_key_weight_SHA256",
    "torch_version", "transformers_version", "cpu_arch",
    "torch_threads", "torch_deterministic_algorithms",
    "model_compute_dtype", "mkldnn_enabled", "router_jitter_forced_zero",
    "requested_dnnl_max_cpu_isa", "requested_onednn_max_cpu_isa",
)


def brief(a,b):
    result=compare(a,b)
    return {
        "full_logits_bitwise_equal":result["full_logits_exact_sha256_equal"],
        "router_masks_bitwise_equal":result["router_masks_exact_sha256_equal"],
        "probed_logits_max_abs_delta":result["sampled_logits_max_abs_delta"],
        "probed_logits_rmse":result["sampled_logits_rmse"],
        "full_vocab_top1_agreement":result["full_vocab_top1_agreement"],
        "next_router_exact_token_agreement":result["next_router_exact_per_token_agreement"],
    }


def main():
    ap=argparse.ArgumentParser()
    for h in ("a","b"):
        for mode in MODES:
            ap.add_argument(f"--{h}-{mode}-json", required=True)
            ap.add_argument(f"--{h}-{mode}-log", required=True)
    ap.add_argument("--output", required=True)
    args=ap.parse_args()
    sources={}
    for short,label in (("a",HOSTS[0]),("b",HOSTS[1])):
        sources[label]={}
        for mode in MODES:
            stem=f"{short}_{mode}"
            path=Path(getattr(args,stem+"_json"))
            doc=json.loads(path.read_text(encoding="utf-8"))
            verbose=one_dnn_trace(Path(getattr(args,stem+"_log")))
            sources[label][mode]={
                "document":doc,"oneDNN_verbose":verbose,
                "source_json_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    reference=sources[HOSTS[0]]["default"]["document"]
    issues=[]
    for host in HOSTS:
        for mode in MODES:
            evidence=sources[host][mode]
            doc=evidence["document"]
            expected={"default":"NO AVX","avx2":"AVX2"}[mode]
            for key in INVARIANTS:
                if doc.get(key)!=reference.get(key):
                    issues.append(host+"."+mode+"."+key)
            if doc.get("experiment")!="CALIBRI_v31_Switch_real_BF16_ATen_dispatch_factor":
                issues.append(host+"."+mode+".experiment")
            if doc.get("torch_ATen_dispatch_requested")!=mode:
                issues.append(host+"."+mode+".ATen_requested")
            if doc.get("torch_ATen_dispatch_selected_reported")!=expected:
                issues.append(host+"."+mode+".ATen_selected")
            if doc.get("requested_dnnl_max_cpu_isa")!="AVX2":
                issues.append(host+"."+mode+".oneDNN_ISA_not_fixed")
            if not doc.get("cpu_supports",{}).get("avx2"):
                issues.append(host+"."+mode+".CPU_has_no_AVX2")
            if not isinstance(doc.get("outputs"),dict):
                issues.append(host+"."+mode+".no_outputs")
                continue
            for corpus in CORPORA:
                out=doc["outputs"].get(corpus)
                if not isinstance(out,dict) or not out.get("full_logit_sha256"):
                    issues.append(host+"."+mode+"."+corpus+".no_logits")
                else:
                    events=out.get("cpu_profiler",{}).get("events",[])
                    if not any(e.get("operator")=="aten::mm" and e.get("calls",0)>0
                               for e in events):
                        issues.append(host+"."+mode+"."+corpus+".ATen_matmul_not_seen")

        a=sources[host]["default"]["document"]
        b=sources[host]["avx2"]["document"]
        if a.get("cpu_brand")!=b.get("cpu_brand") or a.get("cpu_supports")!=b.get("cpu_supports"):
            issues.append(host+".CPU_not_held_constant")

    report={
        "experiment":"CALIBRI_v31_pretrained_BF16_ATen_dispatch_selected_capability",
        "source_invariants_pass":not issues,
        "evidence_errors":sorted(set(issues)),
        "cpu":{h:sources[h]["default"]["document"].get("cpu_brand") for h in HOSTS},
        "hardware_heterogeneous":(
            sources[HOSTS[0]]["default"]["document"].get("cpu_brand") !=
            sources[HOSTS[1]]["default"]["document"].get("cpu_brand")),
        "within_host_default_vs_avx2":{},
        "cross_host_default":{},
        "cross_host_avx2":{},
        "observed_dispatch":{},
        "limits":[
            "One factor ATEN_CPU_CAPABILITY default vs avx2, oneDNN requested cap fixed AVX2.",
            "torch._C._get_cpu_capability reports PyTorch CPU DispatchStub selected capability, not actual per-op assembly.",
            "ATen profiler reveals executed operator kinds, not exact SIMD kernels.",
            "oneDNN verbose reveals listed oneDNN primitive implementations; missing trace is not proof of CPU instruction level.",
            "Full-logit hashes compare all output positions; numeric max absolute delta uses only 24 fixed vocabulary probes per token.",
            "No physical compression, training, PPL, mobile throughput or QMoE 0.8bpw success claimed.",
        ],
    }
    for host in HOSTS:
        report["within_host_default_vs_avx2"][host]={}
        report["observed_dispatch"][host]={}
        for mode in MODES:
            item=sources[host][mode]
            log=item["oneDNN_verbose"]
            doc=item["document"]
            report["observed_dispatch"][host][mode]={
                "requested_ATen_CPU_capability":doc["torch_ATen_dispatch_requested"],
                "PyTorch_reported_ATen_CPU_capability":doc["torch_ATen_dispatch_selected_reported"],
                "requested_oneDNN_cap":doc["requested_dnnl_max_cpu_isa"],
                "executed_oneDNN_primitive_count":log["oneDNN_primitive_execution_count"],
                "oneDNN_implementation_strings":log["oneDNN_observed_primitive_implementations"],
                "ATen_op_counts_WikiText":{
                    e["operator"]:e["calls"] for e in
                    doc["outputs"][CORPORA[0]]["cpu_profiler"]["events"]
                    if e["operator"] in ("aten::mm","aten::bmm","aten::matmul","aten::linear")
                },
                "source_json_sha256":item["source_json_sha256"],
            }
        for corpus in CORPORA:
            try:
                left=sources[host]["default"]["document"]["outputs"][corpus]
                right=sources[host]["avx2"]["document"]["outputs"][corpus]
                if left["sampled_logit_positions"]!=right["sampled_logit_positions"]:
                    raise ValueError("probe positions differ")
                report["within_host_default_vs_avx2"][host][corpus]=brief(left,right)
            except (KeyError,ValueError,TypeError) as exc:
                issues.append(host+"."+corpus+".invalid_pair_"+type(exc).__name__)
    for mode in MODES:
        for corpus in CORPORA:
            try:
                a=sources[HOSTS[0]][mode]["document"]["outputs"][corpus]
                b=sources[HOSTS[1]][mode]["document"]["outputs"][corpus]
                if a["sampled_logit_positions"]!=b["sampled_logit_positions"]:
                    raise ValueError("probe positions differ")
                report["cross_host_"+mode][corpus]=brief(a,b)
            except (KeyError,ValueError,TypeError) as exc:
                issues.append("cross_host_"+mode+"."+corpus+"."+type(exc).__name__)
    report["evidence_errors"]=sorted(set(issues))
    report["source_invariants_pass"]=not issues
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "pass":report["source_invariants_pass"],
        "errors":report["evidence_errors"],
        "cpu":report["cpu"],
        "actual_selected_capability":{
            h:{m:report["observed_dispatch"][h][m]["PyTorch_reported_ATen_CPU_capability"]
               for m in MODES} for h in HOSTS},
        "within_host":report["within_host_default_vs_avx2"],
        "cross_host_avx2":report["cross_host_avx2"],
    },ensure_ascii=False),flush=True)
    if issues:
        raise SystemExit(2)


if __name__=="__main__":
    main()
