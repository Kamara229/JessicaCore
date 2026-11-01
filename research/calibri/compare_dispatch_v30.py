#!/usr/bin/env python3
"""CALIBRI v30 evidence merger: actual oneDNN primitive verbose + ATen ops.

OneDNN verbose logs prove particular oneDNN primitives executed and expose
their implementation strings. ATen profiler alone CANNOT establish hardware
instruction choice; zero oneDNN lines is an observability gap, not proof that
no vector instructions or no optimized kernels were used.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

CORPORA=("WikiText2_validation","TinyStories_validation")
RUNS=("a_default","a_avx2","b_default","b_avx2")
IMPORTANT_OPS=("aten::mm","aten::addmm","aten::bmm",
               "aten::matmul","aten::linear","aten::_softmax",
               "aten::index_select","aten::argmax","aten::topk")


def one_dnn_trace(path):
    isa=[]
    versions=[]
    kernels=Counter()
    ops=Counter()
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        if line.startswith(("onednn_verbose,info,cpu,isa:", "dnnl_verbose,info,cpu,isa:")):
            isa.append(line.split("isa:",1)[1].strip())
        elif line.startswith(("onednn_verbose,info,oneDNN ", "dnnl_verbose,info,oneDNN ")):
            versions.append(line.split("info,",1)[1].strip())
        elif line.startswith(("onednn_verbose,primitive,exec,cpu,", "dnnl_verbose,primitive,exec,cpu,")):
            fields=line.split(",")
            if len(fields)>5:
                ops[fields[4]]+=1
                kernels[fields[4]+" | "+fields[5]]+=1
    return {
        "oneDNN_info_runtime_isa_lines":sorted(set(isa)),
        "oneDNN_version_lines":sorted(set(versions)),
        "oneDNN_primitive_execution_count":int(sum(ops.values())),
        "oneDNN_operation_counts":dict(sorted(ops.items())),
        "oneDNN_observed_primitive_implementations":dict(kernels.most_common(15)),
        "oneDNN_execution_evidence_available":bool(ops),
        "missing_primitives_means":(
            "No oneDNN primitive exec reported by verbose; does NOT prove "
            "no ATen/C++/BLAS or ISA-specific CPU kernels executed."
            if not ops else "Each listed implementation occurred in oneDNN verbose at runtime."
        ),
    }


def aten_trace(data):
    out={}
    for corpus in CORPORA:
        events=data["outputs"][corpus]["cpu_profiler"]["events"]
        if not events or not isinstance(events,list):
            raise ValueError("missing profiler results for "+corpus)
        counts={x["operator"]:int(x["calls"]) for x in events}
        if not any(counts.get(op,0)>0 for op in ("aten::mm","aten::addmm","aten::bmm")):
            raise ValueError("real source forward contains no observed matmul: "+corpus)
        out[corpus]={
            "ATen_cpu_operator_kinds":len(events),
            "ATen_selected_operator_calls":{k:counts.get(k,0) for k in IMPORTANT_OPS},
            "ATen_observation_limit":"Operator name/count, NOT lower-level kernel ISA",
        }
    return out


def main():
    parser=argparse.ArgumentParser()
    for key in RUNS:
        parser.add_argument("--"+key.replace("_","-")+"-json",required=True)
        parser.add_argument("--"+key.replace("_","-")+"-log",required=True)
    parser.add_argument("--v29-verdict",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    verdict=json.loads(Path(args.v29_verdict).read_text(encoding="utf-8"))
    errors=list(verdict.get("validation_errors",[]))
    if not verdict.get("source_integrity_pass"):
        errors.append("v29_SOURCE_INVARIANTS_FAILED")
    out={}
    for key in RUNS:
        name=key.replace("_","-")
        metrics=json.loads(Path(getattr(args,key+"_json")).read_text(encoding="utf-8"))
        log=one_dnn_trace(Path(getattr(args,key+"_log")))
        out[key]={
            "cpu_brand":metrics.get("cpu_brand"),
            "requested_dnnl_isa":metrics.get("requested_dnnl_max_cpu_isa"),
            "cpu_flags":metrics.get("cpu_supports"),
            "oneDNN":log,
            "ATen":aten_trace(metrics),
        }
        expected="AVX2" if "avx2" in key else "DEFAULT"
        if out[key]["requested_dnnl_isa"]!=expected:
            errors.append(key+":wrong-ISA-request")
        if "default" in key and not log["oneDNN_primitive_execution_count"]:
            errors.append(key+":missing-default-oneDNN-primitive-execution-evidence")
    report={
        "experiment":"CALIBRI_v30_real_Switch_ATen_vs_oneDNN_dispatch",
        "hardware_heterogeneous_observed":verdict.get("heterogeneous_CPU_hardware_observed"),
        "v29_source_integrity_pass":verdict.get("source_integrity_pass"),
        "validation_errors":errors,
        "measurements":out,
        "numeric_comparison_from_original_v29":verdict.get("results"),
        "interpretation_limits":[
            "oneDNN verbose reports concrete executed oneDNN primitives and their implementation strings only.",
            "ATen CPU profiler identifies operator labels/call counts, NOT ATen compiled CPU ISA or assembly.",
            "When AVX2 cap leads to no oneDNN primitive verbose, it may indicate a fallback path; it does NOT prove which fallback SIMD ISA executed.",
            "oneDNN ISA cap does not globally constrain every CPU operator or guarantee exact cross-hardware bitwise outputs.",
            "Measured logits compare only 24 selected vocabulary probes; full-logit exact hashes and downstream router exact masks are separate.",
            "No compressed bytes, PPL, QMoE reproduction or packed native speed measurements.",
        ],
    }
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "hardware_heterogeneous":report["hardware_heterogeneous_observed"],
        "validation_errors":errors,
        "kernels":{k:{"cpu":r["cpu_brand"],
                       "request":r["requested_dnnl_isa"],
                       "oneDNN_primitive_executions":r["oneDNN"]["oneDNN_primitive_execution_count"],
                       "oneDNN_runtime_isa":r["oneDNN"]["oneDNN_info_runtime_isa_lines"],
                       "oneDNN_top_implementations":list(r["oneDNN"]["oneDNN_observed_primitive_implementations"].items())[:3],
                       "ATen_selected_op_calls":r["ATen"]["WikiText2_validation"]["ATen_selected_operator_calls"]}
                   for k,r in out.items()},
    },ensure_ascii=False),flush=True)
    if errors:
        raise SystemExit(2)

if __name__=="__main__":
    main()
