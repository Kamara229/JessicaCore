#!/usr/bin/env python3
"""Compare two independent real CALIBRI v32 measurements without cherry-picking.

Negative quality, size or speed outcomes remain valid experimental results.
Only missing/mismatched source identities or impossible byte counts fail CI.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--first",required=True)
    p.add_argument("--second",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.first).read_text(encoding="utf-8"))
    b=json.loads(Path(args.second).read_text(encoding="utf-8"))
    keys=("experiment","source_model","source_revision","corpus",
          "corpus_revision","trained_MLP_projection","projection_shape",
          "source_weight_sha256","token_sha256","tokens","predeclared_selection")
    problems=[k for k in keys if a.get(k)!=b.get(k)]
    rows={}
    for name,doc in (("ubuntu_22_04",a),("ubuntu_24_04",b)):
        metrics=doc["physical_storage"]
        if metrics["actual_multidim_file_bytes"]>metrics["matched_INT4_serialized_bytes"]:
            problems.append(name+".exceeded_matched_budget")
        if (metrics["actual_multidim_file_bytes"]<1 or
            doc["microbenchmark"]["TT_plus_packed_tiled_matmul_vs_same_reconstructed_dense_max_abs"]>1e-4):
            problems.append(name+".invalid_kernel_or_file")
        rows[name]={
            "actual_file_bytes":metrics["actual_multidim_file_bytes"],
            "INT4_control_bytes":metrics["matched_INT4_serialized_bytes"],
            "multidim_file_SHA256":metrics["physical_binary_sha256"],
            "fp16_original_PPL":doc["source_full_model_PPL"]["validation"]["ppl"],
            "INT4_control_PPL":doc["quality_PPL_validation"]["physical_int4_control"]["ppl"],
            "multidim_candidate_PPL":doc["quality_PPL_validation"]["multidim_adaptive_3or4bit"]["ppl"],
            "streaming_python_median_ms":doc["microbenchmark"]["packed_streaming_python_one_tile_scratch_ms"]["median_ms"],
            "decoded_dense_median_ms":doc["microbenchmark"]["dense_reconstructed_matmul_ms"]["median_ms"],
            "tile_counts":doc["physical_storage"]["multidim_encode_metadata"],
            "platform":doc["environment"],
        }
    report={
        "experiment":"CALIBRI_v32_two_independent_CPU_real_Pythia14m",
        "source_invariants_pass":not problems,
        "source_errors":problems,
        "equal_serialized_blob_sha256_between_hosts":
          rows["ubuntu_22_04"]["multidim_file_SHA256"]==rows["ubuntu_24_04"]["multidim_file_SHA256"],
        "results":rows,
        "interpretation":"True physical bytes and WikiText validation perplexity measured for ONE trained Pythia 14m projection; quality uses dense decoded W; streaming timing uses a slow Python no-full-W kernel and is not native fused inference. No claim of novelty or 0.8bpw.",
    }
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "source_invariants_pass":report["source_invariants_pass"],
        "equal_binary_SHA":report["equal_serialized_blob_sha256_between_hosts"],
        "errors":problems,
        "measurements":{k:{z:v[z] for z in (
            "actual_file_bytes","INT4_control_bytes",
            "fp16_original_PPL","INT4_control_PPL","multidim_candidate_PPL",
            "streaming_python_median_ms","decoded_dense_median_ms")}
             for k,v in rows.items()},
    },ensure_ascii=False),flush=True)
    if problems:
        raise SystemExit(2)


if __name__=="__main__":
    main()
