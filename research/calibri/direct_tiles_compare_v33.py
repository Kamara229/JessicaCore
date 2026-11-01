#!/usr/bin/env python3
"""Cross-host artifact & science checker; negative quality and speed allowed."""
import argparse
import json
from pathlib import Path

REQUIRED=("experiment","model","model_revision","dataset","dataset_revision",
          "target_module","source_weight_shape","source_weight_sha256",
          "token_sha256","tokens","predeclared_geometries","training_selected_candidate")

def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("--first",required=True)
    parser.add_argument("--second",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    a=json.loads(Path(args.first).read_text(encoding="utf-8"))
    b=json.loads(Path(args.second).read_text(encoding="utf-8"))
    mismatch=[k for k in REQUIRED if a.get(k)!=b.get(k)]
    results={}
    for label,d in (("ubuntu_22_04",a),("ubuntu_24_04",b)):
        physical=d["physical"]
        orig=int(physical["fixed_INT4_rowwise_serialized_bytes"])
        winner=d["training_selected_candidate"]
        for name,item in physical["candidates"].items():
            if (item["actual_file_bytes"]>orig or item["actual_file_bytes"]<=0
                    or item["tiles_3bit"]+item["tiles_4bit"]!=item["tiles_total"]):
                mismatch.append(label+"."+name+".physical_budget_or_shape")
        if winner not in physical["candidates"] or not d["full_model_single_projection_substituted_PPL"].get(winner):
            mismatch.append(label+".selected_candidate_missing")
        if d["streaming_microbenchmark"]["exact_file_backed_vs_reconstructed_dense_max_abs"]>1e-4:
            mismatch.append(label+".byte_stream_matmul_incorrect")
        results[label]={
            "source_PPL":d["full_model_original_validation_PPL"]["ppl"],
            "INT4_PPL":d["full_model_single_projection_substituted_PPL"]["fixed_INT4_rowwise"]["ppl"],
            "tile_PPL":{key:value["ppl"] for key,value in d["full_model_single_projection_substituted_PPL"].items() if key!="fixed_INT4_rowwise"},
            "physical_int4_bytes":orig,
            "tile_bytes":{key:value["actual_file_bytes"] for key,value in physical["candidates"].items()},
            "tile_SHA256":{key:value["sha256"] for key,value in physical["candidates"].items()},
            "selected_using_training_only":winner,
            "stream_microkernel_ms":d["streaming_microbenchmark"]["file_backed_python_tiled_stream"]["median_ms"],
            "dense_matmul_ms":d["streaming_microbenchmark"]["same_decoded_dense_matmul"]["median_ms"],
            "stream_to_dense_median_ratio":d["streaming_microbenchmark"]["streaming_median_over_dense_median"],
        }
    physical_same=results["ubuntu_22_04"]["tile_SHA256"]==results["ubuntu_24_04"]["tile_SHA256"]
    if not physical_same:
        mismatch.append("tile_physical_sha_differs_across_hosts")
    quality_same=results["ubuntu_22_04"]["tile_PPL"]==results["ubuntu_24_04"]["tile_PPL"]
    # PPL differences alone do not constitute corrupt source, but must be
    # reported rather than smoothing or selecting a favorable host.
    report={
        "experiment":"CALIBRI_v33_TT_free_paired_host_real_Pythia_codec",
        "integrity_pass":not mismatch,
        "integrity_problems":mismatch,
        "physical_binary_SHA_identical_across_CPU":physical_same,
        "all_candidate_PPL_exact_match_between_hosts":quality_same,
        "results":results,
        "limits":"Only one MLP matrix, validation PPL uses restored dense weights, Python stream is not optimized compiled packed inference; no 0.8bpw/QMoE/state-of-art claim.",
    }
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "integrity_pass":report["integrity_pass"],
        "binary_equal":physical_same,
        "ppl_equal":quality_same,
        "measurements":results,
        "errors":mismatch},ensure_ascii=False),flush=True)
    if mismatch:
        raise SystemExit(2)

if __name__=="__main__":
    run()
