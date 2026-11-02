#!/usr/bin/env python3
"""Strict cross-host identity/fidelity for v44 original 12 trained MLP mmap.

Negative speed or memory RSS outcomes MUST NOT fail CI. Compare immutable
trained checkpoint, source MLP SHA, train token and true physical bundle
SHA; report numerical timing/RSS without pretending hosted CPUs equal.
"""
import argparse
import json
import math
from pathlib import Path

COMMON=("experiment","original_source_model","model_revision",
        "source_train_dataset","train_dataset_revision",
        "train_first2048_tokens_sha256","source_MLP_coefficient_count",
        "original_MLP_source_SHA256","selected_ATen_CPU_capability",
        "source_capture_tokens","genuine_activation_tokens_per_matrix",
        "predeclared_row_chunk","timing_warmups","timing_repetitions")
CODECS=("CT33_mixed3_4","CF15_fixed4")
MODES=("mmap_stream","dense_restored")


def inspect(label,doc):
    failures=[]
    if (doc.get("source_MLP_coefficient_count")!=786432 or
        len(doc.get("original_MLP_source_SHA256",{}))!=12 or
        doc.get("selected_ATen_CPU_capability")!="AVX2" or
        doc.get("source_capture_tokens")!=2048 or
        doc.get("genuine_activation_tokens_per_matrix")!=8 or
        doc.get("predeclared_row_chunk")!=16 or
        doc.get("timing_warmups")!=2 or
        doc.get("timing_repetitions")!=5):
        failures.append(label+".model_or_benchmark_predefinition_failed")
    metrics=doc.get("results",{})
    if set(metrics)!=set(CODECS):
        failures.append(label+".missing_physical_codec_variant")
        return failures
    for codec,result in metrics.items():
        expected=343840 if codec=="CT33_mixed3_4" else 401224
        if (result.get("real_disk_bytes")!=expected or
            result.get("real_source_bpw")!=8*expected/786432 or
            len(result.get("binary_sha256",""))!=64):
            failures.append(label+"."+codec+".physical_size_or_sha")
        if not math.isfinite(result.get("median_rowstream_divided_by_dense_wall_time",float("nan"))):
            failures.append(label+"."+codec+".invalid_speed_ratio")
        for mode in MODES:
            meas=result.get(mode,{})
            if (meas.get("worker_mode")!=mode or
                meas.get("codec")!=codec or
                meas.get("pretrained_MLP_matrices")!=12 or
                meas.get("source_parameter_positions")!=786432 or
                meas.get("MLP_forward_calls_per_trial")!=12 or
                meas.get("MLP_forward_repetitions")!=5 or
                meas.get("tokens_per_matrix")!=8 or
                meas.get("physical_bytes")!=expected or
                meas.get("physical_sha256")!=result.get("binary_sha256") or
                len(meas.get("milliseconds",[]))!=5 or
                not 0<=meas.get("max_abs_diff_from_FULL_dense_golden",float("inf"))<=2e-4 or
                not math.isfinite(meas.get("median_12_matmuls_ms",float("nan"))) or
                meas.get("median_12_matmuls_ms",0)<=0 or
                meas.get("peak_process_RSS_KiB_including_numpy_and_mmap",0)<=0):
                failures.append(label+"."+codec+"."+mode+".source_reference_or_timing_invalid")
        stream=result["mmap_stream"]
        dense=result["dense_restored"]
        if (stream.get("prebuilt_dense_FP32_weight_bytes")!=0 or
            stream.get("max_FP32_weight_scratch_bytes",999999)>32768 or
            dense.get("prebuilt_dense_FP32_weight_bytes")!=786432*4 or
            dense.get("max_FP32_weight_scratch_bytes")!=0):
            failures.append(label+"."+codec+".packed_row_scratch_or_dense_alloc_invariant")
    return failures


def reduced(doc):
    return {n:{
        "real_disk_bytes":v["real_disk_bytes"],
        "bpw":v["real_source_bpw"],
        "real_physical_sha":v["binary_sha256"],
        "mmap_stream_median_12_matmuls_ms":v["mmap_stream"]["median_12_matmuls_ms"],
        "full_dense_median_12_matmuls_ms":v["dense_restored"]["median_12_matmuls_ms"],
        "stream_time_divided_by_dense_time":v["median_rowstream_divided_by_dense_wall_time"],
        "stream_peak_RSS_KiB":v["mmap_stream"]["peak_process_RSS_KiB_including_numpy_and_mmap"],
        "dense_peak_RSS_KiB":v["dense_restored"]["peak_process_RSS_KiB_including_numpy_and_mmap"],
        "stream_max_scratch_B":v["mmap_stream"]["max_FP32_weight_scratch_bytes"],
        "restored_full_weight_arrays_B":v["known_full_FP32_weight_storage_avoided_bytes"],
        "max_functional_error":v["mmap_stream"]["max_abs_diff_from_FULL_dense_golden"],
    } for n,v in doc["results"].items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True)
    p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text(encoding="utf-8"))
    b=json.loads(Path(args.b).read_text(encoding="utf-8"))
    problems=inspect("ubuntu22",a)+inspect("ubuntu24",b)
    for key in COMMON:
        if a.get(key)!=b.get(key):
            problems.append("cross-host-trained-source-"+key)
    if set(a.get("results",{}))==set(CODECS) and set(b.get("results",{}))==set(CODECS):
        for codec in CODECS:
            if (a["results"][codec]["binary_sha256"]!=b["results"][codec]["binary_sha256"] or
                a["results"][codec]["real_disk_bytes"]!=b["results"][codec]["real_disk_bytes"]):
                problems.append(codec+".crosshost_real_physical_bitstreams_differ")
    results={"ubuntu22":reduced(a),"ubuntu24":reduced(b)}
    comparison={
        "experiment":"CALIBRI_v44_real_pretrained_mmap_12_MLP_stream_vs_all_dense_cross_cpu",
        "source_and_physical_integrity_pass":not problems,
        "issues":problems,
        "host_timing_note":"Timing/RSS need NOT numerically match independent GitHub CPUs; treat faster/slower as local observations.",
        "measured_results":results,
        "limitations":"12 original MLPs, 8 REAL activations, page-cache-warm Python rowstream vs NumPy dense; NOT full LLM, optimized native packed ISA, guaranteed RAM savings, or deployable compressed checkpoint.",
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(comparison,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(comparison,ensure_ascii=False),flush=True)
    if problems:
        raise SystemExit(2)


if __name__=="__main__":
    main()
