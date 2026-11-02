#!/usr/bin/env python3
"""CALIBRI v44 lightweight SOURCE-INDEPENDENT timing coordinator.

Loads only a 3.5/4bpw pretrained physical-file manifest (prepared
in a PREVIOUS separate GitHub Actions STEP), then spawns fresh Python+
NumPy workers without loading torch/transformers. This prevents parent
model loading from contaminating memory high-water marks inherited via
fork/exec in child processes. Subprocesses are isolated per codec+mode.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CODECS=("CT33_mixed3_4","CF15_fixed4")
MODES=("mmap_stream","dense_restored")


def run():
    p=argparse.ArgumentParser()
    p.add_argument("--prepared",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    prepared=Path(args.prepared)
    root=prepared.parent
    previous=json.loads(prepared.read_text(encoding="utf-8"))
    if set(previous.get("results",{}))!=set(CODECS):
        raise AssertionError("pretrained real source or physically packed MLP metadata missing")
    for codec in CODECS:
        pre=previous["results"][codec]
        blob=root/(codec+".bin")
        samples=root/(codec+"_original_activation_and_dense_reference.npz")
        if not blob.is_file() or not samples.is_file():
            raise AssertionError("original real-weight physical file or golden reference missing")
        actual=blob.read_bytes()
        if (len(actual)!=(343840 if codec==CODECS[0] else 401224) or
            hashlib.sha256(actual).hexdigest()!=pre["binary_sha256"] or
            hashlib.sha256(samples.read_bytes()).hexdigest() !=
                pre["original_real_activation_and_golden_sha256"]):
            raise AssertionError("original trained 12-MLP physical/golden SHA changed since preparation")
        measured={}
        for mode in MODES:
            cmd=[sys.executable,str(Path(__file__).with_name("mmap_rowstream_worker_v44.py")),
                 "--bundle",str(blob),"--examples",str(samples),
                 "--mode",mode,"--warmups","2","--repetitions","5"]
            env=os.environ.copy()
            env["OPENBLAS_NUM_THREADS"]="1"
            env["OMP_NUM_THREADS"]="1"
            env["MKL_NUM_THREADS"]="1"
            cp=subprocess.run(cmd,capture_output=True,text=True,check=False,timeout=240,env=env)
            if cp.returncode:
                raise RuntimeError(f"{codec} {mode} exited {cp.returncode}:\n{cp.stdout[-2000:]}\n{cp.stderr[-5000:]}")
            results=[line for line in cp.stdout.splitlines() if line.startswith("{")]
            if len(results)!=1:
                raise AssertionError("worker must report one real physical measurement")
            record=json.loads(results[0])
            if (record["worker_mode"]!=mode or
                record["physical_bytes"]!=len(actual) or
                record["physical_sha256"]!=pre["binary_sha256"] or
                record["pretrained_MLP_matrices"]!=12 or
                record["source_parameter_positions"]!=786432 or
                record["max_abs_diff_from_FULL_dense_golden"]>2e-4 or
                record["MLP_forward_repetitions"]!=5):
                raise AssertionError("mmap vs independently produced original pretrained dense golden mismatch")
            measured[mode]=record
            print(json.dumps({
                "codec":codec,"mode":mode,
                "median_ms_all12":record["median_12_matmuls_ms"],
                "max_delta_vs_golden":record["max_abs_diff_from_FULL_dense_golden"],
                "RSS_KiB":record["peak_process_RSS_KiB_including_numpy_and_mmap"],
                "tracemalloc_peak_bytes":record["peak_tracemalloc_bytes_including_indices_and_outputs"],
                "weight_scratch_bytes":record["max_FP32_weight_scratch_bytes"],
            },ensure_ascii=False),flush=True)
        packed=measured["mmap_stream"]
        dense=measured["dense_restored"]
        if (packed["prebuilt_dense_FP32_weight_bytes"]!=0 or
            packed["max_FP32_weight_scratch_bytes"]>32768 or
            dense["prebuilt_dense_FP32_weight_bytes"]!=3145728 or
            dense["max_FP32_weight_scratch_bytes"]!=0):
            raise AssertionError("stream unexpectedly allocated full original MLP FP32 matrix")
        previous["results"][codec]={
            **pre,
            "mmap_stream":packed,
            "dense_restored":dense,
            "median_rowstream_divided_by_dense_wall_time":
                packed["median_12_matmuls_ms"]/dense["median_12_matmuls_ms"],
            "peak_RSS_delta_stream_minus_dense_KiB":
                packed["peak_process_RSS_KiB_including_numpy_and_mmap"]-
                dense["peak_process_RSS_KiB_including_numpy_and_mmap"],
            "known_full_FP32_weight_storage_avoided_bytes":3145728,
        }
    previous["separate_process_CI_steps_for_memory_reliability"]=True
    previous["worker_process_parent_does_not_import_torch_or_transformers"]=True
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(previous,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"experiment":previous["experiment"],
                      "source":"real pretrained Pythia12 MLP all weights",
                      "mem_stats":"fresh lightweight Python+NumPy process after separate PyTorch preparation CI step",
                      "measured":{c:{
                          "physical_bytes":d["real_disk_bytes"],
                          "mmap_ms":d["mmap_stream"]["median_12_matmuls_ms"],
                          "dense_ms":d["dense_restored"]["median_12_matmuls_ms"],
                          "slowdown":d["median_rowstream_divided_by_dense_wall_time"],
                          "mmap_RSS_KiB":d["mmap_stream"]["peak_process_RSS_KiB_including_numpy_and_mmap"],
                          "dense_RSS_KiB":d["dense_restored"]["peak_process_RSS_KiB_including_numpy_and_mmap"],
                          "mmap_scratch_B":d["mmap_stream"]["max_FP32_weight_scratch_bytes"],
                          "dense_weight_B":d["dense_restored"]["prebuilt_dense_FP32_weight_bytes"],
                      } for c,d in previous["results"].items()}},
                     ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()
