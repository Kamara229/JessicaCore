#!/usr/bin/env python3
"""CALIBRI v44 real pretrained Pythia 12-MLP physical mmap GEMM benchmark.

Uses EXACT prior signed CT33 rowwise and CF15 INT4 quantizers under the
frozen WikiText TRAIN-A 2048 token calibration, without external eval
text/PPL or hyperparameter search. Saves true complete CB36 binary
archives and eight real trained module-activation inputs per MLP,
generates dense Golden outputs from fully decoded source bytes,
then launches independent isolated pure-NumPy subprocess workers for
mmap-streamed row chunks and fully dense restored reference. A Python
mmap loop is NOT a native fused CPU kernel; negative throughput allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from all_mlp_agnews_v36 import package,unpack_bundle,module_identity
from calibrated_rows_v35 import calibrated_encode
from direct_tiles_v33 import encode as encode_ct33,decode as decode_ct33
from fixedbit_v15 import decode as decode_cf15
from pareto_fixed_highbits_v43 import trained_fixedbit
from quality import MODEL,DATASET,load_token_ids,module_selection,calibrate


def run(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi

    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2"
            and os.getenv("DNNL_MAX_CPU_ISA")=="AVX2"
            and os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2"
            and str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("CPU ATen AVX2/oneDNN ISA cap must be pinned")
    if args.train_tokens!=2048 or args.tokens_per_matrix!=8 or args.repetitions!=5:
        raise AssertionError("must use frozen 2048 trained source tokens, 8 per MLP, 5 timing repeats")
    torch.manual_seed(41)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    api=HfApi()
    model_rev=api.model_info(MODEL).sha
    wiki_rev=api.dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=model_rev,torch_dtype=torch.float32)
    model.eval()
    modules=module_selection(model)
    names=sorted(modules,key=module_identity)
    if len(names)!=12:
        raise AssertionError("must use real pretrained Pythia 12 original MLPs")
    W={name:modules[name].weight.detach().float().cpu().numpy().astype(np.float32).copy()
       for name in names}
    if sum(w.size for w in W.values())!=786432 or any(
        w.shape not in ((512,128),(128,512)) for w in W.values()):
        raise AssertionError("changed Pythia source MLP geometry")
    train=load_token_ids(tok,"train",wiki_rev,args.train_tokens)
    with torch.inference_mode():
        activ=calibrate(model,modules,train,128)
    if set(activ)!=set(names):
        raise AssertionError("not all original source MLP activation inputs captured")
    X={n:np.ascontiguousarray(activ[n][:8].detach().cpu().numpy().astype(np.float32))
       for n in names}
    if any(x.shape!=(8,W[n].shape[1]) for n,x in X.items()):
        raise AssertionError("8 genuine trained module inputs missing")

    blobs3={}
    blobs4={}
    for name in names:
        source=W[name]
        tr=activ[name].detach().cpu().numpy().astype(np.float32)
        moments=np.mean(np.square(tr,dtype=np.float32),axis=0,dtype=np.float32)
        base,meta=encode_ct33(source,moments,int(source.size*3.5/8),
                              br=1,bc=source.shape[1])
        comp,train_meta=calibrated_encode(source,tr,base)
        if len(base)!=len(comp) or not train_meta["bits_are_identical_per_row"]:
            raise AssertionError("original v43 3.5bpw TRAIN-only calibration changed byte allocation")
        blobs3[name]=comp
        baseline4,cal4,info=trained_fixedbit(source,moments,4)
        if len(baseline4)!=len(cal4):
            raise AssertionError("fixed4 TRAIN calibration changed CRC file length")
        blobs4[name]=cal4
    physical={"CT33_mixed3_4":(1,blobs3),
              "CF15_fixed4":(2,blobs4)}

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    path=out.parent
    xs={f"x_{k}_{t}":X[n] for n in names for k,t in [module_identity(n)]}
    evidence={"experiment":"CALIBRI_v44_real_trained_12_MLP_CB36_mmap_rowblock_forward",
              "original_source_model":MODEL,"model_revision":model_rev,
              "source_train_dataset":DATASET,"train_dataset_revision":wiki_rev,
              "train_first2048_tokens_sha256":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
              "source_MLP_coefficient_count":786432,
              "original_MLP_source_SHA256":{n:hashlib.sha256(W[n].tobytes()).hexdigest() for n in names},
              "selected_ATen_CPU_capability":str(torch._C._get_cpu_capability()),
              "source_capture_tokens":len(train),
              "genuine_activation_tokens_per_matrix":8,
              "predeclared_row_chunk":16,"timing_warmups":2,"timing_repetitions":5,
              "method":"pure NumPy/OpenBLAS 1-thread rowwise mmap reference GEMM vs same physical source decoded dense 12-MLP GEMM, subprocess per configuration",
              "results":{}}
    for tag,(mode,components) in physical.items():
        binary=package(components,mode)
        file=path/(tag+".bin")
        file.write_bytes(binary)
        actual=file.read_bytes()
        if actual!=binary:
            raise AssertionError("physical CB36 write/read bytes changed")
        if len(actual)!=(343840 if mode==1 else 401224):
            raise AssertionError("original v43 actual physical size expected")
        modules_bytes=unpack_bundle(actual,mode)
        if set(modules_bytes)!=set(names):
            raise AssertionError("physical bundle missing original source modules")
        golden={}
        for n in names:
            k,t=module_identity(n)
            restored=(decode_ct33(modules_bytes[n]) if mode==1
                      else decode_cf15(modules_bytes[n]))
            if restored.shape!=W[n].shape or not np.isfinite(restored).all():
                raise AssertionError("invalid actual physical pretrained weight decode")
            golden[f"y_{k}_{t}"]=X[n]@restored.T
        examples_file=path/(tag+"_original_activation_and_dense_reference.npz")
        np.savez(examples_file,**xs,**golden)

        workers={}
        for worker_mode in ("mmap_stream","dense_restored"):
            cmd=[sys.executable,str(Path(__file__).with_name("mmap_rowstream_worker_v44.py")),
                 "--bundle",str(file),"--examples",str(examples_file),
                 "--mode",worker_mode,"--warmups","2","--repetitions","5"]
            env=os.environ.copy()
            env["OPENBLAS_NUM_THREADS"]="1"
            env["OMP_NUM_THREADS"]="1"
            env["MKL_NUM_THREADS"]="1"
            cp=subprocess.run(cmd,check=False,capture_output=True,text=True,
                              env=env,timeout=240)
            if cp.returncode:
                raise RuntimeError(f"{tag} {worker_mode} failed:\n{cp.stdout[-2000:]}\n{cp.stderr[-4000:]}")
            lines=[line for line in cp.stdout.splitlines() if line.startswith("{")]
            if len(lines)!=1:
                raise AssertionError(f"missing full source-real worker metrics in {worker_mode}")
            measured=json.loads(lines[0])
            if (measured["worker_mode"]!=worker_mode or
                measured["physical_bytes"]!=file.stat().st_size or
                measured["physical_sha256"]!=hashlib.sha256(actual).hexdigest() or
                measured["pretrained_MLP_matrices"]!=12 or
                measured["source_parameter_positions"]!=786432 or
                measured["max_abs_diff_from_FULL_dense_golden"]>2e-4):
                raise AssertionError("source real physical/fidelity worker failed")
            workers[worker_mode]=measured
            print(json.dumps({"pretrained_binary":tag,"benchmark_mode":worker_mode,
                              "median_12_matmuls_ms":measured["median_12_matmuls_ms"],
                              "max_abs_diff":measured["max_abs_diff_from_FULL_dense_golden"],
                              "peak_RSS_KiB":measured["peak_process_RSS_KiB_including_numpy_and_mmap"],
                              "peak_scratch_B":measured["max_FP32_weight_scratch_bytes"]}),flush=True)
        packed=workers["mmap_stream"]
        dense=workers["dense_restored"]
        if packed["prebuilt_dense_FP32_weight_bytes"]!=0:
            raise AssertionError("packed stream allocated a full source pretrained weight matrix")
        if dense["prebuilt_dense_FP32_weight_bytes"]!=786432*4:
            raise AssertionError("dense control not restored all twelve source matrices")
        if packed["max_FP32_weight_scratch_bytes"]>16*512*4:
            raise AssertionError("stream allocated more than 16 rows of source matrix")
        if dense["max_FP32_weight_scratch_bytes"]!=0:
            raise AssertionError("dense matrix control unexpectedly had block scratch")
        evidence["results"][tag]={
            "real_disk_bytes":len(actual),
            "real_source_bpw":8*len(actual)/786432,
            "binary_sha256":hashlib.sha256(actual).hexdigest(),
            "original_real_activation_and_golden_sha256":hashlib.sha256(examples_file.read_bytes()).hexdigest(),
            "mmap_stream":packed,
            "dense_restored":dense,
            "median_rowstream_divided_by_dense_wall_time":
                packed["median_12_matmuls_ms"]/dense["median_12_matmuls_ms"],
            "peak_RSS_delta_stream_minus_dense_KiB":
                packed["peak_process_RSS_KiB_including_numpy_and_mmap"]-
                dense["peak_process_RSS_KiB_including_numpy_and_mmap"],
            "known_full_FP32_weight_storage_avoided_bytes":dense["prebuilt_dense_FP32_weight_bytes"],
        }
    evidence["limitations"]=[
        "Benchmark is NOT native, fused, on-device or end-to-end LLM: 12 independent real-activation MLP projections at 8 tokens each and warmed OS cache.",
        "True file-backed mmap rowwise decoding with at most 16×512 FP32 scratch; no full W in packed worker, but Python unpack and dispatch overhead may be severe.",
        "Dense reference is full FP32-decoded NumPy/OpenBLAS one-thread, not PyTorch's optimized full-model CPU inference.",
        "Unix peak process RSS includes imported NumPy, file mappings, code, Python and reference outputs; a single ru_maxrss number does not isolate weight-only memory.",
        "Python tracemalloc peak is supplemental and may not account for every allocator/native memory page; independently verified structural scratch/full-dense bytes reported.",
        "Source physical bundle stores only 12 trained Pythia-14M MLP matrices, not attention/embedding/normalization/tokenizer or a deployable full model checkpoint.",
        "No QMoE SwitchTransformer 0.8bpw, new mathematical multidimensional trained weights, mobile/ARM throughput or production changes demonstrated.",
    ]
    out.write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"experiment":evidence["experiment"],"source_model":MODEL,
        "measured":{tag:{
            "physical_bytes":value["real_disk_bytes"],
            "mmap_12matmul_ms":value["mmap_stream"]["median_12_matmuls_ms"],
            "dense_12matmul_ms":value["dense_restored"]["median_12_matmuls_ms"],
            "mmap_to_dense_latency_ratio":value["median_rowstream_divided_by_dense_wall_time"],
            "packed_max_row_scratch_B":value["mmap_stream"]["max_FP32_weight_scratch_bytes"],
            "full_restored_FP32_weight_B":value["known_full_FP32_weight_storage_avoided_bytes"],
            "packed_process_peak_RSS_KiB":value["mmap_stream"]["peak_process_RSS_KiB_including_numpy_and_mmap"],
            "dense_process_peak_RSS_KiB":value["dense_restored"]["peak_process_RSS_KiB_including_numpy_and_mmap"],
            "max_packed_vs_dense_output_abs_delta":value["mmap_stream"]["max_abs_diff_from_FULL_dense_golden"],
        } for tag,value in evidence["results"].items()},
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/calibri_v44_mmap_rowstream.json")
    p.add_argument("--train-tokens",type=int,default=2048)
    p.add_argument("--tokens-per-matrix",type=int,default=8)
    p.add_argument("--repetitions",type=int,default=5)
    run(p.parse_args())
