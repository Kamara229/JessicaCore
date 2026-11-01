#!/usr/bin/env python3
"""CALIBRI v32 real-pretrained Pythia one-projection compression experiment.

Physical adaptive multidimensional stream vs existing physically packed INT4
under the exact same budget. Training activations choose tile bitwidths;
WikiText VALIDATION PPL is measured without tuning on validation results.
Native no-full-W math microbenchmark is a slow Python streaming reference,
not a fused packed kernel or GPU/Android measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from multidim_v32 import encode, decode, streamed_matmul, smoke
from fixedbit_v15 import encode as fixed_encode, decode as fixed_decode
from quality import (
    MODEL, DATASET, load_token_ids, ppl_eval, module_selection, calibrate,
)


def bench(func, warmup=3, repeats=9):
    times=[]
    with torch.inference_mode():
        for _ in range(warmup):
            result=func()
            if not torch.isfinite(result).all():
                raise ValueError("nonfinite kernel output")
        for _ in range(repeats):
            t0=time.perf_counter_ns()
            result=func()
            times.append((time.perf_counter_ns()-t0)/1e6)
    return {"median_ms":float(statistics.median(times)),
            "minimum_ms":float(min(times)),
            "trials":repeats}


def run(args):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if not (os.environ.get("ATEN_CPU_CAPABILITY")=="avx2"
            and os.environ.get("DNNL_MAX_CPU_ISA")=="AVX2"
            and os.environ.get("ONEDNN_MAX_CPU_ISA")=="AVX2"):
        raise AssertionError("research comparison requires selected PyTorch AVX2 and capped oneDNN")
    selected=str(torch._C._get_cpu_capability()).upper()
    if selected!="AVX2":
        raise AssertionError("requested AVX2 dispatch was not selected")
    torch.manual_seed(32)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    smoke()
    api=HfApi()
    rev=api.model_info(MODEL).sha
    d_rev=api.dataset_info(DATASET).sha
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=rev)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=rev,torch_dtype=torch.float32)
    model.eval()
    modules=module_selection(model)
    if len(modules)!=12:
        raise AssertionError("expected 12 MLP projections from 14m trained model")
    name=sorted(modules)[0]
    mod=modules[name]
    w=mod.weight.detach().cpu().numpy().copy().astype(np.float32)
    if tuple(w.shape)!=(128,512):
        raise AssertionError("unexpected selected 4D compatible matrix shape: "+str(w.shape))
    train=load_token_ids(tokenizer,"train",d_rev,args.train_tokens)
    valid=load_token_ids(tokenizer,"validation",d_rev,args.validation_tokens)
    with torch.no_grad():
        xcal=calibrate(model,{name:mod},train,args.context)[name]
    if xcal.shape[-1]!=w.shape[1] or len(xcal)<args.context:
        raise AssertionError("invalid real train-only module input activations")
    input_var=xcal.square().mean(0).numpy()
    original_ppl=ppl_eval(model,valid,args.context)
    blob4=fixed_encode(w,4)
    physical_budget=len(blob4)
    blob,metadata=encode(w,input_var,physical_budget,
                         rank=args.rank,m2=8,n2=32,br=16,bc=32)
    if metadata["tiles_3bit"]==0 or len(blob)>physical_budget:
        raise AssertionError("not a physically compressed multidimensional candidate")
    # Save a real binary payload and restore it from DISK, not an in-memory
    # estimated bitcount; fail on corruption, truncated or wrong CRC payload.
    outdir=Path(args.output).parent
    outdir.mkdir(parents=True,exist_ok=True)
    binpath=outdir/"calibri_v32_multidim_bytes.bin"
    binpath.write_bytes(blob)
    restored_blob=binpath.read_bytes()
    if len(restored_blob)!=len(blob) or hashlib.sha256(restored_blob).digest()!=hashlib.sha256(blob).digest():
        raise AssertionError("physical file write/read failed")
    t0=time.perf_counter_ns()
    reconstruction=decode(restored_blob)
    decode_once_ms=(time.perf_counter_ns()-t0)/1e6
    ref4=fixed_decode(blob4)
    assert reconstruction.shape==ref4.shape==w.shape
    if not (np.isfinite(reconstruction).all() and np.isfinite(ref4).all()):
        raise AssertionError("invalid decoded real pretrained weights")
    originals=mod.weight.detach().clone()
    scores={}
    try:
        for key,recon in (("physical_int4_control",ref4),
                          ("multidim_adaptive_3or4bit",reconstruction)):
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(recon))
            scores[key]=ppl_eval(model,valid,args.context)
            if not np.isfinite(scores[key]["ppl"]):
                raise ValueError("nonfinite held-out PPL")
    finally:
        with torch.no_grad():
            mod.weight.copy_(originals)
    # Test mathematical equality of true byte-backed TT + tile operations,
    # separate from full-model PPL (which uses a decoded DENSE weight).
    activ=xcal[:args.microbatch].contiguous()
    if activ.shape[0]!=args.microbatch:
        raise ValueError("too few genuine train activations for kernel diagnostic")
    stream=streamed_matmul(restored_blob,activ)
    dense=torch.nn.functional.linear(activ,torch.from_numpy(reconstruction))
    worst_abs=float(torch.max(torch.abs(stream-dense)))
    if worst_abs>1e-4:
        raise AssertionError("byte-backed streamed 4D TT+tile matmul diverges from decoded W: "+str(worst_abs))
    native_dense_timing=bench(lambda:torch.nn.functional.linear(activ,torch.from_numpy(reconstruction)))
    byte_stream_timing=bench(lambda:streamed_matmul(restored_blob,activ),warmup=1,repeats=5)
    original_weight_timing=bench(lambda:torch.nn.functional.linear(activ,originals))
    # A control uses the same dense reconstruction, not an unfair original
    # teacher with a different quantization error.
    second=metadata["tiles_total"]*metadata["residual_tile_shape"][0]*metadata["residual_tile_shape"][1]
    if second!=w.size:
        raise AssertionError("tile layout incomplete")
    p0=float(original_ppl["ppl"])
    report={
        "experiment":"CALIBRI_v32_real_Pythia14m_4D_TT_adaptive_3_4bit_physical",
        "source_model":MODEL,"source_revision":rev,"corpus":DATASET,
        "corpus_revision":d_rev,
        "trained_MLP_projection":name,"projection_shape":list(w.shape),
        "model_total_parameter_count":int(sum(p.numel() for p in model.parameters())),
        "source_weight_sha256":hashlib.sha256(w.tobytes()).hexdigest(),
        "tokens":{"training":int(len(train)),"validation":int(len(valid)),
                  "context":args.context,"validation_evaluated":original_ppl["tokens"]},
        "token_sha256":{"train":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
                        "validation":hashlib.sha256(valid.numpy().tobytes()).hexdigest()},
        "predeclared_selection":"one alphabetically first pretrained Pythia MLP matrix; all bit allocation uses train-only activation second moments, never validation PPL",
        "source_full_model_PPL":{"validation":original_ppl},
        "quality_PPL_validation":scores,
        "quality_ratios_to_original":{
            k:float(v["ppl"]/p0) for k,v in scores.items()},
        "physical_storage":{
            "fp16_projection_bytes":int(2*w.size),
            "matched_INT4_serialized_bytes":physical_budget,
            "actual_multidim_file_bytes":binpath.stat().st_size,
            "physical_binary_sha256":hashlib.sha256(restored_blob).hexdigest(),
            "physical_binary_filename":binpath.name,
            "multidim_actual_bits_per_weight":8*len(blob)/w.size,
            "INT4_control_actual_bits_per_weight":8*len(blob4)/w.size,
            "multidim_bytes_vs_INT4_control":len(blob)-len(blob4),
            "multidim_reduction_vs_fp16_pct":100*(1-len(blob)/(2*w.size)),
            "multidim_encode_metadata":metadata},
        "microbenchmark":{
            "input":"genuine training-split MLP module activations",
            "input_shape":list(activ.shape),
            "TT_plus_packed_tiled_matmul_vs_same_reconstructed_dense_max_abs":worst_abs,
            "packed_streaming_python_one_tile_scratch_ms":byte_stream_timing,
            "dense_reconstructed_matmul_ms":native_dense_timing,
            "dense_original_teacher_matmul_ms":original_weight_timing,
            "one_time_full_decode_ms":float(decode_once_ms),
            "streaming_over_decoded_dense_median_latency_ratio":(
                byte_stream_timing["median_ms"]/native_dense_timing["median_ms"]),
            "packed_streaming_resident_full_weight":False,
            "packed_streaming_FUSED_NATIVE":False},
        "environment":{"cpu_ATen_dispatch_reported":selected,
                       "oneDNN_max_isa_requested":"AVX2",
                       "torch":str(torch.__version__),"torch_threads":torch.get_num_threads(),
                       "deterministic_algorithms":torch.are_deterministic_algorithms_enabled()},
        "limitations":[
            "One actual trained Pythia-14m MLP projection only, not full-model compressed checkpoint.",
            "Existing tensor-train + tile quantization components combined experimentally, no mathematical novelty proven.",
            "WikiText-2 validation subset previously encountered by CALIBRI research; exploratory, not independent confirmatory result.",
            "Held-out validation PPL executes dense decoded candidate, NOT physically packed stream.",
            "Only the isolated packed TT+residual Python streaming kernel avoids full reconstructed W; no fused native/GPU/ARM packed acceleration.",
            "Microbench numbers are process-local and hardware-dependent, not model-wide inference speed.",
            "No QMoE GPTQ 0.8-bpw result, no full Switch task benchmark.",
            "No main/Render/Supabase/APK/production or ordinary Jessica chat modifications.",
        ],
    }
    dest=Path(args.output)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"experiment":report["experiment"],
                      "actual_bytes":len(blob),"int4_matched_bytes":len(blob4),
                      "original_PPL":p0,"physical_INT4_PPL":scores["physical_int4_control"]["ppl"],
                      "multidim_PPL":scores["multidim_adaptive_3or4bit"]["ppl"],
                      "tile_allocation":[metadata["tiles_3bit"],metadata["tiles_4bit"]],
                      "stream_ms":byte_stream_timing["median_ms"],
                      "dense_decoded_ms":native_dense_timing["median_ms"],
                      "max_abs_kernel":worst_abs},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/calibri_v32_real_weights.json")
    p.add_argument("--train-tokens",type=int,default=256)
    p.add_argument("--validation-tokens",type=int,default=1024)
    p.add_argument("--context",type=int,default=128)
    p.add_argument("--rank",type=int,default=4)
    p.add_argument("--microbatch",type=int,default=32)
    run(p.parse_args())
