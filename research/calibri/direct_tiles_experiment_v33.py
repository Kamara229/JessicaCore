#!/usr/bin/env python3
"""Evaluate TT-free multidimensional tiled bits on actual pretrained Pythia weights.

Compare real physical byte streams and WikiText PPL to same-physical-byte
INT4 and prior negative v32 TT hybrid. Two block geometries are preregistered;
choose by TRAIN-only weighted reconstruction proxy, not validation PPL.
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

from direct_tiles_v33 import smoke, encode, decode, stream_matmul
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode
from quality import MODEL, DATASET, load_token_ids, ppl_eval, module_selection, calibrate

GEOMETRIES=((16,32),(8,64))

def measure(fn, repeats):
    vals=[]
    with torch.inference_mode():
        for _ in range(2):
            y=fn()
            if not torch.isfinite(y).all():
                raise ValueError("invalid benchmark output")
        for _ in range(repeats):
            start=time.perf_counter_ns()
            y=fn()
            vals.append((time.perf_counter_ns()-start)/1e6)
    return {"median_ms":float(statistics.median(vals)),
            "min_ms":float(min(vals)),"repetitions":len(vals)}


def run(args):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2" and
            os.getenv("DNNL_MAX_CPU_ISA")=="AVX2" and
            os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2"):
        raise AssertionError("ATen=avx2 and oneDNN AVX2 caps must precede torch import")
    if str(torch._C._get_cpu_capability()).upper()!="AVX2":
        raise AssertionError("ATen dispatcher did not actually select AVX2")
    torch.manual_seed(32)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    smoke()

    api=HfApi()
    revision=api.model_info(MODEL).sha
    data_rev=api.dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=revision,torch_dtype=torch.float32)
    model.eval()
    mods=module_selection(model)
    if len(mods)!=12:
        raise AssertionError("expected 12 trained Pythia MLP projections")
    name=sorted(mods)[0]
    module=mods[name]
    w=module.weight.detach().cpu().numpy().astype(np.float32).copy()
    if w.shape!=(128,512):
        raise AssertionError("one-projection MLP geometry changed")
    train=load_token_ids(tok,"train",data_rev,args.train_tokens)
    validation=load_token_ids(tok,"validation",data_rev,args.validation_tokens)
    with torch.inference_mode():
        activ=calibrate(model,{name:module},train,args.context)[name].contiguous()
    moment=activ.square().mean(dim=0).cpu().numpy().astype(np.float32)
    ref_ppl=ppl_eval(model,validation,args.context)

    int4blob=fixed_encode(w,4)
    cap=len(int4blob)
    controls={"fixed_INT4_rowwise":fixed_decode(int4blob)}
    paths={}
    meta={}
    outdir=Path(args.output).parent
    outdir.mkdir(parents=True,exist_ok=True)
    for br,bc in GEOMETRIES:
        key=f"tiles_{br}x{bc}_mixed3or4"
        blob,details=encode(w,moment,cap,br=br,bc=bc)
        p=outdir/(key+".bin")
        p.write_bytes(blob)
        serialized=p.read_bytes()
        if serialized!=blob or len(serialized)>cap:
            raise AssertionError("physical file/INT4 byte cap failure")
        raw=decode(serialized)
        controls[key]=raw
        paths[key]=p
        meta[key]={
            **details,
            "sha256":hashlib.sha256(serialized).hexdigest(),
            "actual_file_bytes":p.stat().st_size,
            "effective_bpw":8*len(serialized)/w.size,
            "training_proxy_original_moment_weighted_error":details["train_weighted_MSE_proxy_sum"],
        }
    # Selection is frozen BEFORE evaluating any held-out validation PPL.
    winner=min(meta, key=lambda key:(
        meta[key]["training_proxy_original_moment_weighted_error"],len(paths[key].read_bytes()),key))
    source=module.weight.detach().clone()
    heldout={}
    try:
        for key,recon in controls.items():
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(recon))
            heldout[key]=ppl_eval(model,validation,args.context)
            if not np.isfinite(heldout[key]["ppl"]):
                raise AssertionError("nonfinite real heldout WikiText PPL")
    finally:
        with torch.no_grad():
            module.weight.copy_(source)

    inp=activ[:args.microbatch]
    if len(inp)!=args.microbatch:
        raise AssertionError("insufficient genuine train activations")
    binpath=paths[winner]
    restored=controls[winner]
    direct=stream_matmul(binpath.read_bytes(),inp)
    dense=torch.nn.functional.linear(inp,torch.from_numpy(restored))
    max_error=float((direct-dense).abs().max())
    if max_error>1e-4:
        raise AssertionError("direct file-backed tile kernel differs from identical reconstructed dense W")

    stream_timing=measure(lambda:stream_matmul(binpath.read_bytes(),inp),5)
    dense_weight=torch.from_numpy(restored)
    dense_timing=measure(lambda:torch.nn.functional.linear(inp,dense_weight),9)
    orig_timing=measure(lambda:torch.nn.functional.linear(inp,source),9)

    report={
        "experiment":"CALIBRI_v33_TT_ablation_direct_adaptive_2D_tiles_pretrained_Pythia",
        "model":MODEL,"model_revision":revision,
        "dataset":DATASET,"dataset_revision":data_rev,
        "target_module":name,"source_weight_shape":list(w.shape),
        "source_weight_sha256":hashlib.sha256(w.tobytes()).hexdigest(),
        "token_sha256":{
            "train":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
            "validation":hashlib.sha256(validation.numpy().tobytes()).hexdigest()},
        "tokens":{"train":len(train),"validation":len(validation),
                  "context":args.context,"ppl_scored_tokens":ref_ppl["tokens"]},
        "selection_rule":"choose block geometry solely by train-activation-weighted reconstruction MSE proxy, fixed BEFORE validation PPL",
        "predeclared_geometries":[list(x) for x in GEOMETRIES],
        "training_selected_candidate":winner,
        "full_model_original_validation_PPL":ref_ppl,
        "full_model_single_projection_substituted_PPL":heldout,
        "physical":{
            "fp16_reference_projection_bytes":2*w.size,
            "fixed_INT4_rowwise_serialized_bytes":cap,
            "fixed_INT4_rowwise_actual_bpw":8*cap/w.size,
            "candidates":meta,
        },
        "streaming_microbenchmark":{
            "selected_candidate":winner,
            "exact_file_backed_vs_reconstructed_dense_max_abs":max_error,
            "input":"genuine Pythia train-split layer activations",
            "input_shape":list(inp.shape),
            "file_backed_python_tiled_stream":stream_timing,
            "same_decoded_dense_matmul":dense_timing,
            "original_teacher_dense_matmul":orig_timing,
            "streaming_median_over_dense_median":float(stream_timing["median_ms"]/dense_timing["median_ms"]),
            "full_dense_weight_allocation_inside_tiled_stream":False,
            "native_fused_packed_kernel":False,
        },
        "hardware":{
            "torch":str(torch.__version__),
            "ATen_dispatch_selected":str(torch._C._get_cpu_capability()),
            "oneDNN_max_requested":"AVX2",
            "threads":torch.get_num_threads(),
            "deterministic":torch.are_deterministic_algorithms_enabled(),
        },
        "limitations":[
            "Only ONE trained Pythia14M MLP projection altered; NOT full-model physical compression.",
            "Tensor-train was omitted as a controlled ablation of failed v32 geometry. Block quantization is prior art.",
            "Validation WikiText subset used by earlier CALIBRI sessions, so exploratory not independent heldout proof.",
            "The train-only proxy selected the block geometry; report both validation PPLs for exploratory transparency.",
            "PPL executes dense restored candidate weights; file-backed streamed Python microbenchmark is a separate operator.",
            "Python tile loops are not compiled packed native inference, no ARM/GPU/model-wide speed claim.",
            "No QMoE 0.8bpw, pretrained Switch MoE PPL or production change.",
        ],
    }
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary={
        "experiment":report["experiment"],
        "selected":winner,
        "original_PPL":ref_ppl["ppl"],
        "rowwise_INT4_PPL":heldout["fixed_INT4_rowwise"]["ppl"],
        "candidate_PPLs":{k:v["ppl"] for k,v in heldout.items() if k!="fixed_INT4_rowwise"},
        "physical_BYTES":{k:x["actual_file_bytes"] for k,x in meta.items()},
        "baseline_INT4_BYTES":cap,
        "streamed_Python_ms":stream_timing["median_ms"],
        "restored_dense_ms":dense_timing["median_ms"],
        "kernel_max_abs":max_error,
    }
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/calibri_v33_real_Pythia.json")
    p.add_argument("--train-tokens",type=int,default=256)
    p.add_argument("--validation-tokens",type=int,default=1024)
    p.add_argument("--context",type=int,default=128)
    p.add_argument("--microbatch",type=int,default=32)
    run(p.parse_args())
