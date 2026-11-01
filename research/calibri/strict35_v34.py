#!/usr/bin/env python3
"""CALIBRI v34: real pretrained Pythia 3.5-bpw physical byte/PPL trial.

The data-dependent selection of row and 2-D geometries is TRAIN-only;
neither validation PPL nor test text influence bit allocation.
All codec outputs are true CRC-protected, serialized streams, not estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from direct_tiles_v33 import encode, decode
from direct_tiles_v33 import HEADER as TILED_HEADER
from fixedbit_v15 import encode as fixed_encode, decode as fixed_decode
from quality import MODEL, DATASET, calibrate, load_token_ids, module_selection, ppl_eval

# Material reduction from v33's effective 4.025 bpw. 128*512 weights:
# 65,536 * 3.5 / 8 = exactly 28,672 bytes INCLUDING headers and scales.
BUDGET_BPW=3.5
GEOMETRIES=((1,512),(16,32),(8,64))


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi
    if (os.getenv("ATEN_CPU_CAPABILITY")!="avx2"
            or os.getenv("DNNL_MAX_CPU_ISA")!="AVX2"
            or os.getenv("ONEDNN_MAX_CPU_ISA")!="AVX2"
            or str(torch._C._get_cpu_capability()).upper()!="AVX2"):
        raise RuntimeError("ATen AVX2 and oneDNN AVX2 caps must be selected before Torch import")
    torch.set_num_threads(2)
    torch.manual_seed(34)
    torch.use_deterministic_algorithms(True)
    api=HfApi()
    revision=api.model_info(MODEL).sha
    dataset_revision=api.dataset_info(DATASET).sha
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,torch_dtype=torch.float32)
    model.eval()
    modules=module_selection(model)
    if len(modules)!=12:
        raise AssertionError("unexpected pretrained MLP projection count")
    name=sorted(modules)[0]
    layer=modules[name]
    w=layer.weight.detach().cpu().numpy().astype(np.float32).copy()
    if w.shape!=(128,512):
        raise AssertionError("selected pretrained matrix dimensions changed")
    budget=int(w.size*BUDGET_BPW/8)
    if budget!=28672 or budget>=len(fixed_encode(w,4)):
        raise AssertionError("invalid or not-stricter physical 3.5bpw budget")
    train=load_token_ids(tokenizer,"train",dataset_revision,args.train_tokens)
    valid=load_token_ids(tokenizer,"validation",dataset_revision,args.valid_tokens)
    with torch.inference_mode():
        activ=calibrate(model,{name:layer},train,args.context)[name]
    importance=activ.square().mean(0).cpu().numpy().astype(np.float32)
    original_ppl=ppl_eval(model,valid,args.context)
    original=layer.weight.detach().clone()

    # Uniform rowwise INT3. Its ~3.05 actual bpw is an inexpensive anchor,
    # NOT claimed equal-byte baseline for 3.5bpw formats.
    row3_base=TILED_HEADER.size + w.shape[0]*(3+int(w.shape[1]*3/8))
    if row3_base >= budget:
        raise AssertionError("rowwise INT3 base does not fit 3.5bpw limit")
    row3_blob,row3_info=encode(w,importance,row3_base,br=1,bc=512)
    if row3_info["tiles_4bit"]!=0:
        raise AssertionError("uniform INT3 anchor unexpectedly upgraded precision")
    candidates={
        "int3_rowwise_low_rate":(row3_blob,row3_info),
    }
    for br,bc in GEOMETRIES:
        key=("rowwise" if br==1 else "tiled")+"_"+str(br)+"x"+str(bc)+"_mixed3_4"
        candidates[key]=encode(w,importance,budget,br=br,bc=bc)
    if len(candidates)!=4:
        raise AssertionError("unexpected candidate count")

    # Rowwise 4bit is an OVER-BUDGET high-quality reference at 4.03bpw.
    # It must never be presented as a matched-byte 3.5bpw control.
    int4=fixed_encode(w,4)
    int4_bytes=len(int4)
    decoded={"int4_over_budget_reference":fixed_decode(int4)}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    physical={}
    for key,(blob,details) in candidates.items():
        p=output.parent/(key+".bin")
        p.write_bytes(blob)
        data=p.read_bytes()
        if data!=blob or p.stat().st_size!=len(blob):
            raise AssertionError("physical serialized bytes differ after roundtrip")
        restored=decode(data)  # CRC, headers, dtype, scales, signed symbols checked
        if not np.isfinite(restored).all() or restored.shape!=w.shape:
            raise AssertionError("corrupt physical restored model weights")
        if len(data)>budget:
            raise AssertionError("candidate exceeds global exact storage ceiling")
        decoded[key]=restored
        physical[key]={
            "physical_file_bytes":p.stat().st_size,
            "bits_per_weight_including_all_overhead":8*p.stat().st_size/w.size,
            "physical_file_sha256":hashlib.sha256(data).hexdigest(),
            "actual_3bit_tile_count":details["tiles_3bit"],
            "actual_4bit_tile_count":details["tiles_4bit"],
            "training_activation_weighted_MSE_proxy":details["train_weighted_MSE_proxy_sum"],
            "tile_shape":details["block"],
        }

    # Select the 3.5bpw-cap candidates using ONLY real train activations
    # and physical size; write this choice before any validation PPL calls.
    eligible=[k for k in physical if k!="int3_rowwise_low_rate"]
    selected=min(eligible,key=lambda k:(
        physical[k]["training_activation_weighted_MSE_proxy"],
        physical[k]["physical_file_bytes"],k))

    ppl={}
    try:
        with torch.inference_mode():
            for key,weights in decoded.items():
                layer.weight.copy_(torch.from_numpy(weights))
                result=ppl_eval(model,valid,args.context)
                if not np.isfinite(result["ppl"]):
                    raise AssertionError("nonfinite PPL")
                ppl[key]=result
    finally:
        with torch.no_grad():
            layer.weight.copy_(original)
    # Explicit avoid conflating strong-rate anchors and over-budget references.
    report={
        "experiment":"CALIBRI_v34_strict_3p5bpw_physical_real_Pythia_PPL",
        "model":MODEL,"model_revision":revision,
        "corpus":DATASET,"corpus_revision":dataset_revision,
        "source_trained_projection":name,"projection_shape":list(w.shape),
        "source_weight_sha256":hashlib.sha256(w.tobytes()).hexdigest(),
        "token_sha256":{
            "train":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
            "validation":hashlib.sha256(valid.numpy().tobytes()).hexdigest(),
        },
        "training_tokens":len(train),"validation_tokens":len(valid),
        "evaluation_context":args.context,
        "source_PPL_validation":original_ppl,
        "material_new_byte_ceiling":budget,
        "exact_byte_cap_bits_per_weight_including_overhead":BUDGET_BPW,
        "physical_packed_formats":physical,
        "uniform_int3_rowwise_not_equal_budget":True,
        "int4_over_budget_reference_bytes":int4_bytes,
        "int4_over_budget_reference_bpw":8*int4_bytes/w.size,
        "selection_rule":"lowest TRAIN-only activation-weighted reconstruction MSE among three fixed-cap 3/4bit formats; NO validation selection",
        "selected_3p5bpw_train_only":selected,
        "single_projection_substituted_full_model_PPL":ppl,
        "cpu":{
            "ATen_dispatch_selected":str(torch._C._get_cpu_capability()),
            "oneDNN_cap_requested":"AVX2",
            "torch":str(torch.__version__),
            "determinism_enabled":torch.are_deterministic_algorithms_enabled(),
            "threads":torch.get_num_threads(),
        },
        "limits":[
            "Only one trained Pythia-14M projection out of twelve MLP matrices, not full-model compression.",
            "WikiText validation has been repeatedly used in past CALIBRI research, so these are exploratory, not independent confirmatory quality gains.",
            "PPL uses fully decoded dense physical weights inside Pythia; no native fused compressed kernel or packed throughput measured.",
            "Uniform INT3 has lower bits than the 3.5bpw candidates and INT4 exceeds their budget; label both references honestly.",
            "Block quantization and sensitivity-weighted allocation are prior art, not proven novel multidimensional weights.",
            "No Switch/QMoE 0.8bpw task result, no native ARM speed, no Render/Supabase/APK/main changes.",
        ],
    }
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "experiment":report["experiment"],
        "physical_budget_bytes":budget,
        "physical_bytes":{k:v["physical_file_bytes"] for k,v in physical.items()},
        "source_ppl":original_ppl["ppl"],
        "validation_PPL":{k:v["ppl"] for k,v in ppl.items()},
        "selected_by_train_only":selected,
        "int4_over_budget_bytes":int4_bytes,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="results/calibri_v34_3p5bpw.json")
    ap.add_argument("--train-tokens",type=int,default=256)
    ap.add_argument("--valid-tokens",type=int,default=1024)
    ap.add_argument("--context",type=int,default=128)
    args=ap.parse_args()
    if (args.train_tokens%args.context or args.valid_tokens%args.context):
        raise SystemExit("token counts must align to evaluation context")
    run(args)
