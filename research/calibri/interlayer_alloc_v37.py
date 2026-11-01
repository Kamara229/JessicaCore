#!/usr/bin/env python3
"""CALIBRI v37: train-only cross-layer INT3/INT4 bit allocation on REAL Pythia.

Compare GLOBAL all-MLP rate allocation vs old LOCAL-per-matrix allocation,
both with unchanged v35 TRAIN-only scale calibration. Physical budget is
hard-frozen to actual v36 343840 bytes INCLUDING CB36 package metadata.
No evaluation PPL or AG-News text influences bit allocation. The row
selection is a simple greedy heuristic, not a proven globally optimal
solution or a new mathematical class of neural weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib

import numpy as np
import torch

from all_mlp_agnews_v36 import package, unpack_bundle, module_identity, AG_NEWS, N_MATRICES
from calibrated_rows_v35 import calibrated_encode
from direct_tiles_v33 import encode as local_encode,decode as tile_decode,HEADER,MAGIC
from multidim_v32 import packed_symbols,quant_tile
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode
from quality import MODEL, DATASET,calibrate,load_token_ids,module_selection,ppl_eval

GLOBAL_LIMIT_BYTES=343840  # actual physically measured v36 full 12-MLP bundle
TOTAL_PARAMETERS=786432
AG_SEALED_SLICE="test[3000:3200]" # disjoint from first-articles v36 AG test probe
ORDER=("local_3p5bpw_baseline","local_3p5bpw_calibrated",
       "global_train_greedy_uncalibrated","global_train_greedy_calibrated",
       "INT4_overbudget_reference")
ALL_CONFIGURATIONS=("source_fp32_original",)+ORDER


def untouched_ag_slice(tokenizer,revision,token_count):
    from datasets import load_dataset
    ds=load_dataset(AG_NEWS,split=AG_SEALED_SLICE,revision=revision)
    parts=[str(row["text"]).strip() for row in ds if str(row.get("text","")).strip()]
    tokens=tokenizer("\n\n".join(parts),return_tensors="pt",
                     add_special_tokens=False).input_ids[0,:token_count]
    if len(tokens)!=token_count:
        raise AssertionError("AG News sealed slice lacks enough independent tokens")
    return tokens,len(parts)


def pack_row_assignment(w, bitwidth):
    """Self-describing actual CT33 bytes for one per-row INT3/INT4 allocation."""
    w=np.asarray(w,dtype=np.float32)
    if w.ndim!=2 or len(bitwidth)!=w.shape[0] or w.shape[1]%8:
        raise ValueError("invalid row allocation geometry")
    m,n=w.shape
    payload=bytearray()
    for i,bits in enumerate(bitwidth):
        if bits not in (3,4):
            raise ValueError("invalid bitwidth")
        scale,symbols,_=quant_tile(w[i:i+1,:],bits)
        payload.extend(struct.pack("<Be",bits,float(scale)))
        payload.extend(symbols)
    raw=bytes(payload)
    return HEADER.pack(MAGIC,m,n,1,n,len(raw),zlib.crc32(raw))+raw


def true_train_row_errors(weights,activations):
    """Per-row ACTUAL trained module output SSE for INT3 vs INT4."""
    w=np.asarray(weights,dtype=np.float32)
    x=np.asarray(activations,dtype=np.float32)
    rows,features=w.shape
    if x.shape[1]!=features or len(x)<64:
        raise AssertionError("not genuine train-layer activation input")
    q3=np.empty_like(w)
    q4=np.empty_like(w)
    for i in range(rows):
        q3[i]=quant_tile(w[i:i+1],3)[2].reshape(-1)
        q4[i]=quant_tile(w[i:i+1],4)[2].reshape(-1)
    e3=x@(w-q3).T
    e4=x@(w-q4).T
    s3=np.sum(e3.astype(np.float64)**2,axis=0)
    s4=np.sum(e4.astype(np.float64)**2,axis=0)
    if not np.isfinite(s3).all() or not np.isfinite(s4).all():
        raise AssertionError("train-only error is nonfinite")
    return s3,s4,q3


def greedy_layout(module_order,matrix,train_inputs,source_train_nll,model):
    """Measure TRAIN-only individual pure-INT3 ablation and allocate bits globally.

    Each candidate 3→4bit row upgrade is scored by
      positive TRAIN output SSE gain / (row incremental bytes),
    multiplied by a clipped TRAIN NLL module sensitivity signal.
    The measured pure-INT3 layer perturbation never occurs on AG News.
    """
    train_errors={}
    penalties={}
    all3={}
    for name in module_order:
        w=matrix[name]
        x=train_inputs[name]
        s3,s4,q3=true_train_row_errors(w,x)
        train_errors[name]={"s3":s3,"s4":s4}
        with torch.inference_mode():
            target=model.get_submodule(name)
            original=target.weight.detach().clone()
            try:
                target.weight.copy_(torch.from_numpy(q3))
                ablated=ppl_eval(model,TRAIN_TOKENS,CONTEXT)["nll"]
            finally:
                target.weight.copy_(original)
        # Positive TRAIN-NLL perturbation indicates local layer sensitivity.
        # Clamp to a bounded range to prevent one highly noisy ablation from
        # taking nearly all global bit allowance.
        penalties[name]=max(float(ablated-source_train_nll),1e-8)
        all3[name]=[3]*w.shape[0]
        print(json.dumps({"train_only_ablation_module":name,
                          "source_train_NLL":source_train_nll,
                          "pure_INT3_train_NLL":ablated,
                          "positive_NLL_penalty":penalties[name]}),flush=True)
    median=float(np.median(list(penalties.values())))
    multiplier={n:float(np.clip(penalties[n]/max(median,1e-8),0.25,4.0))
                for n in module_order}

    original3={n:pack_row_assignment(matrix[n],all3[n]) for n in module_order}
    base_bundle=package(original3,1)
    if len(base_bundle)>GLOBAL_LIMIT_BYTES:
        raise AssertionError("even global pure INT3 exceeds prior recorded physical byte cap")
    candidates=[]
    for n in module_order:
        width=matrix[n].shape[1]
        incremental=width//8
        for index,(e3,e4) in enumerate(zip(train_errors[n]["s3"],train_errors[n]["s4"])):
            gain=max(0.0,float(e3-e4))
            if gain>0.:
                score=gain*multiplier[n]/incremental
                candidates.append((-score,module_identity(n),index,n,incremental,gain))
    candidates.sort()
    available=GLOBAL_LIMIT_BYTES-len(base_bundle)
    upgrades={}
    total_gains=0.
    for _negscore,_ident,i,n,bytes_needed,gain in candidates:
        if bytes_needed<=available:
            all3[n][i]=4
            available-=bytes_needed
            upgrades[n]=upgrades.get(n,0)+1
            total_gains+=gain*multiplier[n]
    physical={n:pack_row_assignment(matrix[n],all3[n]) for n in module_order}
    length=len(package(physical,1))
    if length>GLOBAL_LIMIT_BYTES or length!=GLOBAL_LIMIT_BYTES-available:
        raise AssertionError("byte-count accounting failed for global allocation")
    return physical,{
        "method":"TRAIN-only layer pure-INT3 individual NLL penalty times per-row 3-to-4-bit TRAIN output-SSE positive gain divided by byte increment, greedy",
        "base_all3_total_bundle_bytes":len(base_bundle),
        "global_fixed_max_bytes":GLOBAL_LIMIT_BYTES,
        "greedy_used_bytes":length,
        "unused_global_budget_bytes":available,
        "train_only_pure_INT3_layer_NLL_penalties":penalties,
        "train_only_clipped_layer_multipliers":multiplier,
        "row_upgrades_to_INT4_per_module":upgrades,
        "train_weighted_row_upgrade_proxy_gain":total_gains,
    }


def evaluate(model, originals, substitutions, corpus,context):
    out={}
    try:
        with torch.inference_mode():
            for label,replacement in substitutions.items():
                for name,weight in originals.items():
                    model.get_submodule(name).weight.copy_(weight)
                if replacement is not None:
                    if set(replacement)!=set(originals):
                        raise AssertionError("missing one of 12 full pretrained modules")
                    for name,q in replacement.items():
                        module=model.get_submodule(name)
                        if q.shape!=tuple(module.weight.shape) or not np.isfinite(q).all():
                            raise AssertionError("invalid full matrix replacement")
                        module.weight.copy_(torch.from_numpy(q))
                out[label]={}
                for cname,ids in corpus.items():
                    res=ppl_eval(model,ids,context)
                    if not np.isfinite(res["ppl"]):
                        raise AssertionError("PPL nonfinite")
                    out[label][cname]=res
                    print(json.dumps({"configuration":label,"corpus":cname,
                                      "PPL":res["ppl"]}),flush=True)
    finally:
        with torch.no_grad():
            for name,weight in originals.items():
                model.get_submodule(name).weight.copy_(weight)
    return out


# The inputs below are assigned only to sealed, fixed WikiText TRAIN
# tokens in run(), never to development or new AG News evaluation samples.
TRAIN_TOKENS=None
CONTEXT=None


def run(args):
    global TRAIN_TOKENS,CONTEXT
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer,AutoModelForCausalLM

    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2"
            and os.getenv("DNNL_MAX_CPU_ISA")=="AVX2"
            and os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2"
            and str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("AVX2 selected CPU dispatch/oneDNN cap required")
    torch.set_num_threads(2)
    torch.manual_seed(37)
    torch.use_deterministic_algorithms(True)
    api=HfApi()
    model_sha=api.model_info(MODEL).sha
    wiki_sha=api.dataset_info(DATASET).sha
    ag_sha=api.dataset_info(AG_NEWS).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=model_sha)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=model_sha,
                                             torch_dtype=torch.float32)
    model.eval()
    modules=module_selection(model)
    if len(modules)!=N_MATRICES:
        raise AssertionError("12 source MLPs required")
    names=sorted(modules,key=module_identity)
    sources={n:modules[n].weight.detach().clone() for n in names}
    matrices={n:sources[n].cpu().numpy().astype(np.float32).copy() for n in names}
    if sum(w.size for w in matrices.values())!=TOTAL_PARAMETERS:
        raise AssertionError("original pretrained 12-MLP geometry changed")

    TRAIN_TOKENS=load_token_ids(tok,"train",wiki_sha,args.train_tokens)
    CONTEXT=args.context
    wiki=load_token_ids(tok,"validation",wiki_sha,args.dev_tokens)
    ag,agdocs=untouched_ag_slice(tok,ag_sha,args.ag_tokens)
    with torch.inference_mode():
        train_source_NLL=ppl_eval(model,TRAIN_TOKENS,CONTEXT)["nll"]
        inputs=calibrate(model,modules,TRAIN_TOKENS,CONTEXT)
    if set(inputs)!=set(modules):
        raise AssertionError("not all genuine pretrained layer activations were captured")
    x={n:inputs[n].detach().cpu().numpy().astype(np.float32) for n in names}

    # REPRODUCE exact v36 local-per-matrix baseline, with 12 independent
    # 3.5bpw budgets, before global bits are re-allocated.
    local={}
    for name in names:
        w=matrices[name]
        second=(x[name]*x[name]).mean(axis=0,dtype=np.float32)
        blob,metadata=local_encode(w,second,int(w.size*3.5/8),br=1,bc=w.shape[1])
        if len(blob)>int(w.size*3.5/8):
            raise AssertionError("local baseline 3.5bpw cap exceeded")
        local[name]=blob
    local_file=package(local,1)
    if len(local_file)!=GLOBAL_LIMIT_BYTES:
        raise AssertionError("v36 pre-registered total baseline bytes are no longer 343840")

    global_blobs,layout=greedy_layout(names,matrices,x,train_source_NLL,model)
    adjusted={}
    local_adjusted={}
    per_matrix={}
    for n in names:
        adjusted[n],cal_meta=calibrated_encode(matrices[n],x[n],global_blobs[n])
        local_adjusted[n],local_meta=calibrated_encode(matrices[n],x[n],local[n])
        if len(adjusted[n])!=len(global_blobs[n]) or len(local_adjusted[n])!=len(local[n]):
            raise AssertionError("calibration altered file length or bit allocation")
        per_matrix[n]={
            "shape":list(matrices[n].shape),
            "source_SHA":hashlib.sha256(matrices[n].tobytes()).hexdigest(),
            "local_bytes":len(local[n]),
            "global_bytes":len(global_blobs[n]),
            "local_calibrated_train_SSE_ratio":local_meta["train_output_SSE_ratio"],
            "global_calibrated_train_SSE_ratio":cal_meta["train_output_SSE_ratio"],
            "global_4bit_row_count":cal_meta["row_bitwidths"].count(4),
            "local_4bit_row_count":local_meta["row_bitwidths"].count(4),
        }
    int4={n:fixed_encode(matrices[n],4) for n in names}
    encoded={
        "local_3p5bpw_baseline":local,
        "local_3p5bpw_calibrated":local_adjusted,
        "global_train_greedy_uncalibrated":global_blobs,
        "global_train_greedy_calibrated":adjusted,
        "INT4_overbudget_reference":int4,
    }
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    physical={}
    restored={}
    for label,blobs in encoded.items():
        mode=2 if label=="INT4_overbudget_reference" else 1
        real_binary=package(blobs,mode)
        file=output.parent/(label+".bin")
        file.write_bytes(real_binary)
        read=file.read_bytes()
        if read!=real_binary:
            raise AssertionError("real physical bundle file changed when read")
        modules_bytes=unpack_bundle(read,mode)
        restored[label]={n:fixed_decode(b) if mode==2 else tile_decode(b)
                         for n,b in modules_bytes.items()}
        physical[label]={
            "actual_disk_bytes":file.stat().st_size,
            "effective_bits_per_MLP_weight":8*file.stat().st_size/TOTAL_PARAMETERS,
            "sha256":hashlib.sha256(read).hexdigest(),
            "matrix_count":len(modules_bytes),
        }
    lsize=physical["local_3p5bpw_baseline"]["actual_disk_bytes"]
    gsize=physical["global_train_greedy_uncalibrated"]["actual_disk_bytes"]
    if not (lsize==physical["local_3p5bpw_calibrated"]["actual_disk_bytes"]==
            GLOBAL_LIMIT_BYTES and
            gsize==physical["global_train_greedy_calibrated"]["actual_disk_bytes"]<=GLOBAL_LIMIT_BYTES):
        raise AssertionError("global compression byte budget or matched calibration violated")
    if physical["INT4_overbudget_reference"]["actual_disk_bytes"]<=GLOBAL_LIMIT_BYTES:
        raise AssertionError("INT4 must be reported as higher physical budget")
    corpora={"WikiText2_development":wiki,"AG_NEWS_sealed_test_3000_3200":ag}
    variants={"source_fp32_original":None,**restored}
    scores=evaluate(model,sources,variants,corpora,CONTEXT)
    report={
        "experiment":"CALIBRI_v37_pinned_3p5bpw_global_vs_local_train_layer_sensitivity",
        "model":MODEL,"model_sha":model_sha,
        "datasets":{"wiki":{"name":DATASET,"sha":wiki_sha,"train":"train",
                             "dev":"validation"},
                    "AG_new_sealed":{"name":AG_NEWS,"sha":ag_sha,
                                     "split":AG_SEALED_SLICE,"documents":agdocs}},
        "token_sha256":{
            "wiki_train":hashlib.sha256(TRAIN_TOKENS.numpy().tobytes()).hexdigest(),
            "wiki_dev":hashlib.sha256(wiki.numpy().tobytes()).hexdigest(),
            "AG_sealed":hashlib.sha256(ag.numpy().tobytes()).hexdigest(),
        },
        "token_counts":{"train":len(TRAIN_TOKENS),"wiki_dev":len(wiki),
                        "AG_sealed":len(ag),"context":CONTEXT},
        "source_MLP_coefficient_count":TOTAL_PARAMETERS,
        "source_MLP_FP16_theoretical_bytes":2*TOTAL_PARAMETERS,
        "frozen_global_limit_bytes":GLOBAL_LIMIT_BYTES,
        "actual_physical_bundles":physical,
        "per_matrix":per_matrix,
        "train_only_global_allocation":layout,
        "full_model_PPL_with_all12_MLP_restored":scores,
        "quality_delta_global_minus_local_CALIBRATED":{
            corpus:scores["global_train_greedy_calibrated"][corpus]["ppl"]-
            scores["local_3p5bpw_calibrated"][corpus]["ppl"]
            for corpus in corpora},
        "pinned_cpu":{"ATen_selected":str(torch._C._get_cpu_capability()),
                      "oneDNN_requested":"AVX2",
                      "torch":str(torch.__version__),
                      "torch_threads":torch.get_num_threads(),
                      "deterministic":torch.are_deterministic_algorithms_enabled()},
        "limitations":[
            "This changes rowwise bit precision ALLOCATION between layers but not any underlying neural weight architecture.",
            "Global bit allocation uses WikiText TRAIN actual single-matrix INT3 NLL sensitivities and per-row output SSE, no WikiText dev or AG test.",
            "AG News test slice is disjoint from v36 first article evaluation but is the SAME AG News dataset, not a brand-new dataset or independent broad benchmark.",
            "The maximum byte budget is the exact 343840B v36 physical bundle; original unchanged and INT4 above-budget baselines are separately labeled.",
            "Only 12 MLP matrices are physically bundled; attention/embedding/norms not compressed and no deployable full-model file exists.",
            "Full-model language PPL uses fully decoded dense reconstructed weights; no actual packed native inference latency is established.",
            "Greedy training score is a heuristic, not a proof of globally optimal rate-distortion nor a QMoE 0.8bpw replication.",
        ],
    }
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "experiment":report["experiment"],
        "physical_bytes":{k:v["actual_disk_bytes"] for k,v in physical.items()},
        "rowwise_4bit_by_matrix":layout["row_upgrades_to_INT4_per_module"],
        "PPL":{key:{c:round(score["ppl"],5) for c,score in v.items()}
               for key,v in scores.items()},
        "global_minus_local":report["quality_delta_global_minus_local_CALIBRATED"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/calibri_v37_global_layer_bits.json")
    p.add_argument("--train-tokens",type=int,default=256)
    p.add_argument("--dev-tokens",type=int,default=1024)
    p.add_argument("--ag-tokens",type=int,default=2048)
    p.add_argument("--context",type=int,default=128)
    a=p.parse_args()
    if any(v%a.context for v in (a.train_tokens,a.dev_tokens,a.ag_tokens)):
        raise ValueError("all token limits must be non-overlapping whole contexts")
    run(a)
