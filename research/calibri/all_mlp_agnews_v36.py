#!/usr/bin/env python3
"""CALIBRI v36: physical compression of ALL 12 real Pythia-14m MLP matrices.

Reuses the unchanged v35 rowwise bit-allocation and train-output
covariance scale/rounding calibration. ONE new predeclared AG News TEST
corpus stress-tests it beyond WikiText/TinyStories. Both original and
calibrated physical bundles use EXACT same bits per row and byte counts.
PPL evaluation uses restored dense weights, not optimized packed kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

import numpy as np
import torch

from calibrated_rows_v35 import calibrated_encode
from direct_tiles_v33 import encode, decode
from fixedbit_v15 import encode as fixed_encode, decode as fixed_decode
from quality import MODEL, DATASET, calibrate, load_token_ids, module_selection, ppl_eval

AG_NEWS = "fancyzhx/ag_news"
AG_SPLIT = "test"
BUDGET_BPW_PER_MATRIX = 3.5
BUNDLE_MAGIC = b"CB36"
BUNDLE_HEADER = struct.Struct("<4sBBH")
BUNDLE_ITEM = struct.Struct("<BBI")  # layer idx, projection kind, bytes
N_MATRICES = 12


def module_identity(name):
    m = re.fullmatch(r"gpt_neox\.layers\.(\d+)\.mlp\.(dense_h_to_4h|dense_4h_to_h)", name)
    if m is None or not (0 <= int(m.group(1)) < 6):
        raise ValueError("Unexpected Pythia module naming: " + name)
    return int(m.group(1)), 0 if m.group(2) == "dense_h_to_4h" else 1


def package(blobs, mode):
    """Self-describing fixed-Pythia-layout physical 12-matrix byte container."""
    if len(blobs) != N_MATRICES or mode not in (1, 2):
        raise AssertionError("incomplete Pythia MLP manifest")
    payload=bytearray(BUNDLE_HEADER.pack(BUNDLE_MAGIC,1,mode,len(blobs)))
    seen=set()
    for name,blob in sorted(blobs.items(), key=lambda item: module_identity(item[0])):
        lid,kind=module_identity(name)
        if (lid,kind) in seen or not isinstance(blob,bytes) or len(blob)<5:
            raise AssertionError("invalid or duplicated source module")
        seen.add((lid,kind))
        payload.extend(BUNDLE_ITEM.pack(lid,kind,len(blob)))
        payload.extend(blob)
    if seen!={(layer,kind) for layer in range(6) for kind in range(2)}:
        raise AssertionError("full 12-MLP matrix coverage not achieved")
    return bytes(payload)+hashlib.sha256(payload).digest()


def unpack_bundle(blob, expected_mode):
    if len(blob)<BUNDLE_HEADER.size+32:
        raise ValueError("truncated physical 12-matrix package")
    magic,version,mode,count=BUNDLE_HEADER.unpack_from(blob)
    if magic!=BUNDLE_MAGIC or version!=1 or mode!=expected_mode or count!=12:
        raise ValueError("invalid physical bundle schema")
    payload=blob[:-32]
    if hashlib.sha256(payload).digest()!=blob[-32:]:
        raise ValueError("physical bundle checksum invalid")
    result={}
    off=BUNDLE_HEADER.size
    for _ in range(count):
        if off+BUNDLE_ITEM.size>len(payload):
            raise ValueError("truncated matrix descriptor")
        lid,kind,length=BUNDLE_ITEM.unpack_from(payload,off)
        off+=BUNDLE_ITEM.size
        if lid>=6 or kind not in (0,1) or length<=0 or off+length>len(payload):
            raise ValueError("invalid matrix geometry/length")
        name=f"gpt_neox.layers.{lid}.mlp."+(
            "dense_h_to_4h" if kind==0 else "dense_4h_to_h")
        if name in result:
            raise ValueError("repeated matrix in physical package")
        result[name]=payload[off:off+length]
        off+=length
    if off!=len(payload) or len(result)!=12:
        raise ValueError("extra bytes, wrong matrix count or truncated file")
    return result


def load_ag_news(tokenizer, dataset_rev, limit):
    from datasets import load_dataset
    ds=load_dataset(AG_NEWS,split=AG_SPLIT,revision=dataset_rev)
    chunks=[]
    for item in ds:
        t=str(item["text"]).strip()
        if t:chunks.append(t)
        if sum(len(c) for c in chunks)>limit*14:
            break
    if not chunks:
        raise AssertionError("AG News TEST has no textual examples")
    ids=tokenizer("\n\n".join(chunks),return_tensors="pt",add_special_tokens=False).input_ids[0,:limit]
    if len(ids)<limit:
        raise AssertionError("AG News new test corpus is too short")
    return ids,len(chunks)


@torch.inference_mode()
def evaluate_configs(model,source_matrices,configurations,corpora,context):
    results={}
    try:
        for label,replacement in configurations.items():
            # Critical: restore every source matrix before applying ANY variant.
            for n, original in source_matrices.items():
                model.get_submodule(n).weight.copy_(original)
            if replacement is not None:
                if set(replacement)!=set(source_matrices):
                    raise AssertionError("full 12-MLP replacement missing")
                for n,arr in replacement.items():
                    if not np.isfinite(arr).all():
                        raise AssertionError("nonfinite decoded MLP")
                    mod=model.get_submodule(n)
                    if tuple(mod.weight.shape)!=tuple(arr.shape):
                        raise AssertionError("mismatched trained projection geometry")
                    mod.weight.copy_(torch.from_numpy(arr))
            results[label]={}
            for corpus,ids in corpora.items():
                score=ppl_eval(model,ids,context)
                if not np.isfinite(score["ppl"]) or score["ppl"]<=0:
                    raise AssertionError("nonfinite PPL for "+label+" on "+corpus)
                results[label][corpus]=score
                print(json.dumps({"configuration":label,"corpus":corpus,
                                  "PPL":score["ppl"],"NLL":score["nll"]}),
                      flush=True)
    finally:
        for n,original in source_matrices.items():
            model.get_submodule(n).weight.copy_(original)
    return results


def run(args):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if (os.getenv("ATEN_CPU_CAPABILITY")!="avx2"
        or os.getenv("DNNL_MAX_CPU_ISA")!="AVX2"
        or os.getenv("ONEDNN_MAX_CPU_ISA")!="AVX2"
        or str(torch._C._get_cpu_capability()).upper()!="AVX2"):
        raise AssertionError("v36 CPU dispatcher is not pinned to ATen and oneDNN AVX2")
    torch.manual_seed(36)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    api=HfApi()
    model_rev=api.model_info(MODEL).sha
    wiki_rev=api.dataset_info(DATASET).sha
    ag_rev=api.dataset_info(AG_NEWS).sha
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=model_rev,torch_dtype=torch.float32)
    model.eval()
    modules=module_selection(model)
    if len(modules)!=N_MATRICES:
        raise AssertionError("Expected exactly 12 pretrained Pythia MLP matrices")
    if {module_identity(n) for n in modules}!={(i,k) for i in range(6) for k in range(2)}:
        raise AssertionError("Unexpected MLP module map")
    names=sorted(modules,key=module_identity)
    source={n:modules[n].weight.detach().clone() for n in names}
    original_numpy={
        n:source[n].cpu().numpy().astype(np.float32).copy()
        for n in names
    }
    for n,w in original_numpy.items():
        if w.size!=65536 or w.ndim!=2 or w.shape not in ((512,128),(128,512)):
            raise AssertionError("Incorrect Pythia trained MLP source matrix "+n)
    total_weights=sum(w.size for w in original_numpy.values())
    assert total_weights==786432

    train=load_token_ids(tokenizer,"train",wiki_rev,args.train_tokens)
    wiki=load_token_ids(tokenizer,"validation",wiki_rev,args.eval_tokens)
    ag,ag_docs=load_ag_news(tokenizer,ag_rev,args.eval_tokens)
    with torch.inference_mode():
        inputs=calibrate(model,modules,train,args.context)
    if set(inputs)!=set(modules):
        raise AssertionError("Train activation capture did not reach every source MLP module")

    base_blobs={}
    calibrated_blobs={}
    int4_blobs={}
    per_module={}
    for index,n in enumerate(names):
        w=original_numpy[n]
        x=inputs[n].numpy().astype(np.float32)
        if x.shape!=(len(train),w.shape[1]):
            raise AssertionError("TRAIN layer-input activation dimensions changed")
        var=np.square(x,dtype=np.float32).mean(axis=0)
        cap=int(w.size*BUDGET_BPW_PER_MATRIX/8)
        base,baseline_meta=encode(w,var,cap,br=1,bc=w.shape[1])
        calibrated,calmeta=calibrated_encode(w,x,base)
        if len(base)!=len(calibrated) or len(calibrated)>cap:
            raise AssertionError("same allocation different bytes or 3.5bpw per-matrix cap exceeded")
        bit_count=baseline_meta["tiles_3bit"]+baseline_meta["tiles_4bit"]
        if bit_count!=w.shape[0] or not calmeta["bits_are_identical_per_row"]:
            raise AssertionError("rowwise bit assignment lost")
        base_blobs[n]=base
        calibrated_blobs[n]=calibrated
        int4_blobs[n]=fixed_encode(w,4)
        per_module[n]={
            "shape":list(w.shape),
            "source_sha256":hashlib.sha256(w.tobytes()).hexdigest(),
            "rowwise_3p5bpw_bytes":len(base),
            "calibrated_same_byte_size":len(calibrated),
            "above_budget_int4_bytes":len(int4_blobs[n]),
            "bitwidth_rows":{"INT3":baseline_meta["tiles_3bit"],
                             "INT4":baseline_meta["tiles_4bit"]},
            "train_output_SSE_calibrated_divided_by_baseline":
                calmeta["train_output_SSE_ratio"],
            "train_rows_changed":calmeta["rows_changed"],
        }
        print(json.dumps({
            "module_idx":index+1,"of":len(names),
            "name":n,
            "bytes":len(base),
            "train_SSE_ratio":calmeta["train_output_SSE_ratio"],
        }),flush=True)

    bundle_bytes={
        "uncalibrated_3p5bpw_all12":package(base_blobs,1),
        "calibrated_3p5bpw_all12":package(calibrated_blobs,1),
        "INT4_above_budget_all12":package(int4_blobs,2),
    }
    cap_total=int(total_weights*3.5/8)
    serialized={}
    restored={}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    for label,binary in bundle_bytes.items():
        file=output.parent/(label+".bin")
        file.write_bytes(binary)
        read=file.read_bytes()
        if read!=binary:raise AssertionError("physical model binary roundtrip changed")
        decompressed=unpack_bundle(read,1 if "3p5bpw" in label else 2)
        if set(decompressed)!=set(names):
            raise AssertionError("physical checkpoint missing MLP matrices")
        mat={
            n:decode(decompressed[n]) if "3p5bpw" in label else fixed_decode(decompressed[n])
            for n in names
        }
        restored[label]=mat
        serialized[label]={
            "actual_disk_bytes":file.stat().st_size,
            "effective_bits_per_MLP_weight_including_bundling":8*len(read)/total_weights,
            "sha256":hashlib.sha256(read).hexdigest(),
            "matrix_count":len(decompressed),
        }
    if not (serialized["uncalibrated_3p5bpw_all12"]["actual_disk_bytes"]==
            serialized["calibrated_3p5bpw_all12"]["actual_disk_bytes"]<=cap_total):
        raise AssertionError("3.5bpw cap including bundle/checksums exceeded")
    if serialized["INT4_above_budget_all12"]["actual_disk_bytes"]<=cap_total:
        raise AssertionError("INT4 must be labeled an over-budget control")
    if not all(np.array_equal(decode(base_blobs[n]),restored["uncalibrated_3p5bpw_all12"][n])
               for n in names):
        raise AssertionError("bundle decoding changed a baseline coefficient")

    configs={
        "original_source_model":None,
        "uncalibrated_3p5bpw_all12":restored["uncalibrated_3p5bpw_all12"],
        "calibrated_3p5bpw_all12":restored["calibrated_3p5bpw_all12"],
        "INT4_above_budget_all12":restored["INT4_above_budget_all12"],
    }
    ppl=evaluate_configs(
        model,source,configs,
        {"WikiText2_dev_validation":wiki,"AG_NEWS_NEW_test":ag},args.context)
    report={
        "experiment":"CALIBRI_v36_all12_Pythia_MLP_physical_3p5bpw_AG_NEWS_test",
        "scope":"All 12 pretrained Pythia-14m dense MLP projections, 6 layers; original attention, embeddings, norms unchanged.",
        "model":MODEL,"model_revision":model_rev,
        "datasets":{
            "train_and_dev":{"name":DATASET,"revision":wiki_rev,"train_split":"train","dev_split":"validation"},
            "new_corpus":{"name":AG_NEWS,"revision":ag_rev,"split":AG_SPLIT,"examples_used":ag_docs},
        },
        "token_sha256":{
            "calibration_WikiText2_train":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
            "development_WikiText2_validation":hashlib.sha256(wiki.numpy().tobytes()).hexdigest(),
            "NEW_AG_NEWS_test":hashlib.sha256(ag.numpy().tobytes()).hexdigest(),
        },
        "tokens":{"train":len(train),"development":len(wiki),
                  "NEW_AG_NEWS_test":len(ag),"context":args.context},
        "source_total_parameters":int(sum(p.numel() for p in model.parameters())),
        "source_MLP_projection_weights":total_weights,
        "source_MLP_FP16_reference_bytes_not_saved":total_weights*2,
        "physical_global_MLP_3p5bpw_byte_budget_with_bundle":cap_total,
        "physical_bundled_MLPs":serialized,
        "per_source_MLP_module":per_module,
        "per_row_bits_identical_in_calibrated":True,
        "calibration":"same fixed train-activation covariance FP16 scale grid as v35, no validation or AG News access during calibration",
        "full_model_dense_restored_PPL":ppl,
        "relative_quality":{
            corpus:{
                "calibrated_minus_uncalibrated_PPL":
                    ppl["calibrated_3p5bpw_all12"][corpus]["ppl"]-
                    ppl["uncalibrated_3p5bpw_all12"][corpus]["ppl"],
                "calibrated_divided_by_uncalibrated_PPL":
                    ppl["calibrated_3p5bpw_all12"][corpus]["ppl"]/
                    ppl["uncalibrated_3p5bpw_all12"][corpus]["ppl"],
            } for corpus in ("WikiText2_dev_validation","AG_NEWS_NEW_test")
        },
        "cpu_control":{
            "ATen_selected":str(torch._C._get_cpu_capability()),
            "oneDNN_requested_cap":"AVX2",
            "torch":str(torch.__version__),
            "threads":torch.get_num_threads(),
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        },
        "limits":[
            "AG News TEST is first predeclared CALIBRI use on this model; a single one-shot cross-domain check, not repeated benchmark generalization.",
            "Calibration on WikiText TRAIN; any earlier CALIBRI decisions used WikiText validation and TinyStories, which are no longer untouched.",
            "Full-model PPL still restores all compressed projections as dense weights, NOT physically packed runtime.",
            "Physically stored bundle contains ONLY 12 compressed MLP matrices, NOT a deployable full-model checkpoint.",
            "Source FP16 MLP weight bytes are a theoretical reference, not an original serialized model file.",
            "Simple rowwise INT3/INT4 coding and covariance rounding are known quantization methods, not demonstrated new geometric multidimensional weights.",
            "No hardware/ARM end-to-end speed or QMoE/Switch 0.8-bpw reproduction, no production changes.",
        ],
    }
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "experiment":report["experiment"],
        "all12_weights":total_weights,
        "physical_global_cap":cap_total,
        "physical_actual":{n:item["actual_disk_bytes"] for n,item in serialized.items()},
        "quality_PPL":{k:{c:round(v["ppl"],6) for c,v in result.items()} for k,result in ppl.items()},
        "delta_PPL":{k:round(v["calibrated_minus_uncalibrated_PPL"],6)
                     for k,v in report["relative_quality"].items()},
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/calibri_v36_all12_agnews.json")
    parser.add_argument("--train-tokens",type=int,default=256)
    parser.add_argument("--eval-tokens",type=int,default=1024)
    parser.add_argument("--context",type=int,default=128)
    args=parser.parse_args()
    if (args.train_tokens%args.context or args.eval_tokens%args.context):
        raise ValueError("train/evaluation token count must align with context size")
    run(args)
