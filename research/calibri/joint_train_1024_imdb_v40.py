#!/usr/bin/env python3
"""CALIBRI v40: real all-12 Pythia MLP compression with 256 vs 1024 TRAIN tokens.

Reuse EXACT v39 rowwise 3/4-bit physical codec, fixed 75–125% per-module
precision constraints, 2-round x 20-probe joint full-model TRAIN-NLL
search and existing row-scale calibration. No evaluation data is loaded
until BOTH selections are physically serialized to immutable bytes.
Evaluate simultaneously on WikiText-2 DEV and first-use IMDb TEST.
Full-model PPL operates on reloaded DENSE-decoded weights only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from quality import MODEL,DATASET,load_token_ids,module_selection,calibrate,ppl_eval
from all_mlp_agnews_v36 import package,unpack_bundle,module_identity
from direct_tiles_v33 import encode as encode_tiles,decode as decode_tiles
from fixedbit_v15 import encode as encode_int4,decode as decode_int4
from calibrated_rows_v35 import calibrated_encode
from joint_train_loss_v39 import search_joint_train_nll

NEW_DATASET="stanfordnlp/imdb"
NEW_SPLIT="test[:64]"
SIZES=(256,1024)
GLOBAL_BYTES=343840
ALL_MLP_WEIGHTS=786432
PPL_CONTEXT=128


def fixed_imdb_ids(tokenizer,revision,token_count):
    from datasets import load_dataset
    ds=load_dataset(NEW_DATASET,split=NEW_SPLIT,revision=revision)
    reviews=[str(item["text"]).replace("<br />"," ").strip()
             for item in ds if str(item.get("text","")).strip()]
    if not reviews:
        raise AssertionError("IMDb fixed test[:64] had no text")
    ids=tokenizer("\n\n".join(reviews),return_tensors="pt",
                  add_special_tokens=False).input_ids[0,:token_count]
    if len(ids)!=token_count:
        raise AssertionError("IMDb fixed sample too short")
    return ids,len(reviews)


def evaluate(model,originals,variants,corpora):
    result={}
    try:
        with torch.inference_mode():
            for variant,weights in variants.items():
                for name,source in originals.items():
                    model.get_submodule(name).weight.copy_(source)
                if weights is not None:
                    if set(weights)!=set(originals):
                        raise AssertionError("missing physical matrix in full model")
                    for n,q in weights.items():
                        if tuple(model.get_submodule(n).weight.shape)!=tuple(q.shape) or not np.isfinite(q).all():
                            raise AssertionError("invalid decoded matrix in model")
                        model.get_submodule(n).weight.copy_(torch.from_numpy(q))
                result[variant]={}
                for corpus,ids in corpora.items():
                    score=ppl_eval(model,ids,PPL_CONTEXT)
                    if not np.isfinite(score["ppl"]) or score["ppl"]<=0:
                        raise AssertionError("invalid model PPL")
                    result[variant][corpus]=score
                    print(json.dumps({"variant":variant,"corpus":corpus,
                                      "PPL":score["ppl"]}),flush=True)
    finally:
        with torch.no_grad():
            for n,w in originals.items():
                model.get_submodule(n).weight.copy_(w)
    return result


def run(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi

    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2"
            and os.getenv("DNNL_MAX_CPU_ISA")=="AVX2"
            and os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2"
            and str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("exact selected PyTorch AVX2 CPU dispatcher required")
    if (args.train_tokens!=1024 or args.context!=PPL_CONTEXT or
            args.dev_tokens!=1024 or args.new_tokens!=2048):
        raise AssertionError("frozen preregistered source/train/dev/test token amounts changed")
    torch.manual_seed(40)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    api=HfApi()
    rev=api.model_info(MODEL).sha
    wiki_sha=api.dataset_info(DATASET).sha
    # Metadata fetch does not expose the first-use evaluation text to
    # training. The evaluation dataset itself is read only AFTER selection.
    imdb_sha=api.dataset_info(NEW_DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=rev,torch_dtype=torch.float32)
    model.eval()
    modules=module_selection(model)
    if len(modules)!=12:
        raise AssertionError("all twelve Pythia pretrained MLP modules expected")
    names=sorted(modules,key=module_identity)
    originals={n:modules[n].weight.detach().clone() for n in names}
    weights={n:originals[n].cpu().numpy().astype(np.float32).copy()
             for n in names}
    if sum(v.size for v in weights.values())!=ALL_MLP_WEIGHTS:
        raise AssertionError("unexpected Pythia original MLP parameter counts")

    train=load_token_ids(tok,"train",wiki_sha,args.train_tokens)
    with torch.inference_mode():
        x_all=calibrate(model,modules,train,args.context)
    if set(x_all)!=set(originals):
        raise AssertionError("TRAIN capture did not reach all source modules")
    x={n:v.detach().cpu().numpy().astype(np.float32) for n,v in x_all.items()}
    if any(v.shape!=(args.train_tokens,weights[n].shape[1])
           for n,v in x.items()):
        raise AssertionError("TRAIN layer activation geometry violated")

    encoded_variants={}
    train_selection={}
    for count in SIZES:
        tr=train[:count]
        train_inputs={n:v[:count].copy() for n,v in x.items()}
        base={}
        for n in names:
            source=weights[n]
            moments=(train_inputs[n]*train_inputs[n]).mean(axis=0,dtype=np.float32)
            blob,_=encode_tiles(source,moments,int(source.size*3.5/8),
                                br=1,bc=source.shape[1])
            base[n]=blob
        if len(package(base,1))!=GLOBAL_BYTES:
            raise AssertionError("baseline physical all12 3.5bpw cap changed")
        local_cal={n:calibrated_encode(weights[n],train_inputs[n],base[n])[0]
                   for n in names}
        joint_raw,joint_cal,train_meta=search_joint_train_nll(
            names,weights,train_inputs,model,originals,tr,args.context,base)
        for n in names:
            if len(base[n])!=len(local_cal[n]) or len(joint_raw[n])!=len(joint_cal[n]):
                raise AssertionError("physical calibration changed assigned bits")
        for k,v in {
            f"train{count}_local_uncalibrated":base,
            f"train{count}_local_calibrated":local_cal,
            f"train{count}_joint_calibrated":joint_cal,
        }.items():
            if len(package(v,1))!=GLOBAL_BYTES:
                raise AssertionError("physical size differs between TRAIN widths")
            encoded_variants[k]=v
        train_selection[f"train{count}"]={
            "token_count":count,
            "token_sha256":hashlib.sha256(tr.numpy().tobytes()).hexdigest(),
            "train_NLL_reference":train_meta["original_baseline_joint_train_NLL"],
            "train_NLL_joint_selected":train_meta["selected_joint_train_NLL"],
            "train_NLL_improvement":train_meta["joint_train_NLL_improvement"],
            "selection":train_meta,
            "source_local_binary_sha256":hashlib.sha256(package(base,1)).hexdigest(),
            "selected_joint_binary_sha256":hashlib.sha256(package(joint_cal,1)).hexdigest(),
        }
        print(json.dumps({
            "train_tokens":count,
            "baseline_joint_train_NLL":train_meta["original_baseline_joint_train_NLL"],
            "selected_joint_train_NLL":train_meta["selected_joint_train_NLL"],
            "actual_bytes":len(package(joint_cal,1)),
            "rounds":train_meta["rounds_completed"],
        }),flush=True)

    high_rate={n:encode_int4(weights[n],4) for n in names}
    encoded_variants["INT4_overbudget_reference"]=high_rate
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    physical={}
    decoded={}
    for label,modules_bytes in encoded_variants.items():
        mode=2 if label=="INT4_overbudget_reference" else 1
        binary=package(modules_bytes,mode)
        path=output.parent/(label+".bin")
        path.write_bytes(binary)
        from_disk=path.read_bytes()
        if from_disk!=binary:
            raise AssertionError("physical CB36 bundle disk roundtrip failed")
        restored=unpack_bundle(from_disk,mode)
        if set(restored)!=set(names):
            raise AssertionError("physical bundle missing source MLP module")
        decoded[label]={n:(decode_int4(b) if mode==2 else decode_tiles(b))
                        for n,b in restored.items()}
        physical[label]={
            "disk_bytes":path.stat().st_size,
            "bits_per_MLP_parameter":8*len(binary)/ALL_MLP_WEIGHTS,
            "sha256":hashlib.sha256(from_disk).hexdigest(),
            "actual_number_of_source_MLP_matrices":len(restored),
        }
    if any(v["disk_bytes"]!=GLOBAL_BYTES for k,v in physical.items()
           if k!="INT4_overbudget_reference"):
        raise AssertionError("candidate physical bundle not identical real bytes")
    if physical["INT4_overbudget_reference"]["disk_bytes"]<=GLOBAL_BYTES:
        raise AssertionError("higher-rate INT4 control incorrectly labeled")

    # BLIND-TEST GATE: first read both dev and the new IMDb TEST corpus
    # only AFTER freezing both physical 256/1024 TRAIN encodings to files.
    wiki=load_token_ids(tok,"validation",wiki_sha,args.dev_tokens)
    imdb,imdb_reviews=fixed_imdb_ids(tok,imdb_sha,args.new_tokens)
    evaluation_text_frozen_after_all_training=True
    corpora={"WikiText2_development":wiki,"NEW_IMDB_test":imdb}
    ppl=evaluate(model,originals,{"original_source_teacher":None,**decoded},corpora)
    report={
        "experiment":"CALIBRI_v40_joint_full_model_trainNLL_4x_calibration_first_IMDB",
        "pinned_source_model":MODEL,
        "pinned_model_revision":rev,
        "original_MLP_weights":ALL_MLP_WEIGHTS,
        "source_MLP_weights_SHA256":{
            n:hashlib.sha256(w.tobytes()).hexdigest() for n,w in weights.items()},
        "training_WikiText_dataset_revision":wiki_sha,
        "new_IMDB_dataset_revision":imdb_sha,
        "corpus_test":"stanfordnlp/imdb test[:64], 2048 tokens from fixed first reviews, HTML breaks normalized to spaces",
        "corpus_test_reviews":imdb_reviews,
        "training_choices":train_selection,
        "train_only_full_1024_token_SHA256":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
        "frozen_context":args.context,
        "dev_WikiText_token_SHA256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest(),
        "new_IMDB_test_token_SHA256":hashlib.sha256(imdb.numpy().tobytes()).hexdigest(),
        "new_corpus_first_loaded_after_training_physical_lock":evaluation_text_frozen_after_all_training,
        "physical_bytes_cap":GLOBAL_BYTES,
        "physical_MLP_files":physical,
        "full_model_restored_MLP_PPL":ppl,
        "primary_comparisons":{
            f"train{n}":{corpus:{
                "joint_minus_rate_matched_local_calibrated_PPL":
                    ppl[f"train{n}_joint_calibrated"][corpus]["ppl"]-
                    ppl[f"train{n}_local_calibrated"][corpus]["ppl"],
                "joint_PPL":ppl[f"train{n}_joint_calibrated"][corpus]["ppl"],
                "local_PPL":ppl[f"train{n}_local_calibrated"][corpus]["ppl"],
            } for corpus in corpora} for n in SIZES},
        "environment":{
            "torch":str(torch.__version__),
            "selected_ATen_CPU_capability":str(torch._C._get_cpu_capability()),
            "oneDNN_ISA_max_requested":"AVX2",
            "threads":torch.get_num_threads(),
            "deterministic":torch.are_deterministic_algorithms_enabled(),
        },
        "limits":[
            "Only 12 real Pythia14m MLP matrices stored, not a runnable full-model checkpoint.",
            "PPL always computed by fully decoded dense MLP matrices, no native packed streaming speed measured.",
            "Frozen v39 search exactly 2 rounds, 20 proposals per round; 256 and 1024 TRAIN sample sizes, no new hyperparameter tuning.",
            "IMDb test[:64] first 2048 tokens from 64 fixed reviews is first-use corpus for CALIBRI, not a representative movie sentiment benchmark.",
            "SST2, SciQ, AG News, WikiText and TinyStories were previously seen in CALIBRI research; new IMDb only for this preregistered check.",
            "Source original and >budget INT4 are not same-rate controls. Same-343840B TRAIN-size-specific local calibration is rate-matched.",
            "Any better TRAIN NLL that fails to lower held-out IMDb PPL is a negative scientific result.",
            "No validated new multidimensional neural weights, QMoE 0.8bpw, full-model packed speed, or production changes.",
        ],
    }
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "experiment":report["experiment"],
        "physical_bytes":{k:v["disk_bytes"] for k,v in physical.items()},
        "TRAIN_NLL":{k:[v["train_NLL_reference"],v["train_NLL_joint_selected"]]
                     for k,v in train_selection.items()},
        "PPL":{k:{c:round(v["ppl"],6) for c,v in vals.items()} for k,vals in ppl.items()},
        "matched_rate_PPL_delta":report["primary_comparisons"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="results/calibri_v40_1024_train_imdb.json")
    ap.add_argument("--train-tokens",type=int,default=1024)
    ap.add_argument("--dev-tokens",type=int,default=1024)
    ap.add_argument("--new-tokens",type=int,default=2048)
    ap.add_argument("--context",type=int,default=PPL_CONTEXT)
    run(ap.parse_args())
