#!/usr/bin/env python3
"""CALIBRI v41: never reuse TRAIN-A selection tokens to validate bit swaps.

On REAL pretrained Pythia-14m use WikiText TRAIN tokens 0:2048 for
all 12 MLP row-bit allocation, FP16 row-scale calibration, and the v39
joint full-model NLL optimizer. Use disjoint TRAIN tokens 2048:4096
ONLY ONCE to decide whether to keep the final optimizer candidate.
Only then physically persist MLP bundles and read brand-new SQuAD.
The one-shot internal gate cannot be used to select individual swaps.
"""
from __future__ import annotations

import argparse,hashlib,json,os
from pathlib import Path

import numpy as np
import torch

from quality import MODEL,DATASET,load_token_ids,module_selection,calibrate,ppl_eval
from all_mlp_agnews_v36 import package,unpack_bundle,module_identity
from direct_tiles_v33 import encode as encode_tiles,decode as decode_tiles
from fixedbit_v15 import encode as encode_int4,decode as decode_int4
from calibrated_rows_v35 import calibrated_encode
from joint_train_loss_v39 import search_joint_train_nll

TRAIN_A=2048
TRAIN_B=2048
CONTEXT=128
NEW_TEST_TOKENS=2048
GLOBAL_BYTES=343840
MLP_WEIGHTS=786432
NEW_DATASET="rajpurkar/squad"
NEW_SPLIT="validation[:64]"
GATE_MIN_IMPROVEMENT=1e-6


def squad_new_ids(tokenizer,revision,token_count):
    # First use by CALIBRI. Fixed public validation ordering and content.
    from datasets import load_dataset
    ds=load_dataset(NEW_DATASET,split=NEW_SPLIT,revision=revision)
    paragraphs=[("Context: "+str(item["context"]).strip()+
                 "\nQuestion: "+str(item["question"]).strip()) for item in ds]
    ids=tokenizer("\n\n".join(paragraphs),return_tensors="pt",
                  add_special_tokens=False).input_ids[0,:token_count]
    if len(ids)!=token_count:
        raise AssertionError("SQuAD first 64 validation entries shorter than frozen test")
    return ids,len(paragraphs)


def evaluate_dense(model, originals, variants, corpora):
    result={}
    try:
        with torch.inference_mode():
            for tag,recon in variants.items():
                for n,source in originals.items():
                    model.get_submodule(n).weight.copy_(source)
                if recon is not None:
                    if set(recon)!=set(originals):
                        raise AssertionError("all 12 source MLP modules must be present")
                    for n,q in recon.items():
                        if q.shape!=tuple(originals[n].shape) or not np.isfinite(q).all():
                            raise AssertionError("invalid decoded physical source MLP")
                        model.get_submodule(n).weight.copy_(torch.from_numpy(q))
                result[tag]={}
                for corpus,tokens in corpora.items():
                    p=ppl_eval(model,tokens,CONTEXT)
                    result[tag][corpus]=p
                    print(json.dumps({"variant":tag,"corpus":corpus,
                                      "PPL":p["ppl"],"NLL":p["nll"]}),flush=True)
    finally:
        with torch.no_grad():
            for n,source in originals.items():
                model.get_submodule(n).weight.copy_(source)
    return result


def run(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi

    if (args.train_tokens!=TRAIN_A+TRAIN_B or
            args.dev_tokens!=1024 or args.new_tokens!=NEW_TEST_TOKENS or
            args.context!=CONTEXT):
        raise AssertionError("frozen v41 test/calibration split altered")
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2"
            and os.getenv("DNNL_MAX_CPU_ISA")=="AVX2"
            and os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2"
            and str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("real selected CPU AVX2 dispatcher required")
    torch.manual_seed(41)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    api=HfApi()
    model_revision=api.model_info(MODEL).sha
    wiki_revision=api.dataset_info(DATASET).sha
    # Metadata lookup is safe, SQuAD TEST text is not loaded until final
    # TRAIN-A optimizer/ TRAIN-B gate and immutable binary files complete.
    squad_revision=api.dataset_info(NEW_DATASET).sha
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=model_revision)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=model_revision,torch_dtype=torch.float32)
    model.eval()

    modules=module_selection(model)
    names=sorted(modules,key=module_identity)
    originals={n:modules[n].weight.detach().clone() for n in names}
    weights={n:originals[n].cpu().numpy().astype(np.float32).copy() for n in names}
    if len(names)!=12 or sum(q.size for q in weights.values())!=MLP_WEIGHTS:
        raise AssertionError("unexpected REAL Pythia-14m architecture/MLP size")

    all_train=load_token_ids(tokenizer,"train",wiki_revision,TRAIN_A+TRAIN_B)
    train_a=all_train[:TRAIN_A]
    train_b=all_train[TRAIN_A:]
    assert len(train_a)==len(train_b)==2048
    hash_a=hashlib.sha256(train_a.numpy().tobytes()).hexdigest()
    hash_b=hashlib.sha256(train_b.numpy().tobytes()).hexdigest()
    if hash_a==hash_b:
        raise AssertionError("distinct TRAIN token windows are identical")
    with torch.inference_mode():
        original_activations=calibrate(model,modules,train_a,CONTEXT)
    if set(original_activations)!=set(names):
        raise AssertionError("TRAIN-A activation capture missing MLP")
    x={n:v.detach().cpu().numpy().astype(np.float32) for n,v in original_activations.items()}
    if any(v.shape!=(TRAIN_A,weights[n].shape[1]) for n,v in x.items()):
        raise AssertionError("TRAIN-A MLP input calibration shapes changed")

    # Never examine TRAIN-B, DEV or SQuAD when fitting the quantizer.
    local={}
    for n in names:
        source=weights[n]
        moments=(x[n]*x[n]).mean(axis=0,dtype=np.float32)
        local[n],_=encode_tiles(source,moments,int(source.size*3.5/8),
                                br=1,bc=source.shape[1])
    local_cal={n:calibrated_encode(weights[n],x[n],local[n])[0]
               for n in names}
    raw_joint,cal_joint,search=search_joint_train_nll(
        names,weights,x,model,originals,train_a,CONTEXT,local)
    if any(len(package(blobs,1))!=GLOBAL_BYTES for blobs in (local,local_cal,raw_joint,cal_joint)):
        raise AssertionError("TRAIN-A model compressed bundles do not satisfy exact physical bytes")

    # ONE independent TRAIN-B score for the final source-local and the
    # TRAIN-A-optimized proposal. Never use TRAIN-B within search/ranking.
    internal=evaluate_dense(model,originals,{
        "local_2048_trainA":{n:decode_tiles(blob) for n,blob in local_cal.items()},
        "joint_trainA_candidate":{n:decode_tiles(blob) for n,blob in cal_joint.items()},
    },{"WikiText_TRAIN_B_internal_gate":train_b})
    b_base=internal["local_2048_trainA"]["WikiText_TRAIN_B_internal_gate"]["nll"]
    b_joint=internal["joint_trainA_candidate"]["WikiText_TRAIN_B_internal_gate"]["nll"]
    accepted=b_joint < b_base-GATE_MIN_IMPROVEMENT
    decision="accept_joint" if accepted else "reject_joint_use_local"
    selected=cal_joint if accepted else local_cal
    print(json.dumps({
        "TRAIN_A_local_NLL":search["original_baseline_joint_train_NLL"],
        "TRAIN_A_joint_NLL":search["selected_joint_train_NLL"],
        "TRAIN_B_local_NLL":b_base,"TRAIN_B_joint_NLL":b_joint,
        "train_only_single_GATE":decision,
    }),flush=True)

    # Materialize every real physical variant and read back it with checksum.
    variants={
        "local_uncalibrated":(1,local),
        "local_trainA_calibrated":(1,local_cal),
        "joint_trainA_calibrated_UNGATED":(1,cal_joint),
        "joint_trainA_calibrated_GATED":(1,selected),
        "INT4_higher_rate_anchor":(2,{n:encode_int4(weights[n],4) for n in names}),
    }
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    saved={}
    decoded={}
    for tag,(mode,components) in variants.items():
        archive=package(components,mode)
        dest=output.parent/(tag+".bin")
        dest.write_bytes(archive)
        actual=dest.read_bytes()
        if archive!=actual:
            raise AssertionError("encoded physical disk did not match saved bytes")
        restored=unpack_bundle(actual,mode)
        if set(restored)!=set(names):
            raise AssertionError("physical archive missing modules")
        actual_bytes=dest.stat().st_size
        if mode==1 and actual_bytes!=GLOBAL_BYTES:
            raise AssertionError("unfair physical mixed-rate comparison")
        if mode==2 and actual_bytes<=GLOBAL_BYTES:
            raise AssertionError("INT4 control incorrectly within lower byte budget")
        saved[tag]={
            "real_disk_bytes":actual_bytes,
            "real_sha256":hashlib.sha256(actual).hexdigest(),
            "bpw_for_12_MLP_only":8*actual_bytes/MLP_WEIGHTS,
            "module_count":len(restored),
        }
        decoded[tag]={n:(decode_int4(v) if mode==2 else decode_tiles(v))
                      for n,v in restored.items()}
    if (saved["joint_trainA_calibrated_GATED"]["real_sha256"] !=
       saved["joint_trainA_calibrated_UNGATED" if accepted else "local_trainA_calibrated"]["real_sha256"]):
        raise AssertionError("internal holdout gating not reflected in physical file")

    # FIRST time the new SQuAD corpus is opened, AFTER all training,
    # single-gate decision and actual physical bundle persistence.
    known_dev=load_token_ids(tokenizer,"validation",wiki_revision,args.dev_tokens)
    unseen_squad,n_squad=squad_new_ids(tokenizer,squad_revision,args.new_tokens)
    corpora={
        "WikiText2_development":known_dev,
        "NEW_SQuAD_validation":unseen_squad,
    }
    quality=evaluate_dense(model,originals,{"original_teacher":None,**decoded},corpora)
    baseline=quality["local_trainA_calibrated"]
    raw_candidate=quality["joint_trainA_calibrated_UNGATED"]
    chosen=quality["joint_trainA_calibrated_GATED"]
    diffs={d:{
        "raw_joint_minus_local_PPL":raw_candidate[d]["ppl"]-baseline[d]["ppl"],
        "gated_joint_minus_local_PPL":chosen[d]["ppl"]-baseline[d]["ppl"],
        "gated_strategy":decision,
    } for d in corpora}
    data={
        "experiment":"CALIBRI_v41_split_TRAIN_A_fit_B_one_gate_new_SQuAD_real_12_MLPs",
        "source_model":MODEL,
        "source_revision":model_revision,
        "source_MLP_sha256":{n:hashlib.sha256(w.tobytes()).hexdigest() for n,w in weights.items()},
        "real_source_MLP_parameter_count":MLP_WEIGHTS,
        "datasets":{"WikiText2":{"name":DATASET,"revision":wiki_revision},
                    "new_SQuAD":{"name":NEW_DATASET,"revision":squad_revision,
                                 "split":NEW_SPLIT,"examples":n_squad}},
        "TRAIN_A":{"start":0,"end_exclusive":TRAIN_A,"sha256":hash_a},
        "TRAIN_B":{"start":TRAIN_A,"end_exclusive":TRAIN_A+TRAIN_B,"sha256":hash_b},
        "WikiText2_DEV_sha256":hashlib.sha256(known_dev.numpy().tobytes()).hexdigest(),
        "new_SQuAD_token_sha256":hashlib.sha256(unseen_squad.numpy().tobytes()).hexdigest(),
        "split_epoch":{"TRAIN_A_selection_uses_A_only":True,
                       "TRAIN_B_checked_only_once_after_complete_TRAIN_A_search":True,
                       "new_SQuAD_loaded_after_binary_freeze":True},
        "selection_trainA":search,
        "internal_gate_trainB":{
            "local_NLL":b_base,"ungated_joint_NLL":b_joint,
            "joint_improvement_required":GATE_MIN_IMPROVEMENT,
            "joint_accepted":accepted,"decision":decision,
            "TRAIN_B_scores":internal,
        },
        "physical_MLP_bundles":saved,
        "full_model_decoded_PPL":quality,
        "quality_delta_against_equal_rate_local":diffs,
        "environment":{"torch":str(torch.__version__),
                       "cpu_capability":str(torch._C._get_cpu_capability()),
                       "onednn_max_requested":"AVX2",
                       "threads":torch.get_num_threads(),
                       "determinism":torch.are_deterministic_algorithms_enabled()},
        "limitations":[
          "TRAIN-B used only ONCE to accept or reject the single final candidate. It is no longer an untouched heldout corpus after this session.",
          "Only 2048+2048 training tokens and one small new 2048-token SQuAD sample; not broad statistical evidence.",
          "No inference speed from packed weights: dense decoders used for full model PPL.",
          "Actual CB36 files only store 12 source MLP matrices; NOT a full model checkpoint.",
          "No multidimensional trained weights, QMoE 0.8bpw or deployable runtime demonstrated.",
        ],
    }
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"stage":"FINAL_MEASURED",
       "train_A_local_NLL":search["original_baseline_joint_train_NLL"],
       "train_A_joint_NLL":search["selected_joint_train_NLL"],
       "train_B_local_NLL":b_base,"train_B_joint_NLL":b_joint,
       "gate":decision,
       "file_bytes":{k:x["real_disk_bytes"] for k,x in saved.items()},
       "PPL":{k:{j:round(s["ppl"],6) for j,s in q.items()} for k,q in quality.items()},
       "new_SQuAD_primary_deltas":diffs["NEW_SQuAD_validation"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/calibri_v41_split_train_squad.json")
    parser.add_argument("--train-tokens",type=int,default=4096)
    parser.add_argument("--dev-tokens",type=int,default=1024)
    parser.add_argument("--new-tokens",type=int,default=2048)
    parser.add_argument("--context",type=int,default=128)
    run(parser.parse_args())
