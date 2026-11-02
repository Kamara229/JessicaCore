#!/usr/bin/env python3
"""CALIBRI v50 pretrained full-Pythia INT6 determinism and precision allocation.

PRE: original physically verified complete 6-bit checkpoint with MLP
calibration fitted on WikiText TRAIN A, as v46.
POST: two source-only deterministic row-repair allocation profiles:
balanced 12%/tensor and global. Match total physical max 6.5 bpw,
including a 15480-byte correction reserve. Select ONE by conservative
two-domain TRAIN-only NLL gate; never tune using test PPL. Correct FP16
teacher residual rows are persisted with SHA/CRC and applied once at
cold load; full active weights still expanded into FP32 for fast inference.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np
import torch

from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from huggingface_hub import HfApi
from datasets import load_dataset
from quality import MODEL,DATASET,load_token_ids,module_selection,calibrate,ppl_eval
from all_mlp_agnews_v36 import module_identity
from pareto_fixed_highbits_v43 import trained_fixedbit
from full_checkpoint_v46 import pack_state,unpack_state,read_runtime
from sparse_recovery_v47 import encode_sparse_patch,parse_patch,apply_patch
from deterministic_budget_v49 import (candidates,choose,disk_bytes,
                                      smoke as allocator_smoke)
from activation_aware_v50 import (activation_matched_rows,
                                  smoke as activation_smoke)
from practical_int6_v46 import timing
from gradient_recovery_experiment_v47 import physical_write

A_TOKENS=2048
B_TOKENS=2048
CONTEXT=128
MAX_PHYSICAL_BPW=6.5
POST_RESERVED=15480
RATE_METHODS=("base_int6_pre","sse_balanced_12pct","activation_matched_MLP")
FRESH=(
    ("Yelp_test_448_512","Yelp/yelp_review_full",None,"test[448:512]"),
    ("OpenBookQA_test_256_384","allenai/openbookqa","main","test[256:384]"),
    ("QNLI_validation_256_384","glue","qnli","validation[256:384]"),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def from_texts(tok,rows):
    if not rows:
        raise AssertionError("external evaluation was empty")
    ids=tok("\n\n".join(rows),add_special_tokens=False,
            return_tensors="pt").input_ids[0,:2048]
    if len(ids)!=2048:
        raise AssertionError("frozen first 2048 evaluation tokens missing")
    return ids


def fresh_test(tok,name,repo,cfg,subset,rev):
    ds=(load_dataset(repo,cfg,revision=rev,split=subset) if cfg else
        load_dataset(repo,revision=rev,split=subset))
    chunks=[]
    for row in ds:
        if name=="Yelp_test_448_512":
            txt=str(row["text"]).strip()
        elif name=="OpenBookQA_test_256_384":
            options=row["choices"]
            txt=("Fact: "+str(row.get("fact1","")).strip()+
                 "\nQuestion: "+str(row["question_stem"]).strip()+
                 "\nChoices: "+"; ".join(str(label)+") "+str(choice)
                                       for label,choice in zip(
                                           options.get("label",[]),
                                           options.get("text",[]))))
        elif name=="QNLI_validation_256_384":
            txt=("Question: "+str(row["question"]).strip()+
                 "\nSentence: "+str(row["sentence"]).strip())
        else:
            raise AssertionError("unregistered heldout evaluation domain")
        if txt:
            chunks.append(txt)
    return from_texts(tok,chunks)


def train_agnews_c(tok,rev):
    """Entirely TRAIN records, non-overlapping v48's AG News TRAIN[:128]."""
    ds=load_dataset("fancyzhx/ag_news",revision=rev,split="train[256:384]")
    texts=[str(row["text"]).strip() for row in ds if str(row["text"]).strip()]
    return from_texts(tok,texts)


def eval_ppl(model,domains,tag):
    model.eval()
    out={}
    with torch.inference_mode():
        for name,ids in domains.items():
            stats=ppl_eval(model,ids,CONTEXT)
            out[name]={"nll":float(stats["nll"]),"ppl":float(stats["ppl"])}
            print(json.dumps({"variant":tag,"corpus":name,"ppl":stats["ppl"]}),
                  flush=True)
    return out


def cold_load(cfg,base,sha_base,patches,variant):
    obj=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32)
    loader=read_runtime(obj,str(base))
    if variant!="base_int6_pre":
        loader["physical_residual_patch"]=apply_patch(
            obj,patches[variant]["path"],sha_base)
    obj.eval()
    return obj,loader


def choose_from_train(train):
    """One fixed minimax improvement rule; rejects candidate regressions.

    score = min(base_NLL - candidate_NLL over both TRAIN-only domains);
    require score >=1e-5; higher score wins. Tie-break favor balanced.
    No selection on test corpus or speed.
    """
    domains=("WikiText_TRAIN_B","AGNews_TRAIN_256_384")
    base=train["base_int6_pre"]
    baseline={"base_int6_pre":0.}
    for method in ("sse_balanced_12pct","activation_matched_MLP"):
        baseline[method]=min(base[d]["nll"]-train[method][d]["nll"] for d in domains)
    viable=[m for m in ("sse_balanced_12pct","sse_global")
            if baseline[m]>=1e-5]
    selected=max(viable,key=lambda m:(baseline[m],m=="sse_balanced_12pct")) if viable else "base_int6_pre"
    return {"chosen":selected,
            "fixed_gate_criterion":"max worst-domain NLL reduction versus PRE, if positive by >=1e-5 on BOTH TRAIN-only domains",
            "minimum_improvement_required_nll":1e-5,
            "minimum_train_domain_nll_gains":baseline,
            "all_training_NLL_and_PPL":train}


def run(output):
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2" and
            os.getenv("DNNL_MAX_CPU_ISA")=="AVX2" and
            os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2" and
            str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("same selected AVX2 CPU model baseline required")
    torch.manual_seed(49)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    allocator_smoke()
    activation_smoke()
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    api=HfApi()
    source_rev=api.model_info(MODEL).sha
    wiki_rev=api.dataset_info(DATASET).sha
    ag_rev=api.dataset_info("fancyzhx/ag_news").sha
    eval_revs={name:api.dataset_info(repo).sha for name,repo,_,_ in FRESH}
    tok=AutoTokenizer.from_pretrained(MODEL,revision=source_rev)
    cfg=AutoConfig.from_pretrained(MODEL,revision=source_rev)
    original=AutoModelForCausalLM.from_pretrained(MODEL,revision=source_rev,
                                                  torch_dtype=torch.float32)
    original.eval()
    state={k:v.detach().cpu().contiguous().clone() for k,v in original.state_dict().items()}
    count=sum(v.numel() for v in state.values())
    source_sha=sha(b"".join(k.encode()+v.numpy().tobytes()
                           for k,v in sorted(state.items())))
    mlps=module_selection(original)
    names=sorted(mlps,key=module_identity)
    if len(names)!=12 or sum(state[n+".weight"].numel() for n in names)!=786432:
        raise AssertionError("original pretrained real Pythia architecture mismatch")
    wiki=load_token_ids(tok,"train",wiki_rev,A_TOKENS+B_TOKENS)
    A,B=wiki[:A_TOKENS],wiki[A_TOKENS:]
    if len(A)!=A_TOKENS or len(B)!=B_TOKENS or sha(A.numpy().tobytes())==sha(B.numpy().tobytes()):
        raise AssertionError("calibration train A/B not disjoint")
    with torch.inference_mode():
        captures=calibrate(original,mlps,A,CONTEXT)
    overrides={}
    for n in names:
        w=state[n+".weight"].numpy().astype(np.float32)
        x=captures[n].numpy().astype(np.float32)
        moments=np.mean(x*x,axis=0,dtype=np.float32)
        raw,adjusted,info=trained_fixedbit(w,moments,6)
        if len(raw)!=len(adjusted) or info["TRAIN_diagonal_weighted_MSE_after"]>info["TRAIN_diagonal_weighted_MSE_before"]*1.000001:
            raise AssertionError("PRE calibration failed source v46 loss+same-bytes test")
        overrides[n+".weight"]=adjusted
    blob,manifest=pack_state(state,preencoded=overrides)
    baseline_path=path.parent/"CALIBRI_v50_complete_INT6_pre.cb46"
    baseline=physical_write(baseline_path,blob)
    if baseline["bytes"]!=10803661:
        raise AssertionError("full physical pretrained Pythia 6-bit source count changed")
    decoded=unpack_state(blob,expected_keys=state.keys())
    total_max=math.floor(MAX_PHYSICAL_BPW*count/8)
    available=total_max-baseline["bytes"]-POST_RESERVED
    if available<500000:
        raise AssertionError("unexpected available trained FP16 restoration budget")

    # Global candidate priority is **solely actual source weights and
    # true physical quantization errors**, not even TRAIN-A gradients.
    # Each score is reproducible scalar math.fsum over exact FP32 inputs.
    scores,source_shapes=candidates(state,decoded)
    proposed=choose(scores,available)
    # EXACT same per-layer row quota and actual physical bytes as the v49
    # balanced correction. Re-rank ONLY 12 MLP tensors using WikiText TRAIN-A
    # input-second-moment sensitivity; all non-MLP rows are left intact.
    activation_rows,activation_details=activation_matched_rows(
        state,decoded,captures,proposed["sse_balanced_12pct"]["selection"],scores)
    balanced=proposed["sse_balanced_12pct"]
    proposed["activation_matched_MLP"]={
        "selection":activation_rows,
        "rows":len(activation_rows),
        "actual_patch_bytes_expected":balanced["actual_patch_bytes_expected"],
        "module_counts":balanced["module_counts"]}
    patches={}
    expected_shapes={k:tuple(v.shape) for k,v in state.items() if v.ndim==2}
    for method in ("sse_balanced_12pct","activation_matched_MLP"):
        pick=proposed[method]
        payload=encode_sparse_patch(state,decoded,pick["selection"],baseline["sha256"])
        patchfile=path.parent/("CALIBRI_v50_"+method+".cr47")
        evidence=physical_write(patchfile,payload)
        restored=parse_patch(patchfile.read_bytes(),baseline["sha256"],expected_shapes)
        if (len(restored)!=pick["rows"] or
            evidence["bytes"]!=pick["actual_patch_bytes_expected"] or
            baseline["bytes"]+evidence["bytes"]+POST_RESERVED>total_max):
            raise AssertionError("physical correction row count or budget failed")
        flipped=bytearray(payload);flipped[-1]^=1
        try:
            parse_patch(bytes(flipped),baseline["sha256"],expected_shapes)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted correction byte stream accepted")
        patches[method]={**evidence,
            "selected_real_rows":pick["rows"],
            "module_counts":pick["module_counts"],
            "all_source_plus_correction_bytes":baseline["bytes"]+evidence["bytes"],
            "physical_bpw_excl_reserved_extra_post":
                8*(baseline["bytes"]+evidence["bytes"])/count}
    # One gate: only TRAIN-B and new AGNews TRAIN[256:384]; fitting
    # completed, true physical source+residual persisted BEFORE reading.
    gate_text={"WikiText_TRAIN_B":B,
               "AGNews_TRAIN_256_384":train_agnews_c(tok,ag_rev)}
    gate_metrics={}
    for name in RATE_METHODS:
        model,_=cold_load(cfg,baseline_path,baseline["sha256"],patches,name)
        gate_metrics[name]=eval_ppl(model,gate_text,name)
        del model
    decision=choose_from_train(gate_metrics)
    selected=decision["chosen"]
    # Do not read these test files until AFTER immutable rate choice.
    eval_text={}
    for name,repo,conf,split in FRESH:
        eval_text[name]=fresh_test(tok,name,repo,conf,split,eval_revs[name])
    teacher=eval_ppl(original,eval_text,"original_teacher")
    evaluated={}
    for name in RATE_METHODS:
        model,_=cold_load(cfg,baseline_path,baseline["sha256"],patches,name)
        evaluated[name]=eval_ppl(model,eval_text,name)
        del model

    prompt=A[:CONTEXT].clone()
    follow=A[CONTEXT:CONTEXT+32].clone()
    before=timing(original,prompt,follow)
    del original
    model,loader=cold_load(cfg,baseline_path,baseline["sha256"],patches,selected)
    after=timing(model,prompt,follow)
    del model
    latency=after["total_prefill_and_decode32_ms"]["median"]/before[
        "total_prefill_and_decode32_ms"]["median"]
    selected_bytes=baseline["bytes"]+(
        patches[selected]["bytes"] if selected!="base_int6_pre" else 0)
    report={
        "experiment":"CALIBRI_v50_activation_aware_exact_bytes_INT6_MLP_row_repair",
        "real_pretrained_model":MODEL,
        "source_revision":source_rev,
        "source_checkpoint_fp32_SHA256":source_sha,
        "source_all_parameter_count":count,
        "wiki_revision":wiki_rev,"ag_news_revision":ag_rev,
        "external_revision_by_domain":eval_revs,
        "WikiText_TRAIN_A_SHA256":sha(A.numpy().tobytes()),
        "WikiText_TRAIN_B_SHA256":sha(B.numpy().tobytes()),
        "AGNews_TRAIN_256_384_SHA256":sha(gate_text["AGNews_TRAIN_256_384"].numpy().tobytes()),
        "external_untouched_test_SHA256":{n:sha(v.numpy().tobytes()) for n,v in eval_text.items()},
        "base_INT6_complete_source":baseline,
        "deterministic_row_budget":{
            "score":"fixed Python binary64 error math.fsum -> integer score -> exact row bytes",
            "balance":"v49 one row per tensor and <=12% per tensor, then exact-byte MLP-only TRAIN-A activation sensitivity re-ranking",
            "max_total_physical_bpw":MAX_PHYSICAL_BPW,
            "reserved_other_correction_bytes":POST_RESERVED,
            "max_all_model_plus_all_correction_bytes":total_max,
            "candidate_row_count":len(scores),
            "plans":patches,
            "activation_reallocation":activation_details}, 
        "one_shot_train_only_gate":decision,
        "selected_method":selected,
        "selected_actual_byte_size":selected_bytes,
        "selected_actual_bpw":8*selected_bytes/count,
        "original_teacher_untouched_test_PPL":teacher,
        "all_candidates_untouched_test_PPL":evaluated,
        "actual_full_model_warm_speed":{
            "prompt_tokens":128,"autoregressive_decode_steps":32,
            "warmups":2,"timed_repeats":7,
            "original_teacher":before,"chosen":after,
            "chosen_divided_by_teacher":latency,
            "passes_plus20pct_user_limit":latency<=1.2,
            "fresh_real_physical_CB46_plus_CR47_loader":loader},
        "limitations":[
            "ALL pretrained matrices physically six-bit INT6 on disk, corrected rows separately true FP16 compressed residual sidecars, complete real stored bytes counted.",
            "All running full-model weights are reconstructed to dense FP32 at cold start: no active compressed RAM or native packed INT6 claims.",
            "Old known WikiText training and newly disjoint AGNews TRAIN window are used only for one predeclared acceptance gate; no external TEST results tune this selection.",
            "The 3 heldout slices are new record ranges of previously studied broad dataset families, 2048 tokens each, not representative production Jessica accuracy.",
            "One host may have faster absolute CPU; the measured 20%-latency cap is only within-host against original teacher.",
            "No GitHub main merge, Jessica ordinary chat test, Render, Supabase, APK, settings or secret changes."
        ]}
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "v50":"MEASURED_REAL",
        "base_SHA":baseline["sha256"],
        "int6_base_bytes":baseline["bytes"],
        "patches":{n:{"SHA":d["sha256"],"B":d["bytes"],
                      "rows":d["selected_real_rows"]} for n,d in patches.items()},
        "train_gate":decision["chosen"],
        "train_gate_NLL":{n:{d:round(x["nll"],6) for d,x in score.items()}
                           for n,score in gate_metrics.items()},
        "selected_physical_bpw":report["selected_actual_bpw"],
        "heldout_PPL":{"teacher":{d:round(v["ppl"],5) for d,v in teacher.items()},
                       **{n:{d:round(v["ppl"],5) for d,v in score.items()}
                          for n,score in evaluated.items()}},
        "selected_warm_speed_ratio":latency,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="results/calibri_v50.json")
    args=ap.parse_args()
    run(args.output)
