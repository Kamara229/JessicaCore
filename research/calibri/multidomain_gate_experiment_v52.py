#!/usr/bin/env python3
"""CALIBRI v52 pretrained full-Pythia INT6 with multi-window TRAIN-only acceptance.

PRE: original physically verified complete 6-bit checkpoint with MLP
calibration fitted on WikiText TRAIN A, as v46.
POST: same physically verified 6/12/24% per-tensor allocations as v51.
One preregistered 6-window TRAIN-only selection gate accepts corrections
by majority, mean gain and bounded domain-specific downside. No test tuning. Correct FP16
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
from deterministic_budget_v49 import candidates
from tensor_cap_v51 import (plans_for_budget,smoke as cap_smoke)
from practical_int6_v46 import timing
from gradient_recovery_experiment_v47 import physical_write

A_TOKENS=2048
B_TOKENS=2048
CONTEXT=128
MAX_PHYSICAL_BPW=6.5
POST_RESERVED=15480
RATE_METHODS=("base_int6_pre","sse_cap_06pct","sse_cap_12pct","sse_cap_24pct")
FRESH=(
    ("Yelp_test_576_640","Yelp/yelp_review_full",None,"test[576:640]"),
    ("OpenBookQA_test_448_500","allenai/openbookqa","main","test[448:500]"),
    ("QNLI_validation_512_640","glue","qnli","validation[512:640]"),
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
        if name=="Yelp_test_576_640":
            txt=str(row["text"]).strip()
        elif name=="OpenBookQA_test_448_500":
            options=row["choices"]
            txt=("Fact: "+str(row.get("fact1","")).strip()+
                 "\nQuestion: "+str(row["question_stem"]).strip()+
                 "\nChoices: "+"; ".join(str(label)+") "+str(choice)
                                       for label,choice in zip(
                                           options.get("label",[]),
                                           options.get("text",[]))))
        elif name=="QNLI_validation_512_640":
            txt=("Question: "+str(row["question"]).strip()+
                 "\nSentence: "+str(row["sentence"]).strip())
        else:
            raise AssertionError("unregistered heldout evaluation domain")
        if txt:
            chunks.append(txt)
    return from_texts(tok,chunks)


def six_train_gate_windows(tok,ag_rev,wiki_c,wiki_d):
    """Six new disjoint WikiText/AG News TRAIN-only windows.

    These excerpts are read only AFTER physical candidates are frozen.
    The windows represent two corpus families, not six independent domains.
    """
    out={"WikiText_TRAIN_C_4096_6144":wiki_c,
         "WikiText_TRAIN_D_6144_8192":wiki_d}
    for start,end in ((512,640),(640,768),(768,896),(896,1024)):
        ds=load_dataset("fancyzhx/ag_news",revision=ag_rev,
                        split=f"train[{start}:{end}]")
        rows=[str(rec["text"]).strip() for rec in ds if str(rec["text"]).strip()]
        out[f"AGNews_TRAIN_{start}_{end}"]=from_texts(tok,rows)
    if len(out)!=6:
        raise AssertionError("unregistered number of frozen TRAIN gate windows")
    return out

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
    """One frozen six-window training-only heuristic stability gate.

    Require >=4/6 positive gains, mean NLL improvement >=0.0005,
    worst individual window gain >=-0.0100. Select largest mean gain
    if eligible, tie-break to smaller per-tensor cap. Otherwise PRE.
    Never select on validation or heldout test PPL.
    """
    names=("WikiText_TRAIN_C_4096_6144","WikiText_TRAIN_D_6144_8192",
           "AGNews_TRAIN_512_640","AGNews_TRAIN_640_768",
           "AGNews_TRAIN_768_896","AGNews_TRAIN_896_1024")
    if any(set(train[k])!=set(names) for k in RATE_METHODS):
        raise AssertionError("training-only gate windows missing")
    base=train["base_int6_pre"]
    metrics={}
    for method in RATE_METHODS[1:]:
        improvements={d:base[d]["nll"]-train[method][d]["nll"] for d in names}
        wins=sum(g>0 for g in improvements.values())
        mean_gain=math.fsum(improvements.values())/len(names)
        worst=min(improvements.values())
        eligible=(wins>=4 and mean_gain>=0.0005 and worst>=-0.0100)
        metrics[method]={"gain_NLL_by_training_window":improvements,
                          "positive_windows":wins,
                          "mean_gain":mean_gain,
                          "worst_gain":worst,
                          "passes_fixed_gate":eligible}
    viable=[m for m in RATE_METHODS[1:] if metrics[m]["passes_fixed_gate"]]
    selected=max(viable,key=lambda k:(metrics[k]["mean_gain"],
                        -RATE_METHODS.index(k))) if viable else "base_int6_pre"
    return {"chosen":selected,
            "frozen_acceptance_rule":"at least 4/6 positive, mean NLL gain>=0.0005 and worst>=-0.0100, maximum mean winning",
            "expected_six_training_windows":list(names),
            "candidate_training_stability":metrics,
            "all_training_NLL_and_PPL":train}

def run(output):
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2" and
            os.getenv("DNNL_MAX_CPU_ISA")=="AVX2" and
            os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2" and
            str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("same selected AVX2 CPU model baseline required")
    torch.manual_seed(52)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cap_smoke()
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
    wiki=load_token_ids(tok,"train",wiki_rev,8192)
    A=wiki[:2048]
    C=wiki[4096:6144]
    D=wiki[6144:8192]
    if (len(A)!=2048 or len(C)!=2048 or len(D)!=2048 or
        len({sha(v.numpy().tobytes()) for v in (A,C,D)})!=3):
        raise AssertionError("TRAIN-A/C/D must each contain 2048 disjoint tokens")
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
    baseline_path=path.parent/"CALIBRI_v52_complete_INT6_pre.cb46"
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
    proposed=plans_for_budget(scores,available)
    patches={}
    expected_shapes={k:tuple(v.shape) for k,v in state.items() if v.ndim==2}
    for method in ("sse_cap_06pct","sse_cap_12pct","sse_cap_24pct"):
        pick=proposed[method]
        payload=encode_sparse_patch(state,decoded,pick["selection"],baseline["sha256"])
        patchfile=path.parent/("CALIBRI_v52_"+method+".cr47")
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
    # Six new TRAIN windows only AFTER physical source and ALL three
    # source-only FP16 correction sidecars are persisted and CRC verified.
    gate_text=six_train_gate_windows(tok,ag_rev,C,D)
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
        "experiment":"CALIBRI_v52_six_train_windows_conservative_int6_rate_choice",
        "real_pretrained_model":MODEL,
        "source_revision":source_rev,
        "source_checkpoint_fp32_SHA256":source_sha,
        "source_all_parameter_count":count,
        "wiki_revision":wiki_rev,"ag_news_revision":ag_rev,
        "external_revision_by_domain":eval_revs,
        "WikiText_TRAIN_A_SHA256":sha(A.numpy().tobytes()),
        "six_TRAIN_gate_token_SHA256":{name:sha(ids.numpy().tobytes()) for name,ids in gate_text.items()},
        "external_untouched_test_SHA256":{n:sha(v.numpy().tobytes()) for n,v in eval_text.items()},
        "base_INT6_complete_source":baseline,
        "deterministic_row_budget":{
            "score":"fixed Python binary64 error math.fsum -> integer score -> exact row bytes",
            "balance":"one row per 2D tensor, then 6%,12%,24% of all physical correction budget per tensor, unchanged source-only SSE ranking",
            "max_total_physical_bpw":MAX_PHYSICAL_BPW,
            "reserved_other_correction_bytes":POST_RESERVED,
            "max_all_model_plus_all_correction_bytes":total_max,
            "candidate_row_count":len(scores),
            "plans":patches},
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
        "v52":"MEASURED_REAL",
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
    ap.add_argument("--output",default="results/calibri_v52.json")
    args=ap.parse_args()
    run(args.output)
