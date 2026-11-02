#!/usr/bin/env python3
"""CALIBRI v48: TRAIN-only cross-domain gate for exact-byte INT6 repair.

One preregistered step after v47: no more changing row-selection rules
against an already-seen evaluation corpus. Original trained Pythia14m
source, complete CB46 INT6 physical bytes and two physically identical-
size CR47 patch candidates inherited from v47.

TRAIN A WikiText fit only; independent WikiText TRAIN B plus AG News
TRAIN form a ONE-SHOT conservative accept/reject gate; then physically
freeze selection before loading untouched heldout test slices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
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
from sparse_recovery_v47 import (rank_rows,pick_budget,encode_sparse_patch,
                                parse_patch,apply_patch)
from gradient_recovery_experiment_v47 import gradients_pretrained,physical_write
from practical_int6_v46 import timing

TRAIN_A=2048
TRAIN_B=2048
CONTEXT=128
RATE_CAP=6.5
RESERVED_POST_BYTES=15480
SPLIT_AG_TRAIN="train[:128]"
EXTERNAL=(
    ("Yelp_new_test_320_384","Yelp/yelp_review_full",None,"test[320:384]"),
    ("ARCChallenge_new_validation_128_256","allenai/ai2_arc","ARC-Challenge","validation[128:256]"),
)
METHODS=("base_INT6_PRE","gradient_global","sse_module_size_matched")


def sha(buf):
    return hashlib.sha256(buf).hexdigest()


def fixed_tokens(tokenizer,ds,limit=2048):
    ids=tokenizer("\n\n".join(ds),return_tensors="pt",
                  add_special_tokens=False).input_ids[0,:limit]
    if len(ids)!=limit:
        raise AssertionError("frozen 2048 real-data tokens not available")
    return ids


def training_domain(tokenizer,revision):
    ds=load_dataset("fancyzhx/ag_news",split=SPLIT_AG_TRAIN,revision=revision)
    parts=[str(row["text"]).strip() for row in ds if str(row["text"]).strip()]
    if not parts:
        raise AssertionError("first-use AG News training domain empty")
    return fixed_tokens(tokenizer,parts)


def unseen_test(tokenizer,name,repo,config,subset,revision):
    ds=(load_dataset(repo,config,split=subset,revision=revision)
        if config else load_dataset(repo,split=subset,revision=revision))
    records=[]
    for row in ds:
        if name=="Yelp_new_test_320_384":
            txt=str(row["text"]).strip()
        elif name=="ARCChallenge_new_validation_128_256":
            choice=row["choices"]
            txt=("Question: "+str(row["question"])+"\nOptions: "+
                 "; ".join(str(a)+") "+str(b) for a,b in zip(
                     choice.get("label",[]),choice.get("text",[]))))
        else:
            raise AssertionError("not preregistered untouched text")
        if txt:
            records.append(txt)
    if not records:
        raise AssertionError("no disjoint TEST text")
    return fixed_tokens(tokenizer,records)


def score(model,corpora):
    out={}
    model.eval()
    with torch.inference_mode():
        for name,ids in corpora.items():
            met=ppl_eval(model,ids,CONTEXT)
            out[name]={"ppl":float(met["ppl"]),"nll":float(met["nll"])}
            print(json.dumps({"method":model._calibri_method,"corpus":name,
                              "PPL":out[name]["ppl"]}),flush=True)
    return out


def once_model(cfg,base_file,patch_files,parent_sha,method):
    m=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32)
    load=read_runtime(m,base_file)
    if method!="base_INT6_PRE":
        p=apply_patch(m,patch_files[method]["path"],parent_sha)
        load["correction"]=p
    m._calibri_method=method
    m.eval()
    return m,load


def gate_once(training):
    """Conservative zero-tuning decision using only two non-test domains.

    A candidate must improve NLL versus baseline by >=1e-5 on BOTH
    WikiText TRAIN-B and first-use AGNews TRAIN. Gradient is selected
    only if its NLL is also at least 1e-5 better than equally sized
    same-module SSE on BOTH domains. Otherwise choose SSE if it clears
    both, or plain PRE. The 1e-5 tolerance, arm ordering and rule are
    all frozen before any training-B or AGNews score is evaluated.
    """
    names=("WikiText_TRAIN_B","AGNews_TRAIN_first_use")
    margin=1e-5
    baseline=training["base_INT6_PRE"]
    def wins(variant,control):
        return all(training[variant][name]["nll"]<=
                   training[control][name]["nll"]-margin for name in names)
    if wins("gradient_global","base_INT6_PRE") and wins(
            "gradient_global","sse_module_size_matched"):
        decision="gradient_global"
    elif wins("sse_module_size_matched","base_INT6_PRE"):
        decision="sse_module_size_matched"
    else:
        decision="base_INT6_PRE"
    return {"decision":decision,"frozen_margin_NLL":margin,
            "domain_rule":"must improve on BOTH independent TRAIN-only corpora; gradient must additionally beat exact-byte SSE on both",
            "candidate_scores":training,
            "accepted_sse":wins("sse_module_size_matched","base_INT6_PRE"),
            "accepted_gradient_over_sse_and_pre":wins("gradient_global","base_INT6_PRE") and
                                                  wins("gradient_global","sse_module_size_matched")}


def run(output):
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2" and
            os.getenv("DNNL_MAX_CPU_ISA")=="AVX2" and
            os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2" and
            str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("CPU ISA fixed AVX2 contract changed")
    torch.manual_seed(48)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    api=HfApi()
    model_rev=api.model_info(MODEL).sha
    wiki_rev=api.dataset_info(DATASET).sha
    ag_rev=api.dataset_info("fancyzhx/ag_news").sha
    external_revs={name:api.dataset_info(repo).sha for name,repo,_,_ in EXTERNAL}
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
    config=AutoConfig.from_pretrained(MODEL,revision=model_rev)
    teacher=AutoModelForCausalLM.from_pretrained(MODEL,revision=model_rev,
                                                 torch_dtype=torch.float32)
    teacher.eval()
    state={k:v.detach().cpu().contiguous().clone()
           for k,v in teacher.state_dict().items()}
    originals=module_selection(teacher)
    keys=sorted(originals,key=module_identity)
    if len(keys)!=12 or sum(state[k+".weight"].numel() for k in keys)!=786432:
        raise AssertionError("real pretrained Pythia14m MLP geometry changed")
    count=sum(t.numel() for t in state.values())
    original_sha=sha(b"".join(k.encode()+v.numpy().tobytes()
                          for k,v in sorted(state.items())))
    first=load_token_ids(tokenizer,"train",wiki_rev,TRAIN_A+TRAIN_B)
    a,b=first[:TRAIN_A],first[TRAIN_A:]
    if len(a)!=TRAIN_A or len(b)!=TRAIN_B or sha(a.numpy().tobytes())==sha(b.numpy().tobytes()):
        raise AssertionError("TRAIN splits invalid")
    with torch.inference_mode():
        acts=calibrate(teacher,originals,a,CONTEXT)
    preencoded={}
    for k in keys:
        source=state[k+".weight"].numpy().astype(np.float32)
        x=acts[k].numpy().astype(np.float32)
        mom=(x*x).mean(axis=0,dtype=np.float32)
        raw,adjusted,meta=trained_fixedbit(source,mom,6)
        if len(raw)!=len(adjusted) or meta["TRAIN_diagonal_weighted_MSE_after"]>meta["TRAIN_diagonal_weighted_MSE_before"]*1.000001:
            raise AssertionError("frozen pre quantizer changed rate or train error")
        preencoded[k+".weight"]=adjusted
    full,_=pack_state(state,preencoded=preencoded)
    base=physical_write(out.parent/"CALIBRI_v48_pre_full_int6.cb46",full)
    if base["bytes"]!=10803661:
        raise AssertionError("baseline v46 physical pretrained source changed")
    weights=unpack_state(full,expected_keys=state.keys())
    cap=math.floor(RATE_CAP*count/8)
    # Exact same 15,480B correction reserve for fair comparison to
    # v47 even though this one-step v48 tests only no-hook row correction.
    grad_source=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32)
    grad_source.load_state_dict(weights,strict=True)
    grad,grad_meta=gradients_pretrained(grad_source,a)
    del grad_source
    scores=rank_rows(state,weights,grad)
    choices=pick_budget(scores,cap-RESERVED_POST_BYTES,base["bytes"])
    patches={}
    expected={k:tuple(v.shape) for k,v in state.items() if v.ndim==2}
    for tag in ("gradient_global","sse_module_size_matched"):
        payload=encode_sparse_patch(state,weights,choices[tag],base["sha256"])
        path=out.parent/("CALIBRI_v48_"+tag+".cr47")
        info=physical_write(path,payload)
        rows=parse_patch(path.read_bytes(),base["sha256"],expected)
        if len(rows)!=len(choices[tag]) or base["bytes"]+info["bytes"]+RESERVED_POST_BYTES>cap:
            raise AssertionError("physical full model + row correction exceeds 6.5bpw")
        bad=bytearray(payload)
        bad[-1]^=1
        try: parse_patch(bytes(bad),base["sha256"],expected)
        except ValueError: pass
        else: raise AssertionError("CR47 corruption accepted")
        patches[tag]={**info,"rows":len(rows),
                      "all_model_physical_bytes":base["bytes"]+info["bytes"],
                      "actual_effective_bpw":8*(base["bytes"]+info["bytes"])/count}
    if patches["gradient_global"]["bytes"]!=patches["sse_module_size_matched"]["bytes"]:
        raise AssertionError("candidate correction rates not EXACT equal bytes")
    # Do not load AG News training text before both physical candidate
    # source+patch files have been frozen and verified.
    gate_corpora={
        "WikiText_TRAIN_B":b,
        "AGNews_TRAIN_first_use":training_domain(tokenizer,ag_rev)
    }
    training={}
    for method in METHODS:
        rt,_=once_model(config,base["path"],patches,base["sha256"],method)
        training[method]=score(rt,gate_corpora)
        del rt
    selection=gate_once(training)
    chosen=selection["decision"]
    selected_archive=base if chosen=="base_INT6_PRE" else patches[chosen]

    # ONLY after gate decision: untouched labels-free external test texts
    # may be read for a SINGLE confirmatory evaluation of all candidates.
    test_corpora={}
    for name,repo,cfg,subset in EXTERNAL:
        test_corpora[name]=unseen_test(tokenizer,name,repo,cfg,subset,external_revs[name])
    teacher._calibri_method="original_teacher"
    teacher_result=score(teacher,test_corpora)
    test_result={}
    for method in METHODS:
        rt,_=once_model(config,base["path"],patches,base["sha256"],method)
        test_result[method]=score(rt,test_corpora)
        del rt
    # Actual full-model speed check only for chosen candidate vs
    # uncompressed teacher on true autoregressive real input IDs.
    prompt=a[:128].clone()
    follow=a[128:160].clone()
    reference_time=timing(teacher,prompt,follow)
    del teacher
    chosen_runtime,loadinfo=once_model(
        config,base["path"],patches,base["sha256"],chosen)
    selected_time=timing(chosen_runtime,prompt,follow)
    del chosen_runtime
    speed_ratio=selected_time["total_prefill_and_decode32_ms"]["median"] / (
                reference_time["total_prefill_and_decode32_ms"]["median"])
    result={
        "experiment":"CALIBRI_v48_frozen_multi_train_domain_accept_reject_sparse_INT6",
        "model":MODEL,"model_revision":model_rev,"wiki_revision":wiki_rev,
        "AGNews_training_revision":ag_rev,"AGNews_train_slice":SPLIT_AG_TRAIN,
        "external_test_revisions":external_revs,
        "original_pretrained_full_state_SHA256":original_sha,
        "train_A_token_sha256":sha(a.numpy().tobytes()),
        "train_B_token_sha256":sha(b.numpy().tobytes()),
        "gate_AGNews_token_sha256":sha(gate_corpora["AGNews_TRAIN_first_use"].numpy().tobytes()),
        "external_TEST_token_sha256":{n:sha(v.numpy().tobytes()) for n,v in test_corpora.items()},
        "source_full_model_coefficient_count":count,
        "max_total_all_coefficient_disk_bytes":cap,
        "original_frozen_CB46_complete_INT6_file":base,
        "physical_CR47_correction_files":patches,
        "frozen_train_gradient_metadata":grad_meta,
        "selection_one_shot":selection,
        "selected_real_disk_bytes":base["bytes"]+(
            0 if chosen=="base_INT6_PRE" else patches[chosen]["bytes"]),
        "selected_real_disk_bpw":8*(base["bytes"]+(
            0 if chosen=="base_INT6_PRE" else patches[chosen]["bytes"]))/count,
        "untouched_external_test_original_teacher":teacher_result,
        "untouched_external_test_variants":test_result,
        "true_warm_full_model_speed":{"context_tokens":128,"decode_tokens":32,
             "warmups":2,"timing_repetitions":7,"teacher":reference_time,
             "chosen":selected_time,"chosen_div_teacher":speed_ratio,
             "latency_slower_than_teacher_over_20pct":speed_ratio>1.2,
             "hot_cache_from_full_compressed_checkpoint":loadinfo},
        "limits":[
            "No testing on ordinary Jessica chat, no modifications to production repository branch, Render, Supabase, APK or secrets.",
            "Two-domain gate uses WikiText TRAIN-B (previously exposed to v45 and v47 experiments) plus first AG News TRAIN slice; not a globally independent validation population.",
            "External slices are disjoint from CALIBRI previous text indices but same broader datasets; each only 2048 Pythia tokens, not task accuracy or robust generalization proof.",
            "Selected patch applied only at cold FP32 hot-cache restoration; no six-bit RAM or native packed inference speed claim.",
            "The gate is a preregistered conservative selection between exactly equal physical byte candidate corrections or a smaller baseline; never tuned using external TEST PPL.",
            "Negative external quality or latency must remain visible and cannot cause hash/integrity CI errors."
        ]}
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"v48":"ACTUAL",
                      "one_shot_train_only_gate":selection["decision"],
                      "gated_training_NLL":training,
                      "checkpoint_bytes":base["bytes"],
                      "candidate_sidecar_bytes":{k:v["bytes"] for k,v in patches.items()},
                      "selected_effective_bpw":result["selected_real_disk_bpw"],
                      "external_PPL":{
                          "teacher":{n:v["ppl"] for n,v in teacher_result.items()},
                          **{k:{n:v["ppl"] for n,v in r.items()}
                             for k,r in test_result.items()}},
                      "full_model_speed_ratio":speed_ratio},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/calibri_v48.json")
    args=p.parse_args()
    run(args.output)
