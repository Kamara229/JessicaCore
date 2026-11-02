#!/usr/bin/env python3
"""CALIBRI v47: real pretrained all-weights INT6 plus TRAIN-gradient sparse recovery.

Pre-registers first-order teacher-residual row selection from WikiText TRAIN A,
actual stored FP16 residual-row patches under <=6.5 physical bpw, and a
matched-*physical-byte* row SSE control. Optional post output-affine fitted
only on disjoint TRAIN B. Verify actual complete CB46/CR47/C45C binary files,
then ONLY AFTER freeze read external test/validation text and benchmark
original vs hot-restored compressed full model with true KV-cache generation.

Never treats recovered FP32 hot cache as compressed-in-RAM runtime.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch

from huggingface_hub import HfApi
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from datasets import load_dataset
from full_checkpoint_v46 import pack_state,unpack_state,read_runtime
from pareto_fixed_highbits_v43 import trained_fixedbit
from fixedbit_v15 import decode as fixed_decode
from all_mlp_agnews_v36 import module_identity
from compensation_bridge_v45 import make_sidecar,open_sidecar
from practical_int6_v46 import timing,affine_fit_on_B,add_affine_hooks
from quality import MODEL,DATASET,load_token_ids,module_selection,calibrate,ppl_eval
from sparse_recovery_v47 import (rank_rows,pick_budget,encode_sparse_patch,
                                parse_patch,apply_patch,expected_overhead)

CONTEXT=128
TRAIN_A=2048
TRAIN_B=2048
MAX_BPW=6.5
MAX_QUALITY_PPL_RATIO=1.05
MAX_SPEED_RATIO=1.20
POST_RESERVED=15480  # physical C45C output-affine from v46; verify exactly.
TESTS=(
    ("OpenBookQA_test_new","allenai/openbookqa","main","test[:128]"),
    ("QNLI_validation_new","glue","qnli","validation[:128]"),
    ("Yelp_test_256_320_disjoint","Yelp/yelp_review_full",None,"test[256:320]"),
)
MODES=("teacher","INT6_PRE_only","INT6_PRE_POST_affine",
       "INT6_gradient_recovery","INT6_SSE_module_matched",
       "INT6_SSE_global","INT6_gradient_recovery_POST_affine")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def corpus(tokenizer,name,repo,config,part,rev):
    ds=load_dataset(repo,config,split=part,revision=rev) if config else load_dataset(
        repo,split=part,revision=rev)
    txt=[]
    for rec in ds:
        if name=="OpenBookQA_test_new":
            choices=rec["choices"]
            text=("Fact: "+str(rec.get("fact1","")).strip()+
                  "\nQuestion: "+str(rec["question_stem"]).strip()+
                  "\nChoices: "+"; ".join(str(l)+") "+str(c) for l,c in zip(
                      choices.get("label",[]),choices.get("text",[]))))
        elif name=="QNLI_validation_new":
            text="Question: "+str(rec["question"]).strip()+"\nSentence: "+str(rec["sentence"]).strip()
        elif name=="Yelp_test_256_320_disjoint":
            text=str(rec["text"]).strip()
        else:
            raise AssertionError("not frozen evaluation dataset")
        if text.strip():
            txt.append(text)
    if not txt:
        raise AssertionError("empty disjoint evaluation text")
    ids=tokenizer("\n\n".join(txt),add_special_tokens=False,
                  return_tensors="pt").input_ids[0,:2048]
    if len(ids)!=2048:
        raise AssertionError("2048 test tokens not available")
    return ids


def gradients_pretrained(model,train_ids):
    """True full-model *TRAIN-A* gradient of quantized NLL, never TEST labels.

    Source model is the pretrained original INT6 checkpoint decoded into
    dense PyTorch weights, no source teacher parameters remain on this branch.
    """
    if len(train_ids)!=TRAIN_A:
        raise AssertionError("wrong gradient calibration train sample")
    model.eval()
    model.zero_grad(set_to_none=True)
    total_loss=0.
    chunks=0
    for start in range(0,len(train_ids),CONTEXT):
        ids=train_ids[start:start+CONTEXT]
        if len(ids)!=CONTEXT:
            raise AssertionError("short pretrained NLL gradient context")
        out=model(input_ids=ids.reshape(1,-1),labels=ids.reshape(1,-1),
                  use_cache=False)
        loss=out.loss/float(TRAIN_A//CONTEXT)
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite trained INT6 language model NLL")
        loss.backward()
        total_loss+=float(loss.detach())
        chunks+=1
    if chunks!=16:
        raise AssertionError("full-model quantized gradient evaluated wrong contexts")
    out={}
    for name,p in model.named_parameters():
        if p.ndim==2 and p.grad is not None:
            out[name]=p.grad.detach().cpu().numpy().astype(np.float32).copy()
    model.zero_grad(set_to_none=True)
    return out,{"gradient_source":"real pretrained compressed Pythia14m full-model TRAIN-A NLL",
                "gradient_token_count":TRAIN_A,
                "context":CONTEXT,"trainA_quantized_model_NLL":total_loss,
                "named_2D_gradient_count":len(out)}


def physical_write(path,blob):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(blob)
    after=path.read_bytes()
    if after!=blob:
        raise AssertionError("true physical compressed source reread changed bytes")
    return {"path":str(path),"bytes":len(after),"sha256":sha(after)}


def measure_full(model,token_groups,actual_prompt):
    result={}
    with torch.inference_mode():
        for domain,ids in token_groups.items():
            metric=ppl_eval(model,ids,CONTEXT)
            result[domain]={"ppl":float(metric["ppl"]),"nll":float(metric["nll"])}
            print(json.dumps({"model":model._calibri_v47_kind,
                              "domain":domain,"PPL":metric["ppl"]}),flush=True)
    prompt,follow=actual_prompt[:128],actual_prompt[128:160]
    duration=timing(model,prompt,follow)
    return {"PPL":result,"latency":duration}


def run(args):
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2"
            and os.getenv("DNNL_MAX_CPU_ISA")=="AVX2"
            and os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2"
            and str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("ATen AVX2 and fixed oneDNN ISA cap required")
    torch.manual_seed(47)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    api=HfApi()
    model_rev=api.model_info(MODEL).sha
    wiki_rev=api.dataset_info(DATASET).sha
    external_rev={name:api.dataset_info(repo).sha for name,repo,_,_ in TESTS}
    tok=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
    config=AutoConfig.from_pretrained(MODEL,revision=model_rev)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=model_rev,torch_dtype=torch.float32)
    teacher.eval()
    original={n:v.detach().cpu().contiguous().clone() for n,v in teacher.state_dict().items()}
    coeff_count=sum(v.numel() for v in original.values())
    original_sha=sha(b"".join(name.encode()+w.numpy().tobytes()
                           for name,w in sorted(original.items())))
    original_modules=module_selection(teacher)
    names=sorted(original_modules,key=module_identity)
    if len(names)!=12 or sum(original[n+".weight"].numel() for n in names)!=786432:
        raise AssertionError("expected original pretrained Pythia MLP not loaded")

    all_train=load_token_ids(tok,"train",wiki_rev,TRAIN_A+TRAIN_B)
    a,b=all_train[:TRAIN_A],all_train[TRAIN_A:]
    if len(a)!=TRAIN_A or len(b)!=TRAIN_B or sha(a.numpy().tobytes())==sha(b.numpy().tobytes()):
        raise AssertionError("PRE and POST training split not disjoint")
    with torch.inference_mode():
        acts_A=calibrate(teacher,original_modules,a,CONTEXT)
        acts_B=calibrate(teacher,original_modules,b,CONTEXT)
    preencoded={}
    for name in names:
        w=original[name+".weight"].numpy().astype(np.float32)
        x=acts_A[name].numpy().astype(np.float32)
        moments=np.mean(x*x,axis=0,dtype=np.float32)
        pre,adjusted,meta=trained_fixedbit(w,moments,6)
        if len(pre)!=len(adjusted) or meta["TRAIN_diagonal_weighted_MSE_after"]>meta["TRAIN_diagonal_weighted_MSE_before"]*1.0000001:
            raise AssertionError("frozen pretraining INT6 optimizer changed size or lost train proxy")
        preencoded[name+".weight"]=adjusted
    physical,manifest=pack_state(original,preencoded=preencoded)
    base_path=output.parent/"CALIBRI_v47_all_trained_Pythia_INT6_pre.cb46"
    base_info=physical_write(base_path,physical)
    compressed=unpack_state(base_path.read_bytes(),expected_keys=original.keys())
    if base_info["bytes"]!=10803661:
        raise AssertionError("frozen v46 exact full-model INT6 physical storage changed")
    if sum(v.numel() for v in compressed.values())!=coeff_count:
        raise AssertionError("not all actual pretrained parameters recovered from INT6")
    max_bytes=math.floor(MAX_BPW*coeff_count/8)
    if base_info["bytes"]+POST_RESERVED>=max_bytes:
        raise AssertionError("not enough spare bytes for learned sparse teacher patches")

    # The PRE quantized Q copy (not teacher) supplies full-model
    # training-only gradients that rank which original FP16 teacher-row
    # residuals are worth storing as a *separate physical* POST sidecar.
    grad_model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32)
    grad_model.load_state_dict(compressed,strict=True)
    grads,grad_info=gradients_pretrained(grad_model,a)
    del grad_model
    scores=rank_rows(original,compressed,grads)
    plans=pick_budget(scores,max_bytes-POST_RESERVED,base_info["bytes"])
    sidecars={}
    decoded_patches={}
    for method in ("gradient_global","sse_module_size_matched","sse_global"):
        choice=plans[method]
        binary=encode_sparse_patch(original,compressed,choice,base_info["sha256"])
        file=output.parent/("CALIBRI_v47_"+method+".cr47")
        info=physical_write(file,binary)
        expected_shape={name:tuple(w.shape) for name,w in original.items() if w.ndim==2}
        parsed=parse_patch(file.read_bytes(),base_info["sha256"],expected_shape)
        if len(parsed)!=len(choice) or info["bytes"]+base_info["bytes"]+POST_RESERVED>max_bytes:
            raise AssertionError("wrong sparse teacher correction physical size")
        bad=bytearray(binary)
        bad[-2]^=1
        try:parse_patch(bytes(bad),base_info["sha256"],expected_shape)
        except ValueError:pass
        else:raise AssertionError("sidecar corruption was accepted")
        sidecars[method]={**info,"all_model_plus_patch_bytes":info["bytes"]+base_info["bytes"],
                         "total_bpw":8*(info["bytes"]+base_info["bytes"])/coeff_count,
                         "selected_rows":len(choice),
                         "selected_module_counts":{
                             n:sum(1 for t,_ in choice if t==n)
                             for n in sorted(set(t for t,_ in choice))}}
        decoded_patches[method]=parsed
    if sidecars["gradient_global"]["bytes"]!=sidecars["sse_module_size_matched"]["bytes"]:
        raise AssertionError("gradient and module-matched SSE sidecar bytes differ")

    affine_source={}
    for tag in ("base","gradient"):
        post={}
        patches=decoded_patches["gradient_global"] if tag=="gradient" else []
        module_overrides={}
        for n,i,delta in patches:
            if n in (name+".weight" for name in names):
                if n not in module_overrides:
                    module_overrides[n]=compressed[n].numpy().astype(np.float32).copy()
                module_overrides[n][i]+=delta
        for name in names:
            key=name+".weight"
            original_w=original[key].numpy().astype(np.float32)
            q=module_overrides.get(key,compressed[key].numpy().astype(np.float32))
            xb=acts_B[name].numpy().astype(np.float32)
            post[name]=affine_fit_on_B(original_w,q,xb)
        stored=make_sidecar(post,"affine")
        file=output.parent/("CALIBRI_v47_"+tag+"_post_affine.c45c")
        info=physical_write(file,stored)
        parsed=open_sidecar(file.read_bytes(),"affine")
        if info["bytes"]!=POST_RESERVED:
            raise AssertionError("unexpected physical fixed POST affine bytes")
        affine_source[tag]={"file":info,"parsed":parsed}
    if sidecars["gradient_global"]["all_model_plus_patch_bytes"]+POST_RESERVED>max_bytes:
        raise AssertionError("entire adaptive PRE+POST checkpoint above true 6.5bpw physical limit")

    # AFTER frozen PRE, gradient selection, FP16 teacher-row residual files
    # and TRAIN-B affine post were persisted: first external test read.
    groups={"WikiText2_DEV_known":load_token_ids(tok,"validation",wiki_rev,1024)}
    for name,repo,config_name,split in TESTS:
        groups[name]=corpus(tok,name,repo,config_name,split,external_rev[name])
    evidence={}
    timing_ids=a[:160].clone()
    teacher._calibri_v47_kind="teacher"
    evidence["teacher"]=measure_full(teacher,groups,timing_ids)
    del teacher

    for tag in MODES:
        if tag=="teacher":
            continue
        runtime=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32)
        loadinfo=read_runtime(runtime,str(base_path))
        patch_tag=None
        affine_tag=None
        if tag in ("INT6_gradient_recovery","INT6_gradient_recovery_POST_affine"):
            patch_tag="gradient_global"
        elif tag=="INT6_SSE_module_matched":
            patch_tag="sse_module_size_matched"
        elif tag=="INT6_SSE_global":
            patch_tag="sse_global"
        if patch_tag:
            patch_info=apply_patch(runtime,sidecars[patch_tag]["path"],base_info["sha256"])
            if patch_info["physical_patched_rows"]!=sidecars[patch_tag]["selected_rows"]:
                raise AssertionError("runtime did not apply every original stored residual row")
            loadinfo["patch_runtime"]=patch_info
        if tag=="INT6_PRE_POST_affine":
            affine_tag="base"
        elif tag=="INT6_gradient_recovery_POST_affine":
            affine_tag="gradient"
        hooks=add_affine_hooks(runtime,affine_source[affine_tag]["parsed"]) if affine_tag else []
        runtime._calibri_v47_kind=tag
        try:
            evidence[tag]={"results":measure_full(runtime,groups,timing_ids),
                           "checkpoint_loader":loadinfo,
                           "post_hooks":bool(hooks)}
        finally:
            for h in hooks:h.remove()
            del runtime

    teacher_data=evidence["teacher"]
    for name,row in evidence.items():
        if name=="teacher":
            continue
        measured=row["results"]
        row["latency_ratio_to_teacher"]={
            field:measured["latency"][field]["median"]/
                  teacher_data["latency"][field]["median"]
            for field in ("prefill_ms","decode32_ms","total_prefill_and_decode32_ms")}
        row["PPL_ratio_to_teacher"]={d:measured["PPL"][d]["ppl"]/
                                  teacher_data["PPL"][d]["ppl"] for d in groups}
        row["speed_20pct_pass"]=row["latency_ratio_to_teacher"][
            "total_prefill_and_decode32_ms"]<=MAX_SPEED_RATIO
        row["quality_5pct_all_new_pass"]=all(row["PPL_ratio_to_teacher"][d]<=MAX_QUALITY_PPL_RATIO
                                             for d,_,_,_ in TESTS)
        row["new_corpus_PPL_better_than_preonly"]={
            d:measured["PPL"][d]["ppl"]<evidence["INT6_PRE_only"]["results"]["PPL"][d]["ppl"]
            if name!="INT6_PRE_only" else False for d,_,_,_ in TESTS}
    report={
        "experiment":"CALIBRI_v47_train_gradient_sparse_teacher_row_recovery_6p5bpw_complete_checkpoint",
        "original_model":MODEL,"source_revision":model_rev,
        "source_full_state_SHA256":original_sha,
        "train_dataset":DATASET,"wiki_revision":wiki_rev,
        "external_revisions":external_rev,
        "trainA_SHA256":sha(a.numpy().tobytes()),"trainB_SHA256":sha(b.numpy().tobytes()),
        "validation_test_SHA256":{name:sha(x.numpy().tobytes()) for name,x in groups.items()},
        "actual_pretrained_state_parameter_count":coeff_count,
        "true_max_total_physical_bytes":max_bytes,
        "base_complete_CB46_checkpoint":base_info,
        "sparse_C47_correction_sidecars":sidecars,
        "post_affine_sidecars":{key:val["file"] for key,val in affine_source.items()},
        "sparse_selector":{
            "TRAIN_ONLY_full_quantized_model_NLL_gradient":grad_info,
            "total_candidate_trainable_rows":len(scores),
            "strict_module_size_matched_control":True,
            "global_gradient_proposed_rows":len(plans["gradient_global"]),
            "budget_reserved_exact_POST_bytes":POST_RESERVED,
            "gradient_is_first_order_proxy_not_guaranteed_loss_improvement":True
        },
        "variants":evidence,
        "engineering_gate":{"max_physical_bpw":MAX_BPW,
            "max_warm_full_model_latency_ratio":MAX_SPEED_RATIO,
            "temporary_quality_gate_ratio_PPL":MAX_QUALITY_PPL_RATIO,
            "quality_is_not_user_fixed_tolerance":True,
            "kernel_batch":1,"prompt_tokens":128,"decode_steps":32,
            "warmups":2,"timing_repetitions":7},
        "limitations":[
            "Real physical full-model INT6 disk archive plus stored fp16 teacher residual rows; every correction byte physically counted.",
            "Every runtime expands to full dense FP32 weights ONCE during cold startup; this is speed-first not 6-bit RAM or native packed calculation.",
            "TRAIN-A gradient ranks rows; TRAIN-B fit POST-affine; external text opened only after fixed source/sidecars stored.",
            "Gradient ranks are approximations to global loss, not exactly optimal with coupled layer effects.",
            "SSE module-matched control stores exactly the same bytes as gradient (equal number of rows in each module).",
            "All new evaluation corpora are 2048-token fixed excerpts; no task answer accuracy or user Jessica benchmark yet.",
            "The external datasets OpenBookQA, Yelp, QNLI have predeclared NEW slices; some source dataset families were explored in older CALIBRI sessions.",
            "The +5pct PPL quality target is a provisional research threshold, not a user-declared hard limit.",
            "No main, Render, Supabase, APK, running Jessica or user data changes."
        ]}
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"EXPERIMENT":"CALIBRI_V47_REAL","source_hash":original_sha,
       "base_bytes":base_info["bytes"],"rows":{
           k:v["selected_rows"] for k,v in sidecars.items()},
       "sidecar_bytes":{k:v["bytes"] for k,v in sidecars.items()},
       "total_bpw":{k:v["total_bpw"] for k,v in sidecars.items()},
       "result":{k:{"PPL":{d:round(v["ppl"],5)
                             for d,v in (val["PPL"] if k=="teacher" else val["results"]["PPL"]).items()},
                    "latency_ms":(val["latency"] if k=="teacher" else val["results"]["latency"])[
                        "total_prefill_and_decode32_ms"]["median"],
                    "ratio":None if k=="teacher" else val["latency_ratio_to_teacher"][
                        "total_prefill_and_decode32_ms"]}
                 for k,val in evidence.items()}
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/calibri_v47.json")
    run(parser.parse_args())
