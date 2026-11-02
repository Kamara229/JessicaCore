#!/usr/bin/env python3
"""CALIBRI v71: independent cross-corpus full-model validation of FROZEN v70 choice.

Teacher original FP32 and students native packed K32V8 C CPU run in
separate processes. 16 x 128 token fresh windows for each of two corpora.
No learned changes, no TEST-guided tuning, no live deployment.
Only original CB59 and TRAIN-selected v70 'l0_hadamard' run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from direct_codebook_v59 import CB59
from direct_runtime_v59 import NativeKernel, bind_into_llama, compile_kernel
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from quality import DATASET,load_token_ids

WINDOW=128
WINDOWS=16
N=WINDOW*WINDOWS
TARGET_BYTES=11229303
WIKI_START=32768
SOURCE_KINDS=("baseline","l0_hadamard")
CORPORA=("wikitext","agnews")
AG_DATASET="fancyzhx/ag_news"
TEACHER_DTYPE="float16"  # explicitly quantized only to store source reference logits

def source_ids(corpus):
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    if corpus=="wikitext":
        sha=HfApi().dataset_info(DATASET).sha
        all_ids=load_token_ids(tok,"test",sha,WIKI_START+N)
        ids=all_ids[WIKI_START:WIKI_START+N]
        provenance={"dataset":DATASET,"split":"test","revision":sha,
                    "token_span":[WIKI_START,WIKI_START+N]}
    elif corpus=="agnews":
        sha=HfApi().dataset_info(AG_DATASET).sha
        rows=load_dataset(AG_DATASET,split="test[:300]",revision=sha)
        # Predeclared official TEST news records; no labels ever used.
        raw="\n\n".join(str(row["text"]) for row in rows)
        ids=tok(raw,return_tensors="pt",add_special_tokens=False).input_ids[0,:N]
        provenance={"dataset":AG_DATASET,"split":"test[:300]","revision":sha,
                    "used_text_only":True,"token_span":[0,N]}
    else:raise ValueError(corpus)
    if len(ids)!=N:raise AssertionError("not enough original untouched corpus tokens")
    return ids,{"corpus":corpus,"source":provenance,
                "tokens_sha256":hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                "token_count":N,"window_count":WINDOWS,"context":WINDOW}

def create_model(which,folder):
    if which=="teacher":
        m=AutoModelForCausalLM.from_pretrained(
            MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
            attn_implementation="eager").eval()
        return m,None
    if which not in SOURCE_KINDS:raise ValueError(which)
    source=folder/"sources"/(which+".cb59")
    archive=CB59(source.read_bytes())
    if (archive.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE or
        archive.physical_bytes()!=TARGET_BYTES):
        raise AssertionError("wrong real full 135M physical checkpoint")
    model_cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    model_cfg._attn_implementation="eager"
    m=AutoModelForCausalLM.from_config(model_cfg,torch_dtype=torch.float32).eval()
    so=compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"),
                      folder/("lib71_"+which+".so"))
    kernel=NativeKernel(so)
    evidence=bind_into_llama(m,archive,kernel)
    if evidence["runtime_dense_multi_dim_source_parameters"]!=0:
        raise AssertionError("trained dense source weights in packed runtime")
    return m,{"physical_bytes":archive.physical_bytes(),
              "physical_sha256":archive.hash(),
              "original_2D_dense_hot_weights":0,
              "native_C_direct_5bit_embedding_and_GEMV":True,
              "tied_embeddings_verified":evidence["tied_embed_and_output_codebook_SHARED_IN_RAM"]}

def phase(args):
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    ids,provenance=source_ids(args.corpus)
    model,evidence=create_model(args.kind,args.folder)
    # Do one unscored warmup, then run ALL windows with genuine source logits.
    with torch.inference_mode():
        model(input_ids=ids[:4][None],use_cache=False)
        teacher_path=args.folder/("teacher_"+args.corpus+".f16")
        if args.kind=="teacher":
            mapped=np.memmap(teacher_path,mode="w+",dtype="<f2",shape=(WINDOWS,WINDOW,49152))
        else:
            mapped=np.memmap(teacher_path,mode="r",dtype="<f2",shape=(WINDOWS,WINDOW,49152))
        per=[]
        timings=[]
        for wi in range(WINDOWS):
            one=ids[wi*WINDOW:(wi+1)*WINDOW][None]
            tic=time.perf_counter()
            logits=model(input_ids=one,use_cache=False).logits[0].detach().cpu().numpy().astype(np.float32)
            timings.append(time.perf_counter()-tic)
            if logits.shape!=(WINDOW,49152) or not np.isfinite(logits).all():
                raise AssertionError("bad real LLM logits")
            if args.kind=="teacher":
                mapped[wi]=logits.astype("<f2")
                continue
            x=torch.from_numpy(logits[:-1].astype(np.float64))
            t=torch.from_numpy(np.array(mapped[wi,:-1],dtype=np.float64))
            p=t.log_softmax(-1)
            q=x.log_softmax(-1)
            probs=p.exp()
            kl=float((probs*(p-q)).sum(-1).mean())
            truth=one[0,1:]
            nll=float(-q[torch.arange(WINDOW-1),truth].mean())
            top1=float((t.argmax(-1)==x.argmax(-1)).double().mean())
            if not (math.isfinite(kl) and math.isfinite(nll) and math.isfinite(top1)):
                raise AssertionError("non-finite genuine model quality")
            per.append({"window":wi,"teacher_student_KL":kl,
                        "next_token_NLL":nll,"teacher_top1_agreement":top1})
        if args.kind=="teacher":mapped.flush()
    audit={"stage":"CALIBRI_v71_crosscorpus_real_135M",
           "kind":args.kind,"provenance":provenance,
           "pinned_pretrained_model":MODEL,"model_revision":MODEL_SHA,
           "physical_source":evidence,
           "teacher_logits_file_quantization":TEACHER_DTYPE,
           "unscored_warmup_tokens":4,
           "all_inference_full_128_token_windows":True,
           "window_cpu_times_seconds":timings,
           "cpu_prefill_total_seconds":float(sum(timings)),
           "scores_per_window":per,
           "pretrained_teacher_NOT_loaded_with_packed":args.kind!="teacher",
           "original_pretrained_2D_weights_absent_from_packed_hot_model":args.kind!="teacher"}
    name=args.folder/("v71_"+args.corpus+"_"+args.kind+".json")
    name.write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"phase":args.kind,"corpus":args.corpus,
                      "physical_bytes":(evidence or {}).get("physical_bytes"),
                      "mean_KL":round(float(np.mean([v["teacher_student_KL"] for v in per])),5) if per else None,
                      "mean_NLL":round(float(np.mean([v["next_token_NLL"] for v in per])),5) if per else None,
                      "prefill_sec":round(sum(timings),3)}),flush=True)

def summarize(args):
    build=json.loads((args.folder/"v70_build.json").read_text())
    sel=json.loads((args.folder/"v70_train_selection.json").read_text())
    if sel["chosen"]!="l0_hadamard":
        raise AssertionError("frozen v70 selection changed; v71 must not reselect by new data")
    assert build["variants"]["baseline"]["actual_bytes"]==TARGET_BYTES
    assert build["variants"]["l0_hadamard"]["actual_bytes"]==TARGET_BYTES
    assert build["variants"]["baseline"]["source_sha256"]!=build["variants"]["l0_hadamard"]["source_sha256"]
    out={}
    rng=np.random.default_rng(7101)
    for corpus in CORPORA:
        loaded={which:json.loads((args.folder/("v71_"+corpus+"_"+which+".json")).read_text())
                for which in ("teacher",*SOURCE_KINDS)}
        h={r["provenance"]["tokens_sha256"] for r in loaded.values()}
        if len(h)!=1:raise AssertionError("teacher and students used different token inputs")
        for name in SOURCE_KINDS:
            d=loaded[name]["physical_source"]
            if d["physical_bytes"]!=TARGET_BYTES or d["physical_sha256"]!=build["variants"][name]["source_sha256"]:
                raise AssertionError("original trained packed source integrity lost")
            if d["original_2D_dense_hot_weights"]!=0 or not d["tied_embeddings_verified"]:
                raise AssertionError("packed model not legitimately native full source")
        a=loaded["baseline"]["scores_per_window"]
        b=loaded["l0_hadamard"]["scores_per_window"]
        if len(a)!=WINDOWS or len(b)!=WINDOWS:
            raise AssertionError("incomplete paired test")
        z={}
        for key in ("teacher_student_KL","next_token_NLL","teacher_top1_agreement"):
            xa=np.array([u[key] for u in a],dtype=np.float64)
            xb=np.array([u[key] for u in b],dtype=np.float64)
            d=xb-xa
            ix=rng.integers(0,WINDOWS,size=(2000,WINDOWS))
            lo,hi=np.quantile(d[ix].mean(1),[0.025,0.975])
            z[key]={"baseline_mean":float(np.mean(xa)),
                    "selected_v70_mean":float(np.mean(xb)),
                    "selected_minus_baseline":float(np.mean(d)),
                    "paired_16_window_bootstrap_95pct_interval":[float(lo),float(hi)],
                    "improved_window_count":int(np.sum(d<0)),
                    "worse_window_count":int(np.sum(d>0))}
        out[corpus]={"source":loaded["teacher"]["provenance"]["source"],
                     "token_sha256":loaded["teacher"]["provenance"]["tokens_sha256"],
                     "next_token_positions":WINDOWS*(WINDOW-1),
                     "signed_metrics":z,
                     "CPU_window_prefill_seconds":{
                         k:loaded[k]["cpu_prefill_total_seconds"] for k in ("teacher",*SOURCE_KINDS)},
                     "blind_selection_no_fit":True}
    success=all(out[c]["signed_metrics"]["teacher_student_KL"]["selected_minus_baseline"]<0 and
                out[c]["signed_metrics"]["next_token_NLL"]["selected_minus_baseline"]<0
                for c in CORPORA)
    report={"experiment":"CALIBRI_v71_crosscorpus_blind_replication_of_v70_2D_gauge",
            "trained_source":MODEL,"model_rev":MODEL_SHA,"unique_pretrained_params":EXPECTED_UNIQUE,
            "original_and_selected_physical_bytes_each":TARGET_BYTES,
            "bpw_full_model":8*TARGET_BYTES/EXPECTED_UNIQUE,
            "frozen_earlier_v70_TRAIN_winner":"l0_hadamard",
            "v71_heldout_data_NOT_used_to_train_select_or_revise_v70_gauge":True,
            "all_packed_full_model_inference_direct_5bit_C_no_original_dense_2D":True,
            "source_reference_logits_saved_float16_for_storage":True,
            "real_human_task_QA_NOT_measured":True,
            "warm_cpu_times_exploratory_not_controlled_microbenchmark":True,
            "no_claim_of_usable_model_at_sub0_7_bpw":True,
            "corpora":out,
            "strict_both_corpora_KL_and_NLL_better":success,
            "status":"PASS cross-corpus sign replication (small scale)" if success
                     else "NEGATIVE or MIXED; no cross-corpus quality replication"}
    (args.folder/"v71_crosscorpus_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"stage":"v71_verified_real_crosscorpus","rate_bpw":round(report["bpw_full_model"],9),
                      "both_corpora_KL_NLL_better":success,
                      "corpora":{c:{
                          "KL_delta":round(out[c]["signed_metrics"]["teacher_student_KL"]["selected_minus_baseline"],5),
                          "NLL_delta":round(out[c]["signed_metrics"]["next_token_NLL"]["selected_minus_baseline"],5),
                          "KL95":out[c]["signed_metrics"]["teacher_student_KL"]["paired_16_window_bootstrap_95pct_interval"]
                          } for c in CORPORA}}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("run","summarize"))
    p.add_argument("--kind",choices=("teacher",*SOURCE_KINDS),default="teacher")
    p.add_argument("--corpus",choices=CORPORA,default="wikitext")
    p.add_argument("--folder",type=Path,default=Path("results"))
    args=p.parse_args()
    args.folder.mkdir(parents=True,exist_ok=True)
    if args.phase=="run":phase(args)
    else:summarize(args)

if __name__=="__main__":main()
