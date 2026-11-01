#!/usr/bin/env python3
"""Real pretrained Switch CPU diagnostic: ONE factor only, torch deterministic flag.

All other model, data, dtype, MKLDNN, thread, jitter settings are fixed.
This experiment does NOT train, compress or measure PPL/packed speed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import torch
import numpy as np
from huggingface_hub import HfApi
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

from moe_allocation_v24 import (
    MODEL, DATASET, EXPERTS, PINNED_SWITCH_SHA,
    token_data, load_new_corpus, make_windows, identify_next_router,
)
from moe_replay_v25 import fingerprint, hash_bytes
from precision_router_v27 import trace

CORPORA=("WikiText2_validation","TinyStories_validation")
MODES=("deterministic_on_1","deterministic_off","deterministic_on_2")

def compare(a,b):
    x=np.asarray(a["sampled_token_logits"],dtype=np.float64)
    y=np.asarray(b["sampled_token_logits"],dtype=np.float64)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AssertionError("invalid logit probes")
    m1=np.asarray(a["exact_eight_expert_mask_per_token"],dtype=np.int8)
    m2=np.asarray(b["exact_eight_expert_mask_per_token"],dtype=np.int8)
    top1=np.asarray(a["full_vocab_top1_token_id"],dtype=np.int64)
    top2=np.asarray(b["full_vocab_top1_token_id"],dtype=np.int64)
    if m1.shape!=m2.shape or top1.shape!=top2.shape or m1.shape[0]!=top1.size:
        raise AssertionError("model/router token alignment changed")
    return {
        "full_logits_exact_sha256_equal":a["full_logit_sha256"]==b["full_logit_sha256"],
        "router_masks_exact_sha256_equal":a["router_mask_sha256"]==b["router_mask_sha256"],
        "sampled_logits_max_abs_delta":float(np.max(np.abs(x-y))),
        "sampled_logits_rmse":float(np.sqrt(np.mean((x-y)**2))),
        "full_vocab_top1_agreement":float(np.mean(top1==top2)),
        "next_router_exact_per_token_agreement":float(np.mean(np.all(m1==m2,axis=1))),
    }

def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(27)
    torch.backends.mkldnn.enabled=True
    api=HfApi()
    revision=api.model_info(MODEL).sha
    if revision!=PINNED_SWITCH_SHA:
        raise AssertionError("upstream model has changed")
    wiki_rev=api.dataset_info(DATASET).sha
    tiny_rev=api.dataset_info("roneneldan/TinyStories").sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=PINNED_SWITCH_SHA)
    model,loading=SwitchTransformersForConditionalGeneration.from_pretrained(
        MODEL,revision=PINNED_SWITCH_SHA,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
        raise AssertionError("source checkpoint did not load exactly")
    model.eval()
    for name,mod in model.named_modules():
        if name.endswith(".mlp.router"):
            mod.jitter_noise=0.0
    router=identify_next_router(model)
    if router is None:
        raise AssertionError("downstream router not found")
    wiki=token_data(tok,wiki_rev,"validation",args.tokens)
    tiny,_=load_new_corpus(tok,args.tokens,tiny_rev)
    windows={
        CORPORA[0]:list(make_windows(wiki,args.encoder_len,args.decoder_len,tok.pad_token_id)),
        CORPORA[1]:list(make_windows(tiny,args.encoder_len,args.decoder_len,tok.pad_token_id)),
    }
    if not all(windows.values()):
        raise AssertionError("empty dataset windows")
    hashes={
        n:fingerprint(model.get_submodule(n).weight)
        for n in ("decoder.block.1.layer.2.mlp.router.classifier",
                  f"{EXPERTS}.expert_0.wi",f"{EXPERTS}.expert_0.wo")
    }
    report={
        "experiment":"CALIBRI_v28_Switch_BF16_deterministic_algorithms_factor",
        "source_model":MODEL,"model_revision":revision,
        "dataset_revisions":{"WikiText2":wiki_rev,"TinyStories":tiny_rev},
        "source_token_sha256":{
            CORPORA[0]:hash_bytes(wiki.numpy().tobytes()),
            CORPORA[1]:hash_bytes(tiny.numpy().tobytes()),
        },
        "original_BF16_weight_sha256":hashes,
        "torch_version":str(torch.__version__),
        "transformers_version":"4.46.3",
        "cpu_arch":platform.machine(),
        "cpu_brand":next((line.split(":",1)[1].strip()
            for line in open("/proc/cpuinfo",encoding="utf-8")
            if line.lower().startswith("model name")), platform.processor()),
        "cpu_capability":str(torch.backends.cpu.get_cpu_capability()),
        "torch_threads":torch.get_num_threads(),
        "compute_dtype":"bfloat16",
        "mkldnn_enabled":True,
        "router_jitter_forced_zero":True,
        "modes":{},
    }
    try:
        for name,flag in zip(MODES,(True,False,True)):
            torch.use_deterministic_algorithms(flag)
            report["modes"][name]={}
            for corpus,w in windows.items():
                report["modes"][name][corpus]=trace(model,w,router)
        report["within_host_pairwise"]={}
        for corpus in CORPORA:
            first=report["modes"][MODES[0]][corpus]
            off=report["modes"][MODES[1]][corpus]
            final=report["modes"][MODES[2]][corpus]
            report["within_host_pairwise"][corpus]={
                "flag_on_vs_off":compare(first,off),
                "on_before_vs_on_after":compare(first,final),
            }
    finally:
        torch.use_deterministic_algorithms(False)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"experiment":report["experiment"],
        "cpu":report["cpu_brand"],"within_host_pairwise":report["within_host_pairwise"]},
        ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="results/calibri_v28_determinism.json")
    ap.add_argument("--tokens",type=int,default=128)
    ap.add_argument("--encoder-len",type=int,default=64)
    ap.add_argument("--decoder-len",type=int,default=32)
    run(ap.parse_args())
