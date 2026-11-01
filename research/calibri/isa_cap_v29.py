#!/usr/bin/env python3
"""CALIBRI v29: real pinned Switch BF16 forward under explicit oneDNN ISA cap.

This script is invoked as FRESH separate Python processes for default and
DNNL_MAX_CPU_ISA=AVX2. A requested cap is not proof of actual effective ISA;
log real CPU features and report uncertainty. No compression/PPL claims.
"""
from __future__ import annotations
import argparse, json, os, pathlib, platform
import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from huggingface_hub import HfApi
from moe_allocation_v24 import (MODEL,DATASET,EXPERTS,PINNED_SWITCH_SHA,
    token_data,load_new_corpus,make_windows,identify_next_router)
from moe_replay_v25 import fingerprint,hash_bytes
from precision_router_v27 import trace

def cpu_info():
    raw=pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8")
    fields={}
    for line in raw.splitlines():
        if ":" in line:
            k,v=line.split(":",1)
            if k.strip().lower() in ("model name","flags") and k.strip().lower() not in fields:
                fields[k.strip().lower()]=v.strip()
    flags=set(fields.get("flags","").split())
    return fields.get("model name",platform.processor()),{
        k:(k in flags) for k in ("avx2","avx512f","avx512_bf16","amx_bf16")
    }

def run(args):
    req=os.environ.get("DNNL_MAX_CPU_ISA")
    req_alias=os.environ.get("ONEDNN_MAX_CPU_ISA")
    if args.cap=="AVX2":
        if req!="AVX2" or req_alias!="AVX2":
            raise AssertionError("AVX2 cap must be set BEFORE importing Torch")
    elif req or req_alias:
        raise AssertionError("Uncapped control must unset oneDNN ISA caps")
    torch.set_num_threads(2)
    torch.manual_seed(27)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=True
    cpu,flags=cpu_info()
    if not flags.get("avx2"):
        raise AssertionError("host lacks AVX2, cannot run experiment")
    api=HfApi()
    revision=api.model_info(MODEL).sha
    if revision!=PINNED_SWITCH_SHA:
        raise AssertionError("pinned Switch source revision changed")
    wiki_rev=api.dataset_info(DATASET).sha
    tiny_rev=api.dataset_info("roneneldan/TinyStories").sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=PINNED_SWITCH_SHA)
    model,loading=SwitchTransformersForConditionalGeneration.from_pretrained(
        MODEL,revision=PINNED_SWITCH_SHA,
        torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
        output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
        raise AssertionError("pretrained checkpoint load incomplete")
    model.eval()
    for name,m in model.named_modules():
        if name.endswith(".mlp.router"):
            m.jitter_noise=0.0
    router=identify_next_router(model)
    if router is None:
        raise AssertionError("next decoder router absent")
    wiki=token_data(tok,wiki_rev,"validation",args.tokens)
    tiny,_=load_new_corpus(tok,args.tokens,tiny_rev)
    windows={
        "WikiText2_validation":list(make_windows(wiki,args.encoder_len,args.decoder_len,tok.pad_token_id)),
        "TinyStories_validation":list(make_windows(tiny,args.encoder_len,args.decoder_len,tok.pad_token_id)),
    }
    if not all(windows.values()):
        raise AssertionError("source token windows empty")
    expected={
        n:fingerprint(model.get_submodule(n).weight)
        for n in ("decoder.block.1.layer.2.mlp.router.classifier",
                  f"{EXPERTS}.expert_0.wi",f"{EXPERTS}.expert_0.wo")
    }
    report={
        "experiment":"CALIBRI_v29_Switch_BF16_oneDNN_ISA_cap",
        "source_model":MODEL,"model_revision":revision,
        "dataset_revisions":{"WikiText2":wiki_rev,"TinyStories":tiny_rev},
        "source_token_sha256":{
            "WikiText2_validation":hash_bytes(wiki.numpy().tobytes()),
            "TinyStories_validation":hash_bytes(tiny.numpy().tobytes()),
        },
        "source_key_weight_SHA256":expected,
        "torch_version":str(torch.__version__),
        "transformers_version":"4.46.3",
        "cpu_arch":platform.machine(),
        "cpu_brand":cpu,
        "cpu_supports":flags,
        "torch_cpu_capability_hardware_not_effective_ISA":
            str(torch.backends.cpu.get_cpu_capability()),
        "torch_threads":torch.get_num_threads(),
        "torch_deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "model_compute_dtype":"bfloat16",
        "mkldnn_enabled":torch.backends.mkldnn.enabled,
        "router_jitter_forced_zero":True,
        "requested_dnnl_max_cpu_isa":req or "DEFAULT",
        "requested_onednn_max_cpu_isa":req_alias or "DEFAULT",
        "effective_oneDNN_ISA_confirmed":False,
        "outputs":{},
        "scope":"Only oneDNN ISA cap requested; there may be Torch CPU kernels not governed by oneDNN. 24 fixed vocab probes, full-logit hashes, true expert masks; not speed/PPL/compression result.",
    }
    for corpus,w in windows.items():
        report["outputs"][corpus]=trace(model,w,router)
    out=pathlib.Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"cap":args.cap,"cpu":cpu,"cpu_supports":flags,
        "source_token_sha256":report["source_token_sha256"],
        "logit_hashes":{k:v["full_logit_sha256"] for k,v in report["outputs"].items()}},
        ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cap",required=True,choices=["DEFAULT","AVX2"])
    p.add_argument("--output",required=True)
    p.add_argument("--tokens",type=int,default=128)
    p.add_argument("--encoder-len",type=int,default=64)
    p.add_argument("--decoder-len",type=int,default=32)
    run(p.parse_args())
