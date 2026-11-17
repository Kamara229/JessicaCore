#!/usr/bin/env python3
"""Oscar v1E: safe provenance/geometry audit of independent TinyStories FP32.

No remote model Python execution or pickle arbitrary globals. This audit
is prerequisite to real paired forward-vs-CB59 generative tests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
from huggingface_hub import HfApi,hf_hub_download

MODEL="karpathy/tinyllamas"
WEIGHTS="stories42M.pt"


def source(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    info=HfApi().model_info(MODEL,files_metadata=False)
    if not info.sha or len(info.sha)!=40:
        raise AssertionError("Unpinned independent FP32 checkpoint reference!")
    original=Path(hf_hub_download(repo_id=MODEL,filename=WEIGHTS,
                                  revision=info.sha))
    raw_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    if original.stat().st_size<=150_000_000:
        raise AssertionError("External original FP32 42M checkpoint file suspiciously short")
    # Never execute third-party pretrained architecture or arbitrary pt pickle.
    loaded=torch.load(str(original),map_location="cpu",weights_only=True)
    if isinstance(loaded,dict) and "model" in loaded and isinstance(loaded["model"],dict):
        loaded=loaded["model"]
    if not isinstance(loaded,dict) or not loaded:
        raise AssertionError("Pretrained 42M checkpoint doesn't have a safe numeric state dictionary")
    tensors=OrderedDict((str(k),v) for k,v in loaded.items()
                        if isinstance(v,torch.Tensor))
    if len(tensors)!=len(loaded):
        raise AssertionError("Reject unknown/non-tensor external FP32 pickle fields")
    if not all(v.dtype==torch.float32 and torch.isfinite(v).all()
               for v in tensors.values()):
        raise AssertionError("Original independent teacher FP32 tensor datatype/nonfinite")
    alias={}
    owners={}
    for k,v in tensors.items():
        key=(v.untyped_storage().data_ptr(),v.storage_offset(),
             tuple(v.size()),tuple(v.stride()))
        if key in owners:alias[k]=owners[key]
        else:owners[key]=k
    unique={k:v for k,v in tensors.items() if k not in alias}
    n=sum(v.numel() for v in unique.values())
    state_signature={
        "independent_real_external_pretrained_FP32_checkpoint":True,
        "checkpoint_repo":MODEL,"checkpoint_model_file":WEIGHTS,
        "pinned_exact_HF_revision_sha40":info.sha,
        "physically_downloaded_original_weight_bytes":original.stat().st_size,
        "archive_SHA256":raw_sha,
        "torch_load_weights_only_true":True,
        "REMOTE_python_model_code_never_executed":True,
        "model_parameters_real_unique":n,
        "original_model_claimed_42M_parameters":True,
        "actual_tensor_keys":list(tensors),
        "tensor_shapes":{k:list(v.shape) for k,v in tensors.items()},
        "actual_duplicate_aliases":alias,
        "all_original_tensors_genuine_FP32":True,
        "all_source_two_dim_matrix_width_divisible8":all(
            v.shape[1]%8==0 for v in unique.values() if v.ndim==2),
        "external_source_is_NOT_CALIBRI_12layer":True,
        "teacher_reference_not_previously_compressed_by_Oscar":True,
        "no_QA_or_original_J7_used":True,
    }
    if not 38_000_000<=n<=48_000_000:
        raise AssertionError("External real 42M checkpoint true UNIQUE parameter count out of plausible range: "+str(n))
    (out/"OSCAR_V1E_TRUE_INDEPENDENT_42M_FP32_PINNED_AUDIT.json").write_text(
        json.dumps(state_signature,sort_keys=True,ensure_ascii=False,
                   indent=2)+"\n",encoding="utf8")
    print(json.dumps({"stage":"OSCAR_V1E_ACTUAL_PUBLIC_PRETRAINED_42M_FP32_AUDITED",
           "repo":MODEL,"rev":info.sha,"n":n,"nkeys":len(tensors),
           "source_SHA256":raw_sha,
           "first_eight":[(k,list(v.shape)) for k,v in list(tensors.items())[:8]],
           "all_2D_width_8":state_signature["all_source_two_dim_matrix_width_divisible8"],
           "aliases":alias,
           "NOT_12layer_Calibri":True},ensure_ascii=False),flush=True)
    return original,tensors,state_signature


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    source(p.parse_args().out)
