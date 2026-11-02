#!/usr/bin/env python3
"""CALIBRI v73: exact-rate real 135M CB59 vs V/O-gauged CB59 full-model quality.

Four frozen 64-token WikiText TEST windows at offsets 40000..40255.
Teacher/packed phases are separate processes. Native compressed C-GEMV
uses packed 5-bit code resident in RAM with ZERO trained dense 2D tensors.
Whole-model KL, next-token NLL, top1, prefill time are measured honestly.
No changes to live Jessica or production.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi

from direct_codebook_v59 import CB59
from direct_runtime_v59 import compile_kernel, NativeKernel, bind_into_llama
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from quality import DATASET, load_token_ids

WINDOW=64
SPLIT="test"
START=40000
NUM_WINDOWS=4
END=START+WINDOW*NUM_WINDOWS

def tokens():
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    ds_sha=HfApi().dataset_info(DATASET).sha
    ids=load_token_ids(tokenizer,SPLIT,ds_sha,END)[START:END]
    assert len(ids)==WINDOW*NUM_WINDOWS
    return ids, ds_sha

@torch.inference_mode()
def compute(model,ids):
    chunks=[]
    timestamps=[]
    for i in range(0,len(ids),WINDOW):
        one=ids[i:i+WINDOW][None]
        tic=time.perf_counter()
        output=model(input_ids=one,use_cache=False,return_dict=True)
        timestamps.append(time.perf_counter()-tic)
        logits=output.logits.detach().cpu().numpy()[0].astype(np.float32)
        assert logits.shape[0]==WINDOW and np.isfinite(logits).all()
        chunks.append(logits)
    return np.stack(chunks),timestamps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=("teacher","original","gauged","compare"),required=True)
    ap.add_argument("--source",type=Path)
    ap.add_argument("--candidate",type=Path)
    ap.add_argument("--folder",type=Path,default=Path("results"))
    args=ap.parse_args()
    args.folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    if args.phase=="compare":
        evaluate(args.folder)
        return
    ids,ds_sha=tokens()
    metadata={"phase":args.phase,"model":MODEL,"model_revision":MODEL_SHA,
              "dataset":DATASET,"dataset_rev":ds_sha,"split":SPLIT,
              "token_span":[START,END],"context_window":WINDOW,
              "windows":NUM_WINDOWS,"token_sha256":hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
    if args.phase=="teacher":
        model=AutoModelForCausalLM.from_pretrained(
            MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
            attn_implementation="eager").eval()
        metadata["pretrained_teacher_loaded"]=True
    else:
        source=args.source if args.phase=="original" else args.candidate
        if source is None or not source.is_file():
            raise AssertionError("real physical source not available: "+args.phase)
        archive=CB59(source.read_bytes())
        assert archive.meta["original_unique_parameter_count"]==EXPECTED_UNIQUE
        cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
        cfg._attn_implementation="eager"
        model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
        native=NativeKernel(compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"),
                            args.folder/("libv73_"+args.phase+".so")))
        evidence=bind_into_llama(model,archive,native)
        assert evidence["runtime_dense_multi_dim_source_parameters"]==0
        metadata.update({"pretrained_teacher_loaded":False,
                         "native_direct_packed_5bit_GEMV":True,
                         "physical_complete_source_bytes":archive.physical_bytes(),
                         "full_unique_parameters":EXPECTED_UNIQUE,
                         "full_source_bpw":8*archive.physical_bytes()/EXPECTED_UNIQUE,
                         "source_sha256":archive.hash(),
                         "packed_module_evidence":evidence})
    logits,times=compute(model,ids)
    np.save(args.folder/("v73_"+args.phase+"_logits.npy"),logits)
    metadata.update({"logits_shape":list(logits.shape),
                     "total_context_prefill_seconds":sum(times),
                     "per_window_seconds":times})
    (args.folder/("v73_"+args.phase+".json")).write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"phase":args.phase,"logits_shape":list(logits.shape),
                     "bytes":metadata.get("physical_complete_source_bytes"),
                     "seconds":round(sum(times),3)}),flush=True)

def evaluate(folder):
    meta={k:json.loads((folder/("v73_"+k+".json")).read_text())
          for k in ("teacher","original","gauged")}
    xs={k:np.load(folder/("v73_"+k+"_logits.npy"))
        for k in ("teacher","original","gauged")}
    assert xs["teacher"].shape==xs["original"].shape==xs["gauged"].shape
    assert all(np.isfinite(x).all() for x in xs.values())
    assert len({m["token_sha256"] for m in meta.values()})==1
    assert all(meta[k].get("native_direct_packed_5bit_GEMV") for k in ("original","gauged"))
    assert all(meta[k]["packed_module_evidence"]["runtime_dense_multi_dim_source_parameters"]==0
               for k in ("original","gauged"))
    assert meta["original"]["physical_complete_source_bytes"]==meta["gauged"]["physical_complete_source_bytes"]==11229303
    # Unchanged candidate is a valid negative/no-change outcome, not an integrity failure.
    # compare on CONTEXT next-token distributions at all 31 positions per window;
    # last token in each window has no known next-token target, excluded.
    t=torch.from_numpy(xs["teacher"][:,:-1,:].astype(np.float64))
    target=t.log_softmax(-1)
    P=target.exp()
    out={}
    tokenids,rev=tokens()
    assert rev==meta["teacher"]["dataset_rev"]
    for k in ("teacher","original","gauged"):
        x=torch.from_numpy(xs[k][:,:-1,:].astype(np.float64))
        q=x.log_softmax(-1)
        true_idx=tokenids.reshape(NUM_WINDOWS,WINDOW)[:,1:].reshape(-1)
        target_nll=-q.reshape(-1,q.shape[-1])[torch.arange(len(true_idx)),true_idx]
        delta=x-t
        norm=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(t).clamp_min(1e-30))
        kl=float(torch.sum(P*(target-q),dim=-1).mean())
        top=float((t.argmax(-1)==x.argmax(-1)).double().mean())
        nll=float(target_nll.mean())
        out[k]={"teacher_KL_nats_per_position":kl,
                "teacher_top1_agreement":top,
                "heldout_next_token_NLL":nll,
                "heldout_next_token_ppl":float(math.exp(min(nll,700))),
                "normalized_logits_L2_error":norm,
                "four_window_total_prefill_seconds":meta[k]["total_context_prefill_seconds"]}
    report={"experiment":"CALIBRI_v73_genuine_full_original_and_TRAIN_8D_covariance_0_6678bpw_native_LLModel",
            "source_model":MODEL,"unique_original_parameters":EXPECTED_UNIQUE,
            "same_total_real_bytes":11229303,
            "full_source_bpw":8*11229303/EXPECTED_UNIQUE,
            "split":SPLIT,"window_tokens":WINDOW,"windows":NUM_WINDOWS,
            "evaluated_next_token_positions":NUM_WINDOWS*(WINDOW-1),
            "TEST_or_VALID_labels_used_for_gauge_training":False,
            "teacher_and_two_packed_checkpoints_run_in_distinct_processes":True,
            "original_and_gauged_direct_native_C_GEMV":True,
            "results":out,
            "paired_heldout_KL_gauged_minus_CB59":out["gauged"]["teacher_KL_nats_per_position"]-
                                               out["original"]["teacher_KL_nats_per_position"],
            "paired_heldout_NLL_gauged_minus_CB59":out["gauged"]["heldout_next_token_NLL"]-
                                                out["original"]["heldout_next_token_NLL"],
            "TRAIN_8D_improvement_NOT_assumed":True,
            "no_task_QA_or_long_context_speed_claims":True,
            "candidate_is_covariance_8D_not_gauge":True,
            "TEST_token_span_frozen_before_test":[START,END]}
    (folder/"calibri_v73_full_model_quality.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"stage":"v73_real_full_model_check","rate":report["full_source_bpw"],
                      "baseline_KL":round(out["original"]["teacher_KL_nats_per_position"],4),
                      "gauged_KL":round(out["gauged"]["teacher_KL_nats_per_position"],4),
                      "baseline_NLL":round(out["original"]["heldout_next_token_NLL"],4),
                      "gauged_NLL":round(out["gauged"]["heldout_next_token_NLL"],4),
                      "next_token_count":report["evaluated_next_token_positions"]}),flush=True)

if __name__=="__main__":main()
