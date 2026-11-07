"""Recover SEALED S4 physical CB59 after v200 max_new_tokens=14 API error.

S4's 2048 full-model TRAIN updates ALREADY COMPLETED in CI #42381140396,
which uploaded the physical file BEFORE its posttrain probe mistakenly passed
14, violating v200's frozen [2,12] token bound. Never rerun training merely
to recover the immutable physically paid S4. Independently validate SHA,
full storage, official TRAIN outputs from reloaded archive and canonical
1D/alias layout. The QA/VALID questions are not read in this recovery.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config, CANDIDATES
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.diverse_task_replay import make_new_pairs
from calibri2.s4.last_mile import START_SHA, START_BYTES, PARAMS, STEPS, new_pairs
from v186.pairs import build_pairs
from v200.rollout import probe_greedy_train
from v174.run import save

ORIGIN_RUN = 42381140396
ORIGIN_UPLOADED_ARTIFACT_ID = 13578581638
MAX_NEW_TOKENS = 12
EXPECTED_SOURCE_SHA = START_SHA

def evaluate(model, sources, pairs):
    observed = probe_greedy_train(model, sources, pairs, max_new_tokens=MAX_NEW_TOKENS)
    data = []
    for p in pairs:
        for side in ("left","right"):
            v = observed[p["pair_id"]]["sides"][side]
            data.append({
              "pair_id": p["pair_id"], "domain": p["domain"],
              "side": side, "gold": p[side]["answer"],
              "actual_generated_ids": v["generated_tokens"],
              "strict_full_EOS": bool(v["exact_with_EOS"]),
            })
    return data

def main():
    a = argparse.ArgumentParser()
    a.add_argument("archive",type=Path)
    a.add_argument("out",type=Path)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    raw=args.archive.read_bytes()
    if len(raw)!=START_BYTES:
        raise AssertionError("Recovery must use exact paid S4 size")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=PARAMS or 8*len(raw)/PARAMS>.700:
        raise AssertionError("Full original compressed storage violates 0.7bpp")
    if len([x for x in archive.meta["tensor_descriptors"].values()
            if x["dtype"]=="CB59_K32V8"])!=43:
        raise AssertionError("S4 changed 43 operators")
    if archive.handles["model.embed_tokens.weight"] is not archive.handles["lm_head.weight"]:
        raise AssertionError("S4 failed tied embedding alias")
    src_sha=hashlib.sha256(raw).hexdigest()
    if src_sha==EXPECTED_SOURCE_SHA:
        raise AssertionError("S4 output unchanged from S3 starting source")
    cfg=build_config(next(s for s in CANDIDATES if s[0]=="small_320x6"))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    pairs=build_pairs(tok,"TRAIN")
    s3=make_new_pairs(tok)
    s4=new_pairs(tok)
    small=pairs[:4]+[p for p in s3 if "two_word" in p["domain"]][:4]+s4[:8]
    if len(small)!=16:
        raise AssertionError("Recovery changed the original S4 32-TRAIN-response probe")
    model,sources=create_small_paid_model(archive,cfg)
    source_first=evaluate(model,sources,small)
    del model,sources
    model,sources=create_small_paid_model(CB59(args.archive.read_bytes()),cfg)
    source_second=evaluate(model,sources,small)
    if source_first!=source_second:
        raise AssertionError("S4 sealed archive not reproducible on independent reload")
    summary={
      "stage":"S4_SEALED_RECOVERY_NO_RETRAINING",
      "origin_run":ORIGIN_RUN,"artifact_id":ORIGIN_UPLOADED_ARTIFACT_ID,
      "physical_sha256":src_sha,"result_sha256":src_sha,
      "initial_S3_sha256":EXPECTED_SOURCE_SHA,
      "total_steps_originally_completed":STEPS,
      "original_failure":"v200 probe cap 14 > 12 after archive write",
      "recovery_not_a_new_training_run":True,
      "full_physical_bytes":len(raw),"complete_bpp":8*len(raw)/PARAMS,
      "original_unique_parameters":PARAMS,
      "paid_5bit_operators":43,
      "reloaded_twice_same_actual_answers":True,
      "TRAIN_only_32_response_probe":source_second,
      "strict_correct_TRAIN_32":sum(v["strict_full_EOS"] for v in source_second),
      "no_QA_or_VALID_used_to_select_weights":True,
      "model_not_yet_FP32_quality_parity":True,
    }
    save(args.out/"ACTUAL_CALIBRI2_S4_RECOVERED_SEALED.json",summary)
    print(json.dumps({k:v for k,v in summary.items()
        if k!="TRAIN_only_32_response_probe"}),flush=True)

if __name__=="__main__":main()
