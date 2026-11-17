#!/usr/bin/env python3
"""Oscar v1F: PHYSICAL GT-only research package from existing trained 8layer run.

The observed trained GT-only ablation has more first-token matches than
GT+head (5/8 versus 3/8 on EIGHT already-exposed DEBUG prompts). This is
EXPLORATORY SELECTION ON PUBLIC DEBUG, NOT BLIND HELDOUT model acceptance.
Do not place the unused output-head correction in this physical package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file,save_file

from oscar.v1f.green_tunnel import (
    FullLayerGreenTunnel,PaidHeadCorrection,
    PREFIX,DEPTH,RANK,
)

UNIQUE=41689600
LIMIT=int(.7*UNIQUE/8)
RESERVED_MANIFEST=3072


def digest(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(result:Path):
    result=Path(result)
    full=result/"OSCAR_V1F_REAL_PAID_RUNTIME_SOURCE"
    core=full/"OSCAR_V1F_REAL_INDEPENDENT_42M_FP32_ENCODED_CORE.cb61"
    original=full/"OSCAR_V1F_FULL8_RECURRENT_GT_AND_32K_HEAD_FP16.safetensors"
    report=result/"OSCAR_V1F_FINAL_TRUE_EXTERNAL_42M_FP32_ALL8_GT_REPAIR_REPORT.json"
    if not all(p.is_file() for p in (core,original,report)):
        raise FileNotFoundError("Cannot select green-tunnel-only physical package without COMPLETE verified original real model evidence")
    raw=load_file(str(original))
    gt=FullLayerGreenTunnel()
    wanted=set(gt.source_paid_tensors())
    if len(raw)!=6 or len(wanted)!=4 or not wanted<=set(raw):
        raise AssertionError("Oscar v1F original source must include ALL eight trained layers plus two paid head tensors")
    source= {k:v.contiguous() for k,v in raw.items() if k in wanted}
    gt.restore_paid(source)
    if (not torch.isfinite(gt.up).all() or
        int(torch.count_nonzero(gt.up))==0):
        raise AssertionError("Oscar GT-only cannot export zero/untrained eight-layer corrector")
    record=json.loads(report.read_text(encoding="utf8"))
    summaries=record["metric_summaries"]
    cold=summaries["eight_debug_original_cold_before"]
    gt_results=summaries["eight_debug_ablated_trained_GT_only"]
    joint=summaries["eight_debug_after"]
    if (cold["n"]!=8 or gt_results["n"]!=8 or joint["n"]!=8):
        raise AssertionError("GT-only parent must prove actual 8 public debug samples compared")
    # Research artifact chosen AFTER inspecting already-exposed DEBUG metrics.
    # 5/8 here is not independent genuine heldout acceptance.
    target=result/"OSCAR_V1F_GT_ONLY_EXPLORATORY_RESEARCH_RUNTIME"
    target.mkdir(parents=True,exist_ok=True)
    packed_cb61=target/"ORIGINAL_INDEPENDENT_FP32_42M_OSCAR.c61"
    packed_gt=target/"OSCAR_V1F_FULL8_ONLY_R32_FP16.safetensors"
    manifest=target/"ACTUAL_PHYSICAL_GT_ONLY_MANIFEST.json"
    shutil.copyfile(core,packed_cb61)
    save_file(source,str(packed_gt))
    reloaded=load_file(str(packed_gt))
    if set(reloaded)!=wanted or any(not torch.equal(source[k],reloaded[k]) for k in wanted):
        raise AssertionError("Oscar physical GT-only FP16 all8 restoration failed")
    total=packed_cb61.stat().st_size+packed_gt.stat().st_size+RESERVED_MANIFEST
    payload={
      "research_stage":"OSCAR_v1F_recurrent_GT_ONLY_from_actual_trained_FP32_original42M",
      "parent_GitHub_CI":"44156097254",
      "parent_S142_Calibri_12layer_NOT_MODIFIED":True,
      "original_real_trained_external_source_SHA256":
        "fbe124df8794c2e033432c22b59cff64485455b4644046c52a456d0bbd8859fe",
      "original_8layer_512hidden_TinyStories_NOT_12layer_Calibri":True,
      "original_unique_pretrained_weights":UNIQUE,
      "original_source_CB61_SHA256":digest(packed_cb61),
      "new_paid_FP16_full8_GT_SHA256":digest(packed_gt),
      "new_all8_GT_carry_hidden_rank":RANK,
      "new_32k_extra_output_head_correction_PRESENT":False,
      "real_pretrained_FP32_teacher_at_inference":False,
      "actual_CB61_bytes":packed_cb61.stat().st_size,
      "actual_full8_GT_FP16_bytes":packed_gt.stat().st_size,
      "actual_fixed_paid_manifest_bytes":RESERVED_MANIFEST,
      "total_actual_inference_source_plus_GT_manifest_bytes":total,
      "original_trained_unique_no_43M_seed_inflation_bpp":8*total/UNIQUE,
      "physical_bpp_le_0p7":total<=LIMIT,
      "prior_public_eight_debug_original_FP32_first":cold["real_fp32_teacher_first_match"],
      "exploratory_actual_GT_only_first_on_same8_PUBLIC_DEBUG":
          gt_results["real_fp32_teacher_first_match"],
      "exploratory_actual_joint_GT_head_first_on_same8_PUBLIC_DEBUG":
          joint["real_fp32_teacher_first_match"],
      "exploratory_GT_only_original_FP32_complete8_match":
          gt_results["real_fp32_teacher_FULL_8token_match"],
      "trained_GT_only_acts_at_all8_original_pretrained_decoder_layers":True,
      "selection_based_on_ALREADY_EXPOSED_DEBUG_NOT_VALID_BLIND_EVAL":True,
      "real_C_5bit_kernel_native_CPU_runtime_not_tested":True,
      "tokenizer_and_executable_runtime_excluded":True,
      "no_original_616_J7_or_independent_heldout_QA":True,
      "not_an_accepted_learnt_general_FP32_to_CALIBRI_conversion":True,
      "no_PRODUCTION_or_LIVE":True
    }
    data=(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode()
    if len(data)>RESERVED_MANIFEST:
        raise AssertionError("New runtime manifest not physically accounted")
    manifest.write_bytes(data+b" "*(RESERVED_MANIFEST-len(data)))
    if (total>LIMIT or
        sum(p.stat().st_size for p in (packed_cb61,packed_gt,manifest))!=total):
        raise AssertionError("Oscar GT-only actual full physical model not <=0.7")
    (result/"OSCAR_V1F_GT_ONLY_PUBLIC_DEBUG_RESULT.json").write_text(
      json.dumps({"cold":cold,"trained_GT_only":gt_results,
                  "trained_GT_plus_head":joint,
                  "selection_debug_only_not_blind_QA":True,
                  "paid_runtime_receipt":payload},
                 ensure_ascii=False,indent=2,sort_keys=True)+"\n",
      encoding="utf8")
    print(json.dumps({"stage":"OSCAR_V1F_FULL8_GT_ONLY_ACTUAL_PAID_PHYSICAL_MODEL",
                      "new_GT_only_bytes":total,
                      "full_noninflated_FP32_bpp":8*total/UNIQUE,
                      "real_public_debug_first_before":cold["real_fp32_teacher_first_match"],
                      "real_public_debug_first_after_GT_ONLY":gt_results["real_fp32_teacher_first_match"],
                      "real_public_debug_original_FP32_full8_after":
                        gt_results["real_fp32_teacher_FULL_8token_match"],
                      "no_independent_blind_QA_test":True,
                      "no_PRODUCTION":True},
                     ensure_ascii=False),flush=True)
    return payload


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("result_dir",type=Path)
    pack(p.parse_args().result_dir)
