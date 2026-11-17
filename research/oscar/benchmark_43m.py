#!/usr/bin/env python3
"""Oscar actual 43.261760M 12-layer GreenTunnel FP32 conversion experiment.

Only S142 trained source-derived dense FP32 mirror exists here: NOT a
separately pretrained 43M open FP32 model. Real 12-layer GT, S94/S95/S98,
S142 trained progressive rank896 and KV-cache generation are installed.
Compare cold/locked/adaptive REAL physically packed source with FP32 and
baseline. Keep honest conservative bpp excluding seed/zero-only growth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s7a_43m_fp32_transfer.actual_43m_fp32_conversion import (
    validate_parent, Stack, selected_S93_existing_train, BASE_36M,
    TOTAL_43M, digest, dump, log, MAX_BPP,
)
from oscar.encoder import (
    encode_fp32, to_cb61, MODE_COLD, MODE_LOCKED, MODE_ADAPT,
)

ORIGINAL_S142="5a706d9012a61f24678929a2a6ee20f77c922e7e0f4bb516c061f32742cc56c7"
MODES=(MODE_COLD,MODE_LOCKED,MODE_ADAPT)


def score(fp32,reference,candidate,tok,rows,copy,mode):
    outcomes=[]
    for i,row in enumerate(rows):
        q=row["question"];feature=row["feature"]
        fp_h,fp_logits=fp32.layer_probe(tok,q,feature)
        old_h,old_logits=reference.layer_probe(tok,q,feature)
        trial_h,trial_logits=candidate.layer_probe(tok,q,feature)
        reference_ids,_=reference.tokens(tok,q,feature,copy)
        fp32_ids,_=fp32.tokens(tok,q,feature,copy)
        trial_ids,_=candidate.tokens(tok,q,feature,copy)
        layer_comparison=[]
        for idx,(a,b,c) in enumerate(zip(fp_h,old_h,trial_h)):
            def cosine(left,right):
                return float(torch.nn.functional.cosine_similarity(
                    left[None],right[None]).item())
            layer_comparison.append({
               "after_original_decoder":idx-1,
               "FP32_to_original_43M_cosine":cosine(a,b),
               "FP32_to_Oscar_compressed_cosine":cosine(a,c),
               "FP32_to_Oscar_relative_L2":float(
                   torch.linalg.vector_norm(a-c)/
                   torch.linalg.vector_norm(a).clamp_min(1e-10)),
            })
        if len(layer_comparison)!=13:
            raise AssertionError("Oscar genuine original 12 decoder layer diagnostics missing")
        result={
           "source_original_TRAIN_id":row["TRAIN_id"],
           "question_sha256":digest(q.encode()),
           "mode":mode,
           "FP32_mirror_full_token_ids":fp32_ids,
           "original_working_S142_full_token_ids":reference_ids,
           "Oscar_recompressed_full_token_ids":trial_ids,
           "real_original_pretrained_FP32_mirror_trained_independently":False,
           "FP32_vs_original_S142_full_EOS_match":fp32_ids==reference_ids,
           "FP32_vs_Oscar_full_EOS_match":fp32_ids==trial_ids,
           "original_gold_id_list":row["expected"],
           "Oscar_full_gold_correct":trial_ids==row["expected"],
           "original_FP32_mirror_text":tok.decode(fp32_ids,skip_special_tokens=True),
           "Oscar_compressed_text":tok.decode(trial_ids,skip_special_tokens=True),
           "FP32_vs_original_max_abs_logits":float(
                torch.max(torch.abs(fp_logits-old_logits))),
           "FP32_vs_Oscar_max_abs_logits":float(
                torch.max(torch.abs(fp_logits-trial_logits))),
           "all_12layer_real_GT_hidden":layer_comparison,
        }
        outcomes.append(result)
        log("OSCAR_REAL_43M_GT12_FP32_actual_output",
            mode=mode,index=i+1,
            full_and_EOS=fp32_ids==trial_ids,
            original_43M=tok.decode(fp32_ids,skip_special_tokens=True)[:55],
            new_Oscar=tok.decode(trial_ids,skip_special_tokens=True)[:55],
            last_hidden_cos=round(layer_comparison[-1][
                "FP32_to_Oscar_compressed_cosine"],5))
    return outcomes


def paid_package(mode,result,benchmark_dir,a,per_model_report):
    target=benchmark_dir/mode
    target.mkdir(parents=True,exist_ok=True)
    cb=to_cb61(result.blob)
    file=target/"OSCAR_AUTHENTICATED_43M_CB61.s61"
    file.write_bytes(cb)
    # This is a REAL self-contained inference weight package; one source,
    # one S142 36-tensor learned checkpoint and one EACH router/bank/copier.
    # We do not duplicate prior's CB61. FP32 reference is not shipped.
    sources={
      "compressed_native_original_12layer_weights":file,
      "physically_trained_S142_all_36_neural_tensors":a.S142_FP16,
      "original_S94_three_profile_bank":a.S94_bank,
      "original_S95_prompt_router":a.S95_router,
      "original_S98_copy_policy":a.S98_copy_net
    }
    physical={}
    for label,src in sources.items():
        if label!="compressed_native_original_12layer_weights":
            dest=target/("OSCAR_"+src.name)
            shutil.copyfile(src,dest)
        else:dest=file
        if not dest.exists() or dest.stat().st_size<=0:
            raise AssertionError("Oscar missing physically exported standalone model component")
        physical[label]={
           "file":dest.name,
           "bytes":dest.stat().st_size,
           "sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
        }
    summary={
      "project":"Oscar","version":"v0","mode":mode,
      "real_trained_12layer_core_plus_other_mechanisms":True,
      "same_source_CALIBRI_derived_FP32_not_independent":True,
      "original_FP32_teacher_weights_NOT_shipped":True,
      "prior_learned_CB59_required_at_INFERENCE":False,
      "pretested_original_S142_TRAIN_456plus128":["362/456","17/128"],
      "full_456_plus_128_not_retested_in_this_debug_run":True,
      "only_preexisting_visible_TRAIN_questions_in_this_test":True,
      "new_heldout_QA_run":False,
      "source_codebook_original_K32_V8":True,
      "original_43M_core_geometries":{"layers":12,"hidden":320,
          "base_unique":BASE_36M,
          "structural_zero_and_seed_generated_expansion":TOTAL_43M-BASE_36M,
          "complete_architecture":TOTAL_43M},
      "preserved_original_S142_sidecar_sha256":ORIGINAL_S142,
      "all_runtime_files_with_SHA":physical,
      "physical_manifest_bytes":0,
      "actual_inference_physical_all_in_bytes":0,
      "actual_nominal_43M_bpp":0.,
      "conservative_original_36M_without_growth_bpp":0.,
      "nominal_under_0p7":False,
      "conservative_under_0p7":False,
      "physical_quality_model_accepted":False,
      "research_result":per_model_report,
      "OSCAR_FP32_input_digest":result.report["FP32_input_full_state_sha256"],
      "compiled_CB59_original_sha":result.report["source_sha256"],
    }
    manifest=target/"OSCAR_REPRODUCIBLE_MODEL_MANIFEST.json"
    # Reserve a fixed physical JSON size BEFORE filling self-referencing
    # byte fields. A naive iterative float repr can oscillate by 1-2 bytes.
    # Charge every trailing whitespace byte: valid readable JSON.
    template=(json.dumps(summary,ensure_ascii=False,sort_keys=True,
                         indent=2)+"\n").encode("utf-8")
    charged_manifest_bytes=len(template)+1024
    total=charged_manifest_bytes+sum(x["bytes"] for x in physical.values())
    summary["physical_manifest_bytes"]=charged_manifest_bytes
    summary["actual_inference_physical_all_in_bytes"]=total
    summary["actual_nominal_43M_bpp"]=8*total/TOTAL_43M
    summary["conservative_original_36M_without_growth_bpp"]=8*total/BASE_36M
    summary["nominal_under_0p7"]=8*total/TOTAL_43M<=MAX_BPP
    summary["conservative_under_0p7"]=8*total/BASE_36M<=MAX_BPP
    data=(json.dumps(summary,ensure_ascii=False,sort_keys=True,
                     indent=2)+"\n").encode("utf-8")
    if len(data)>charged_manifest_bytes:
        raise AssertionError("Oscar physically reserved manifest bytes too small")
    manifest.write_bytes(data+b" "*(charged_manifest_bytes-len(data)))
    if (manifest.stat().st_size!=summary["physical_manifest_bytes"] or
        sum(x["bytes"] for x in physical.values())+manifest.stat().st_size!=
             summary["actual_inference_physical_all_in_bytes"]):
        raise AssertionError("Oscar real original physical package components not counted exactly")
    if not summary["nominal_under_0p7"]:
        raise AssertionError("Oscar fails even 43.26M structural 0.7 budget")
    return summary


def run(a):
    torch.manual_seed(430917)
    torch.set_num_threads(2)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    prior,source,s93,side,bank,router,copy=validate_parent(a)
    audit={
      "OSCAR_algorithm":"0 independent offline FP32→CB59",
      "original_trained_43M_reused_in_reference_only":True,
      "original_source_S142_sidecar_sha":ORIGINAL_S142,
      "independent_open_43M_FP32_checkpoint_was_loaded":False,
      "baseline_source_36p38048M_decoder":True,
      "expanded_12layer_architecture_43p26176M":True,
      "original_Green_Tunnel_connected_to_12_layers":True,
      "comparison_modes":list(MODES),
      "original_43M_same_source_FP32_reference":True,
      "no_new_heldout_QA_or_TEST":True,
      "NRA_NOT_PRESENT":True,
      "no_Jessica_chat_Chrome_Render_Supabase_APK_LIVE":True,
      "no_inference_FP32_ghost_weights":True,
      "no_blind_quality_claim":True,
    }
    dump(out/"OSCAR_PRECOMMIT_BEFORE_SOURCE_RECONSTRUCTION.json",audit)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    rows=selected_S93_existing_train(tok,router)
    dense_source=s30.dense_from_paid(prior).eval().requires_grad_(False)
    if sum(p.numel() for p in dense_source.parameters())!=BASE_36M:
        raise AssertionError("Original independently pretrained 43M source wrongly assumed")
    state=dense_source.state_dict()
    fp32=Stack(prior,source,side,dense=True)
    baseline=Stack(prior,source,side,dense=False)
    modes={}
    packaged={}
    try:
        for mode in MODES:
            start=time.monotonic()
            encoded=encode_fp32(state,mode=mode,
                 prior_blob=prior.blob if mode!=MODE_COLD else None,
                 adapt_samples=4096,adapt_steps=5,prior_anchor=.15)
            restored=CB59(encoded.blob)
            if (restored.meta["original_unique_parameter_count"]!=BASE_36M or
                len([d for d in restored.meta["tensor_descriptors"].values()
                      if d["dtype"]=="CB59_K32V8"])!=85):
                raise AssertionError("Oscar original 85 matrices were not all converted")
            converted=Stack(restored,encoded.blob,side,dense=False)
            try:
                outcomes=score(fp32,baseline,converted,tok,rows,copy,mode)
            finally:
                converted.close()
            full=sum(int(x["FP32_vs_Oscar_full_EOS_match"]) for x in outcomes)
            old=sum(int(x["FP32_vs_original_S142_full_EOS_match"]) for x in outcomes)
            if old!=len(rows):
                raise AssertionError("Oscar authentic FP32 mirror to original trained CALIBRI lost 8/8 baseline")
            # This is the genuine S7B same-learned-codebook inverse control
            # now implemented by a separate coder, not copying old 5bit ids.
            if mode==MODE_LOCKED and full!=len(rows):
                raise AssertionError("Oscar original 12layer trained dictionary FP32 inverse is broken")
            stats={
               "mode":mode,"FP32_base_checkpoint_source":"derived from original S93",
               "not_independently_pretrained_FP32":True,
               "exact_FP32_mirror_original43M_matches":old,
               "exact_FP32_mirror_NEW_43M_matches":full,
               "original_43M_S142_full_per_token_plus_EOS_n":len(rows),
               "FP32_vs_Oscar_original_12layer_GT_output":outcomes,
               "source_PY_CPU_offline_encode_seconds":round(time.monotonic()-start,3),
               "encoder_physical_metrics":encoded.report,
               "same_original_S142_GreenTunnel_other_sidecars":True,
               "FP32_source_weights_not_at_INFERENCE":True,
               "new_blind_QA_evaluated":False,
               "independent_FP32→CALIBRI_universal_conversion_proven":False,
            }
            modes[mode]=stats
            pkg=paid_package(mode,encoded,out,a,stats)
            packaged[mode]={
              "CB61_SHA":pkg["all_runtime_files_with_SHA"][
                  "compressed_native_original_12layer_weights"]["sha256"],
              "bpp_43M":pkg["actual_nominal_43M_bpp"],
              "bpp_conservative_36M":pkg["conservative_original_36M_without_growth_bpp"],
              "nominal_pass":pkg["nominal_under_0p7"],
              "conservative_pass":pkg["conservative_under_0p7"],
              "full_FP32_mirror_match":full,
            }
            dump(out/"OSCAR_EXPERIMENT_INCREMENTAL_REAL_43M.json",
                 {"modes":modes,"physical_packages":packaged})
            log("OSCAR_ACTUAL_43M_MODE_DONE",
                mode=mode,FP32_same_source_full_EOS=full,
                total=len(rows),bpp_nominal=pkg["actual_nominal_43M_bpp"],
                bpp_conservative=pkg["conservative_original_36M_without_growth_bpp"])
    finally:
        fp32.close()
        baseline.close()
    report={
      **audit,
      "completed":True,
      "original_GT12_actual_active_S142_FP16_checksum_authenticated":True,
      "real_three_modes":modes,
      "real_full_physically_paid_inference_packages":packaged,
      "warm_locked_FP32_mirror_parity_passed":
           packaged[MODE_LOCKED]["full_FP32_mirror_match"]==len(rows),
      "not_a_43M_pretrained_open_checkpoint_transfer":True,
      "independent_43M_FP32_training_proof":False,
      "runtime_dependency_on_teacher_FP32":False,
      "any_valid_new_independent_FP32_blind_quality":False,
      "READY_FOR_135M_TRANSFER":False,
      "next_science":"Train warm-adaptive FP32 source on a separate real pretrained FP32 same 12layer checkpoint, protect full S142 GT and independent heldout QA. For 135M 30×576→12×320 distill, do not relabel tensors.",
    }
    dump(out/"OSCAR_FINAL_43M_GT12_FP32_CONVERTER_RESULT.json",report)
    log("OSCAR_FINAL_ACTUAL_FP32_to_43M_GT12",
        real_full_EOS=packaged[MODE_LOCKED]["full_FP32_mirror_match"],
        n=len(rows),
        cold_full=packaged[MODE_COLD]["full_FP32_mirror_match"],
        warmed_new_FP32_adapt_full=packaged[MODE_ADAPT]["full_FP32_mirror_match"],
        physical_locked_bpp=packaged[MODE_LOCKED]["bpp_43M"],
        conservative_locked_bpp=packaged[MODE_LOCKED]["bpp_conservative_36M"],
        independent_FP32_model_tested=False,
        accepted_for_deployment=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("S93_CB61","S93_sidecar","S142_FP16","S94_bank",
                "S94_report","S95_router","S95_report",
                "S98_copy_net","S98_report","out"):
        p.add_argument(arg,type=Path)
    run(p.parse_args())
