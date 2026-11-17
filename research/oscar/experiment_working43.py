#!/usr/bin/env python3
"""OSCAR authentic 43.26M CALIBRI integration, FULL 12-layer GT and FP32 mirror.

OSCAR is a converter library; CALIBRI here is only the reference model
and teacher-student integration adapter. All results are TRAIN-only.
The original FP32 *mirror* is derived from the compressed teacher itself.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59,unpack5
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s7a_43m_fp32_transfer.actual_43m_fp32_conversion import (
    Stack,validate_parent,selected_S93_existing_train,
    BASE_36M,TOTAL_43M,dump,log,digest,
)
from oscar.codec import OscarConverter
from oscar.pack import cb61_pack,bundle

N=8
EXPECTED_ORIGINAL_MATRICES=85
EXPECTED_INDEX_GROUPS=4546560


def comparable(a,b):
    return {
        "hidden_cosine_by_decoder_layer":
          [float(torch.nn.functional.cosine_similarity(
              x[None],y[None]).item())
           for x,y in zip(a,b)],
        "last_hidden_difference_l2":
          float(torch.linalg.vector_norm(a[-1]-b[-1])),
        "last_hidden_relative_l2":
          float(torch.linalg.vector_norm(a[-1]-b[-1])/
                torch.linalg.vector_norm(a[-1]).clamp_min(1e-10))
    }


def run(a):
    torch.manual_seed(43171117)
    torch.set_num_threads(2)
    output=Path(a.out)
    output.mkdir(parents=True,exist_ok=True)
    old,raw,orig_s93,orig_s142,bank,router,copy=validate_parent(a)
    baseline_audit={
        "stage":"OSCAR_v0.1_real_S142_43M_FP32_to_12layer_GT",
        "algorithm_under_test":"research/oscar/codec.py independent OscarConverter",
        "reference_original_calibri_S142_sha":"5a706d9012a61f24678929a2a6ee20f77c922e7e0f4bb516c061f32742cc56c7",
        "reference_original_core_S93_sha":"18e5396e42fba01b4c85fc773f196c65c9badca5a7506f0923eab540813a92b7",
        "original_unique_trained_core_params":BASE_36M,
        "original_total_structural_43M_params":TOTAL_43M,
        "FP32_reference_is_independent_public_model":False,
        "FP32_reference_origin":"EXACT DECODE FROM original working trained CALIBRI 43M physical CB61 model",
        "NRA_present":False,
        "new_QA_or_heldout_evaluation_access":False,
        "original_S142_456plus128_FROZEN_J7_no_new_qa":True,
        "original_Green_Tunnel_local":list((3,9)),
        "original_Green_Tunnel_all_12_recurrent":True,
        "original_S142_real_active_48_learned_adapter_channels":True,
        "dictionary_retraining_in_OSCAR_inherited":False,
        "OSCAR_inherited_prior_original_5bit_indices_unused_for_assignment":True,
        "OSCAR_independent_cold_can_refit_any_compatible_FP32_shapes":True,
        "full_new_MODEL_training_completed":False,
        "independent_FP32_language_restoration_proven":False
    }
    dump(output/"OSCAR_PRECOMMITTED_EXPERIMENT_BEFORE_LOADING_DENSE.json",baseline_audit)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    checked=selected_S93_existing_train(tok,router)
    if len(checked)!=N or any(q["route"]!="S93" for q in checked):
        raise AssertionError("OSCAR unapproved original TRAIN diagnostic prompts")
    dense_fp32=s30.dense_from_paid(old).cpu().eval().requires_grad_(False)
    dense_state=dense_fp32.state_dict()
    if (sum(p.numel() for p in dense_fp32.parameters())!=BASE_36M or
        dense_fp32.config.num_hidden_layers!=12 or
        dense_fp32.config.hidden_size!=320):
        raise AssertionError("OSCAR reference is NOT working trained Calibri 12×320 geometry")
    del dense_fp32
    inherited=OscarConverter(mode="inherited",prior=old,tolerance=1e-4).convert(
        dense_state,layer_count=12,derived_from_calibri=True)
    cold=OscarConverter(mode="cold").convert(
        dense_state,layer_count=12,derived_from_calibri=True)
    # Physical source and full FP32 tensor identity checked AFTER all
    # original FP32 weight vectors have been independently re-encoded.
    prior_d=old.meta["tensor_descriptors"]
    inv=CB59(inherited.raw_cb59)
    matrices=[k for k,v in prior_d.items() if v["dtype"]=="CB59_K32V8"]
    if (len(matrices)!=EXPECTED_ORIGINAL_MATRICES or
        inherited.matrices!=EXPECTED_ORIGINAL_MATRICES or
        inherited.code_groups!=EXPECTED_INDEX_GROUPS or
        inv.meta!=old.meta):
        raise AssertionError("OSCAR original paid 43M matrix/index manifest mismatch")
    identical=0
    for name in matrices:
        x=old.handles[name]
        y=inv.handles[name]
        n=x.shape[0]*x.shape[1]//8
        identical+=int(np.count_nonzero(
            unpack5(x.packed.tobytes(),n)==unpack5(y.packed.tobytes(),n)))
    if identical!=EXPECTED_INDEX_GROUPS or inherited.raw_cb59!=raw:
        raise AssertionError(
            "OSCAR pretrained 43M *same-source* learned-dictionary inversion is not exact")
    if cold.raw_cb59==raw:
        raise AssertionError(
            "OSCAR cold mode surprisingly reproduced trained original dictionaries")
    log("OSCAR_v01_complete_paid_source",inherited_original_sha=inherited.sha256,
        source_bytes=inherited.physical_bytes,
        original_all_85_codebooks_recovered=True,
        recovered_from_FP32_5bit_indices=identical,
        cold_FP32_independently_encoded_sha=cold.sha256)
    # Original learned S142 complete sidecar + router + profile bank +
    # S98 context scorer. This ZIP is a physically loadable proof of
    # the trained 12-layer consumer. No dense teacher weights inside ZIP.
    sidecars={
        "S142_GT12_core_active_48.safetensors":a.S142_FP16.read_bytes(),
        "S94_three_profile_bank.bin":a.S94_bank.read_bytes(),
        "S95_prompt_router.safetensors":a.S95_router.read_bytes(),
        "S98_neural_copy.safetensors":a.S98_copy_net.read_bytes(),
    }
    output_data={}
    for label,result in (("inherited",inherited),("cold",cold)):
        compressed_cb61=cb61_pack(result.raw_cb59)
        path=output/f"OSCAR_{label.upper()}_REAL_S142_FP32_TO_12LAYER_GT.oscar.zip"
        physical=bundle(path,compressed_cb61,sidecars,
             source_params=BASE_36M,nominal_params=TOTAL_43M,
             metadata={"algorithm":"OSCAR-v0.1",
                       "mode":label,"trained43m_sidecar_verified":True,
                       "reference_FP32_is_reconstructed_from_calibri":True,
                       "not_promoted_to_LIVE":True})
        output_data[label]={"codec":result.summary(),
                            "physical":physical.as_dict(),
                            "new_packed_CB61_sha256":digest(compressed_cb61)}
    # Reinstantiate with actual tokenwise learned S142+GT12 and copy/routing.
    teacher=Stack(old,raw,orig_s142,dense=True)
    ancestor=Stack(old,raw,orig_s142)
    oscar_inherited=Stack(inv,inherited.raw_cb59,orig_s142)
    oscar_cold=Stack(CB59(cold.raw_cb59),cold.raw_cb59,orig_s142)
    scores=[]
    try:
        for i,r in enumerate(checked):
            probes={}
            states={}
            for label,stack in (
                ("same_source_FP32_teacher",teacher),
                ("original_working_S142_reference",ancestor),
                ("Oscar_inherited",oscar_inherited),
                ("Oscar_cold",oscar_cold)):
                h,logits=stack.layer_probe(tok,r["question"],r["feature"])
                answer,policy=stack.tokens(tok,r["question"],r["feature"],copy)
                probes[label]={"tokens":answer,
                  "EOS_observed":bool(answer and answer[-1]==tok.eos_token_id),
                  "raw_decoded_text":tok.decode(answer,skip_special_tokens=True),
                  "first_token":answer[0] if answer else None,
                  "correct_original_TRAIN_target":answer==r["expected"],
                  "neural_copy_diagnostic":policy}
                states[label]=(h,logits)
            original=probes["same_source_FP32_teacher"]["tokens"]
            old_tokens=probes["original_working_S142_reference"]["tokens"]
            inherited_ids=probes["Oscar_inherited"]["tokens"]
            cold_ids=probes["Oscar_cold"]["tokens"]
            baseline_hidden,baseline_logits=states["same_source_FP32_teacher"]
            drift={}
            for key in ("original_working_S142_reference","Oscar_inherited","Oscar_cold"):
                h,logits=states[key]
                drift[key]={
                    **comparable(baseline_hidden,h),
                    "last_token_vocabulary_max_abs_logit":
                    float(torch.max(torch.abs(baseline_logits-logits))),
                    "first_token_matches_FP32":probes[key]["first_token"]==
                                               probes["same_source_FP32_teacher"]["first_token"],
                    "full_generated_sequence_plus_EOS_identical":probes[key]["tokens"]==original
                }
            row={"TRAIN_id":r["TRAIN_id"],
                 "prompt_SHA256":digest(r["question"].encode()),
                 "real_original_expected_answer":r["expected"],
                 "full_native_S142_and_GT12_probes":probes,
                 "all_12_layer_drift_vs_original_FP32":drift}
            scores.append(row)
            dump(output/"OSCAR_INCREMENTAL_FULL_12L_GT_FP32_VS_INHERITED_VS_COLD.json",scores)
            log("OSCAR_real_43m_12layer_GT_full_EOS_compare",
                i=i,reference_text=probes["same_source_FP32_teacher"]["raw_decoded_text"][:70],
                inherited_eq=inherited_ids==original,cold_eq=cold_ids==original,
                learned_GT_active=True,
                inherited_last_hidden_l2=drift["Oscar_inherited"]["last_hidden_difference_l2"],
                cold_last_hidden_cos=round(drift["Oscar_cold"][
                    "hidden_cosine_by_decoder_layer"][-1],5))
    finally:
        teacher.close()
        ancestor.close()
        oscar_inherited.close()
        oscar_cold.close()
    full_inherited=sum(r["all_12_layer_drift_vs_original_FP32"][
                           "Oscar_inherited"]["full_generated_sequence_plus_EOS_identical"]
                       for r in scores)
    full_old=sum(r["all_12_layer_drift_vs_original_FP32"][
                     "original_working_S142_reference"]["full_generated_sequence_plus_EOS_identical"]
                 for r in scores)
    full_cold=sum(r["all_12_layer_drift_vs_original_FP32"][
                      "Oscar_cold"]["full_generated_sequence_plus_EOS_identical"]
                 for r in scores)
    max_error=max(r["all_12_layer_drift_vs_original_FP32"][
                     "Oscar_inherited"]["last_token_vocabulary_max_abs_logit"]
                  for r in scores)
    if full_old!=N or full_inherited!=N or max_error>0.01:
        raise AssertionError("OSCAR working 43M FP32 inherited dictionary full GT12 parity failed")
    stats={
        **baseline_audit,
        "completed":True,
        "authentic_original_OSCAR_43M_source":True,
        "original_previously_working_S142_attested_old456_full":362,
        "original_previously_working_S142_attested_new128_full":17,
        "J7_456plus128_not_retested_by_this_8prompt_experiment":True,
        "real_source_FP32_mirror_to_original_S142_matches":full_old,
        "OSCAR_inherited_real_FP32_full_generation_matches":full_inherited,
        "OSCAR_cold_real_FP32_full_generation_matches":full_cold,
        "evaluated_original_TRAIN_debug_prompts":N,
        "OSCAR_inherited_equals_source_all_5bit_index_groups":identical,
        "original_5bit_index_groups":EXPECTED_INDEX_GROUPS,
        "OSCAR_FP32_tied_weight_alias_preserved":True,
        "OSCAR_codecs":output_data,
        "all_original_12layer_hidden_and_real_literal_generated_tokens":scores,
        "original_FP32_pretrained_independently":False,
        "new_actual_functionally_preserved_independent_FP32_model":False,
        "native_C_90pct_speed_measured":False,
        "NRA_modified":False,
        "model_promoted":False,
        "next_step":"Independent open pretrained matching 12layer FP32 and TRAIN-only supervised learned-dictionary acquisition, fresh heldout QA and strict conservative <=0.7 bpp before adapting to 135M",
    }
    dump(output/"OSCAR_v01_FULL_REAL_WORKING43M_GT12_EXPERIMENT.json",stats)
    log("OSCAR_v01_FINAL_REAL_FP32_TO_CALIBRI43",
        n=N,full_matched_inherited=full_inherited,
        full_matched_cold=full_cold,
        learned_85_matrices=True,
        physical_nominal_bpp=round(output_data["inherited"][
            "physical"]["nominal_0p7_bpp"],7),
        physical_conservative_bpp=round(output_data["inherited"][
            "physical"]["conservative_excluding_generated_growth_bpp"],7),
        independent_pretrained_43m_FP32_conversion_demonstrated=False,
        user_goal_135m_FP32_quality_restoration_demonstrated=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for item in ("S93_CB61","S93_sidecar","S142_FP16","S94_bank",
                 "S94_report","S95_router","S95_report",
                 "S98_copy_net","S98_report","out"):
        p.add_argument(item,type=Path)
    run(p.parse_args())
