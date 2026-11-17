#!/usr/bin/env python3
"""Oscar v1L: original pretrained42M FP32 exact PHYSICAL source/GT ABlation.

Six options made only from previously trained, literally physically paid
original 5bit K32/V8 core and two original full8 recurrent GT FP16 files.
No new gradient/teacher-calibration, no unpaid original FP32 *model* hybrid.
The actual original pretrained independent FP32 teacher is a SEPARATE
evaluation comparator, NOT shipped at inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from oscar.v1e.audit_checkpoint import source as real_fp32_source
from oscar.v1e.safe_pretrained_llama import from_checkpoint,materialize_decoded_cb59
from oscar.v1f.green_tunnel import FullLayerGreenTunnel
from oscar.v1f.train_real_all8_green_tunnel import (
    SOURCE_SHA,SOURCE_REV,TOKENIZER,TOKENIZER_REV,
    EXPECTED_ORIGINAL_UNIQUE,EVAL_OLD,EVAL_OTHER,prompt_ids,true_greedy,
)
from oscar.v1g.real_onpolicy_gt8 import (
    V1F_CB59_SHA,V1F_CB61_SHA,V1F_GT_SHA,DEBUG_NEW,
)
from oscar.v1h.real_dual_phase_inference import (
    V1G_SHA,NEW_DEBUG,PHASES,dual_phase_greedy,
)
from oscar.v1i.real_functional_codebook import NEW_TEST_DEBUG
from oscar.v1k.real_joint_source_gt8 import EXTRA_DEBUG

NEW_CB59_SHA="c03f1a2e798ff0549385bb49d341180b6136d4213bdae5797044e840517592dc"
NEW_CB61_SHA="2d3c3f1388e7c8211b665fa30318b1af3a3531c7fb63a50e462d2de21f7c41ba"
NEW_FIRST_SHA="a29f79656e71c5873b200809c7cb9611578ee6754c41a8483986d9a98b7a2016"
NEW_LATER_SHA="2fa6e700e9a0fbc92374056d361a6f476c8ed22ee4700441620f22e6cc290ffe"

SETS=(
   ("baseline_old_source_old_FIRST_old_LATER","old","old","old"),
   ("SOURCE_only_NEW_old_FIRST_old_LATER","new","old","old"),
   ("GTs_only_old_source_NEW_FIRST_NEW_LATER","old","new","new"),
   ("new_SOURCE_plus_old_FIRST_new_LATER","new","old","new"),
   ("new_SOURCE_plus_new_FIRST_old_LATER","new","new","old"),
   ("v1K_joint_NEW_SOURCE_new_FIRST_new_LATER","new","new","new"),
)
PREDECLARED=("old_v1H_20","old_v1I_8","old_v1K_8")
MAX_BPP=.7
RUNTIME_MANIFEST=3072


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def dump(p,data):
    Path(p).write_text(json.dumps(data,sort_keys=True,indent=2,
                        ensure_ascii=False)+"\n",encoding="utf8")


def log(stage,**details):
    print(json.dumps({"stage":stage,**details},ensure_ascii=False),flush=True)


def stats(rows):
    return {
       "n":len(rows),
       "correct_original_FP32_FIRST_token":sum(
           int(x["FP32_pretrained_FIRST_token_exact"]) for x in rows),
       "correct_original_FP32_FULL_8tokens_EOS":sum(
           int(x["FP32_pretrained_FULL_greedy8_EOS_exact"]) for x in rows),
       "mean_unigram_repeat":sum(
           x["original_student_greedy"]["actual_repetition"]["unigram_repeat"]
           for x in rows)/len(rows),
       "mean_bigram_repeat":sum(
           x["original_student_greedy"]["actual_repetition"]["bigram_repeat"]
           for x in rows)/len(rows),
       "mean_last_decoder_hidden_cosine_on_actual_student_prefix":sum(
           x["original_student_greedy"][
               "mean_last_hidden_cosine_on_OWN_student_prefix"] for x in rows)/len(rows),
    }


def run(a):
    torch.manual_seed(4219127)
    torch.set_num_threads(2)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar v1L research output directory must initially be empty")
    refs={
       "old_cb59":(Path(a.old_CB59),V1F_CB59_SHA),
       "old_cb61":(Path(a.old_CB61),V1F_CB61_SHA),
       "old_first":(Path(a.old_first_GT),V1F_GT_SHA),
       "old_later":(Path(a.old_later_GT),V1G_SHA),
       "new_cb59":(Path(a.new_CB59),NEW_CB59_SHA),
       "new_cb61":(Path(a.new_CB61),NEW_CB61_SHA),
       "new_first":(Path(a.new_first_GT),NEW_FIRST_SHA),
       "new_later":(Path(a.new_later_GT),NEW_LATER_SHA),
    }
    for key,(p,hash_) in refs.items():
        if not p.exists() or sha(p.read_bytes())!=hash_:
            raise AssertionError("Oscar v1L pre-existing original pretrained physical source/GT mismatch "+key)
    texts=EVAL_OLD+EVAL_OTHER+DEBUG_NEW+NEW_DEBUG+NEW_TEST_DEBUG+EXTRA_DEBUG
    if len(texts)!=36 or len(set(texts))!=36:
        raise AssertionError("Oscar v1L exact predefined PUBLIC DEBUG cohort mismatch")
    groups={
       "old_v1H_20":texts[:20],
       "old_v1I_8":texts[20:28],
       "old_v1K_8":texts[28:],
    }
    prereg={
      "experiment":"Oscar v1L original external FP32 source+both GT physically paid factorial ABlation",
      "original_pretrained_external_checkpoint_SHA256":SOURCE_SHA,
      "original_41p6896M_trained_unique":EXPECTED_ORIGINAL_UNIQUE,
      "same_geometry_8x512_not_Calibri12x320":True,
      "literal_authenticated_all_eight_original_packed_CB59_CB61_GT_SHA":{
          k:v for k,(_,v) in refs.items()},
      "no_NEW_gradient_training":True,
      "no_NEW_model_acceptance_or_heldout_QA":True,
      "six_precommitted_existing_physical_source_GT_combinations":SETS,
      "old_and_new_public_debug_PREVIOUSLY_EXPOSED":groups,
      "no_unpaid_original_FP32_matrix_substitutions_into_student":True,
      "all_real_original_teacher_next_token_targets_from_OWN_student_prefix":True,
      "all_greedy_full8_original_model_gold_recomputed":True,
      "actual_source_plus_2GT_config_target_le_0p7":True,
      "native_CPU_5bit_runtime_and_tokenizer_excluded":True,
      "original_616_J7_S142_NRA_LIVE_main_Render_Supabase_APK_UNTOUCHED":True,
    }
    dump(out/"OSCAR_V1L_PRECOMMIT_BEFORE_REAL_SOURCE_FP32_LOAD.json",prereg)
    _,source,audit=real_fp32_source(out)
    if (audit["archive_SHA256"]!=SOURCE_SHA or
        audit["pinned_exact_HF_revision_sha40"]!=SOURCE_REV or
        audit["model_parameters_real_unique"]!=EXPECTED_ORIGINAL_UNIQUE):
        raise AssertionError("Original independently pretrained TinyStories FP32 wrong provenance")
    teacher=from_checkpoint(source).eval().requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(TOKENIZER,revision=TOKENIZER_REV)
    if tok.vocab_size!=32000 or tok.eos_token_id is None:
        raise AssertionError("Original pretrained 42M teacher tokenizer mismatch")
    prebuilt=[]
    with torch.inference_mode():
        for name,items in groups.items():
            for q in items:
                context=prompt_ids(tok,q)
                expected=true_greedy(teacher,tok,context)
                prebuilt.append({
                    "cohort":name,"prompt":q,
                    "prompt_sha256":sha(q.encode()),
                    "context":context,
                    "original_FP32_reference_full_tokens":expected,
                })
    fullrows={}
    summary={}
    physical={}
    began=time.monotonic()
    for mode,source_choice,first_choice,later_choice in SETS:
        source_file=refs[source_choice+"_cb59"][0]
        student=materialize_decoded_cb59(teacher,source_file.read_bytes())
        student.eval().requires_grad_(False)
        gt0=FullLayerGreenTunnel().install(student)
        gt1=FullLayerGreenTunnel().install(student)
        gt0.restore_paid(load_file(str(refs[first_choice+"_first"][0])))
        gt1.restore_paid(load_file(str(refs[later_choice+"_later"][0])))
        gt0.requires_grad_(False)
        gt1.requires_grad_(False)
        rows=[]
        with torch.inference_mode():
            for item in prebuilt:
                actual=dual_phase_greedy(
                    student,teacher,gt0,gt1,tok,
                    item["context"],PHASES[2])
                generated=actual["generated_original_Oscar_token_ids"]
                expected=item["original_FP32_reference_full_tokens"]["ids"]
                rows.append({
                    "PUBLIC_DEBUG_cohort":item["cohort"],
                    "prompt_SHA256":item["prompt_sha256"],
                    "original_pretrained_FP32_reference_ids":expected,
                    "original_student_greedy":actual,
                    "FP32_pretrained_FIRST_token_exact":generated[0]==expected[0],
                    "FP32_pretrained_FULL_greedy8_EOS_exact":generated==expected,
                })
        gt0.uninstall()
        gt1.uninstall()
        checks={
            "original_same_pretrained_source_CB61_bytes":
                refs[source_choice+"_cb61"][0].stat().st_size,
            "first_real_original8_GT_FP16_bytes":
                refs[first_choice+"_first"][0].stat().st_size,
            "later_real_original8_GT_FP16_bytes":
                refs[later_choice+"_later"][0].stat().st_size,
            "fully_charged_runtime_JSON_config_bytes":RUNTIME_MANIFEST,
        }
        total=sum(checks.values())
        checks.update({
           "total_actual_source_plus2GT_config_bytes":total,
           "original_41689600_unique_source_plusGT_bpp":
               8*total/EXPECTED_ORIGINAL_UNIQUE,
           "actual_model_0p7_source_rate_pass":
               total<=int(MAX_BPP*EXPECTED_ORIGINAL_UNIQUE/8),
           "original_FP32_teacher_not_shipped":True,
           "tokenizer_binary_runtime_not_included":True,
        })
        if not checks["actual_model_0p7_source_rate_pass"]:
            raise AssertionError("Oscar physically PAID v1L source/GT combination exceeds 0.7 bpp")
        summary[mode]={
          "all36_public_debug":stats(rows),
          "old20_public_debug":stats(rows[:20]),
          "old28_public_debug":stats(rows[:28]),
          "last8_public_debug":stats(rows[28:]),
        }
        physical[mode]=checks
        fullrows[mode]=rows
        log("OSCAR_V1L_REAL42M_CAUSAL_SOURCE_GT_ABLATION_MODE",
            mode=mode,
            actual_first_36=summary[mode]["all36_public_debug"][
                "correct_original_FP32_FIRST_token"],
            actual_full_36=summary[mode]["all36_public_debug"][
                "correct_original_FP32_FULL_8tokens_EOS"],
            repetition=round(summary[mode]["all36_public_debug"][
                "mean_unigram_repeat"],5),
            hidden_cos=round(summary[mode]["all36_public_debug"][
                "mean_last_decoder_hidden_cosine_on_actual_student_prefix"],5),
            true_paid_bpp=checks[
                "original_41689600_unique_source_plusGT_bpp"])
    if (summary[SETS[0][0]]["all36_public_debug"][
           "correct_original_FP32_FIRST_token"]!=19 or
        summary[SETS[-1][0]]["all36_public_debug"][
           "correct_original_FP32_FIRST_token"]!=11):
        raise AssertionError("Oscar v1L source+GT original baseline 19/36 and full v1K 11/36 did NOT reproduce")
    if (summary[SETS[0][0]]["all36_public_debug"][
           "correct_original_FP32_FULL_8tokens_EOS"]!=0 or
        summary[SETS[-1][0]]["all36_public_debug"][
           "correct_original_FP32_FULL_8tokens_EOS"]!=0):
        raise AssertionError("Oscars original proven 0/36 complete response benchmark changed unexpectedly")
    for key,(p,h) in refs.items():
        if sha(p.read_bytes())!=h:
            raise AssertionError("Oscar v1L original 5bit/GT source mutated by inference-only ablation: "+key)
    result={
      **prereg,
      "all_six_existing_physically_paid_source_GreenTunnel_variant_results":summary,
      "original_full_36_references_AND_every_generated_real_token_prefix":fullrows,
      "each_variant_verified_total_physically_charged_source_plusGT":physical,
      "real_cpu_wall_seconds":round(time.monotonic()-began,3),
      "independent_pretrained_original_FP32_reference_not_runtime_model":True,
      "no_new_BOTH_GT_training_or_model_promotion":True,
      "posthoc_public_debug_selection_NOT_BLIND_QA":True,
      "original_12layer_Calibri_J7_not_retested":True,
    }
    dump(out/"OSCAR_V1L_FINAL_REAL_PRETRAINED_FP32_SOURCE_GT_FACTORIAL_ABLATION.json",
         result)
    log("OSCAR_V1L_FINAL_ORIGINAL42M_SOURCE_2GT_CAUSAL_AUDIT",
        real_original_42M_pretrained_unique=EXPECTED_ORIGINAL_UNIQUE,
        variants=6,public_DEBUG=36,
        first_by_variant={
            key:r["all36_public_debug"]["correct_original_FP32_FIRST_token"]
            for key,r in summary.items()},
        full_by_variant={
            key:r["all36_public_debug"]["correct_original_FP32_FULL_8tokens_EOS"]
            for key,r in summary.items()},
        no_new_FP32_inference_copies=True,
        no_acceptance_new_sealed_QA_native_CPU_or_616J7=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("old_CB59","old_CB61","old_first_GT","old_later_GT",
                 "new_CB59","new_CB61","new_first_GT","new_later_GT","out"):
        parser.add_argument(name,type=Path)
    run(parser.parse_args())
