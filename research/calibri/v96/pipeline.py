"""FROZEN CALIBRI v96 scientific-generation forensic audit.

Independent FP32 pretrained model compared on ACTUAL generated answer content
to pinned native CB59 and pinned v92 same-byte block. No KL/NLL, no teacher
probability metrics, no changes to packed model, no already-used official TEST.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v94.provenance import verify_sources
from v92.validate import swap_one_block
from .runner import run_canaries,run_open_ended,quality_stats
from .grade import paired_behavior
from .spec import CANARIES,OPEN_ENDED,SOURCE

def run(original_path:Path,v92_path:Path,v92_report:Path,locked:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    lock=json.loads(locked.read_text())
    if lock.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04":
        raise AssertionError("no frozen source baseline")
    origin=verify_sources(original_path,v92_path,v92_report)
    if (origin["CB59_SHA256"]!=lock["frozen_reference_sources"]["CB59"]["sha256"]
        or origin["candidate_SHA256"]!=lock["frozen_reference_sources"]["v92_whole_block_inplace_candidate"]["sha256"]):
        raise AssertionError("original compressed source SHA changed")
    packed,info,_=make_model("baseline",SimpleNamespace(original=original_path,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("packed native real trained runtime expected")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    full_fp32=original_teacher()
    name=origin["rewritten_complete_operator"]
    parent,leaf,original,new=swap_one_block(packed,original_path.read_bytes(),v92_path.read_bytes(),name)
    try:
        setattr(parent,leaf,original)
        base_questions=run_canaries(packed,tokenizer,"CB59_NATIVE_FROZEN")
        base_open=run_open_ended(packed,tokenizer,"CB59_NATIVE_FROZEN")
        setattr(parent,leaf,new)
        rewrite_questions=run_canaries(packed,tokenizer,"v92_INPLACE_NO_EXTRA_BITS")
        rewrite_open=run_open_ended(packed,tokenizer,"v92_INPLACE_NO_EXTRA_BITS")
    finally:setattr(parent,leaf,original)
    reference_questions=run_canaries(full_fp32,tokenizer,"ORIGINAL_FP32_PRETRAINED")
    reference_open=run_open_ended(full_fp32,tokenizer,"ORIGINAL_FP32_PRETRAINED")
    correctness=paired_behavior(base_questions,rewrite_questions,reference_questions)
    models={"CB59":{"questions":base_questions,"open_ended":base_open,
                     "stats":quality_stats(base_questions,base_open)},
            "v92":{"questions":rewrite_questions,"open_ended":rewrite_open,
                     "stats":quality_stats(rewrite_questions,rewrite_open)},
            "original_FP32":{"questions":reference_questions,"open_ended":reference_open,
                     "stats":quality_stats(reference_questions,reference_open)}}
    # The first canary verification is a diagnostic only: no new candidates
    # trained or selected against its observed answers.
    if models["original_FP32"]["stats"]["correct_short_answers"]==0:
        status="INCONCLUSIVE_ORIGINAL_MODEL_FAILED_ALL_EXACT_CANARIES"
    elif models["v92"]["stats"]["correct_short_answers"]<models["original_FP32"]["stats"]["correct_short_answers"]:
        status="QUALITY_NOT_PRESERVED_BEHAVIORAL_REPAIR_NEEDED"
    else:
        status="CANARY_ACCURACY_NOT_BELOW_ORIGINAL_NOT_A_GENERAL_QUALITY_PASS"
    full_speed_ratio=(models["v92"]["stats"]["total_generated_tokens"]/
        models["v92"]["stats"]["full_model_CPU_total_elapsed_seconds"])/(
        models["CB59"]["stats"]["total_generated_tokens"]/
        models["CB59"]["stats"]["full_model_CPU_total_elapsed_seconds"])
    report={"experiment":"CALIBRI_v96_original_answer_and_semantic_content_canary_audit",
        "original_pretrained_model_revision":MODEL_SHA,
        "frozen_native_archives_source":origin,
        "actual_canary_source":SOURCE,
        "canary_answer_keys_scored_against_real_correct_answers_not_FP32_logits":True,
        "paired_correctness":correctness,
        "full_original_models_genuine_greedy_decoding":models,
        "native_CPU_paired_speed_ratio_v92_vs_CB59":full_speed_ratio,
        "status":status,
        "generation_lexical_score_is_only_a_gibberish_or_repetition_alarm":True,
        "full_semantic_factual_judgement_NOT_Automated":True,
        "conclusions_limited_to_fresh_synthetic_canaries":True,
        "previous_official_TEST_never_loaded":True,
        "original_model_or_packed_sources_never_retrained_or_changed":True,
        "CB59_CB88_v92_Jessica_Live_Render_Supabase_main_APK_untouched":True}
    path=out/"v96_FP32_CB59_v92_genuine_answer_and_generation_diagnosis.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v96_completed_answer_generative_canaries",
       "correct_answers":{name:v["stats"]["correct_short_answers"] for name,v in models.items()},
       "repeated_over_75pct":{name:v["stats"]["repeat_over_75pct"] for name,v in models.items()},
       "full_native_speed_ratio":full_speed_ratio,"result":status,
       "complete_compressed_archive_bytes":origin["complete_archive_bytes"],
       "official_TEST_and_Jessica_live_untouched":True},ensure_ascii=False),flush=True)
    return report
