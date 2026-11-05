"""Real original 135M operator-aware reencoding, no new bit per parameter.

Uses only FP32 source INPUT energy to improve stored original matrix
representations; neither question labels nor teacher output distributions
are optimized. Actually serializes and reloads a fully packed model,
then compares real 40 generated answers and real CPU throughput with
original FP32 and original CB59 on identical frozen prompts.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from time import perf_counter
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import TOTAL_UNIQUE,ORIGINAL_CB59_SHA,MAX_SOURCE_BYTES
from v153.codec import unpack
from v162.evidence import frozen_v161
from v169.sources import all_cases,digest
from v169.grade import detect
from v94.generation import generate_one
from v108.generation import build_chat

from .protocol import ARMS,CASE_IDS,NUM_CASES,MAX_GENERATED_TOKENS,check
from .activations import collect_original_linear_input_energy
from .codec import recode


@torch.inference_mode()
def evaluate_real_answers(models,tok,cases):
    outputs={a:[] for a in ARMS}
    for i,case in enumerate(cases):
        actual_prompt=build_chat(tok,case["prompt"])
        for arm in ARMS[i%3:]+ARMS[:i%3]:
            r=generate_one(models[arm],tok,actual_prompt,stop_at_eos=True)
            ids=r["ids"]
            if not 1<=len(ids)<=MAX_GENERATED_TOKENS or r["prompt_SHA"]!=digest(actual_prompt):
                raise ValueError("not a genuine matched pretrained model response")
            status,parsed=detect({**case,"verbatim_text":r["text"]})
            outputs[arm].append({
                "question_id":case["question_id"],"domain":case["domain"],
                "style":case["style"],"prompt_sha":case["prompt_sha"],
                "question":case["question"],"reference_answer":case["gold"],
                "real_model_verbatim_text":r["text"],
                "actual_generated_token_ids":ids,
                "token_count":len(ids),"EOS":r["eos_generated"],
                "full_wall_seconds":r["elapsed_seconds"],
                "prefill_s":r["prefill_seconds"],
                "decode_s":r["decode_seconds"],
                "distinct_token_ratio":len(set(ids))/len(ids),
                "machine_extraction_status_ONLY":status,
                "machine_parsed_answer":parsed,
                "machine_exact_answer_match_IF_PARSED":
                    str(parsed).lower()==str(case["gold"]).lower()
                    if status=="parsed" and parsed is not None else None,
            })
    for arm in ARMS:
        outputs[arm].sort(key=lambda x:x["question_id"])
        if [x["question_id"] for x in outputs[arm]]!=list(CASE_IDS):
            raise ValueError("missing or mismatched real answer")
    summary={}
    for arm,rows in outputs.items():
        answered=[r for r in rows if r["machine_exact_answer_match_IF_PARSED"] is not None]
        tokens=sum(r["token_count"] for r in rows)
        elapsed=sum(r["full_wall_seconds"] for r in rows)
        summary[arm]={
            "questions":len(rows),
            "machine_parsed":len(answered),
            "machine_exact_correct":sum(x["machine_exact_answer_match_IF_PARSED"] for x in answered),
            "unparsed_or_prose_require_human_review":len(rows)-len(answered),
            "mean_token_distinctness":sum(x["distinct_token_ratio"] for x in rows)/len(rows),
            "full_model_CPU_seconds":elapsed,
            "actual_generated_tokens":tokens,
            "actual_CPU_tokens_per_second":tokens/max(elapsed,1e-9),
        }
    summary["activation_aware_CB172"]["speed_ratio_vs_original_CB59"]=(
        summary["activation_aware_CB172"]["actual_CPU_tokens_per_second"]/
        summary["original_CB59"]["actual_CPU_tokens_per_second"])
    return outputs,summary


def run(physical_path,v161_json,v161_seal,outdir):
    torch.set_num_threads(2)
    check()
    outdir.mkdir(parents=True,exist_ok=True)
    physically=unpack(physical_path.read_bytes())
    original=physically.original
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("not original SHA-locked native FP32-compressed source")
    evidence=frozen_v161(physically,v161_json,v161_seal)
    cases=[x for x in all_cases() if x["question_id"] in CASE_IDS and x["style"]=="direct"]
    if len(cases)!=NUM_CASES:
        raise ValueError("registered new QA data changed")
    frozen={
        "stage":"CALIBRI_v172_ORIGINAL_OPERATOR_INPUT_ENERGY_SAME_BIT_HYPOTHESIS",
        "physical_source_original_CB59_SHA":ORIGINAL_CB59_SHA,
        "original_full_CB59_bytes":len(original),
        "source_baseline_CB161_bytes":physically.size,
        "all_original_unique_weights":TOTAL_UNIQUE,
        "same_native_embedding_head_alias_required":True,
        "FP32_pretrained_SHA":MODEL_SHA,
        "old_16_unlabeled_activation_FIT_only":True,
        "40_real_unseen_nonfit_v169_source_questions":[x["question_id"] for x in cases],
        "locked_prompt_shas":[x["prompt_sha"] for x in cases],
        "physical_complete_source_cannot_grow_one_bit":True,
        "no_teacher_output_logit_matching_or_answers_in_fit":True,
        "official_independent_v89_TEST_not_opened":True,
        "no_live_Jessica_Render_Supabase_APK_or_main_changes":True,
    }
    (outdir/"v172_frozen_source_and_behavior_protocol_BEFORE_ANY_FIT.json").write_text(
        json.dumps(frozen,ensure_ascii=False,indent=2)+"\n")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("unexpected different source 135M pretrained architecture")
    start=perf_counter()
    input_energy,origin=collect_original_linear_input_energy(teacher,tok)
    print(json.dumps({"stage":"v172 real FP32 source input activation energy",
                      "measured_operators":origin["actual_original_FP32_linear_operators"]}),flush=True)
    new_full,fit=recode(original,teacher,input_energy)
    fit_seconds=perf_counter()-start
    model_file=outdir/"CALIBRI_v172_original_activation_aware_same_source_bytes.cb59"
    model_file.write_bytes(new_full)
    verification=CB59(model_file.read_bytes())
    if verification.hash()!=fit["candidate_sha256"] or len(new_full)!=len(original):
        raise ValueError("physically persisted strict 0.6678bpp source not valid")
    if len(new_full)>MAX_SOURCE_BYTES or len(new_full)>physically.size:
        raise ValueError("source bpp grew above owner-approved previous original CB161")
    with TemporaryDirectory(dir=outdir) as tmp:
        root=Path(tmp)
        (root/"baseline").mkdir()
        (root/"new").mkdir()
        old_path=root/"frozen_CB59_original.cb59"
        old_path.write_bytes(original)
        cb59,old_meta,_=make_model("baseline",SimpleNamespace(
            original=old_path,folder=root/"baseline"))
        cb172,new_meta,_=make_model("baseline",SimpleNamespace(
            original=model_file,folder=root/"new"))
        for mod,meta in ((cb59,old_meta),(cb172,new_meta)):
            if not meta["zero_dense_pretrained_2D_hot_weights"]:
                raise ValueError("unpaid FP32 original 2D source in memory")
            if mod.model.embed_tokens.handle is not mod.lm_head.handle:
                raise ValueError("duplicated physical tied head/embedding")
            if any(p.ndim>=2 for p in mod.parameters()):
                raise ValueError("unpaid pretrained dense source matrix")
        real_answers,summary=evaluate_real_answers(
            dict(zip(ARMS,(teacher,cb59,cb172))),tok,cases)
    report={
        "stage":frozen["stage"],"frozen":frozen,
        "original_source_verification":evidence,
        "source_original_FP32_activation_energy_summary":origin,
        "source_only_actual_K32_vector_bit_recoding":fit,
        "whole_new_source_real_physically_persisted_bytes":len(new_full),
        "unique_original_bpp":8*len(new_full)/TOTAL_UNIQUE,
        "new_full_original_pretrained_2D_codebook_SHA":verification.hash(),
        "physical_archive":model_file.name,
        "source_activation_and_FIT_CPU_seconds":fit_seconds,
        "real_40_generated_answers_all_arms":real_answers,
        "actual_behavior_and_cpu_speed":summary,
        "no_Green_or_external_model_or_output_mask":True,
        "NLL_KL_PPL_not_used_as_criterion":True,
        "no_FP32_quality_claim_without_actual_responses":True,
        "all_failure_cases_retained":True,
    }
    destination=outdir/"v172_ACTIVATION_AWARE_same_bytes_40_real_QA.json"
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report["stage"],
        "new_bpp":report["unique_original_bpp"],
        "refit_all_original_2D":fit["source_2D_operator_count"],
        "real_quality_and_speed":summary,
        "report":str(destination)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_CB161",type=Path)
    p.add_argument("v161_prior_JSON",type=Path)
    p.add_argument("v161_source_seal",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.original_CB161,a.v161_prior_JSON,a.v161_source_seal,a.out)
