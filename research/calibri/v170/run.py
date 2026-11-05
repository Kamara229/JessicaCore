"""CALIBRI v170: changed REAL codebook source, native packed real answers.

Unlike v169 (QA only), this actually creates a new physically serialized
candidate: EXACT same total number of bytes as original CB59 source.
Refit from original pretrained FP32 matrix weights, never from QA answers.
Measure equal-source-bit vs previous original and full FP32.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from direct_codebook_v59 import CB59
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v153.codec import unpack,report
from v162.evidence import frozen_v161
from v94.generation import generate_one
from v108.generation import build_chat
from v169.sources import all_cases,digest
from v169.grade import detect
from .protocol import ARMS,CASE_IDS,NUM_CASES,GENERATION_TOKENS,check
from .codec import reencode_full


@torch.inference_mode()
def run_probes(models,tok,cases):
    records={a:[] for a in ARMS}
    for i,source in enumerate(cases):
        chat=build_chat(tok,source["prompt"])
        order=ARMS[i%3:]+ARMS[:i%3]
        for arm in order:
            output=generate_one(models[arm],tok,chat,stop_at_eos=True)
            ids=output["ids"]
            if not 1<=len(ids)<=GENERATION_TOKENS or output["prompt_SHA"]!=digest(chat):
                raise ValueError("invalid unmodified real native generation")
            status,extracted=detect({**source,"verbatim_text":output["text"]})
            records[arm].append({
                "question_id":source["question_id"],
                "domain":source["domain"],
                "question":source["question"],
                "prompt_sha":source["prompt_sha"],
                "gold_for_POST_generation_review":source["gold"],
                "gold_was_never_in_prompt":source["gold"].casefold() not in source["prompt"].casefold()
                    if source["kind"]=="number" else True,
                "verbatim_generated_answer":output["text"],
                "actual_native_generated_token_ids":ids,
                "num_tokens":len(ids),"EOS":output["eos_generated"],
                "distinct_token_ratio":len(set(ids))/len(ids),
                "full_model_wall_s":output["elapsed_seconds"],
                "prefill_s":output["prefill_seconds"],
                "decode_s":output["decode_seconds"],
                "parser_status_only_not_semantic_grade":status,
                "parser_extracted_answer_if_unambiguous":extracted,
                "reference_matches_parser_exactly":(str(extracted).casefold()==str(source["gold"]).casefold()
                    if extracted is not None and status=="parsed" else None),
            })
    for arm in ARMS:
        records[arm].sort(key=lambda r:r["question_id"])
        if [r["question_id"] for r in records[arm]]!=list(CASE_IDS):
            raise ValueError("one model skipped or reordered frozen response tasks")
    return records


def report_performance(records):
    out={}
    for arm,rows in records.items():
        total=sum(x["num_tokens"] for x in rows)
        seconds=sum(x["full_model_wall_s"] for x in rows)
        known=[x for x in rows if x["reference_matches_parser_exactly"] is not None]
        out[arm]={
            "questions":len(rows),
            "machine_parsed_answers":len(known),
            "machine_verified_exact_correct":sum(x["reference_matches_parser_exactly"] for x in known),
            "ambiguous_or_prose_ungraded":len(rows)-len(known),
            "full_answer_distinct_ratio_mean":sum(x["distinct_token_ratio"] for x in rows)/len(rows),
            "actual_generated_tokens":total,
            "cpu_full_wall_seconds":seconds,
            "cpu_tokens_per_second":total/max(seconds,1e-9)
        }
    out["equal_byte_CB170"]["real_CPU_throughput_ratio_vs_CB59"]=(
        out["equal_byte_CB170"]["cpu_tokens_per_second"]/
        out["original_CB59"]["cpu_tokens_per_second"])
    return out


def run(source,report_v161,seal_v161,destination):
    torch.set_num_threads(2)
    check()
    destination.mkdir(parents=True,exist_ok=True)
    original_container=unpack(source.read_bytes())
    if hashlib.sha256(original_container.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("unapproved original pretrained compressed source")
    evidence=frozen_v161(original_container,report_v161,seal_v161)
    original=original_container.original
    if len(original)>=len(source.read_bytes()) or len(original)>MAX_SOURCE_BYTES:
        raise ValueError("invalid original source budget or wrapper")
    # Commit codebook approach, full evaluated tasks and physical rate prior
    # to the source-only original FP32 weight fit.
    benchmark=[x for x in all_cases()
               if x["question_id"] in CASE_IDS and x["style"]=="direct"]
    if len(benchmark)!=NUM_CASES:
        raise ValueError("missing precommitted diverse actual QA prompts")
    fixed={
        "stage":"v170_EXACT_SAME_SOURCE_BYTES_ALL_ORIGINAL_MATRICES_RECODE",
        "original_FULL_CB59_SHA":ORIGINAL_CB59_SHA,
        "baseline_full_CB161_sha":original_container.sha256,
        "baseline_bpp":8*original_container.size/TOTAL_UNIQUE,
        "original_source_physical_bytes":len(original),
        "new_source_bytes_must_equal_exact_original":len(original),
        "original_FP32_model_SHA":MODEL_SHA,
        "predeclared_40_question_ids":list(CASE_IDS),
        "question_prompt_SHA256":[x["prompt_sha"] for x in benchmark],
        "original_v89_TEST_unopened":True,
        "source_fitting_only_uses_pretrained_FP32_weights":True,
        "original_5bit_group_indices_physically_replaced_at_exact_same_bytes":True,
        "no_teacher_kl_nll_or_answer_loss_selection":True,
        "no_patches_to_Jessica_main_Render_Supabase_APK":True,
    }
    (destination/"v170_SOURCE_CASES_AND_BITS_SEALED_BEFORE_FIT.json").write_text(
        json.dumps(fixed,ensure_ascii=False,indent=2)+"\n")
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("wrong original pretrained FP32 teacher")
    print(json.dumps({"stage":"v170 begin same-byte recoding original pretrained 2D matrices",
                      "before_bpp":fixed["baseline_bpp"]}),flush=True)
    started=perf_counter()
    candidate,fit=reencode_full(original,teacher)
    fitting_s=perf_counter()-started
    new_file=destination/"CALIBRI_v170_FULL_original_source_same_bytes.cb59"
    new_file.write_bytes(candidate)
    if new_file.read_bytes()!=candidate:
        raise ValueError("physically persisted archive bytes mismatch")
    verify=CB59(new_file.read_bytes())
    if verify.hash()!=fit["new_same_byte_source_sha"] or len(candidate)!=len(original):
        raise ValueError("fitted physical source not an independently readable equal-byte archive")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    with TemporaryDirectory(dir=destination) as temp:
        t=Path(temp)
        baseline=t/"frozen_original_CB59.cb59"
        baseline.write_bytes(original)
        cb59,old_meta,_=make_model("baseline",SimpleNamespace(original=baseline,folder=t/"baseline"))
        cb170,new_meta,_=make_model("baseline",SimpleNamespace(original=new_file,folder=t/"new"))
        for meta in (old_meta,new_meta):
            if not meta["zero_dense_pretrained_2D_hot_weights"]:
                raise ValueError("new model decompressed to unpaid FP32 2D trained weights")
        for mod in (cb59,cb170):
            if mod.model.embed_tokens.handle is not mod.lm_head.handle:
                raise ValueError("no single physical tied embedding/head weights")
            if any(p.ndim>=2 for p in mod.parameters()):
                raise ValueError("unpaid trained 2D source weights in runtime")
        models=dict(zip(ARMS,(teacher,cb59,cb170)))
        responses=run_probes(models,tok,benchmark)
        speed=report_performance(responses)
    result={
        "stage":fixed["stage"],"pretest_source_fit_protocol":fixed,
        "original_v161_source_metadata":evidence,
        "entire_new_5bit_original_source_physical_SHA":verify.hash(),
        "physical_actual_new_source_bytes":new_file.stat().st_size,
        "all_in_unique_original_parameter_bpp":8*new_file.stat().st_size/TOTAL_UNIQUE,
        "actual_archive_artifact":new_file.name,
        "source_only_all_matrix_refit":fit,
        "real_SOURCE_fit_cpu_seconds":fitting_s,
        "actual_40_FP32_CB59_CB170_greedy_answers":responses,
        "measured_answers_and_CPU_speed":speed,
        "all_negative_outcomes_retained":True,
        "no_QA_correctness_claim_from_source_weight_RMSE":True,
        "not_a_PASSED_FP32_parity_claim_without_real_answer_review":True,
        "no_KL_NLL_PPL_gate":True,
        "new_archive_not_promoted_to_live_Jessica":True,
    }
    report_path=destination/"v170_full_same_byte_reencode_40_real_QA.json"
    report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":result["stage"],
        "fitted_matrices":fit["number_original_2D_matrices_reencoded"],
        "new_bpp":result["all_in_unique_original_parameter_bpp"],
        "real_answers_and_speed":speed,
        "report":str(report_path)
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_CB161_complete",type=Path)
    p.add_argument("v161_completed_report",type=Path)
    p.add_argument("v161_original_seal",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.original_CB161_complete,a.v161_completed_report,a.v161_original_seal,a.out)
