"""CALIBRI v169: real FP32/CB59/CB161 free-answer model inference.

All 96 cases and both prompt templates are fixed in code, 192 separate
real greedy generations per arm. No correct answer tokens are passed in
any prompt. Never score KL, NLL, PPL or vocabulary distribution matching.
"""
import argparse,hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v162.evidence import frozen_v161
from v108.generation import build_chat
from v94.generation import generate_one

from .protocol import (ARMS,FULL_ARCHIVE_BYTES,MAX_NEW_TOKENS,
                       EXPECTED_GENERATIONS_PER_ARM,check)
from .sources import all_cases,digest
from .grade import summarize

@torch.inference_mode()
def gen(models,packed,root,tokenizer,cases):
    rec={a:[] for a in ARMS}
    for i,case in enumerate(cases):
        prompt=build_chat(tokenizer,case["prompt"])
        order=ARMS[i%len(ARMS):]+ARMS[:i%len(ARMS)]
        for arm in order:
            which=models["FP32"] if arm=="FP32" else models["CB59"]
            ctx=install_paid_block(which,packed,root) if arm=="CB161" else nullcontext()
            with ctx:
                result=generate_one(which,tokenizer,prompt,stop_at_eos=True)
            ids=result["ids"]
            if not 1<=len(ids)<=MAX_NEW_TOKENS or result["prompt_SHA"]!=digest(prompt):
                raise ValueError("bad genuine model output")
            rec[arm].append({
                "question_id":case["question_id"],"domain":case["domain"],
                "style":case["style"],"question":case["question"],
                "prompt_sha":case["prompt_sha"],
                "chat_prompt_sha":result["prompt_SHA"],
                "verbatim_text":result["text"],
                "generated_token_ids":ids,
                "output_tokens":len(ids),
                "full_wall_seconds":result["elapsed_seconds"],
                "prefill_seconds":result["prefill_seconds"],
                "decode_seconds":result["decode_seconds"],
                "eos_generated":result["eos_generated"],
                "distinct_token_fraction":len(set(ids))/len(ids),
            })
    for arm in ARMS:
        rec[arm].sort(key=lambda x:(x["question_id"],x["style"]=="brief"))
        if len(rec[arm])!=EXPECTED_GENERATIONS_PER_ARM:
            raise ValueError("missing real generation")
        # all_cases() is direct-then-brief; sorting key preserves this.
        for target,source in zip(rec[arm],cases):
            if target["question_id"]!=source["question_id"] or target["style"]!=source["style"]:
                raise ValueError("bad per-arm order")
    return rec

def run(archive,old_report,seal,output):
    check()
    torch.set_num_threads(2)
    output.mkdir(parents=True,exist_ok=True)
    raw=archive.read_bytes()
    physical=unpack(raw)
    if len(raw)!=FULL_ARCHIVE_BYTES or len(raw)>MAX_SOURCE_BYTES:
        raise ValueError("not actually accounted original source")
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("CB59 original physically frozen archive changed")
    evidence=frozen_v161(physical,old_report,seal)
    cases=all_cases()
    protocol={
        "stage":"v169_REAL_DIVERSE_BALANCED_FREE_QA_NOT_TEACHER_MATCH",
        "physically_paid_source_bytes":len(raw),
        "original_parameter_count":TOTAL_UNIQUE,
        "all_in_bpp":8*len(raw)/TOTAL_UNIQUE,
        "original_packed_sha256":physical.sha256,
        "arms":ARMS,"exact_base_questions":96,"per_arm_generated_answers":len(cases),
        "case_shas":[c["prompt_sha"] for c in cases],
        "all_cases_frozen_BEFORE_loading_models":True,
        "no_correct_text_added_to_model_inputs":True,
        "no_source_or_tunnel_weights_updated":True,
        "no_teacher_lexical_alignment_target":True,
        "official_independent_v89_TEST_never_used":True,
    }
    seal_path=output/"v169_protocol_and_SHA_BEFORE_inference.json"
    seal_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("wrong original model")
    with TemporaryDirectory(dir=output) as temp:
        root=Path(temp)
        original=root/"original_unchanged_CB59.cb59"
        original.write_bytes(physical.original)
        student,meta,_=make_model("baseline",SimpleNamespace(original=original,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("packed model contains original FP32 2D matrices")
        if student.model.embed_tokens.handle is not student.lm_head.handle:
            raise ValueError("original tied source was not packed")
        actual=gen({"FP32":teacher,"CB59":student},physical,root,tok,cases)
        evaluation=summarize(cases,actual)
        if any(p.ndim>=2 for p in student.parameters()):
            raise ValueError("unpaid dense original FP32 weights in packed runtime")
    out={
        "protocol":protocol,
        "origins":{"base":"HuggingFaceTB/SmolLM2-135M-Instruct",
                   "model_sha":MODEL_SHA,"original_paid_source_v161":evidence},
        "physical_source":report(raw),
        "exact_questions_and_gold_NOT_in_prompt":cases,
        "true_real_greedy_output_review":evaluation,
        "NO_KL_NLL_PPL_metric_used":True,
        "no_production_or_main_or_GreenTunnel_changes":True,
        "this_is_a_source_research_benchmark_NOT_an_independent_official_TEST":True,
        "response_text_must_be_human_reviewed_before_quality_conclusions":True,
    }
    path=output/"v169_real_fully_packed_vs_fp32_balanced_greedy_192each.json"
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":protocol["stage"],"all_in_bpp":protocol["all_in_bpp"],
        "real_behavior_summary":evaluation["summary"],
        "report":str(path)
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("packed_archive",type=Path)
    p.add_argument("v161_completed_report",type=Path)
    p.add_argument("v161_original_seal",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.packed_archive,a.v161_completed_report,a.v161_original_seal,a.output)
