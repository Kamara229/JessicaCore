"""Behavioral reality check of the entire paid original block0 archive.

The *physical* candidate is v161's immutable complete CB59+block0,
11,292,174 bytes / 0.6715785 bit per original parameter. Earlier v161
NLL and teacher KL on tiny 8x64 source VALID improved, but the true
teacher top1 alignment was 1/504 and 0/504. Real QA/greedy generation
may well FAIL, and all negative outputs and exact texts are retained.

v162 32-window source VALID is a separate parallel experiment, not
smuggled into this QA selection. This run is ONLY a fixed fresh official
VALIDATION subset of 16 SciQ and 16 BoolQ, plus 4 x 32-token REAL
autoregressive chat responses from each native original model.

FP32 teacher is ONLY the read-only QA evaluator; all compressed arms use
identical pretrained original CB59 weights with the paid original 7 1bit
layer0 operator residuals toggled, NO extra head priors/green weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v162.evidence import frozen_v161
from v146.scoring import encode_prompt,label_tokens
from v148.scoring import make_score
from v94.generation import NEW_PROMPTS,generate_one
from v108.generation import build_chat

from .protocol import (check,DATASETS,ARMS,PACKED_ARMS,LABELS,
                       GENERATION_INDICES,COUNT)
from .sources import load_frozen_validation
from .audit import summarize_choices,summarize_generation


@torch.inference_mode()
def last_logits(model,ids):
    result=model(input_ids=ids,use_cache=False).logits[0,-1].detach().clone()
    if result.shape!=(49152,) or not torch.isfinite(result).all():
        raise ValueError("real 135m model did not return valid source token scores")
    return result


def qa_one(model,teacher,tokenizer,tasks):
    answer_ids={domain:label_tokens(tokenizer,LABELS[domain])
                for domain in DATASETS}
    records={arm:{domain:[] for domain in DATASETS} for arm in ARMS}
    durations={arm:0.0 for arm in ARMS}

    for domain in DATASETS:
        for question in tasks[domain]:
            ids=encode_prompt(tokenizer,question.prompt)
            t0=perf_counter()
            ref=last_logits(teacher,ids)
            durations[ARMS[0]]+=perf_counter()-t0
            t0=perf_counter()
            original=last_logits(model,ids)
            durations[ARMS[1]]+=perf_counter()-t0
            base=make_score(original,answer_ids[domain],question.correct_index)
            teacher_rec=make_score(ref,answer_ids[domain],question.correct_index)
            for arm,record in ((ARMS[0],teacher_rec),(ARMS[1],base)):
                record.update({
                    "row":question.index,
                    "prompt_sha":question.prompt_sha,
                    "passage_original_context_token_count":len(ids[0]),
                    "no_correct_option_sent_to_model_as_target":True,
                })
                records[arm][domain].append(record)

    return records,durations


def paid_qa(model,physical,root,tokenizer,tasks,records,durations):
    token_ids={domain:label_tokens(tokenizer,LABELS[domain])
               for domain in DATASETS}
    with install_paid_block(model,physical,root) as paid_meta:
        if paid_meta["paid_2D_modules"]!=7:
            raise ValueError("not exactly 7 original decoder operator source patches")
        for domain in DATASETS:
            for question in tasks[domain]:
                ids=encode_prompt(tokenizer,question.prompt)
                t0=perf_counter()
                values=last_logits(model,ids)
                durations[ARMS[2]]+=perf_counter()-t0
                score=make_score(values,token_ids[domain],question.correct_index)
                score.update({
                    "row":question.index,
                    "prompt_sha":question.prompt_sha,
                    "passage_original_context_token_count":len(ids[0]),
                    "no_correct_option_sent_to_model_as_target":True,
                })
                records[ARMS[2]][domain].append(score)


def generate_both(model,physical,root,tokenizer):
    out={arm:[] for arm in PACKED_ARMS}
    for ordinal,idx in enumerate(GENERATION_INDICES):
        category,question=NEW_PROMPTS[idx]
        prompt=build_chat(tokenizer,question)
        order=PACKED_ARMS if ordinal%2==0 else PACKED_ARMS[::-1]
        for arm in order:
            if arm==PACKED_ARMS[1]:
                context=install_paid_block(model,physical,root)
            else:
                from contextlib import nullcontext
                context=nullcontext({"native_original_CB59":True})
            with context:
                with torch.inference_mode():
                    actual=generate_one(model,tokenizer,prompt,stop_at_eos=True)
            tokens=actual["ids"]
            if not 1<=len(tokens)<=32:
                raise ValueError("real greedy output not a proper 32-token sample")
            out[arm].append({
                "prior_historical_prompt_index":idx,
                "category":category,
                "question":question,
                "prompt_SHA256":actual["prompt_SHA"],
                "actual_full_generated_token_ids":tokens,
                "verbatim_real_greedy_text":actual["text"],
                "full_model_wall_seconds":actual["elapsed_seconds"],
                "full_prefill_seconds":actual["prefill_seconds"],
                "full_greedy_decode_seconds":actual["decode_seconds"],
                "EOS_generated":actual["eos_generated"],
                "token_distinctness":len(set(tokens))/len(tokens),
                "no_unjustified_semantic_accuracy_claim":True,
            })
    for arm in PACKED_ARMS:
        rows={x["prior_historical_prompt_index"]:x for x in out[arm]}
        out[arm]=[rows[idx] for idx in GENERATION_INDICES]
    for r1,r2 in zip(out[PACKED_ARMS[0]],out[PACKED_ARMS[1]]):
        if r1["prompt_SHA256"]!=r2["prompt_SHA256"]:
            raise AssertionError("real greedy paired outputs used different prompts")
    return out


def run(archive,old_v161_report,v161_seal,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    raw=archive.read_bytes()
    physical=unpack(raw)
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original pretrained tied source changed")
    prior=frozen_v161(physical,old_v161_report,v161_seal)
    if len(raw)>MAX_SOURCE_BYTES or physical.size!=len(raw):
        raise ValueError("actual original model has unpaid archive bytes")
    # No question or generation is permitted until whole physical source
    # file is verified, SHA sealed, and exact cases/generation settings frozen.
    locked=out/"v163_source_and_choices_SEALED_BEFORE_QA.json"
    locked.write_text(json.dumps({
        "source":{"whole_SHA256":physical.sha256,"source_bytes":len(raw),
                  "all_original_unique_parameters":TOTAL_UNIQUE,
                  "bpp_all_matrices_and_codebooks":8*len(raw)/TOTAL_UNIQUE},
        "old_completed_v161_source_only_VALID":prior,
        "predeclared_arms":list(ARMS),
        "QA_domains":list(DATASETS),
        "fresh_QA_validation_row_interval":[704,720],
        "predeclared_greedy_indices":list(GENERATION_INDICES),
        "predeclared_real_generated_token_cap":32,
        "QA_examples_NOT_loaded_before_seal":True,
        "source_never_refit_from_QA_gold":True,
    },indent=2,ensure_ascii=False)+"\n")

    tasks,origins=load_frozen_validation()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("original pretrained teacher wrong architecture")
    with TemporaryDirectory(dir=out) as temp:
        root=Path(temp)
        file=root/"same_original_native_CB59.cb59"
        file.write_bytes(physical.original)
        student,metadata,_=make_model(
            "baseline",SimpleNamespace(original=file,folder=root))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("original 2D trained weight not compressed")
        emb=student.model.embed_tokens
        head=student.lm_head
        if emb.handle is not head.handle:
            raise ValueError("original pretrained 5bit tied source changed")

        choice_records,seconds=qa_one(student,teacher,tokenizer,tasks)
        paid_qa(student,physical,root,tokenizer,tasks,choice_records,seconds)
        choice_result=summarize_choices(choice_records)
        raw_generation=generate_both(student,physical,root,tokenizer)
        generation_result=summarize_generation(raw_generation)
        if student.model.embed_tokens is not emb or student.lm_head is not head:
            raise ValueError("source paid intervention modified tied embedding")
        if any(p.ndim>=2 for p in student.parameters()):
            raise ValueError("unpaid pretrained FP32 matrices survived")

    del teacher
    report_data={
        "stage":"v163_REAL_PAID_ORIGINAL_BLOCK0_NEW_QA_AND_GREEDY_GENERATION",
        "immutable_physical_archive_SHA256":physical.sha256,
        "all_original_weight_and_extra_1bit_bytes_paid":report(raw),
        "original_pretrained_CB59_SHA256":ORIGINAL_CB59_SHA,
        "previous_narrow_v161_NLL_KL_VALID_not_semantics":prior,
        "source_and_QA_generations_precommitted_before_loading":locked.name,
        "untouched_SciQ_and_BoolQ_official_validation_only":origins,
        "real_model_arms":list(ARMS),
        "16_sciq_16_boolq_gold_forced_choice_rows":choice_records,
        "actual_forced_choice_comparison":choice_result,
        "matched_real_greedy_32token_original_prompts":raw_generation,
        "genuine_autoregressive_text_and_speed":generation_result,
        "QA_first_token_full_forward_CPU_seconds":seconds,
        "no_training_or_source_weight_changes":True,
        "no_QA_labels_in_optimizer_or_model_input_targets":True,
        "original_FP32_teacher_ONLY_reference":True,
        "no_independent_v89_TEST_opened":True,
        "no_semantic_or_speed_success_claim_without_actual_human_review":True,
        "no_live_Jessica_Render_Supabase_APK_changes":True,
    }
    path=out/"v163_original_block0_true_SciQ_BoolQ_and_greedy.json"
    path.write_text(json.dumps(report_data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report_data["stage"],
        "physically_paid_all_in_bpp":report_data[
            "all_original_weight_and_extra_1bit_bytes_paid"]["all_in_bpp_unique_original"],
        "real_QA_correct_and_agreement":choice_result,
        "real_generation_speed_ratio":generation_result[
            "genuine_packed_tokens_per_second_ratio_paid_vs_CB59"],
        "real_generation_distinct_token_ratio":{
            arm:generation_result["genuine_full_model_autoregressive_measures"][arm][
                "distinct_token_ratio_over_all_four_responses"] for arm in PACKED_ARMS},
        "report":str(path),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("immutable_complete_CB161",type=Path)
    parser.add_argument("completed_v161_original_SOURCE_VALID_JSON",type=Path)
    parser.add_argument("v161_SHA_source_seal_BEFORE_valid",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    run(a.immutable_complete_CB161,a.completed_v161_original_SOURCE_VALID_JSON,
        a.v161_SHA_source_seal_BEFORE_valid,a.out)
