"""Actual no-training 360M original teacher capability vs sealed v188 135M.

Run original fp32 360M on EXACT same public historical 192 prompts;
record model resolved SHA and genuine unique parameter count.
Never infer <=0.7 bpp or test score for a compressed source.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import model_info

from v169.sources import all_cases
from v174.answers import annotate
from v174.run import save
from v175.evaluate import generate
from v188.run import precommit,report
from . import protocol as p


def source_integrity(original,cases):
    rows=original["actual_original_FP32_192_answers"]
    if not original.get("complete") or len(rows)!=p.EXPECTED_GENERATIONS:
        raise ValueError("Missing complete original FP32 v188 raw answer audit")
    precommit(cases)
    for before,case in zip(rows,cases):
        if (before["question_id"]!=case["question_id"] or
             before["style"]!=case["style"] or
             before["domain"]!=case["domain"] or
             before["question"]!=case["question"] or
             before["source_prompt_sha"]!=case["prompt_sha"]):
            raise AssertionError("External original model answer has DIFFERENT question")
    if original["protocol"]["original_source"]!=p.ORIGINAL_MODEL:
        raise AssertionError("FP32 provenance is not original trained 135M")
    return {"verified_old_original_135m_questions":len(rows),
            "old_original_model":original["protocol"]["original_source"],
            "old_original_revision":original["protocol"]["original_revision"],
            "old_fp32_unique_parameters":
                original["protocol"]["unique_original_trainable_parameters"],
            "no_historical_question_or_style_drift":True}


def score_both(old,new):
    if len(old)!=len(new) or len(new)!=p.EXPECTED_GENERATIONS:
        raise AssertionError("Incomplete aligned historical model answers")
    report_old=report(old)
    report_new=report(new)
    compare={}
    for style in p.STYLE_CHOICES:
        index=[i for i,x in enumerate(old) if x["style"]==style]
        compare[style]={"old_135m_confirmed":report_old["styles"][style]["machine_correct"],
          "new_360m_confirmed":report_new["styles"][style]["machine_correct"],
          "new_minus_old_confirmed":report_new["styles"][style]["machine_correct"]-
                                     report_old["styles"][style]["machine_correct"],
          "old_135m_ungraded":report_old["styles"][style]["ungraded"],
          "new_360m_ungraded":report_new["styles"][style]["ungraded"],
          "both_confirmed":sum(old[i]["machine_correct"] is True and
                               new[i]["machine_correct"] is True for i in index),
          "new_correct_old_not":sum(new[i]["machine_correct"] is True and
                               old[i]["machine_correct"] is not True for i in index),
          "old_correct_new_not":sum(old[i]["machine_correct"] is True and
                               new[i]["machine_correct"] is not True for i in index)}
    return {"old":report_old,"new":report_new,"paired_summary":compare,
        "same_historical_DEV_not_blind_test":True,
        "ungraded_not_equivalent_to_incorrect":True,
        "tokenizer_IDs_not_compared_across_different_model_sources":True}


@torch.inference_mode()
def run(original_path,out):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    cases=all_cases()
    if len(cases)!=p.EXPECTED_GENERATIONS:
        raise AssertionError("Missing precommitted original questions")
    original=json.loads(Path(original_path).read_text())
    source_audit=source_integrity(original,cases)
    commit=model_info(p.OTHER_MODEL,revision="main").sha
    if not commit or len(commit)!=40:
        raise ValueError("Unable to pin stronger original pretrained revision")
    frozen={"experiment":"v189 original stronger model behavioral FP32 ONLY",
        "candidate_checkpoint":p.OTHER_MODEL,"candidate_resolved_HF_sha":commit,
        "old_actual_original_135m_reference":source_audit,
        "old_real_answer_source":"completed v188 raw artifact #13444347281",
        "historical_questions":96,"styles":list(p.STYLE_CHOICES),
        "greedy_cap":p.TOKEN_CAP,
        "actual_HF_model_unique_parameters_to_measure_after_loading":True,
        "not_compressing_or_training_any_weights":True,
        "no_independent_blind_TEST":True,
        "no_KL_NLL_PPL_as_QA_gate":True,
        "no_live_Jessica_Render_Supabase_APK_changes":True,
        "no_Jessica_chats_or_tests":True}
    save(out/"locked_HF_revision_and_benchmark_before_generation.json",frozen)
    tokenizer=AutoTokenizer.from_pretrained(p.OTHER_MODEL,revision=commit)
    model=AutoModelForCausalLM.from_pretrained(
         p.OTHER_MODEL,revision=commit,torch_dtype=torch.float32).eval().cpu()
    count=sum(x.numel() for x in model.parameters())
    tied=(model.get_input_embeddings().weight.data_ptr() ==
          model.get_output_embeddings().weight.data_ptr())
    if count<=source_audit["old_fp32_unique_parameters"]:
        raise AssertionError("Original 360M source is not larger than original 135M")
    if not tied:raise AssertionError("360M has unaccounted untied output head")
    frozen["actual_360m_unique_parameters"]=count
    frozen["same_tied_original_embedding_head"]=tied
    frozen["physically_max_0p7_bytes_for_THAT_original_model"]=int(
        (count*7)//80)
    frozen["no_360M_compressed_file_exists_in_this_study"]=True
    save(out/"stronger_source_model_provenance.json",frozen)
    records=[]
    for i,case in enumerate(cases):
        result=generate(model,tokenizer,case["prompt"],cap=p.TOKEN_CAP,stop=True)
        prompt=tokenizer.apply_chat_template(
            [{"role":"user","content":case["prompt"]}],
            tokenize=False,add_generation_prompt=True)
        if result["prompt_sha"]!=hashlib.sha256(prompt.encode()).hexdigest():
            raise AssertionError("360M prompt template mismatch")
        graded=annotate(case,result)
        graded["style"]=case["style"]
        graded["pair_id"]=case["pair_id"]
        graded["source_prompt_sha"]=case["prompt_sha"]
        records.append(graded)
        if (i+1)%16==0:
            save(out/"original_360M_verbatim_PARTIAL.json",records)
            print(json.dumps({"stage":"original_360m_dev","completed":i+1,
                "confirmed":sum(x["machine_correct"] is True for x in records),
                "ungraded":sum(x["machine_correct"] is None for x in records)}),flush=True)
    if len(records)!=192:raise AssertionError("Missing original 360M answers")
    evaluation=score_both(original["actual_original_FP32_192_answers"],records)
    saved={"complete":True,"frozen":frozen,"original_360m_raw_answers":records,
        "old_135m_to_new_360m":evaluation,
        "no_weights_modified":True,"no_compression_created":True,
        "no_PRIVATE_TEST":True,"no_Jessica_tests":True}
    save(out/"v189_original_135m_vs_360m_actual_192DEV_complete.json",saved)
    print(json.dumps({"stage":"complete","unique_parameters_360M":count,
        "new":evaluation["new"]["all"],
        "old":evaluation["old"]["all"],
        "paired":evaluation["paired_summary"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("v188",type=Path)
    parser.add_argument("output",type=Path)
    options=parser.parse_args()
    run(options.v188,options.output)
