"""Original FP32 135M vs 360M new independent public balanced QA pilot.

Fifty/six? no: exactly 64 source questions by four fixed domains;
the opposing behavioral pairs are graded on the COMPLETE answer,
not just on an arbitrary token ID that may be common to both.
No model training, no source compression, and no private TEST.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from v84.sources import original_teacher
from v174.run import save
from v175.evaluate import generate
from . import protocol as p
from .cases import cases
from .grade import grade,summarize


@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    dataset=cases()
    frozen={"task":"new public heldout balanced original FP32 capability QA",
        "domain_counts":{d:sum(x["domain"]==d for x in dataset) for d in p.DOMAINS},
        "all_64_questions_golds_hashes":dataset,
        "original_old_model":p.OLD_MODEL,
        "original_old_SHA":p.OLD_SHA,
        "original_old_unique_parameters":p.OLD_UNIQUE,
        "original_new_model":p.NEW_MODEL,
        "original_new_SHA":p.NEW_SHA,
        "original_new_unique_parameters":p.NEW_UNIQUE,
        "new_64_questions_do_not_duplicate_historical_96_DEV":True,
        "original_generated_answer_length_cap":p.MAX_GENERATION,
        "complete_answer_semantic_string_grader_version":"v190.grade.v1",
        "original_weight_parameters_frozen":True,
        "no_model_fitting_at_all":True,
        "no_compressed_360M_model_exists_in_this_study":True,
        "not_original_360M_0p7_bpp_pass":True,
        "independent_private_v89_TEST_unread":True,
        "no_Jessica_chat_test":True,
        "production_main_Render_Supabase_APK_untouched":True}
    save(out/"frozen_new_64QA_golds_before_model_load.json",frozen)
    model135=original_teacher().eval()
    tok135=AutoTokenizer.from_pretrained(p.OLD_MODEL,revision=p.OLD_SHA)
    model360=AutoModelForCausalLM.from_pretrained(
         p.NEW_MODEL,revision=p.NEW_SHA,torch_dtype=torch.float32).eval().cpu()
    tok360=AutoTokenizer.from_pretrained(p.NEW_MODEL,revision=p.NEW_SHA)
    if sum(x.numel() for x in model135.parameters())!=p.OLD_UNIQUE:
        raise AssertionError("Old source weights/architecture drift")
    if sum(x.numel() for x in model360.parameters())!=p.NEW_UNIQUE:
        raise AssertionError("New 360M original weights/architecture drift")
    for name,model in (("135M",model135),("360M",model360)):
        if model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr():
            raise AssertionError("Unknown duplicate/tied original source parameter count "+name)
    trained_controls={"original_fp32_models":{"135M":{
        "revision":p.OLD_SHA,"unique_parameters":p.OLD_UNIQUE},
        "360M":{"revision":p.NEW_SHA,"unique_parameters":p.NEW_UNIQUE}},
        "future_physical_0p7_bpp_360M_byte_ceiling":p.NEW_UNIQUE*7//80,
        "no_compression_touched":True}
    save(out/"both_actual_original_FP32_sources_provenance.json",trained_controls)
    models={"135M":(model135,tok135),"360M":(model360,tok360)}
    raw={arm:[] for arm in models}
    strict_grade={arm:[] for arm in models}
    for i,case in enumerate(dataset):
        for arm,(model,tok) in models.items():
            response=generate(model,tok,case["question"],cap=p.MAX_GENERATION,stop=True)
            expected_prompt=tok.apply_chat_template(
                [{"role":"user","content":case["question"]}],
                tokenize=False,add_generation_prompt=True)
            if response["prompt_sha"]!=hashlib.sha256(expected_prompt.encode()).hexdigest():
                raise AssertionError("Original model received altered frozen question")
            measured=grade(case,response)
            raw[arm].append({"case_id":case["case_id"],"domain":case["domain"],
                "question":case["question"],"gold":case["gold"],
                "prompt_sha256":response["prompt_sha"],"response":response,
                "strict_grade":measured})
            strict_grade[arm].append(measured)
        if (i+1)%8==0:
            save(out/"new_64QA_partial_raw.json",raw)
            print(json.dumps({"stage":"new_public_original_FP32_QA",
                "completed_cases_per_original_model":i+1,
                "strict_correct_so_far":{
                    arm:sum(x["machine_correct"] is True for x in val)
                    for arm,val in strict_grade.items()}}),flush=True)
    if any(len(x)!=p.TOTAL for x in raw.values()):
        raise AssertionError("Missing original pretrained generated answers")
    results={arm:summarize(dataset,grades)
             for arm,grades in strict_grade.items()}
    if len({item["question_sha256"] for item in dataset})!=p.TOTAL:
        raise AssertionError("Original public QA reuse detected")
    completed={"complete":True,"locked_before_original_model_inference":frozen,
        "source_provenance":trained_controls,
        "original_FP32_64_full_verbatim_generations_each":raw,
        "conservative_whole_answer_grading":results,
        "independent_public_DEVELOPMENT_not_hidden_PRIVATE_TEST":True,
        "no_360M_compressed_source_constructed":True,
        "all_results_including_failures_preserved":True}
    save(out/"v190_original_135M_vs_360M_new_public_64QA_complete.json",completed)
    print(json.dumps({"stage":"complete","original_sources":trained_controls,
        "grades":results}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.output)
