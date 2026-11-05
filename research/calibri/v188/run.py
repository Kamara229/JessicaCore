"""Actual original FP32 answer capability, 96 historical questions × 2 styles.

This audit is NOT independent new QA, since the 96 public DEV
questions are already part of research history. It establishes
what the *original source model* can demonstrably answer.
No development result trains/modifies any model/source/weights.
"""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from v169.sources import all_cases
from v174.answers import annotate,summary
from v175.evaluate import generate
from v174.run import save


def precommit(cases):
    if len(cases)!=192 or len({c["prompt_sha"] for c in cases})!=192:
        raise AssertionError("Exactly two intact 96-case historical formats")
    if {c["question_id"] for c in cases}!=set(range(96)):
        raise AssertionError("Missing genuine original benchmark questions")
    if {c["style"] for c in cases}!={"direct","brief"}:
        raise AssertionError("One original public prompt style lost")
    shapes={}
    for d in sorted({x["domain"] for x in cases}):
        styles={style:len([x for x in cases if x["domain"]==d and x["style"]==style])
                for style in ("direct","brief")}
        if styles["direct"]!=styles["brief"]:raise ValueError("Unbalanced domains")
        shapes[d]=styles
    frozen={"task":"original pretrained SmolLM2-135M actual FP32 answers ONLY",
        "original_source":MODEL,"original_revision":MODEL_SHA,
        "unique_original_trainable_parameters":EXPECTED_UNIQUE,
        "historical_96_public_DEV_questions":96,
        "two_prompt_styles":["direct","brief"],
        "total_actual_original_answers":192,
        "domain_counts":shapes,
        "exact_prompt_hashes":[{"id":c["question_id"],"style":c["style"],
             "sha256":c["prompt_sha"]} for c in cases],
        "evaluation_type":"historical DEVELOPMENT benchmark, NOT a new blind TEST",
        "new_teacher_fit_or_quantizer_tuning":False,
        "fixed_greedy_decode":True,
        "max_answer_tokens":32,
        "conservative_machine_grade_unparsed_is_NOT_wrong":True,
        "no_NLL_KL_PPL_as_QA_acceptance":True,
        "private_v89_TEST_unread":True,
        "Jessica_does_NOT_run_tests":True,
        "production_main_Render_Supabase_APK_unchanged":True}
    return frozen


def paired_rows(rows,style,domain):
    chosen=[r for r in rows if r["style"]==style and r["domain"]==domain]
    groups=defaultdict(list)
    for item in chosen:
        if item.get("pair_id"):groups[item["pair_id"]].append(item)
    pairs=[]
    for key,item in sorted(groups.items()):
        if len(item)!=2:raise AssertionError("Missing opposite original historical pair")
        correct=sum(x["machine_correct"] is True for x in item)
        different_first=len({x["token_ids"][0] for x in item})>1
        pairs.append({"id":key,"questions":[x["question"] for x in item],
            "correct_both":correct==2,
            "both_are_parseable":all(x["machine_correct"] is not None for x in item),
            "distinct_first_tokens":different_first,
            "first_token_ids":[x["token_ids"][0] for x in item],
            "actual_texts":[x["verbatim_text"] for x in item]})
    return pairs


def report(rows):
    out={"all":summary(rows),"styles":{},"domains":{},"opposite_pairs":{}}
    for style in ("direct","brief"):
        subset=[x for x in rows if x["style"]==style]
        out["styles"][style]=summary(subset)
        out["domains"][style]={dom:summary([x for x in subset if x["domain"]==dom])
             for dom in sorted({x["domain"] for x in subset})}
        out["opposite_pairs"][style]={}
        for dom in ("binding","yesno"):
            pairs=paired_rows(rows,style,dom)
            out["opposite_pairs"][style][dom]={
                "paired_groups":len(pairs),
                "exact_two_answers_correct_pairs":sum(x["correct_both"] for x in pairs),
                "exact_two_parsed_pairs":sum(x["both_are_parseable"] for x in pairs),
                "genuine_first_token_changes":sum(x["distinct_first_tokens"] for x in pairs),
                "paired_verbatim":pairs}
    out["first_token_diversity"]={
        style:len({row["token_ids"][0] for row in rows if row["style"]==style})
        for style in ("direct","brief")}
    out["same_question_style_agreement"]={
        "first_token_both_styles_identical":sum(
            rows[2*i]["token_ids"][0]==rows[2*i+1]["token_ids"][0]
            for i in range(96)),
        "same_question_answer_text_identical":sum(
            rows[2*i]["verbatim_text"]==rows[2*i+1]["verbatim_text"]
            for i in range(96))}
    return out


@torch.inference_mode()
def run(output):
    torch.set_num_threads(2)
    output.mkdir(parents=True,exist_ok=True)
    cases=all_cases()
    locked=precommit(cases)
    save(output/"precommitted_actual_original_192QA.json",locked)
    from v84.sources import original_teacher
    model=original_teacher()
    count=sum(p.numel() for p in model.parameters())
    if count!=EXPECTED_UNIQUE or (
        model.get_input_embeddings().weight.data_ptr()!=
        model.get_output_embeddings().weight.data_ptr()):
        raise AssertionError("Not the original FP32 134.5M tied pretrained source")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    records=[]
    for i,case in enumerate(cases):
        response=generate(model,tok,case["prompt"],cap=32,stop=True)
        if response["prompt_sha"]!=hashlib.sha256(
                tok.apply_chat_template([{"role":"user","content":case["prompt"]}],
                    tokenize=False,add_generation_prompt=True).encode()).hexdigest():
            raise ValueError("Changed original evaluation question")
        if not response["token_ids"]:
            raise AssertionError("Empty generated response")
        result=annotate(case,response)
        result["style"]=case["style"]
        result["pair_id"]=case["pair_id"]
        result["source_prompt_sha"]=case["prompt_sha"]
        records.append(result)
        if (i+1)%16==0:
            save(output/"FP32_actual_texts_partial.json",records)
            print(json.dumps({"stage":"original_FP32_QA",
                "actual_generated_cases":i+1,
                "confirmed_machine_correct_so_far":sum(
                    x["machine_correct"] is True for x in records),
                "ungraded":sum(x["machine_correct"] is None for x in records)}),flush=True)
    if len(records)!=192:raise AssertionError("Skipped original question")
    scored=report(records)
    completed={"complete":True,"protocol":locked,
        "actual_original_FP32_192_answers":records,
        "real_answer_quality":scored,
        "this_is_historical_DEV_not_blind_independent_TEST":True,
        "no_compression_acceptance_claim":True,
        "no_weights_updated":True,
        "no_Jessica_model_tests":True}
    save(output/"v188_actual_original_FP32_192QA_complete.json",completed)
    short={style:{"machine_correct":scored["styles"][style]["machine_correct"],
                  "machine_parsed":scored["styles"][style]["machine_parsed"],
                  "ungraded":scored["styles"][style]["ungraded"],
                  "first_token_diversity":scored["first_token_diversity"][style]}
           for style in ("direct","brief")}
    print(json.dumps({"stage":"complete","all":scored["all"],
                      "style_summary":short,
                      "paired_consistency":{
                          style:{dom:{"complete_pairs":v["exact_two_answers_correct_pairs"],
                                      "first_token_changes":v["genuine_first_token_changes"]}
                            for dom,v in scored["opposite_pairs"][style].items()}
                          for style in ("direct","brief")}}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    arg=parser.parse_args()
    run(arg.output)
