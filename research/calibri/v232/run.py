"""CALIBRI v232: natural-answer read-only diagnostic of sealed 135M FP32 vs v228 CB59.

v231 found 326/960 valid choice letters for FP32 and 0/960 CB59; CB59
emitted 'Yes' in 901/960. Determine if that failure is only an A-D
instruction mismatch or answer collapse. NEVER optimize, modify model, or
select any weights on these public VALIDATION/diagnostic tasks.

Evaluation A: 24 previously unseen symmetric opposing VALID prompts,
with direct Yes/No, numbers, red/blue. Same strict frozen training-style
chat templates as original v186; no gold text in input beyond instructions.
Evaluation B: same physically frozen 480 v231 public validation questions,
now single natural-language answer (no A/B/C/D output request); this is a
post-hoc format sensitivity DIAGNOSTIC, not independent new validation.
Both models see byte-identical prompts; all 504 paired predictions preserved.
Scoring is exact normalized answer, plus format/nonanswer diagnostics.
Original CB59 file SHA and FP32 revision are authenticated; no new weights.
Jessica performs no tests; main/LIVE/Render/Supabase/APK remain untouched.
"""
import argparse
from collections import Counter,defaultdict
import gc
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from v174.run import save
from v176.operators import clear
from v186.pairs import build_pairs
from v196 import protocol as P
from v196.recover import build_trainable_from_paid_source
from v229.run import sha_checked,ACCEPTED_V228_STEP12_SHA256
from v231.run import (TASKS,SOURCES,PER_DOMAIN,build_cases,sha as v231_sha,
                      MAX_CONTEXT_TOKENS)

# All static choices fixed BEFORE loading either neural weight set.
MAX_NEW_TOKENS=32
TOTAL_PUBLIC_CASES=480
VALID_SIDES=24
PINNED_DATASETS={
 "google/boolq":"35b264d03638db9f4ce671b711558bf7ff0f80d5",
 "allenai/sciq":"2c94ad3e1aafab77146f384e23536f97a4849815",
 "allenai/ai2_arc":"210d026faf9955653af8916fad021475a3f00453",
 "lighteval/piqa":"41782e6bf0ef7de82a2ca8a9feb1dca042837fae",
 "Rowan/hellaswag":"218ec52e09a7e7462a5400043bb9a69a41d06b76",
}
# Taken from v231's PRECOMMITTED manifest, generated before first QA model use.
FROZEN_V231_CASE_SHA={
 "boolq":"c729a3a51969ab221895439f60b662dc20be206008be16216b66e58b0c0fbd23",
 "sciq":"4f51829cc5df9fff93a081c19d4e1694db6f78ccfdc0537641605dd8e70e456f",
 "arc_easy":"d66bdf7cf59daff3981824f2a9370c4be6bae24a736e6169ef5ca89311a07dff",
 "piqa":"f1c8a9b9cd6778f7a66c91afaeb9bbe88fd278327c302ac928b060515b9dc260",
 "hellaswag":"c8a52dfdd1e500c2f27de1fad760c34879e87821245fcd624f3c86aefadb12d2"
}
LETTERS="ABCD"

def normalize(s):
    s=unicodedata.normalize("NFKC",s).lower().strip()
    s=s.replace("\u2019","'").replace("\u2018","'")
    s=re.sub(r"^(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*","",s)
    s=s.strip(" \t\r\n\"'.,:;!?*")
    s=re.sub(r"\s+"," ",s)
    return s

def normalized_answer(s):
    first=s.strip().splitlines()[0] if s.strip() else ""
    return normalize(first)

def natural_prompt(case):
    question=case["question"]
    task=case["task"]
    if task=="boolq":
        return (f"Passage: {case['passage']}\nQuestion: {question}\n"
                "Answer with the word Yes or No, without any explanation.")
    if task in ("sciq","arc_easy"):
        return (f"Question: {question}\n"
                "Give the answer directly, in a few words. Do not reply "
                "with a multiple-choice letter.")
    if task in ("piqa","hellaswag"):
        # Options remain NECESSARY for comparative choice/continuation tasks,
        # but the correct output is the actual option text, NOT an A-D label.
        options="\n".join(f"Candidate {k+1}: {s}"
                          for k,s in enumerate(case["choices"]))
        return (f"Question or situation: {question}\n{options}\n"
                "Copy the wording of the single best candidate exactly. "
                "Do not reply with an option number or letter.")
    raise AssertionError("Unexpected task: "+task)

def freeze_input(tok):
    public=[]
    plan={"datasets":PINNED_DATASETS,"model_revision":MODEL_SHA,
          "sealed_paid_SHA256":ACCEPTED_V228_STEP12_SHA256,
          "part_A_new_frozen_opposing_VALID_sides":VALID_SIDES,
          "part_B_same_480_v231_cases_with_natural_language_prompt":True,
          "part_B_is_posthoc_format_diagnostic_not_independent_QA":True,
          "part_B_previous_results_forbidden_for_training":True,
          "max_new_tokens":MAX_NEW_TOKENS,
          "no_dev_or_valid_used_for_training_or_selection":True,
          "no_model_weight_changes":True,"Jessica_tests_executed":False}
    for task in TASKS:
        repo,config,split=SOURCES[task]
        assert split=="validation"
        ds=load_dataset(repo,config,split=split,
                        revision=PINNED_DATASETS[repo],trust_remote_code=False)
        cases,_=build_cases(task,ds)
        digest=v231_sha([x["case_id"] for x in cases])
        if digest!=FROZEN_V231_CASE_SHA[task]:
            raise AssertionError("v231 frozen case identities drift: "+task)
        for c in cases:
            text=natural_prompt(c)
            prompt=tok.apply_chat_template([{"role":"user","content":text}],
                  tokenize=False,add_generation_prompt=True)
            ids=tok.encode(prompt,add_special_tokens=False)
            if not 1<len(ids)<=MAX_CONTEXT_TOKENS:
                raise AssertionError("Outside precommitted max context "+c["case_id"])
            gold=("Yes" if c["choices"][LETTERS.index(c["gold"])]=="Yes"
                  else "No") if task=="boolq" else c["choices"][LETTERS.index(c["gold"])]
            public.append({"task":task,"case_id":c["case_id"],
                "question":c["question"],"gold":gold,
                "prompt":prompt,"input_ids":ids,
                "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()})
        print(json.dumps({"stage":"v232_frozen_NATURAL_480", "task":task,
            "cases":len(cases),"case_SHA":digest}),flush=True)
    valid=[]
    pairset=build_pairs(tok,"VALID")
    if len(pairset)!=12:raise AssertionError("Expected exactly 12 VALID pairs")
    for p in pairset:
        for side in ("left","right"):
            x=p[side]
            ids=[int(t) for t in x["prompt_ids"]]
            valid.append({"task":"VALID_"+p["domain"],
                "case_id":f"VALID_{p['pair_id']}_{side}",
                "question":x["question"],
                "gold":x["answer"],
                "prompt":tok.decode(ids,skip_special_tokens=False,
                      clean_up_tokenization_spaces=False),
                "input_ids":ids,
                "prompt_sha256":hashlib.sha256(
                  x["prompt_ids"].numpy().tobytes()).hexdigest()})
    if len(valid)!=VALID_SIDES or len(public)!=TOTAL_PUBLIC_CASES:
        raise AssertionError("Frozen evaluation cases incomplete")
    if len({x["case_id"] for x in public+valid})!=len(public+valid):
        raise AssertionError("Duplicate cross-task cases")
    return valid+public,plan

def generate(model,sources,tok,item):
    context=torch.tensor(item["input_ids"],dtype=torch.long)[None]
    generated=[]
    start=time.perf_counter()
    stopped=False
    with torch.inference_mode():
        try:
            for _ in range(MAX_NEW_TOKENS):
                if sources is not None:clear(sources)
                logits=model(input_ids=context,use_cache=False).logits[0,-1,:]
                if not torch.isfinite(logits).all():
                    raise AssertionError("Nonfinite logits on frozen QA question")
                t=int(logits.argmax())
                if t==tok.eos_token_id:
                    stopped=True
                    break
                generated.append(t)
                context=torch.cat((context,context.new_tensor([[t]])),dim=1)
                # First complete newline stops direct answer, avoids model
                # running into unsolicited next explanations or questions.
                raw=tok.decode(generated,skip_special_tokens=True,
                         clean_up_tokenization_spaces=False)
                if "\n" in raw:
                    stopped=True
                    break
        finally:
            if sources is not None:clear(sources)
    raw=tok.decode(generated,skip_special_tokens=True,
                   clean_up_tokenization_spaces=False)
    actual=normalized_answer(raw)
    expected=normalize(item["gold"])
    correct=(actual==expected)
    return {"task":item["task"],"case_id":item["case_id"],
        "question":item["question"],"prompt_sha256":item["prompt_sha256"],
        "expected":item["gold"],"actual":raw,"normalized_actual":actual,
        "strict_correct":correct,
        "nonempty":bool(actual),
        "response_is_yes_no":actual in ("yes","no"),
        "response_is_choice_letter":actual in ("a","b","c","d"),
        "generated_token_count":len(generated),
        "ended_by_eos_or_newline":stopped,
        "possible_output_truncation":not stopped,
        "elapsed_s":round(time.perf_counter()-start,5)}

def eval_one(name,model,sources,tok,items,out):
    path=out/(name+"_504_natural_generated_answers.jsonl")
    result=[]
    with path.open("w",encoding="utf8") as stream:
        for i,item in enumerate(items):
            row=generate(model,sources,tok,item)
            result.append(row)
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            stream.flush()
            if i==VALID_SIDES-1 or (i-VALID_SIDES+1)%48==0:
                group=result[-(VALID_SIDES if i==VALID_SIDES-1 else 48):]
                print(json.dumps({"stage":"v232_actual_direct_answers",
                    "model":name,"completed":i+1,"of":len(items),
                    "last_task":row["task"],
                    "latest_group_correct":sum(r["strict_correct"] for r in group),
                    "latest_group_yes":sum(r["normalized_actual"]=="yes" for r in group),
                    "latest_group_no":sum(r["normalized_actual"]=="no" for r in group),
                    "latest_group_empty":sum(not r["nonempty"] for r in group)}),flush=True)
    return result

def compare(fp32,cb59):
    if (len(fp32)!=VALID_SIDES+TOTAL_PUBLIC_CASES or
        len(cb59)!=VALID_SIDES+TOTAL_PUBLIC_CASES):
        raise AssertionError("Unpaired diagnostic")
    summaries={}
    keys=sorted({r["task"] for r in fp32})
    for domain in keys:
        a=[r for r in fp32 if r["task"]==domain]
        b=[r for r in cb59 if r["task"]==domain]
        if [(r["case_id"],r["prompt_sha256"],r["expected"]) for r in a]!=[
            (r["case_id"],r["prompt_sha256"],r["expected"]) for r in b]:
            raise AssertionError("Unmatched question or gold")
        sums={
          "n":len(a),
          "FP32_correct":sum(x["strict_correct"] for x in a),
          "CALIBRI_correct":sum(x["strict_correct"] for x in b),
          "both_correct":sum(x["strict_correct"] and y["strict_correct"]
                        for x,y in zip(a,b)),
          "FP32_only_correct":sum(x["strict_correct"] and not y["strict_correct"]
                        for x,y in zip(a,b)),
          "CALIBRI_only_correct":sum(y["strict_correct"] and not x["strict_correct"]
                        for x,y in zip(a,b)),
          "FP32_yes":sum(x["normalized_actual"]=="yes" for x in a),
          "CALIBRI_yes":sum(x["normalized_actual"]=="yes" for x in b),
          "FP32_no":sum(x["normalized_actual"]=="no" for x in a),
          "CALIBRI_no":sum(x["normalized_actual"]=="no" for x in b),
          "FP32_truncated":sum(x["possible_output_truncation"] for x in a),
          "CALIBRI_truncated":sum(x["possible_output_truncation"] for x in b)}
        sums["FP32_accuracy"]=sums["FP32_correct"]/len(a)
        sums["CALIBRI_accuracy"]=sums["CALIBRI_correct"]/len(a)
        summaries[domain]=sums
    return {"v231_mc_choice_experiment_already_showed": {
               "FP32":326,"CALIBRI":0,"prompts_each":960,
               "measured_on_same_public_questions_with_choice_letters":True},
            "strict_normalized_text_comparison_not_semantic_grader":True,
            "same_gold_and_chat_prompts_for_both_models":True,
            "natural_language_format_diagnostic_not_new_model_training":True,
            "full_fp32_CB59_case_count":len(fp32),
            "by_task":summaries,
            "total_FP32_correct":sum(x["strict_correct"] for x in fp32),
            "total_CALIBRI_correct":sum(x["strict_correct"] for x in cb59),
            "valid_subgroup": {
              "FP32":sum(x["strict_correct"] for x in fp32[:VALID_SIDES]),
              "CALIBRI":sum(x["strict_correct"] for x in cb59[:VALID_SIDES]),
              "n":VALID_SIDES},
            "FP32_output_histogram_top":Counter(
                r["normalized_actual"] for r in fp32).most_common(20),
            "CALIBRI_output_histogram_top":Counter(
                r["normalized_actual"] for r in cb59).most_common(20),
            "no_checkpoint_selection_on_QA":True,
            "no_model_weight_change":True,
            "not_full_proof_of_other_instructions_or_CPU_speed":True}

def run(source,out):
    torch.set_num_threads(2)
    torch.manual_seed(232)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases,plan=freeze_input(tok)
    save(out/"PRECOMMITTED_v232_natural_answer_diagnostic_plan.json",plan)
    save(out/"v232_frozen_504_case_manifest.json",[
       {k:v for k,v in e.items() if k not in ("input_ids",)}
       for e in cases])
    print(json.dumps({"stage":"v232_FREEZE_COMPLETE",
        "public_reused_cases":TOTAL_PUBLIC_CASES,
        "prior_independent_VALID_cases":VALID_SIDES,
        "answers_per_model":len(cases)}),flush=True)
    cb=sha_checked(source,ACCEPTED_V228_STEP12_SHA256)
    fp32=AutoModelForCausalLM.from_pretrained(
       MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).cpu().eval()
    if sum(p.numel() for p in fp32.parameters())!=EXPECTED_UNIQUE:
        raise AssertionError("Unexpected FP32 parameter count")
    output_fp32=eval_one("FP32",fp32,None,tok,cases,out)
    del fp32;gc.collect()
    packed,paid=build_trainable_from_paid_source(cb)
    packed.cpu().eval()
    if len(paid)!=P.CODEBOOK_MATRIX_COUNT:
        raise AssertionError("Missing original compressed paid layers")
    output_packed=eval_one("CALIBRI_v228",packed,paid,tok,cases,out)
    del packed,paid;gc.collect()
    result=compare(output_fp32,output_packed)
    result["FP32_revision"]=MODEL_SHA
    result["CALIBRI_archive_sha256"]=ACCEPTED_V228_STEP12_SHA256
    result["physical_bpp"]=0.6678394131307638
    save(out/"ACTUAL_v232_FP32_CALIBRI_natural_answers_480_plus24.json",result)
    print(json.dumps({"stage":"v232_FINAL_NATURAL_ANSWERS",
        "FP32":result["total_FP32_correct"],
        "CALIBRI":result["total_CALIBRI_correct"],
        "VALID24":result["valid_subgroup"],
        "per_task":{k:{"FP32":v["FP32_correct"],"CALIBRI":v["CALIBRI_correct"],
                       "n":v["n"]} for k,v in result["by_task"].items()},
        "CB59_outputs_top":result["CALIBRI_output_histogram_top"][:6],
        "unchanged_bpp":result["physical_bpp"]}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("sealed_paid_v228",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.sealed_paid_v228,a.output)
