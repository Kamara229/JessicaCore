#!/usr/bin/env python3
"""CALIBRI v72: actual zero-shot QA answers from native packed full pretrained model.

Freeze original v70 TRAIN-selected l0_hadamard transform. Compare original
SmolLM2-135M-Instruct FP32, original physical CB59 and same-rate Hadamard CB59
on pinned official SciQ TEST (4-way) and BoolQ VALID (yes/no).
Student inference uses the unchanged C direct 5-bit kernel, no original
pretrained dense 2D weights. Scores are constrained-choice token decisions,
NOT free-text conversational quality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from direct_codebook_v59 import CB59
from direct_runtime_v59 import NativeKernel,bind_into_llama,compile_kernel
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE

TOTAL_BYTES=11229303
TASKS=("sciq","boolq")
MODELS=("teacher","baseline","l0_hadamard")
SCI_Q_ID="allenai/sciq"
BOOLQ_ID="google/boolq"
SCI_Q_RANGE=(480,520)
BOOLQ_RANGE=(480,520)
BOOLQ_PASSAGE_TOKENS=128
LETTERS=("A","B","C","D")
# The fixed tasks, order, prompts, examples, options and scoring were set
# before the CI run; nothing here optimizes against QA TEST data.

def fixed_one_token_ids(tok,options):
    ids=[]
    for suffix in options:
        tid=tok.encode(" "+suffix,add_special_tokens=False)
        if len(tid)!=1:
            raise AssertionError(f"option token {suffix!r} not exactly one token: {tid}")
        ids.append(int(tid[0]))
    if len(set(ids))!=len(ids):
        raise AssertionError("option labels collided")
    return ids

def dataset_questions(tok):
    tasks={}
    for task in TASKS:
        key=SCI_Q_ID if task=="sciq" else BOOLQ_ID
        revision=HfApi().dataset_info(key).sha
        ds=load_dataset(key,split="test" if task=="sciq" else "validation",revision=revision)
        start,end=SCI_Q_RANGE if task=="sciq" else BOOLQ_RANGE
        if len(ds)<end:raise AssertionError("official source has fewer examples than pinned indices")
        prompts=[]
        gold=[]
        for idx in range(start,end):
            row=ds[idx]
            if task=="sciq":
                original=[row["correct_answer"],row["distractor1"],row["distractor2"],row["distractor3"]]
                if len(set(original))!=4:raise AssertionError("ambiguously duplicated SciQ answer options")
                choices=original[:]
                random.Random(720000+idx).shuffle(choices)
                correct=choices.index(row["correct_answer"])
                letters="\n".join(f"{LETTERS[i]}. {choice}" for i,choice in enumerate(choices))
                # No SciQ support paragraph: closed-book original science QA.
                prompt=("Answer the science multiple-choice question. "
                        "Reply with one letter: A, B, C, or D.\n"
                        f"Question: {row['question']}\n{letters}\nAnswer:")
                suffixes=LETTERS
            else:
                passage_ids=tok.encode(str(row["passage"]),add_special_tokens=False)
                shortened=tok.decode(passage_ids[:BOOLQ_PASSAGE_TOKENS],skip_special_tokens=True)
                prompt=("Read the context and answer yes or no.\n"
                        f"Context: {shortened}\nQuestion: {row['question']}\nAnswer:")
                suffixes=("Yes","No")
                correct=0 if bool(row["answer"]) else 1
            ids=tok.encode(prompt,add_special_tokens=False)
            if not 1<=len(ids)<=400:
                raise AssertionError(f"unexpected prompt length {len(ids)}, sample {idx}")
            candidate_ids=fixed_one_token_ids(tok,suffixes)
            prompts.append({"record_index":idx,"input_ids":ids,
                            "option_token_ids":candidate_ids,"gold_option":correct,
                            "input_sha256":hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest()})
            gold.append(correct)
        combined=hashlib.sha256(
            json.dumps([{k:v for k,v in p.items() if k!="input_ids"}
                        for p in prompts],sort_keys=True).encode()).hexdigest()
        tasks[task]={"dataset_name":key,"dataset_revision":revision,
                     "split":"test" if task=="sciq" else "validation",
                     "fixed_range":[start,end],"prompts":prompts,
                     "prompt_signature":combined,
                     "baseline_chance":1/len(prompts[0]["option_token_ids"]),
                     "rules":"single next-token score among labelled options; no free text",
                     "sciq_support_paragraph_excluded":task=="sciq",
                     "boolq_context_truncated_to_first_128_tokens":task=="boolq"}
    return tasks

def make_model(kind,folder):
    if kind=="teacher":
        model=AutoModelForCausalLM.from_pretrained(
            MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
            attn_implementation="eager").eval()
        return model,{"teacher_pretrained_weights_loaded":True}
    if kind not in MODELS:raise ValueError(kind)
    path=folder/"sources"/(kind+".cb59")
    archive=CB59(path.read_bytes())
    if archive.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE or archive.physical_bytes()!=TOTAL_BYTES:
        raise AssertionError("full actual trained physical archive size or unique count changed")
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    so=compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"),
                      folder/("lib_v72_"+kind+".so"))
    native=NativeKernel(so)
    evidence=bind_into_llama(model,archive,native)
    if evidence["runtime_dense_multi_dim_source_parameters"]!=0:
        raise AssertionError("compressed hot model still contains original trained 2D dense")
    return model,{"teacher_pretrained_weights_loaded":False,
                  "complete_archive_bytes":archive.physical_bytes(),
                  "complete_archive_SHA256":archive.hash(),
                  "physical_source_bpw":8*archive.physical_bytes()/EXPECTED_UNIQUE,
                  "native_direct_C_5bit_GEMV":True,
                  "original_trained_2D_hot_weights":0,
                  "tied_embedding_alias":evidence["tied_embed_and_output_codebook_SHARED_IN_RAM"]}

@torch.inference_mode()
def measure(model,task_data):
    outputs={}
    for task,data in task_data.items():
        cases=[]
        times=[]
        for sample in data["prompts"]:
            x=torch.tensor(sample["input_ids"],dtype=torch.int64)[None,:]
            tic=time.perf_counter()
            scores=model(input_ids=x,use_cache=False).logits[0,-1,:].float()
            times.append(time.perf_counter()-tic)
            select=scores[torch.tensor(sample["option_token_ids"],dtype=torch.int64)]
            if not torch.isfinite(select).all():raise AssertionError("non-finite real answer scores")
            answer=int(select.argmax())
            gold=int(sample["gold_option"])
            normalized=torch.softmax(select.to(torch.float64),dim=0)
            cases.append({"record_index":sample["record_index"],
                          "correct":int(answer==gold),
                          "model_selected_option_index":answer,
                          "gold_option_index":gold,
                          "gold_option_probability_conditioned_on_candidates":float(normalized[gold]),
                          "gold_option_logit_margin_to_best_wrong":float(
                              select[gold]-torch.max(select[torch.arange(len(select))!=gold])),
                          "prompt_sha256":sample["input_sha256"]})
        outputs[task]={"dataset_name":data["dataset_name"],
                       "dataset_revision":data["dataset_revision"],
                       "split":data["split"],"fixed_range":data["fixed_range"],
                       "total_questions":len(cases),
                       "prompt_signature":data["prompt_signature"],
                       "chance_rate":data["baseline_chance"],
                       "sciq_no_support":data["sciq_support_paragraph_excluded"],
                       "boolq_truncated_passage":data["boolq_context_truncated_to_first_128_tokens"],
                       "correct_answers":int(sum(x["correct"] for x in cases)),
                       "accuracy":float(np.mean([x["correct"] for x in cases])),
                       "mean_gold_choice_probability":float(np.mean([x["gold_option_probability_conditioned_on_candidates"] for x in cases])),
                       "mean_gold_logit_margin":float(np.mean([x["gold_option_logit_margin_to_best_wrong"] for x in cases])),
                       "native_CPU_time_seconds":float(sum(times)),
                       "question_results":cases}
        print(json.dumps({"task":task,
                          "accuracy":round(outputs[task]["accuracy"],4),
                          "correct":outputs[task]["correct_answers"],
                          "total":outputs[task]["total_questions"],
                          "seconds":round(sum(times),2)}),flush=True)
    return outputs

def run(args):
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    tests=dataset_questions(tok)
    model,source=make_model(args.kind,args.folder)
    with torch.inference_mode():
        # One independent warm-up, same for all models and not scored.
        ids=torch.tensor(tests["sciq"]["prompts"][0]["input_ids"][:4],dtype=torch.int64)[None,:]
        model(input_ids=ids,use_cache=False)
    results=measure(model,tests)
    audit={"stage":"CALIBRI_v72_real_labeled_answer_quality",
           "kind":args.kind,"model":MODEL,"revision":MODEL_SHA,
           "source":source,"fixed_128_total_zero_shot_questions":80,
           "no_training_or_model_selection_on_answer_labels":True,
           "original_v70_TRAIN_winner_not_reselected":True,
           "forced_choice_next_token_decision_NOT_freeform_generation":True,
           "teacher_and_packed_in_distinct_processes":True,
           "tasks":results}
    name=args.folder/("v72_"+args.kind+".json")
    name.write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"phase":args.kind,
                      "sciq_correct":results["sciq"]["correct_answers"],
                      "boolq_correct":results["boolq"]["correct_answers"],
                      "archive_bytes":source.get("complete_archive_bytes")}),flush=True)

def exact_mcnemar_p(a,b):
    # Two-sided exact binomial for discordant paired mistakes, no SciPy dependency.
    a_only=int(np.sum((a==1)&(b==0)))
    b_only=int(np.sum((a==0)&(b==1)))
    n=a_only+b_only
    if n==0:return 1.0,a_only,b_only
    probability=sum(math.comb(n,k) for k in range(min(a_only,b_only)+1))*2/(2**n)
    return min(1.0,probability),a_only,b_only

def compare(args):
    raw={name:json.loads((args.folder/("v72_"+name+".json")).read_text())
         for name in MODELS}
    build=json.loads((args.folder/"v70_build.json").read_text())
    source=build["variants"]
    for k in ("baseline","l0_hadamard"):
        a=raw[k]["source"]
        if a["complete_archive_SHA256"]!=source[k]["source_sha256"] or a["complete_archive_bytes"]!=TOTAL_BYTES:
            raise AssertionError("real physical K32V8 source changed, cannot compare")
        if a["original_trained_2D_hot_weights"]!=0 or not a["native_direct_C_5bit_GEMV"]:
            raise AssertionError("packed model source not direct trained 5-bit")
    report={"experiment":"CALIBRI_v72_frozen_l0_hadamard_independent_real_SciQ_BoolQ",
            "teacher":MODEL,"revision":MODEL_SHA,
            "original_unique_parameters":EXPECTED_UNIQUE,
            "baseline_and_candidate_complete_source_bytes":TOTAL_BYTES,
            "both_real_full_model_bpw":8*TOTAL_BYTES/EXPECTED_UNIQUE,
            "v70_frozen_training_only_source_selection":"l0_hadamard",
            "v72_label_test_never_used_for_selection":True,
            "all_sciq_and_boolq_cases_are_frozen_heldout":True,
            "this_is_single_token_forced_choice_QA_not_free_text":True,
            "sample_size_warning":"40 cases per task; limited uncertainty power and official tasks simplified",
            "results":{}}
    rng=np.random.default_rng(72)
    for task in TASKS:
        d={k:raw[k]["tasks"][task] for k in MODELS}
        hashes={k:(d[k]["prompt_signature"],d[k]["dataset_revision"],
                   d[k]["total_questions"]) for k in MODELS}
        if len(set(hashes.values()))!=1:raise AssertionError("different dataset/prompt/test sample")
        for a,b in zip(d["baseline"]["question_results"],d["l0_hadamard"]["question_results"]):
            if a["prompt_sha256"]!=b["prompt_sha256"] or a["gold_option_index"]!=b["gold_option_index"]:
                raise AssertionError("not the same original questions")
        a=np.array([q["correct"] for q in d["baseline"]["question_results"]],dtype=np.int8)
        b=np.array([q["correct"] for q in d["l0_hadamard"]["question_results"]],dtype=np.int8)
        p,a_only,b_only=exact_mcnemar_p(a,b)
        indexes=rng.integers(0,len(a),size=(3000,len(a)))
        bootstrap=(b-a)[indexes].mean(1)
        lo,hi=np.quantile(bootstrap,[0.025,0.975])
        teacher=np.array([q["correct"] for q in d["teacher"]["question_results"]],dtype=np.int8)
        report["results"][task]={
            "data_name":d["teacher"]["dataset_name"],
            "dataset_revision":d["teacher"]["dataset_revision"],
            "split":d["teacher"]["split"],"fixed_range":d["teacher"]["fixed_range"],
            "question_count":len(a),
            "chance_accuracy":d["teacher"]["chance_rate"],
            "original_teacher_correct":int(teacher.sum()),
            "original_teacher_accuracy":float(teacher.mean()),
            "original_CB59_correct":int(a.sum()),
            "original_CB59_accuracy":float(a.mean()),
            "selected_gauge_correct":int(b.sum()),
            "selected_gauge_accuracy":float(b.mean()),
            "candidate_minus_baseline_accuracy":float(b.mean()-a.mean()),
            "paired_3000_bootstrap_accuracy_diff_95pct":[float(lo),float(hi)],
            "exact_two_sided_McNemar_p":float(p),
            "base_only_correct_candidate_wrong":a_only,
            "base_wrong_candidate_only_correct":b_only,
            "selected_gauge_teacher_answer_agreement":float(np.mean([
                x["model_selected_option_index"]==y["model_selected_option_index"]
                for x,y in zip(d["teacher"]["question_results"],d["l0_hadamard"]["question_results"])])),
            "baseline_teacher_answer_agreement":float(np.mean([
                x["model_selected_option_index"]==y["model_selected_option_index"]
                for x,y in zip(d["teacher"]["question_results"],d["baseline"]["question_results"])])),
            "mean_gold_choice_probability":{
                k:d[k]["mean_gold_choice_probability"] for k in MODELS},
            "CPU_total_warm_prefill_seconds":{
                k:d[k]["native_CPU_time_seconds"] for k in MODELS},
            "benchmark_limitations":{
                "SciQ_no_support":d["teacher"]["sciq_no_support"],
                "BoolQ_passage_first_128_tokens_only":d["teacher"]["boolq_truncated_passage"]}}
    (args.folder/"v72_actual_qa_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"stage":"v72_verifiable_original_full_model_labeled_QA",
                      "source_bpw":round(report["both_real_full_model_bpw"],9),
                      "tasks":{task:{
                        "teacher_correct":v["original_teacher_correct"],
                        "baseline_correct":v["original_CB59_correct"],
                        "gauged_correct":v["selected_gauge_correct"],
                        "acc_delta":round(v["candidate_minus_baseline_accuracy"],4),
                        "mcnemar_p":round(v["exact_two_sided_McNemar_p"],4),
                        "interval":v["paired_3000_bootstrap_accuracy_diff_95pct"]}
                        for task,v in report["results"].items()}}),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=("run","compare"),required=True)
    ap.add_argument("--kind",choices=MODELS,default="teacher")
    ap.add_argument("--folder",type=Path,default=Path("results"))
    a=ap.parse_args()
    a.folder.mkdir(parents=True,exist_ok=True)
    run(a) if a.phase=="run" else compare(a)

if __name__=="__main__":main()
