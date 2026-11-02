#!/usr/bin/env python3
"""CALIBRI v77: frozen v76 paid rank4-vs-bias physical-source ablations.

Real trained 135M FP32 teacher vs native original CB59 and THREE truly
same-size paid CB76 decompositions of original v76 TRAIN-selected factors:
full rank4+bias, bias-only, rank4-only; also paid all-zero negative control.
NO Q/A TEST selection/training, no production Jessica or runtime changes.

Scores disjoint precommitted WikiText+AGNews TEST full-model KL/NLL, SciQ
TEST and BoolQ validation constrained answers, and KV-cache greedy CPU
decode speed. Physical UTF16 factor tensors + CRC/SHA included in all
source bytes, direct original trained 2D weights remain packed K32V8 C.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from quality import DATASET,load_token_ids
from paid_lowrank_v76 import (
    SOURCE_BYTES,REQUIRED_PHYSICAL_BYTES,MAX_PHYSICAL_BYTES,
    DIM,VOCAB,RANK,pack,unpack,zero_factors,make_model,
)

KINDS=("teacher","baseline","zero_paid","bias_only","rank4_only","full")
PAID=("zero_paid","bias_only","rank4_only","full")
CORPORA=("wiki_test","news_test")
SCI_Q_ID="allenai/sciq"
BOOLQ_ID="google/boolq"
TASKS=("sciq","boolq")
SCI_Q_RANGE=(680,728)
BOOLQ_RANGE=(960,1024)
BOOLQ_PASSAGE_TOKENS=128
CONTEXT=64
WINDOWS=8
WIKI_START=80000
NEWS_RECORDS=(4000,4100)
AUTOREGRESSIVE_PROMPTS=(
    "The following scientific observation suggests that",
    "Once upon a time in a small town",
    "The first step in solving the problem is",
)
GENERATED_TOKENS=16
OPTION_LETTERS=("A","B","C","D")

def get_source_ids(corpus,tok):
    if corpus not in CORPORA:raise ValueError(corpus)
    n=WINDOWS*CONTEXT
    if corpus=="wiki_test":
        revision=HfApi().dataset_info(DATASET).sha
        ids=load_token_ids(tok,"test",revision,WIKI_START+n)[WIKI_START:WIKI_START+n]
        meta={"name":DATASET,"revision":revision,"split":"test",
              "token_range":[WIKI_START,WIKI_START+n]}
    else:
        revision=HfApi().dataset_info("fancyzhx/ag_news").sha
        start,end=NEWS_RECORDS
        rows=load_dataset("fancyzhx/ag_news",split=f"test[{start}:{end}]",revision=revision)
        raw="\n\n".join(str(r["text"]) for r in rows)
        ids=tok(raw,add_special_tokens=False,return_tensors="pt").input_ids[0,:n]
        meta={"name":"fancyzhx/ag_news","revision":revision,"split":f"test[{start}:{end}]",
              "token_range":[0,n],"text_only_no_labels":True}
    if len(ids)!=n:raise AssertionError("not enough untouched TEST source text")
    meta["input_sha256"]=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    return ids,meta

def label_tokens(tok,suffixes):
    out=[]
    for s in suffixes:
        t=tok.encode(" "+s,add_special_tokens=False)
        if len(t)!=1:raise AssertionError(f"not a single token label {s!r}: {t}")
        out.append(int(t[0]))
    if len(set(out))!=len(out):raise AssertionError("candidate option tokens collide")
    return out

def qa_dataset(tok):
    data={}
    for task in TASKS:
        name=SCI_Q_ID if task=="sciq" else BOOLQ_ID
        rev=HfApi().dataset_info(name).sha
        split="test" if task=="sciq" else "validation"
        rows=load_dataset(name,split=split,revision=rev)
        lo,hi=SCI_Q_RANGE if task=="sciq" else BOOLQ_RANGE
        if len(rows)<hi:raise AssertionError("original task has too few cases")
        cases=[]
        rejected_duplicates=[]
        for idx in range(lo,hi):
            r=rows[idx]
            if task=="sciq":
                answers=[r["correct_answer"],r["distractor1"],
                         r["distractor2"],r["distractor3"]]
                if len(set(answers))<4:
                    # Some official SciQ rows have repeated distractors.
                    # A correct answer cannot be uniquely evaluated among
                    # duplicate choices; exclude by an explicit source-data
                    # validity rule, identically for EVERY model. Keep the
                    # source row index in signed outcome metadata.
                    rejected_duplicates.append(idx)
                    continue
                random.Random(770000+idx).shuffle(answers)
                target=answers.index(r["correct_answer"])
                formatted="\n".join(f"{OPTION_LETTERS[i]}. {answer}"
                                    for i,answer in enumerate(answers))
                prompt=("Answer the science multiple-choice question. "
                        "Reply with one letter: A, B, C, or D.\n"
                        f"Question: {r['question']}\n{formatted}\nAnswer:")
                opts=label_tokens(tok,OPTION_LETTERS)
            else:
                text_ids=tok.encode(str(r["passage"]),add_special_tokens=False)
                short=tok.decode(text_ids[:BOOLQ_PASSAGE_TOKENS],skip_special_tokens=True)
                prompt=("Read the context and answer yes or no.\n"
                        f"Context: {short}\nQuestion: {r['question']}\nAnswer:")
                opts=label_tokens(tok,("Yes","No"))
                target=0 if bool(r["answer"]) else 1
            prompt_ids=tok.encode(prompt,add_special_tokens=False)
            if not (1<len(prompt_ids)<400):raise AssertionError("bad prompt length")
            cases.append({"index":idx,"input_ids":prompt_ids,
                          "options":opts,"correct":target,
                          "input_sha256":hashlib.sha256(
                              np.asarray(prompt_ids,dtype="<i8").tobytes()).hexdigest()})
        signature=hashlib.sha256(json.dumps(
            [{k:v for k,v in r.items() if k!="input_ids"} for r in cases],
            sort_keys=True).encode()).hexdigest()
        data[task]={"dataset":name,"revision":rev,"split":split,
                    "range":[lo,hi],"cases":cases,"signature":signature,
                    "chance_accuracy":1/(4 if task=="sciq" else 2),
                    "original_rows_precommitted":hi-lo,
                    "excluded_original_duplicate_choice_rows":rejected_duplicates,
                    "nonduplicate_source_row_exclusion_rule_uniform_across_models":True,
                    "limitations":{
                      "sciq_support_paragraph_excluded":task=="sciq",
                      "boolq_passage_first_128_tokens_only":task=="boolq"}}
    return data

def build(args):
    original=args.original.read_bytes()
    if len(original)!=SOURCE_BYTES:raise AssertionError("original fully trained 135M CB59 not supplied")
    selection=json.loads((args.folder/"v76_selection.json").read_text())
    prior=(args.folder/"CALIBRI_v76_paid_full.cb76").read_bytes()
    cb,U,V,b=unpack(prior)
    if selection["chosen_source_mode"]!="rank4" or selection["chosen_source_step"]!=20:
        raise AssertionError("cannot alter original frozen v76 TRAIN-selected parent source")
    if hashlib.sha256(prior).hexdigest()!=selection["full_paid_CB76_sha256"]:
        raise AssertionError("v76 original frozen TRAIN source checksum mismatch")
    if original!=prior[:SOURCE_BYTES]:
        raise AssertionError("the paid source does not contain identical original trained weights")
    base_U,base_V,base_b=zero_factors()
    variants={
        "zero_paid":(base_U,base_V,base_b),
        "bias_only":(base_U,base_V,b),
        "rank4_only":(U,V,base_b),
        "full":(U,V,b),
    }
    folder=args.folder/"v77_sources";folder.mkdir(exist_ok=True,parents=True)
    audit={"experiment":"CALIBRI v77 paid exactly same-bit source decomposition",
           "original_model":MODEL,"pinned_revision":MODEL_SHA,
           "original_trained_unique_params":EXPECTED_UNIQUE,
           "original_CB59_actual_bytes":len(original),
           "same_complete_paid_CB76_bytes":REQUIRED_PHYSICAL_BYTES,
           "all_in_paid_bpw":8*REQUIRED_PHYSICAL_BYTES/EXPECTED_UNIQUE,
           "budget_max_bytes":MAX_PHYSICAL_BYTES,
           "prior_frozen_v76_training_choice":selection["chosen_source_mode"],
           "prior_frozen_v76_training_step":selection["chosen_source_step"],
           "original_v76_source_SHA256":hashlib.sha256(prior).hexdigest(),
           "sidecar_no_tuning_or_retraining":True,
           "all_paid_source_versions":{}}
    for name,factors in variants.items():
        packed=pack(original,*factors)
        path=folder/(name+".cb76");path.write_bytes(packed)
        _,up,vp,bp=unpack(path.read_bytes())
        if len(packed)!=REQUIRED_PHYSICAL_BYTES or len(packed)>MAX_PHYSICAL_BYTES:
            raise AssertionError("paid physical source rate changed")
        if name=="full" and packed!=prior:raise AssertionError("full version not identical to v76")
        actual_parts=[int(np.count_nonzero(arr)) for arr in (up,vp,bp)]
        audit["all_paid_source_versions"][name]={
            "file_sha256":hashlib.sha256(packed).hexdigest(),
            "complete_actual_bytes":len(packed),
            "all_in_bpw":8*len(packed)/EXPECTED_UNIQUE,
            "nonzero_U_V_bias":actual_parts,
            "entire_CB59_trained_source_bitwise_original":True,
            "still_native_original_K32V8_GEMV":True,
            "paid_sidecar_CPU_torch_not_native_C":True}
    (args.folder/"v77_physical_build.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"stage":"v77_actual_physical_paid_4way_ablations",
                      "real_full_bytes_each":REQUIRED_PHYSICAL_BYTES,
                      "physical_bpw":round(audit["all_in_paid_bpw"],9),
                      "additional_over_CB59":REQUIRED_PHYSICAL_BYTES-SOURCE_BYTES,
                      "original_v76_source_frozen_sha_match":True}),flush=True)

def trained_model(args):
    # make_model() from v76 is the REAL original direct-packed CPU
    # Llama runtime. For paid models it loads the exact whole CB76 source,
    # not an FP32 checkpoint and not an uncounted sidecar.
    if args.kind in ("teacher","baseline"):
        model,info,_=make_model(args.kind,SimpleNamespace(
            original=args.original,folder=args.folder))
        return model,info
    full=(args.folder/"v77_sources"/(args.kind+".cb76")).read_bytes()
    if len(full)!=REQUIRED_PHYSICAL_BYTES:raise AssertionError("not fully packed paid model")
    # v76 model() looks up a single frozen CB76 name; preserve it without
    # editing the original v76 TRAIN-selection checkpoint.
    path=args.folder/"CALIBRI_v76_paid_full.cb76"
    prior=path.read_bytes()
    try:
        path.write_bytes(full)
        model,info,_=make_model("corrected",SimpleNamespace(
            original=args.original,folder=args.folder))
    finally:
        path.write_bytes(prior)
    expected=hashlib.sha256(full).hexdigest()
    if info["complete_physical_SHA256"]!=expected:
        raise AssertionError("trained packed source changed after model loading")
    return model,info

@torch.inference_mode()
def eval_language(model,kind,folder,tok):
    stats={}
    for corpus in CORPORA:
        ids,origin=get_source_ids(corpus,tok)
        nwin=len(ids)//CONTEXT
        teacher_ref=folder/("v77_teacher_"+corpus+".npy")
        if kind=="teacher":
            array=np.lib.format.open_memmap(teacher_ref,mode="w+",
                shape=(nwin,CONTEXT,VOCAB),dtype="<f2")
        else:
            array=np.load(teacher_ref,mmap_mode="r")
            if tuple(array.shape)!=(nwin,CONTEXT,VOCAB):
                raise AssertionError("real teacher reference logits absent")
        samples=[];times=[]
        for i in range(nwin):
            x=ids[i*CONTEXT:(i+1)*CONTEXT][None,:]
            start=time.perf_counter()
            logits=model(input_ids=x,use_cache=False).logits.detach().float().cpu().numpy()[0]
            times.append(time.perf_counter()-start)
            if logits.shape!=(CONTEXT,VOCAB) or not np.isfinite(logits).all():
                raise AssertionError("invalid original pretrained full model forward")
            if kind=="teacher":
                array[i]=logits.astype("<f2")
                continue
            p=torch.from_numpy(np.asarray(array[i,:-1],dtype=np.float64))
            q=torch.from_numpy(logits[:-1].astype(np.float64))
            lp=p.log_softmax(-1);lq=q.log_softmax(-1)
            kl=float(torch.sum(lp.exp()*(lp-lq),dim=-1).mean())
            y=x[0,1:]
            nll=float(-lq[torch.arange(CONTEXT-1),y].mean())
            top=float((p.argmax(-1)==q.argmax(-1)).double().mean())
            if not all(math.isfinite(v) for v in (kl,nll,top)):
                raise AssertionError("non-finite true full-model language quality")
            samples.append({"window":i,"KL_teacher_to_student":kl,
                            "NLL_actual_next_token":nll,"top1_teacher_agreement":top})
        if kind=="teacher":array.flush()
        stats[corpus]={"source":origin,"number_windows":nwin,
                       "true_scored_next_tokens":nwin*(CONTEXT-1),
                       "teacher_logits_temp_storage_FP16":True,
                       "windows":samples,"total_CPU_prefill_seconds":sum(times)}
        print(json.dumps({"stage":"v77_language","model":kind,"corpus":corpus,
                          "KL":round(np.mean([z["KL_teacher_to_student"] for z in samples]),5)
                            if samples else None,
                          "NLL":round(np.mean([z["NLL_actual_next_token"] for z in samples]),5)
                            if samples else None,
                          "prefill_sec":round(sum(times),3)}),flush=True)
    return stats

@torch.inference_mode()
def eval_answers(model,kind,tok):
    data=qa_dataset(tok)
    results={}
    for task,part in data.items():
        cases=[];times=[]
        for example in part["cases"]:
            x=torch.tensor(example["input_ids"],dtype=torch.long)[None,:]
            start=time.perf_counter()
            logits=model(input_ids=x,use_cache=False).logits[0,-1].float()
            times.append(time.perf_counter()-start)
            opts=torch.tensor(example["options"],dtype=torch.long)
            sub=logits[opts]
            if not torch.isfinite(sub).all():raise AssertionError("non-finite QA answer")
            pred=int(sub.argmax())
            conditional=torch.softmax(sub.double(),dim=-1)
            gold=example["correct"]
            cases.append({"record_index":example["index"],
                          "correct":int(pred==gold),
                          "prediction_option":pred,
                          "gold_option":gold,
                          "gold_conditional_probability":float(conditional[gold]),
                          "prompt_sha256":example["input_sha256"]})
        results[task]={"source":{k:v for k,v in part.items() if k!="cases"},
                       "count":len(cases),"correct":sum(q["correct"] for q in cases),
                       "accuracy":float(np.mean([q["correct"] for q in cases])),
                       "CPU_prefill_seconds":sum(times),
                       "items":cases}
        print(json.dumps({"stage":"v77_heldout_QA","model":kind,"task":task,
                          "correct":results[task]["correct"],"count":len(cases),
                          "accuracy":round(results[task]["accuracy"],4)}),flush=True)
    return results

@torch.inference_mode()
def eval_generation(model,kind,tok):
    # Genuine greedy autoregressive token generation. The SAME three
    # predeclared fixed prompts and 16 output positions for each source;
    # model may emit eos before 16 tokens, but this fixed-load benchmark
    # deliberately continues after eos; NOT a conversational quality test.
    outputs=[]
    for prompt in AUTOREGRESSIVE_PROMPTS:
        ids=tok.encode(prompt,add_special_tokens=False,return_tensors="pt")
        t0=time.perf_counter()
        out=model(input_ids=ids,use_cache=True)
        prefill=time.perf_counter()-t0
        past=out.past_key_values
        if past is None:raise AssertionError("native packed model cache not returned")
        tokid=out.logits[:,-1,:].argmax(-1,keepdim=True)
        emitted=[int(tokid.item())]
        tic=time.perf_counter()
        for _ in range(GENERATED_TOKENS-1):
            out=model(input_ids=tokid,past_key_values=past,use_cache=True)
            past=out.past_key_values
            tokid=out.logits[:,-1,:].argmax(-1,keepdim=True)
            emitted.append(int(tokid.item()))
        decode=time.perf_counter()-tic
        if len(emitted)!=GENERATED_TOKENS:
            raise AssertionError("did not complete actual autoregressive generation")
        outputs.append({"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                        "prompt_tokens":ids.numel(),
                        "first_step_prefill_plus_first_token_seconds":prefill,
                        "subsequent_decode_seconds":decode,
                        "subsequent_decode_tokens_per_second":(GENERATED_TOKENS-1)/max(decode,1e-9),
                        "generated_16_token_sha256":hashlib.sha256(
                            np.asarray(emitted,dtype="<i8").tobytes()).hexdigest(),
                        "generated_distinct_token_ratio":len(set(emitted))/len(emitted),
                        "eos_appeared":tok.eos_token_id in emitted,
                        "fixed_16_steps_even_after_EOS":True})
    mean_tokens=(GENERATED_TOKENS-1)*len(outputs)
    elapsed=sum(r["subsequent_decode_seconds"] for r in outputs)
    print(json.dumps({"stage":"v77_autoregressive_cpu_speed",
                      "kind":kind,"decode_tokens":mean_tokens,
                      "decode_token_per_second":round(mean_tokens/max(elapsed,1e-9),4)}),flush=True)
    return {"fixed_prompt_count":len(outputs),
            "per_prompt":outputs,
            "measured_decode_tokens_after_prefill":mean_tokens,
            "true_native_packed_KV_cache_token_per_second":mean_tokens/max(elapsed,1e-9),
            "distinguish_teacher_fp32_vs_native_5bit_plus_CPU_residual":True,
            "single_cpu_runner_only":True}

def execute(args):
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    model,source=trained_model(args)
    with torch.inference_mode():
        model(input_ids=torch.tensor([[1,2,3,4]],dtype=torch.long),use_cache=False)
    language=eval_language(model,args.kind,args.folder,tok)
    qa=eval_answers(model,args.kind,tok)
    generation=eval_generation(model,args.kind,tok)
    r={"experiment":"CALIBRI v77 actual trained 135M independent paid factor QA+generation",
       "kind":args.kind,"model":MODEL,"pinned_original_revision":MODEL_SHA,
       "source":source,
       "original_trained_dense_2D_absent_from_paid_hot_model":args.kind=="teacher" or
            source.get("zero_dense_pretrained_2D_hot_weights",False),
       "QA_not_used_for_v76_TRAIN_selection":True,
       "all_ablations_physical_bytes_accounted":True,
       "language":language,"QA":qa,"greedy_generation_speed":generation}
    (args.folder/("v77_"+args.kind+".json")).write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps({"stage":"v77_completed_real_model","kind":args.kind,
                      "source_bytes":source.get("physical_complete_model_bytes"),
                      "BoolQ":qa["boolq"]["correct"],"SciQ":qa["sciq"]["correct"],
                      "greedy_decode_tps":round(generation["true_native_packed_KV_cache_token_per_second"],3)}),
          flush=True)

def exact_mcnemar(a,b):
    a_only=int(np.sum((a==1)&(b==0)))
    b_only=int(np.sum((b==1)&(a==0)))
    n=a_only+b_only
    if n==0:return 1.0,a_only,b_only
    p=2*sum(math.comb(n,k) for k in range(min(a_only,b_only)+1))/(2**n)
    return min(p,1.0),a_only,b_only

def summarize(args):
    records={k:json.loads((args.folder/("v77_"+k+".json")).read_text())
             for k in KINDS}
    build_info=json.loads((args.folder/"v77_physical_build.json").read_text())
    original_hash=hashlib.sha256(args.original.read_bytes()).hexdigest()
    for kind in PAID:
        actual=records[kind]["source"]
        known=build_info["all_paid_source_versions"][kind]
        if actual["physical_complete_model_bytes"]!=REQUIRED_PHYSICAL_BYTES:
            raise AssertionError("paid source didn't include sidecar or footer")
        if actual["complete_physical_SHA256"]!=known["file_sha256"]:
            raise AssertionError("paid source archive changed during evaluation")
        if actual["all_in_bits_per_original_parameter"]>.700:
            raise AssertionError("paid source exceeds fixed global bpw")
        if not records[kind]["original_trained_dense_2D_absent_from_paid_hot_model"]:
            raise AssertionError("hot model contains original trained full source")
    if records["baseline"]["source"]["complete_physical_SHA256"]!=original_hash:
        raise AssertionError("original source doesn't match original CB59")
    output={"experiment":"CALIBRI v77 genuine independent ablation",
            "model":MODEL,"model_pinned_revision":MODEL_SHA,
            "original_unique_trained_parameters":EXPECTED_UNIQUE,
            "actual_original_CB59_bytes":SOURCE_BYTES,
            "each_paid_variant_same_entire_source_bytes":REQUIRED_PHYSICAL_BYTES,
            "same_rate_paid_variants_bpw":8*REQUIRED_PHYSICAL_BYTES/EXPECTED_UNIQUE,
            "original_baseline_bpw":8*SOURCE_BYTES/EXPECTED_UNIQUE,
            "parent_v76_TRAIN_choice_frozen":True,
            "v77_QA_and_TEXT_data_never_used_to_fit_or_select":True,
            "source_and_artifact_hashes_valid":True,
            "all_full_model_native_C_K32V8_and_small_paid_CPU_torch":True,
            "note":"Only 1 CPU host and few windows/questions. NOT a robust intelligence/speed claim.",
            "language":{},"QA":{},"generation_speed":{}}
    rng=np.random.default_rng(77)
    for domain in CORPORA:
        source=records["teacher"]["language"][domain]["source"]["input_sha256"]
        for kind in KINDS:
            if records[kind]["language"][domain]["source"]["input_sha256"]!=source:
                raise AssertionError("unmatched frozen genuine TEST corpus inputs")
        result={}
        for kind in KINDS[1:]:
            a=records[kind]["language"][domain]["windows"]
            nll=np.array([r["NLL_actual_next_token"] for r in a],dtype=np.float64)
            kl=np.array([r["KL_teacher_to_student"] for r in a],dtype=np.float64)
            result[kind]={"KL":float(kl.mean()),"NLL":float(nll.mean()),
                          "prefill_seconds":records[kind]["language"][domain]["total_CPU_prefill_seconds"]}
        for kind in PAID:
            a=records["baseline"]["language"][domain]["windows"]
            b=records[kind]["language"][domain]["windows"]
            metrics={}
            for measure in ("KL_teacher_to_student","NLL_actual_next_token"):
                old=np.array([r[measure] for r in a],dtype=np.float64)
                current=np.array([r[measure] for r in b],dtype=np.float64)
                delta=current-old
                pick=rng.integers(0,len(delta),size=(2000,len(delta)))
                q=np.quantile(delta[pick].mean(1),[.025,.975])
                metrics[measure]={"delta_vs_original":float(delta.mean()),
                                  "bootstrap_window_95":[float(q[0]),float(q[1])]}
            result[kind]["signed_vs_original"]=metrics
        output["language"][domain]=result
    for task in TASKS:
        ref=records["teacher"]["QA"][task]
        signature=ref["source"]["signature"]
        allitems=ref["items"]
        for kind in KINDS:
            if records[kind]["QA"][task]["source"]["signature"]!=signature:
                raise AssertionError("QA prompts/labels differ between original models")
            if any(x["prompt_sha256"]!=y["prompt_sha256"] or
                   x["gold_option"]!=y["gold_option"]
                   for x,y in zip(allitems,records[kind]["QA"][task]["items"])):
                raise AssertionError("QA answers refer to different original question")
        m={}
        base=np.array([v["correct"] for v in records["baseline"]["QA"][task]["items"]],dtype=int)
        for kind in KINDS:
            items=records[kind]["QA"][task]["items"]
            a=np.array([v["correct"] for v in items],dtype=int)
            measure={"correct":int(sum(a)),"total":len(a),"accuracy":float(np.mean(a)),
                     "CPU_prefill_seconds":records[kind]["QA"][task]["CPU_prefill_seconds"]}
            if kind in PAID:
                p,base_only,kind_only=exact_mcnemar(base,a)
                idx=rng.integers(0,len(base),size=(3000,len(base)))
                q=np.quantile((a-base)[idx].mean(-1),[.025,.975])
                measure.update({"paid_minus_original_accuracy":float((a-base).mean()),
                                "paired_accuracy_bootstrap95":[float(q[0]),float(q[1])],
                                "exact_McNemar_p_two_sided":float(p),
                                "baseline_only_correct":base_only,"candidate_only_correct":kind_only})
            m[kind]=measure
        output["QA"][task]={"dataset":ref["source"]["dataset"],
                             "split":ref["source"]["split"],"range":ref["source"]["range"],
                             "teacher_chance_accuracy":ref["source"]["chance_accuracy"],
                             "limited_forced_choice_protocol":True,
                             "results":m}
    for kind in KINDS:
        r=records[kind]["greedy_generation_speed"]
        output["generation_speed"][kind]={
             "model_cpu_decode_tokens_per_sec":r["true_native_packed_KV_cache_token_per_second"],
             "decoded_tokens_after_prefill":r["measured_decode_tokens_after_prefill"],
             "prompt_count":r["fixed_prompt_count"],
             "mean_generated_distinct_token_ratio":float(np.mean([
                 p["generated_distinct_token_ratio"] for p in r["per_prompt"]]))}
    output["same_budget_rank4_and_bias_vs_bias_only_KL_gain_test"]=all(
         output["language"][c]["full"]["KL"]<output["language"][c]["bias_only"]["KL"]
         for c in CORPORA)
    output["full_corrected_vs_original_KL_and_NLL_both_domains"]=all(
         output["language"][c]["full"]["KL"]<output["language"][c]["baseline"]["KL"] and
         output["language"][c]["full"]["NLL"]<output["language"][c]["baseline"]["NLL"]
         for c in CORPORA)
    output["QA_gain_not_assumed_from_KL"]=True
    (args.folder/"v77_independent_QA_ablation_speed.json").write_text(
        json.dumps(output,indent=2)+"\n")
    print(json.dumps({"stage":"v77_real_pinned_135m_QA_and_KL_paid_ablation",
                      "paid_physical_bpw":round(output["same_rate_paid_variants_bpw"],9),
                      "rank4_adds_test_KL_beyond_bias_only":
                             output["same_budget_rank4_and_bias_vs_bias_only_KL_gain_test"],
                      "full_both_domain_KL_NLL_gain":
                             output["full_corrected_vs_original_KL_and_NLL_both_domains"],
                      "QA":{t:{k:v["correct"] for k,v in output["QA"][t]["results"].items()}
                            for t in TASKS},
                      "decode_tps":{k:round(v["model_cpu_decode_tokens_per_sec"],4)
                                    for k,v in output["generation_speed"].items()}}),flush=True)

def smoke():
    U,V,b=zero_factors()
    assert U.shape==(VOCAB,RANK) and V.shape==(DIM,RANK) and b.shape==(VOCAB,)
    assert REQUIRED_PHYSICAL_BYTES<=MAX_PHYSICAL_BYTES
    assert len(KINDS)==6 and len(PAID)==4
    print("CALIBRI v77 paid same-budget ablation geometry and rate PASS")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("build","evaluate","summarize","smoke"),required=True)
    p.add_argument("--kind",choices=KINDS,default="teacher")
    p.add_argument("--original",type=Path)
    p.add_argument("--folder",type=Path,default=Path("results"))
    args=p.parse_args()
    args.folder.mkdir(parents=True,exist_ok=True)
    if args.phase=="smoke":smoke()
    elif args.phase=="build":
        if args.original is None:p.error("--original required")
        build(args)
    elif args.phase=="evaluate":
        if args.kind=="baseline" and args.original is None:p.error("--original needed for baseline")
        if args.kind in PAID and not (args.folder/"v77_physical_build.json").exists():
            raise AssertionError("physically paid ablation source was not frozen")
        execute(args)
    else:
        if args.original is None:p.error("--original required for audit")
        summarize(args)
if __name__=="__main__":main()
