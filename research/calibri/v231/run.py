"""CALIBRI v231: independent, sealed FP32-v228 multi-domain ANSWER benchmark.

Five public labeled validation splits; 96 distinct cases per domain, two
equivalent zero-shot prompt variants per case. 480 questions / 960 prompts
per model. Frozen options, order, parsing, sampling, and decision policy.
NEVER use these labels to train, choose checkpoints, or tune prompts.
FP32 is the exact original HuggingFaceTB/SmolLM2-135M-Instruct revision;
CB59 is independently SHA-verified v228 step12 archive (0.66784 bpp).
All scores are literal generated ANSWERS, never log-prob proxies.
Neither model's prompts contain gold labels; all incorrect/unparsable count
wrong; paired resampling clusters BOTH format variants per original case.
No Jessica tests; no production/main/Render/Supabase/APK modifications.
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import time

from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np
from scipy.stats import binomtest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from v174.run import save
from v176.operators import clear
from v196 import protocol as P
from v196.recover import build_trainable_from_paid_source
from v229.run import sha_checked,ACCEPTED_V228_STEP12_SHA256

SEED=231
PER_DOMAIN=96
TASKS=("boolq","sciq","arc_easy","piqa","hellaswag")
FORMATS=("direct","rephrased")
MAX_NEW_TOKENS=6
MAX_CONTEXT_TOKENS=900
MAX_BOOLQ_PASSAGE_CHARS=1400
N_BOOTSTRAP=3000
LETTERS="ABCD"
RESPONSE=re.compile(
    r"^\s*(?:(?:the\s+)?(?:correct\s+)?answer\s*(?::|is)\s*|option\s*)?"
    r"\(?([A-D])\)?[.!)\s]*$",re.I)
# Compare one answer per prompt. Wrong/invalid is always scored incorrect;
# never infer an answer from letter occurrences later in an explanation.
SOURCES={
  "boolq":("google/boolq",None,"validation"),
  "sciq":("allenai/sciq",None,"validation"),
  "arc_easy":("allenai/ai2_arc","ARC-Easy","validation"),
  # Safe dataset-author-endorsed PIQA parquet mirror; upstream ybisk/piqa\n  # only offers untrusted remote Python loader at this library version.\n  "piqa":("lighteval/piqa","plain_text","validation"),
  "hellaswag":("Rowan/hellaswag",None,"validation"),
}
def sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
        separators=(",",":")).encode("utf-8")).hexdigest()

def hashed_shuffle(items,key):
    r=random.Random(int(hashlib.sha256(
        (str(SEED)+"|"+str(key)).encode()).hexdigest(),16))
    result=list(items);r.shuffle(result)
    return result

def build_cases(name,data):
    available=[]
    for index,row in enumerate(data):
        try:
            if name=="boolq":
                question=row["question"].strip()
                passage=row["passage"].strip()
                if (not question or not passage or
                    len(passage)>MAX_BOOLQ_PASSAGE_CHARS):continue
                choices=["Yes","No"];winner=0 if bool(row["answer"]) else 1
                detail=question
            elif name=="sciq":
                question=row["question"].strip()
                choices=[row["correct_answer"].strip(),
                  row["distractor1"].strip(),row["distractor2"].strip(),
                  row["distractor3"].strip()]
                if not all(choices) or len(set(x.lower() for x in choices))!=4:continue
                winner=0;passage="";detail=question
            elif name=="arc_easy":
                question=row["question"].strip()
                choices=list(row["choices"]["text"])
                labels=list(row["choices"]["label"])
                if len(choices)!=4 or row["answerKey"] not in labels:continue
                winner=labels.index(row["answerKey"])
                if not all(choices) or len(set(x.lower() for x in choices))!=4:continue
                passage="";detail=question
            elif name=="piqa":
                question=row["goal"].strip()
                choices=[row["sol1"].strip(),row["sol2"].strip()]
                winner=int(row["label"])
                if winner not in (0,1):continue
                passage="";detail=question
            elif name=="hellaswag":
                question=row["ctx"].strip()
                choices=[x.strip() for x in row["endings"]]
                winner=int(row["label"])
                if winner not in range(4):continue
                passage="";detail=question
            else:raise AssertionError(name)
            if not question or not all(choices) or len(question)>850:
                continue
            if len(choices) not in (2,4) or len(set(choices))!=len(choices):
                continue
            key=f"{name}|{index}|{question}"
            order=hashed_shuffle(range(len(choices)),key+"|choices")
            options=[choices[j] for j in order]
            gold=LETTERS[order.index(winner)]
            available.append({
                "task":name,"case_id":f"{name}-{index}",
                "source_row_index":index,"question":question,
                "passage":passage,"choices":options,"gold":gold,
                "source_text_sha256":sha([question,passage,choices,winner]),
            })
        except (TypeError,ValueError,KeyError,IndexError):
            continue
    # Deterministic label-blind order. BoolQ/PIQA balanced by original
    # ground truth to prevent one majority answer bias in the sample.
    ordered=hashed_shuffle(available,name+"|case-selection")
    if name in ("boolq","piqa"):
        # This fixed prior balancing is based on dataset annotations ONLY
        # before model inference and never on either model's predictions.
        # Original answers recovered from visible option text, not model.
        groups=defaultdict(list)
        for x in ordered:
            if name=="boolq":
                label=0 if x["choices"][LETTERS.index(x["gold"])]=="Yes" else 1
            else:
                label=int(data[x["source_row_index"]]["label"])
            groups[label].append(x)
        if not 0 in groups or not 1 in groups:
            raise AssertionError("Binary domain missing a gold class")
        selected=groups[0][:PER_DOMAIN//2]+groups[1][:PER_DOMAIN//2]
        selected=hashed_shuffle(selected,name+"|balanced-selection")
    else:
        selected=ordered[:PER_DOMAIN]
    if len(selected)!=PER_DOMAIN or len({x["case_id"] for x in selected})!=PER_DOMAIN:
        raise AssertionError(f"Insufficient distinct labeled {name} cases")
    return selected,len(available)

def make_prompt(case,variant):
    lines="\n".join(f"{LETTERS[i]}. {value}"
                    for i,value in enumerate(case["choices"]))
    # Identical task, options and correct answer for both variants.
    if case["task"]=="boolq":
        if variant=="direct":
            return ("Read the passage and answer the question.\n"
                f"Passage: {case['passage']}\nQuestion: {case['question']}\n"
                f"Options:\n{lines}\nWrite only the correct option letter.")
        return (f"Question: {case['question']}\n"
            f"Relevant passage: {case['passage']}\n"
            f"Possible answers:\n{lines}\n"
            "Select the best answer. Reply with only A or B.")
    if variant=="direct":
        return (f"Question: {case['question']}\nOptions:\n{lines}\n"
                "Write only the correct option letter.")
    return (f"Choose the single best completion or answer.\n"
            f"Problem: {case['question']}\nPossible answers:\n{lines}\n"
            f"Reply with only one letter ({'/'.join(LETTERS[:len(case['choices'])])}).")

def build_manifest(tokenizer):
    api=HfApi()
    records=[];revs={}
    selection_meta={}
    for name in TASKS:
        repo,config,split=SOURCES[name]
        revision=api.dataset_info(repo).sha
        revs[repo]=revision
        ds=load_dataset(repo,config,split=split,revision=revision,
                        trust_remote_code=False)
        rows,eligible=build_cases(name,ds)
        selection_meta[name]={
            "eligible_prior_to_model_inference":eligible,
            "sampled":len(rows),"split":split,
            "case_ids_sha256":sha([x["case_id"] for x in rows]),
            "n_options":dict(Counter(len(x["choices"]) for x in rows)),
            "gold_letter_distribution":dict(Counter(x["gold"] for x in rows))}
        for x in rows:
            for variant in FORMATS:
                content=make_prompt(x,variant)
                prompt=tokenizer.apply_chat_template(
                    [{"role":"user","content":content}],
                    tokenize=False,add_generation_prompt=True)
                encoded=tokenizer.encode(prompt,add_special_tokens=False)
                if not 1<len(encoded)<=MAX_CONTEXT_TOKENS:
                    raise AssertionError(f"Precommitted case out of context: {x['case_id']} {len(encoded)}")
                records.append({"task":x["task"],"case_id":x["case_id"],
                    "variant":variant,"gold":x["gold"],
                    "options":len(x["choices"]),
                    "prompt":prompt,"prompt_sha256":hashlib.sha256(
                        prompt.encode()).hexdigest(),
                    "input_ids":encoded})
        print(json.dumps({"stage":"v231_FROZEN_INDEPENDENT_CASES",
            "task":name,"cases":len(rows),"prompts":len(rows)*len(FORMATS),
            "dataset_revision":revision}),flush=True)
    if (len(records)!=len(TASKS)*PER_DOMAIN*len(FORMATS) or
        len({(r["case_id"],r["variant"]) for r in records})!=len(records)):
        raise AssertionError("Paired 960 independent prompts incomplete")
    plan={
        "frozen_before_FP32_or_CB59_weight_load":True,
        "original_fp32_model":MODEL,"original_fp32_revision":MODEL_SHA,
        "immutable_sealed_CB59_sha256":ACCEPTED_V228_STEP12_SHA256,
        "original_FP32_parameters":EXPECTED_UNIQUE,
        "original_CB59_complete_bpp":0.6678394131307638,
        "unseen_for_current_model_recovery_not_guaranteed_unseen_for_pretraining":True,
        "datasets_SHA":revs,"selected_cases":selection_meta,
        "dataset_splits_only_validation":True,
        "task_count":len(TASKS),"questions_per_task":PER_DOMAIN,
        "unique_cases":len(TASKS)*PER_DOMAIN,
        "prompt_variants_per_case":len(FORMATS),
        "total_real_answers_per_model":len(records),
        "max_new_tokens":MAX_NEW_TOKENS,
        "max_context_tokens":MAX_CONTEXT_TOKENS,
        "max_boolq_passage_chars_without_truncation":MAX_BOOLQ_PASSAGE_CHARS,
        "seed":SEED,"sampling": "SHA256-seeded label-blind (balanced 48/48 for BoolQ and PIQA)",
        "prompt_variants":FORMATS,"invalid_or_unparseable_scored_incorrect":True,
        "only_generated_exact_option_letter_scored_not_logit_probabilities":True,
        "pair_bootstrap_unit":"source case grouping both prompt variants",
        "num_bootstrap":N_BOOTSTRAP,
        "quality_acceptance_precommitted":
          "packed accuracy >= FP32 across all prompts AND paired clustered 95pct CI lower >= -5pp AND no task worse by >10pp AND packed format-valid rate no more than 5pp below FP32",
        "no_training_no_finetuning_no_checkpoint_choice_from_this_validation":True,
        "do_not_claim_native_compressed_kernel_CPU_speed":True,
        "no_live_main_render_supabase_apk":True,
        "Jessica_tests_ran":False,
    }
    return records,plan

def get_prediction(raw,n):
    m=RESPONSE.fullmatch(raw.strip())
    if not m:return None
    answer=m.group(1).upper()
    return answer if answer in LETTERS[:n] else None

@torch.inference_mode()
def generate(model,sources,tokenizer,entry):
    context=torch.tensor(entry["input_ids"],dtype=torch.long)[None]
    generated=[]
    start=time.perf_counter()
    try:
        for _ in range(MAX_NEW_TOKENS):
            if sources:clear(sources)
            logits=model(input_ids=context,use_cache=False).logits
            if not torch.isfinite(logits[:,-1,:]).all():
                raise AssertionError("Nonfinite model logits")
            token=int(logits[0,-1,:].argmax())
            if token==tokenizer.eos_token_id:break
            generated.append(token)
            text=tokenizer.decode(generated,skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
            if get_prediction(text,entry["options"]) is not None:
                break
            context=torch.cat((context,torch.tensor([[token]],dtype=torch.long)),dim=1)
        output=tokenizer.decode(generated,skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        answer=get_prediction(output,entry["options"])
    finally:
        if sources:clear(sources)
    return {"task":entry["task"],"case_id":entry["case_id"],
       "variant":entry["variant"],"prompt_sha256":entry["prompt_sha256"],
       "gold":entry["gold"],"prediction":answer,"format_valid":answer is not None,
       "correct":answer==entry["gold"],"response_preview":output[:90],
       "generated_tokens":len(generated),
       "elapsed_seconds":round(time.perf_counter()-start,5)}

def evaluate(model_name,model,sources,tok,records,out):
    result=[];t0=time.perf_counter()
    path=out/f"{model_name}_all_960_generated_answers.jsonl"
    with path.open("w") as f:
        for idx,entry in enumerate(records):
            row=generate(model,sources,tok,entry)
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
            f.flush()
            result.append(row)
            if (idx+1)%96==0:
                partial=result[-96:]
                print(json.dumps({"stage":"v231_ACTUAL_independent_answers",
                  "model":model_name,"done":idx+1,"total":len(records),
                  "task":row["task"],
                  "last96_correct":sum(x["correct"] for x in partial),
                  "last96_parse_valid":sum(x["format_valid"] for x in partial),
                  "elapsed_seconds":round(time.perf_counter()-t0,1)}),flush=True)
    return result,round(time.perf_counter()-t0,2)

def paired_results(fp32,packed):
    if (len(fp32)!=len(TASKS)*PER_DOMAIN*len(FORMATS) or
        len(packed)!=len(TASKS)*PER_DOMAIN*len(FORMATS)):
        raise AssertionError("Paired benchmark count mismatched")
    keys=lambda rows:[(r["case_id"],r["variant"],r["gold"],r["prompt_sha256"])
                         for r in rows]
    if keys(fp32)!=keys(packed):
        raise AssertionError("Both models MUST receive identical case/options/prompt")
    by_domain={};b0=[];b1=[]
    for task in TASKS:
        p0=[r for r in fp32 if r["task"]==task]
        p1=[r for r in packed if r["task"]==task]
        if len(p0)!=2*PER_DOMAIN or len(p1)!=2*PER_DOMAIN:
            raise AssertionError("Incomplete domain paired outcomes")
        v0=sum(r["format_valid"] for r in p0)/len(p0)
        v1=sum(r["format_valid"] for r in p1)/len(p1)
        a0=sum(r["correct"] for r in p0)/len(p0)
        a1=sum(r["correct"] for r in p1)/len(p1)
        by_domain[task]={
            "unique_questions":PER_DOMAIN,"prompt_variants_per_question":len(FORMATS),
            "FP32_correct":sum(r["correct"] for r in p0),
            "CB59_correct":sum(r["correct"] for r in p1),
            "FP32_accuracy":a0,"CB59_accuracy":a1,
            "CB59_minus_FP32_percentage_points":round(100*(a1-a0),3),
            "FP32_answer_format_valid":v0,"CB59_answer_format_valid":v1}
        b0+=p0;b1+=p1
    # Ordered within one task and prompt variant; nonindependent formats
    # are grouped by original case, not treated as independent samples.
    d0={};d1={}
    for rows,store in ((fp32,d0),(packed,d1)):
        for r in rows:
            store.setdefault(r["case_id"],{})[r["variant"]]=int(r["correct"])
    if set(d0)!=set(d1) or len(d0)!=len(TASKS)*PER_DOMAIN:
        raise AssertionError("Case-level paired comparison mismatch")
    diffs=[]
    for cid in sorted(d0):
        if set(d0[cid])!=set(FORMATS) or set(d1[cid])!=set(FORMATS):
            raise AssertionError("Missing source-case prompt variant")
        diffs.append(sum(d1[cid][variant]-d0[cid][variant]
                         for variant in FORMATS)/len(FORMATS))
    rng=np.random.default_rng(SEED)
    sample=np.asarray(diffs,dtype=np.float32)
    boot=np.empty(N_BOOTSTRAP,dtype=np.float32)
    for i in range(N_BOOTSTRAP):
        boot[i]=sample[rng.integers(0,len(sample),len(sample))].mean()
    lo,hi=np.quantile(boot,[.025,.975]).tolist()
    fp_acc=sum(x["correct"] for x in fp32)/len(fp32)
    cb_acc=sum(x["correct"] for x in packed)/len(packed)
    fp_valid=sum(x["format_valid"] for x in fp32)/len(fp32)
    cb_valid=sum(x["format_valid"] for x in packed)/len(packed)
    primary0={r["case_id"]:bool(r["correct"]) for r in fp32
              if r["variant"]=="direct"}
    primary1={r["case_id"]:bool(r["correct"]) for r in packed
              if r["variant"]=="direct"}
    fp_only=sum(primary0[c] and not primary1[c] for c in primary0)
    cb_only=sum(primary1[c] and not primary0[c] for c in primary0)
    pvalue=float(binomtest(min(fp_only,cb_only),fp_only+cb_only,.5).pvalue) if fp_only+cb_only else 1.
    quality_floor=(cb_acc>=fp_acc and lo>=-.05 and cb_valid>=fp_valid-.05
                   and all(x["CB59_minus_FP32_percentage_points"]>=-10
                           for x in by_domain.values()))
    return {"total_unique_questions":len(d0),
        "exact_same_prompts_per_model":len(fp32),
        "scores_are_correct_generated_answers_not_NLL_or_KL":True,
        "per_task":by_domain,
        "FP32_total_correct":sum(r["correct"] for r in fp32),
        "CB59_total_correct":sum(r["correct"] for r in packed),
        "FP32_total_accuracy":fp_acc,"CB59_total_accuracy":cb_acc,
        "CB59_minus_FP32_accuracy_percentage_points":round((cb_acc-fp_acc)*100,3),
        "FP32_format_valid":fp_valid,"CB59_format_valid":cb_valid,
        "case_cluster_bootstrap_95pct_CI_delta": [round(100*lo,3),round(100*hi,3)],
        "paired_primary_format_FP32_only":fp_only,
        "paired_primary_format_CB59_only":cb_only,
        "paired_primary_format_exact_McNemar_p":pvalue,
        "precommitted_quality_noninferiority_gate":bool(quality_floor),
        "evidence_of_clear_superiority_not_just_point_estimate":bool(
            lo>0 and cb_acc>fp_acc),
        "gate_requires_confirmed_original_bpp_below_0_7":True,
        "if_gate_pass_then_scale_next_without_setting_higher_135M_accuracy_target":True,
        "heldout_unseen_only_to_compression_training_not_guaranteed_teacher_pretraining":True,
        "cannot_establish_free_form_language_quality_or_90pct_native_CPU_speed":True,
    }

def run(path,out):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    if tok.eos_token_id is None:
        raise AssertionError("Original model EOS token not defined")
    records,plan=build_manifest(tok)
    save(out/"PRECOMMITTED_v231_480_case_five_task_paired_plan.json",plan)
    save(out/"FROZEN_v231_960_prompts_manifest_no_answers.json",[
        {k:v for k,v in row.items() if k not in ("input_ids","gold")}
        for row in records])
    print(json.dumps({"stage":"v231_FROZEN_before_weight_inference",
         "n_cases":480,"n_prompts":960,
         "original_135M_revision":MODEL_SHA,
         "CB59_source_SHA":ACCEPTED_V228_STEP12_SHA256}),flush=True)
    archive=sha_checked(path,ACCEPTED_V228_STEP12_SHA256)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).cpu().eval()
    n_fp32=sum(p.numel() for p in teacher.parameters())
    if n_fp32!=EXPECTED_UNIQUE:
        raise AssertionError("Original FP32 source params differ from 135M baseline")
    fp32,t_fp32=evaluate("FP32",teacher,None,tok,records,out)
    del teacher;gc.collect()
    student,paid=build_trainable_from_paid_source(archive)
    student.cpu().eval()
    if len(paid)!=211:
        raise AssertionError("Missing original 211 paid codebooks")
    cb59,t_cb59=evaluate("CB59_v228",student,paid,tok,records,out)
    del student,paid;gc.collect()
    metrics=paired_results(fp32,cb59)
    metrics["physically_sealed_CB59_source_sha256"]=ACCEPTED_V228_STEP12_SHA256
    metrics["actual_original_135M_FP32_revision"]=MODEL_SHA
    metrics["original_FP32_unique_parameters"]=n_fp32
    metrics["physically_paid_CB59_bpp"]=0.6678394131307638
    metrics["runtime_seconds_FP32_diagnostic_not_native"]=t_fp32
    metrics["runtime_seconds_CB59_python_diagnostic_not_native"]=t_cb59
    metrics["completed"]=True
    metrics["no_new_model_training"]=True
    metrics["Jessica_test_requested_or_executed"]=False
    save(out/"ACTUAL_v231_independent_FP32_vs_sealed_CB59_480_cases_960_prompts_each.json",metrics)
    print(json.dumps({"stage":"v231_FINAL_real_answers_FP32_vs_CB59",
      "unique_questions":metrics["total_unique_questions"],
      "prompts_each":metrics["exact_same_prompts_per_model"],
      "FP32_correct":metrics["FP32_total_correct"],
      "CB59_correct":metrics["CB59_total_correct"],
      "FP32_accuracy":round(metrics["FP32_total_accuracy"],4),
      "CB59_accuracy":round(metrics["CB59_total_accuracy"],4),
      "paired_cluster_CI_delta_pp":metrics["case_cluster_bootstrap_95pct_CI_delta"],
      "noninferiority_gate":metrics["precommitted_quality_noninferiority_gate"],
      "clear_superiority":metrics["evidence_of_clear_superiority_not_just_point_estimate"],
      "physical_bpp":metrics["physically_paid_CB59_bpp"]}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("v228_paid_cb59",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.v228_paid_cb59,a.output)
