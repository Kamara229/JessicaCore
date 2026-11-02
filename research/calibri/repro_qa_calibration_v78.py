#!/usr/bin/env python3
"""CALIBRI v78 genuine pretrained source reproducibility + QA.

No TEST-guided weight tuning. Re-run identical original TRAIN optimizer twice,
record bitwise vs numerical cross-run reproducibility, choose one of FOUR
predeclared quantized output-residual scales by TWO independent TRAIN-VALID
full-vocabulary KL objectives, then open new disjoint heldout TEST corpora,
SciQ full-answer choice likelihood with support, BoolQ, real cache decoding.

Physical source always = complete original CB59 + paid CB76 FP16 factors,
header CRC32 SHA256, exact <=0.700bpw; packed original 2D weights still
executed by unchanged native C K32V8 5-bit GEMV. All changes research only.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, shutil, time
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
    VALID_CORPORA,tensors,pack,unpack,zero_factors,make_model,
)

WEIGHT_SCALES=(0.0,0.5,0.75,1.0)
TRAIN_MIN_BOTH_GAIN=0.10
TWO_VALID=("wiki_valid","news_valid")
RUN_MODES=("teacher","baseline","full_original","calibrated")
TEST_CORPORA=("wiki_test","news_test")
SCI_Q="allenai/sciq"
BOOLQ="google/boolq"
SCI_Q_ROWS=(840,864)
BOOLQ_ROWS=(1200,1228)
WIKI_TEST_FIRST=90000
NEWS_TEST_ROW_RANGE=(5000,5100)
WINDOW=64
WINDOW_COUNT=8
NEW_TOKENS_PER_PROMPT=24
PROMPTS=(
  "In the laboratory, a surprising result revealed",
  "The old town had one important secret",
  "To explain the solution, the teacher wrote",
)
NUMERIC_CROSSRUN_PREVIOUS_SHA=(
  "bf5eed8a06fb5d01843fca1bd23d62066ed04813403e71f9cba0ed80ebd1ee62",
  "4d5c1258924b77a458b1d08d962914f6c8845632f8604a0a04e929cd43ef759c",
)

def h(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()

def sha_file(path):
    return h(path.read_bytes())

def support_prompt(tok,row):
    # No published test examples appear in calibration, ever.
    support_tokens=tok.encode(str(row.get("support","")),add_special_tokens=False)
    context=tok.decode(support_tokens[:320],skip_special_tokens=True)
    return f"Context: {context}\nQuestion: {row['question']}\nAnswer:"

def qa_source(tok):
    tasks={}
    for task,(name,split,span) in {
      "sciq":(SCI_Q,"test",SCI_Q_ROWS),
      "boolq":(BOOLQ,"validation",BOOLQ_ROWS),
    }.items():
        rev=HfApi().dataset_info(name).sha
        lo,hi=span
        rows=load_dataset(name,split=f"{split}[{lo}:{hi}]",revision=rev)
        cases=[];skipped=[]
        for ix,row in enumerate(rows):
            idx=lo+ix
            if task=="sciq":
                options=[str(row["correct_answer"]),str(row["distractor1"]),
                         str(row["distractor2"]),str(row["distractor3"])]
                if len(set(options))!=4:
                    skipped.append({"source_index":idx,"reason":"duplicate_choice"});continue
                random.Random(780000+idx).shuffle(options)
                correct=options.index(str(row["correct_answer"]))
                prefix=support_prompt(tok,row)
                prompt_tokens=tok.encode(prefix,add_special_tokens=False)
                suffix_tokens=[tok.encode(" "+ans,add_special_tokens=False)
                               for ans in options]
                if any(not t or len(t)>64 for t in suffix_tokens):
                    skipped.append({"source_index":idx,"reason":"length_or_empty_choice"});continue
                case={"index":idx,"prefix_tokens":prompt_tokens,
                      "answer_options_tokens":suffix_tokens,"gold":correct,
                      "promptSHA":h(np.asarray(prompt_tokens,dtype="<i8").tobytes())}
            else:
                evidence=tok.encode(str(row["passage"]),add_special_tokens=False)
                short=tok.decode(evidence[:256],skip_special_tokens=True)
                prefix=f"Context: {short}\nQuestion: {row['question']}\nAnswer:"
                prompt_tokens=tok.encode(prefix,add_special_tokens=False)
                answers=[tok.encode(" Yes",add_special_tokens=False),
                         tok.encode(" No",add_special_tokens=False)]
                if any(len(q)!=1 for q in answers):
                    raise AssertionError("Yes/No completion not one token")
                case={"index":idx,"prefix_tokens":prompt_tokens,
                      "answer_options_tokens":answers,
                      "gold":0 if bool(row["answer"]) else 1,
                      "promptSHA":h(np.asarray(prompt_tokens,dtype="<i8").tobytes())}
            if not 1<len(prompt_tokens)<=600:raise AssertionError("prompt too long")
            cases.append(case)
        if len(cases)<10:raise AssertionError("too many invalid unseen QA rows")
        fingerprint=h(json.dumps(
            [{k:v for k,v in v.items() if k!="prefix_tokens"}
             for v in cases],sort_keys=True).encode())
        tasks[task]={"name":name,"revision":rev,"official_split":split,
                     "original_row_range":[lo,hi],"source_invalid_rows_uniformly_omitted":skipped,
                     "sample_count":len(cases),"cases":cases,
                     "fixed_input_fingerprint":fingerprint,
                     "chance_guess":0.25 if task=="sciq" else 0.5,
                     "limitations":(
                        "SciQ support first 320 model tokens; full answer option length-normalized "
                        "per-token likelihood instead of one-letter token."
                        if task=="sciq" else
                        "BoolQ passage first 256 model tokens; constrained Yes/No output.")}
    return tasks

def four_scale_factors(factors,scale):
    U,V,b=factors
    return (np.asarray(np.float32(scale)*U.astype(np.float32),dtype="<f2"),
            V.copy(),
            np.asarray(np.float32(scale)*b.astype(np.float32),dtype="<f2"))

def train_valid_KL(sample,U,V,b):
    teacher,base,hidden=sample
    with torch.inference_mode():
        x=hidden/torch.sqrt(torch.mean(hidden.square(),dim=-1,keepdim=True)+1e-6)
        uu=torch.from_numpy(U.astype(np.float32))
        vv=torch.from_numpy(V.astype(np.float32))
        bb=torch.from_numpy(b.astype(np.float32))
        target=teacher.log_softmax(dim=-1)
        prediction=(base+(x@vv)@uu.T+bb).log_softmax(dim=-1)
        kl=(target.exp()*(target-prediction)).sum(dim=-1).mean()
    return float(kl)

def freeze(args):
    first=(args.folder/"v78_first_train.cb76").read_bytes()
    second=(args.folder/"v78_second_train.cb76").read_bytes()
    meta1=json.loads((args.folder/"v78_first_selection.json").read_text())
    meta2=json.loads((args.folder/"v78_second_selection.json").read_text())
    if len(first)!=len(second)!=REQUIRED_PHYSICAL_BYTES:
        raise AssertionError("cannot verify actually complete original trained source")
    if meta1["chosen_source_step"]!=20 or meta1["chosen_source_mode"]!="rank4":
        raise AssertionError("reproduction no longer freezes v76 TRAIN-only choice")
    if meta2["chosen_source_step"]!=meta1["chosen_source_step"] or meta2["chosen_source_mode"]!=meta1["chosen_source_mode"]:
        raise AssertionError("TRAIN source-selection diverged between same-CPU repeats")
    original=args.original.read_bytes()
    if len(original)!=SOURCE_BYTES or first[:SOURCE_BYTES]!=original or second[:SOURCE_BYTES]!=original:
        raise AssertionError("bitwise exact entire original native 5bit CB59 source not present")
    _,U,V,b=unpack(first)
    _,U2,V2,b2=unpack(second)
    reproducibility={"same_runner_bitwise_identical":first==second,
      "first_cb76_SHA":h(first),"second_cb76_SHA":h(second),
      "rank4_U_max_abs_FP16_diff":float(np.max(np.abs(U.astype("f4")-U2.astype("f4")))),
      "rank4_V_max_abs_FP16_diff":float(np.max(np.abs(V.astype("f4")-V2.astype("f4")))),
      "output_bias_max_abs_FP16_diff":float(np.max(np.abs(b.astype("f4")-b2.astype("f4")))),
      "v76_original_pr206_source_SHA_match":h(first)==NUMERIC_CROSSRUN_PREVIOUS_SHA[0],
      "v77_original_pr207_source_SHA_match":h(first)==NUMERIC_CROSSRUN_PREVIOUS_SHA[1],
      "sha_variation_across_independent_runs_observed_previously":True}
    # Nonidentical bitwise source is reported, not silently called exact.
    # v76 source-selection comes from genuine TRAIN only; v78 trial scaling
    # depends ONLY on two original disjoint TRAIN VALID text corpora.
    data={name:tensors(args.folder,name) for name in TWO_VALID}
    scores={}
    base={c:train_valid_KL(data[c],*four_scale_factors((U,V,b),0.0))
          for c in TWO_VALID}
    for a in WEIGHT_SCALES:
        factors=four_scale_factors((U,V,b),a)
        scores[str(a)]={c:train_valid_KL(data[c],*factors) for c in TWO_VALID}
    improvements={a:{c:base[c]-scores[a][c] for c in TWO_VALID} for a in scores}
    sorted_candidates=sorted(WEIGHT_SCALES,
        key=lambda a:(min(improvements[str(a)].values()),-WEIGHT_SCALES.index(a)),reverse=True)
    raw_best=sorted_candidates[0]
    selected=(raw_best if min(improvements[str(raw_best)].values())>=TRAIN_MIN_BOTH_GAIN else 0.0)
    factors=four_scale_factors((U,V,b),selected)
    final=pack(original,*factors)
    if len(final)!=REQUIRED_PHYSICAL_BYTES or len(final)>MAX_PHYSICAL_BYTES:
        raise AssertionError("paid correction violates complete physical source budget")
    _,u,v,bias=unpack(final)
    out=args.folder/"CALIBRI_v78_calibrated.cb76"
    out.write_bytes(final)
    # Restore first original v76 MODEL file (if second trial overwrote it).
    (args.folder/"CALIBRI_v76_paid_full.cb76").write_bytes(first)
    (args.folder/"v76_selection.json").write_text(
        json.dumps(meta1,indent=2)+"\n")
    audit={"experiment":"CALIBRI_v78_two_domain_TRAIN_ONLY_alpha_validation",
      "model":MODEL,"pinned_real_trained_weights_revision":MODEL_SHA,
      "original_CB59_source_SHA":h(original),
      "source_unique_pretrained_parameters":EXPECTED_UNIQUE,
      "original_baseline_complete_bytes":SOURCE_BYTES,
      "all_paid_complete_CB76_bytes":REQUIRED_PHYSICAL_BYTES,
      "actual_all_source_and_lowrank_factor_bpw":8*len(final)/EXPECTED_UNIQUE,
      "all_original_pretrained_2D_weights_still_packed_in_original_C_5bit":True,
      "paid_rank4_factor_CPU_torch_not_native_C":True,
      "frozen_previous_v76_TRAIN_model_source":h(first),
      "reproducibility":reproducibility,
      "alpha_search_frozen_pre_TEST":list(WEIGHT_SCALES),
      "two_independent_TRAIN_VALID_datasets":list(TWO_VALID),
      "selection_min_two_domain_teacher_KL_gain_nats":TRAIN_MIN_BOTH_GAIN,
      "baseline_validation_KL_by_corpus":base,
      "VALID_KL_by_calibration_alpha":scores,
      "minimaxi_gain_by_calibration_alpha":improvements,
      "raw_best_TRAIN_alpha":raw_best,"selected_TRAIN_alpha":selected,
      "selected_frozen_CB76_source_SHA":h(final),
      "TEST_corpus_and_QA_unseen_at_selection":True,
      "actual_paid_factor_shapes":{"U":list(u.shape),"V":list(v.shape),"bias":list(bias.shape)},
      "same_runner_hashes_different_is_reported_not_ignored":True}
    (args.folder/"v78_frozen_training_choice.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"stage":"v78_training_repro_and_calibration_frozen",
                      "same_runner_SHA":reproducibility["same_runner_bitwise_identical"],
                      "first_reprod_sha":h(first),"second_reprod_sha":h(second),
                      "selected_alpha":selected,
                      "both_VALID_gains":improvements[str(selected)],
                      "all_in_bpw":round(audit["actual_all_source_and_lowrank_factor_bpw"],9)}),flush=True)

def model_for(args):
    if args.kind in ("teacher","baseline"):
        model,info,_=make_model(args.kind,SimpleNamespace(original=args.original,folder=args.folder))
        return model,info
    source_file=args.folder/"CALIBRI_v76_paid_full.cb76"
    original=source_file.read_bytes()
    path=(args.folder/"v78_first_train.cb76" if args.kind=="full_original"
          else args.folder/"CALIBRI_v78_calibrated.cb76")
    candidate=path.read_bytes()
    _,U,V,b=unpack(candidate)
    try:
        source_file.write_bytes(candidate)
        model,info,_=make_model("corrected",SimpleNamespace(original=args.original,folder=args.folder))
    finally:
        source_file.write_bytes(original)
    if info["complete_physical_SHA256"]!=h(candidate):
        raise AssertionError("actual loaded physically complete trained model mismatch")
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("trained dense original 2D source in paid runtime")
    return model,info

def actual_TEST_tokens(tok,domain):
    n=WINDOW_COUNT*WINDOW
    if domain=="wiki_test":
        rev=HfApi().dataset_info(DATASET).sha
        ids=load_token_ids(tok,"test",rev,WIKI_TEST_FIRST+n)[WIKI_TEST_FIRST:WIKI_TEST_FIRST+n]
        note={"dataset":DATASET,"split":"test","revision":rev,
              "token_range":[WIKI_TEST_FIRST,WIKI_TEST_FIRST+n]}
    else:
        rev=HfApi().dataset_info("fancyzhx/ag_news").sha
        lo,hi=NEWS_TEST_ROW_RANGE
        ds=load_dataset("fancyzhx/ag_news",split=f"test[{lo}:{hi}]",revision=rev)
        text="\n\n".join(str(r["text"]) for r in ds)
        ids=tok(text,add_special_tokens=False,return_tensors="pt").input_ids[0,:n]
        note={"dataset":"fancyzhx/ag_news","split":f"test[{lo}:{hi}]",
              "revision":rev,"token_range":[0,n],"text_only_no_labels":True}
    if len(ids)!=n:raise AssertionError("real TEST not long enough")
    note["input_SHA"]=h(ids.numpy().tobytes())
    return ids,note

@torch.inference_mode()
def metric_language(model,kind,tok,folder):
    scored={}
    for domain in TEST_CORPORA:
        ids,origin=actual_TEST_tokens(tok,domain)
        path=folder/("v78_teacher_"+domain+".npy")
        if kind=="teacher":
            original=np.lib.format.open_memmap(path,mode="w+",
              shape=(WINDOW_COUNT,WINDOW,VOCAB),dtype="<f2")
        else:
            original=np.load(path,mmap_mode="r")
            if tuple(original.shape)!=(WINDOW_COUNT,WINDOW,VOCAB):
                raise AssertionError("missing actual pretrained original teacher reference")
        per=[];times=[]
        for ix in range(WINDOW_COUNT):
            x=ids[ix*WINDOW:(ix+1)*WINDOW][None,:]
            tic=time.perf_counter()
            logits=model(input_ids=x,use_cache=False).logits[0].detach().float().numpy()
            times.append(time.perf_counter()-tic)
            if logits.shape!=(WINDOW,VOCAB) or not np.isfinite(logits).all():
                raise AssertionError("invalid trained full model logits")
            if kind=="teacher":
                original[ix]=logits.astype("<f2")
                continue
            ref=torch.from_numpy(np.array(original[ix,:-1],dtype=np.float64))
            val=torch.from_numpy(logits[:-1].astype(np.float64))
            lp=ref.log_softmax(-1); lq=val.log_softmax(-1)
            kl=float((lp.exp()*(lp-lq)).sum(-1).mean())
            answer=x[0,1:]
            nll=float(-lq[torch.arange(WINDOW-1),answer].mean())
            top1=float((ref.argmax(-1)==val.argmax(-1)).double().mean())
            per.append({"window":ix,"teacher_student_KL":kl,
                        "heldout_token_NLL":nll,"teacher_top1":top1})
        if kind=="teacher":original.flush()
        scored[domain]={"source":origin,"scored_next_tokens":WINDOW_COUNT*(WINDOW-1),
                        "data":per,"CPU_prefill_seconds":sum(times),
                        "teacher_original_logits_storage_FP16":True}
        print(json.dumps({"stage":"v78_frozen_language_TEST","model":kind,
          "domain":domain,"KL":round(np.mean([i["teacher_student_KL"] for i in per]),5) if per else None,
          "NLL":round(np.mean([i["heldout_token_NLL"] for i in per]),5) if per else None}),flush=True)
    return scored

@torch.inference_mode()
def QA_teacher_forced_likelihood(model,kind,tok):
    tasks=qa_source(tok)
    result={}
    for task,source in tasks.items():
        records=[];times=[]
        for item in source["cases"]:
            prefix=item["prefix_tokens"]
            if task=="sciq":
                # Evaluate ENTIRE answer's teacher-forced tokens rather than
                # unreliable single-letter A/B/C/D logit:
                # per-token average logprob of the candidate answer.
                scores=[]
                for suffix in item["answer_options_tokens"]:
                    ids=torch.tensor(prefix+suffix[:-1],dtype=torch.long)[None,:]
                    start=time.perf_counter()
                    out=model(input_ids=ids,use_cache=False).logits[0].float()
                    times.append(time.perf_counter()-start)
                    tokens=torch.tensor(suffix,dtype=torch.long)
                    relevant=out[len(prefix)-1:len(prefix)-1+len(suffix)]
                    if len(relevant)!=len(suffix):raise AssertionError("incorrect sequence alignment")
                    logp=relevant.log_softmax(-1)
                    indexes=torch.arange(len(tokens))
                    score=float(logp[indexes,tokens].mean())
                    scores.append(score)
            else:
                ids=torch.tensor(prefix,dtype=torch.long)[None,:]
                start=time.perf_counter()
                logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
                times.append(time.perf_counter()-start)
                scores=[float(logits[ids[0]]-torch.logsumexp(logits,0))
                        for ids in item["answer_options_tokens"]]
            predicted=int(np.argmax(scores))
            records.append({"record_index":item["index"],
                            "correct":int(predicted==item["gold"]),
                            "gold":item["gold"],"prediction":predicted,
                            "prompt_SHA":item["promptSHA"],
                            "option_conditional_prob":float(torch.softmax(
                               torch.tensor(scores,dtype=torch.float64),dim=-1)[item["gold"]])})
        count=len(records)
        result[task]={"source":{
            k:v for k,v in source.items() if k!="cases"},
            "correct_count":sum(r["correct"] for r in records),
            "total_valid":count,
            "accuracy":float(np.mean([r["correct"] for r in records])),
            "score_method":"length_normalized_full_answer_log_likelihood" if task=="sciq"
                    else "single_token_yes_no_relative_likelihood",
            "CPU_forward_seconds":sum(times),
            "items":records}
        print(json.dumps({"stage":"v78_holdout_real_answer_correctness","kind":kind,
                          "task":task,"correct":result[task]["correct_count"],
                          "valid":count,"accuracy":round(result[task]["accuracy"],4)}),flush=True)
    return result

@torch.inference_mode()
def real_generation(model,kind,tok):
    stats=[]
    for prompt in PROMPTS:
        x=tok(prompt,return_tensors="pt",add_special_tokens=False).input_ids
        start=time.perf_counter()
        output=model(input_ids=x,use_cache=True)
        prefill=time.perf_counter()-start
        cache=output.past_key_values
        token=output.logits[:,-1,:].argmax(dim=-1,keepdim=True)
        emitted=[int(token.item())]
        start=time.perf_counter()
        for j in range(NEW_TOKENS_PER_PROMPT-1):
            output=model(input_ids=token,past_key_values=cache,use_cache=True)
            cache=output.past_key_values
            token=output.logits[:,-1,:].argmax(dim=-1,keepdim=True)
            emitted.append(int(token.item()))
        elapsed=time.perf_counter()-start
        stats.append({"prompt_SHA":h(prompt.encode()),
                      "prefill_seconds":prefill,
                      "decode_seconds":elapsed,
                      "decode_tokens":NEW_TOKENS_PER_PROMPT-1,
                      "tokens_per_sec":(NEW_TOKENS_PER_PROMPT-1)/elapsed,
                      "distinct_token_ratio":len(set(emitted))/len(emitted),
                      "most_frequent_token_share":max(emitted.count(v) for v in set(emitted))/len(emitted),
                      "emitted_source_SHA":h(np.asarray(emitted,dtype="<i8").tobytes()),
                      "genuine_KV_cached_greedy":True})
    total=(NEW_TOKENS_PER_PROMPT-1)*len(stats)
    sec=sum(a["decode_seconds"] for a in stats)
    out={"genuine_tokens_after_prefill":total,
          "tokens_per_second_one_cpu":total/sec,
          "mean_distinct_token_ratio":float(np.mean([z["distinct_token_ratio"] for z in stats])),
          "mean_top_token_share":float(np.mean([z["most_frequent_token_share"] for z in stats])),
          "prompts":stats}
    print(json.dumps({"stage":"v78_greedy_actual_KV_cache","kind":kind,
                     "decode_tokens_per_s":round(out["tokens_per_second_one_cpu"],4),
                     "mean_distinct_token_ratio":round(out["mean_distinct_token_ratio"],4)}),flush=True)
    return out

def evaluate(args):
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    model,source=model_for(args)
    with torch.inference_mode():
        model(input_ids=torch.tensor([[1,2,3,4]]),use_cache=False)
    scores=metric_language(model,args.kind,tok,args.folder)
    qa=QA_teacher_forced_likelihood(model,args.kind,tok)
    gen=real_generation(model,args.kind,tok)
    audit={"study":"CALIBRI v78 true real model post-freeze TEST evidence",
           "kind":args.kind,"model":MODEL,"pinned_revision":MODEL_SHA,
           "source_physical":source,"source_TEST_not_used_to_choose_weights":True,
           "language":scores,"QA":qa,"greedy_actual_token_generation":gen}
    (args.folder/("v78_"+args.kind+".json")).write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"stage":"v78_completed_model","kind":args.kind,
      "sciq":qa["sciq"]["correct_count"],"boolq":qa["boolq"]["correct_count"],
      "greedy_distinct":round(gen["mean_distinct_token_ratio"],4)}),flush=True)

def exact_mcnemar(a,b):
    better_old=int(np.sum((a==1)&(b==0)))
    better_new=int(np.sum((b==1)&(a==0)))
    n=better_old+better_new
    if n==0:return 1.0,better_old,better_new
    p=2*sum(math.comb(n,k) for k in range(min(better_old,better_new)+1))/(2**n)
    return min(1.0,p),better_old,better_new

def summarize(args):
    frozen=json.loads((args.folder/"v78_frozen_training_choice.json").read_text())
    variants={name:json.loads((args.folder/("v78_"+name+".json")).read_text())
              for name in RUN_MODES}
    if variants["baseline"]["source_physical"]["physical_complete_model_bytes"]!=SOURCE_BYTES:
        raise AssertionError("CB59 not complete")
    for name in ("full_original","calibrated"):
        meta=variants[name]["source_physical"]
        if meta["physical_complete_model_bytes"]!=REQUIRED_PHYSICAL_BYTES or not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise AssertionError("miscounted original trained source or hidden dense weights")
    if variants["full_original"]["source_physical"]["complete_physical_SHA256"]!=frozen["frozen_previous_v76_TRAIN_model_source"]:
        raise AssertionError("full regenerated source SHA not frozen")
    if variants["calibrated"]["source_physical"]["complete_physical_SHA256"]!=frozen["selected_frozen_CB76_source_SHA"]:
        raise AssertionError("TRAIN-selected source SHA drift")
    output={"study":"CALIBRI_v78_post_freeze_heldout_QA_weight_repro_and_REAL_generated_tokens",
            "trained_original_model":MODEL,"pinned_revision":MODEL_SHA,
            "original_unique_parameters":EXPECTED_UNIQUE,
            "all_paid_actual_source_bytes":REQUIRED_PHYSICAL_BYTES,
            "full_packed_source_bpw":8*REQUIRED_PHYSICAL_BYTES/EXPECTED_UNIQUE,
            "frozen_TRAIN_only_selected_alpha":frozen["selected_TRAIN_alpha"],
            "no_TEST_tuning_or_prompt_choice":True,
            "same_runner_reproducibility":frozen["reproducibility"],
            "rank4_cpu_torch_correction_not_native_5bit_C":True,
            "original_trained_2D_source_all_packed_natively":True,
            "task_score_limitations":"SciQ length-normalized full answer with first 320-token support; BoolQ only first 256-token passage.",
            "language":{},"QA":{},"generation":{}}
    rng=np.random.default_rng(78)
    for domain in TEST_CORPORA:
        src=variants["teacher"]["language"][domain]["source"]["input_SHA"]
        if any(m["language"][domain]["source"]["input_SHA"]!=src for m in variants.values()):
            raise AssertionError("different original heldout source inputs")
        res={}
        for kind in RUN_MODES[1:]:
            original=variants[kind]["language"][domain]["data"]
            res[kind]={
              "KL":float(np.mean([x["teacher_student_KL"] for x in original])),
              "NLL":float(np.mean([x["heldout_token_NLL"] for x in original])),
              "prefill_CPU_s":variants[kind]["language"][domain]["CPU_prefill_seconds"]}
        for kind in ("full_original","calibrated"):
            a=variants["baseline"]["language"][domain]["data"]
            b=variants[kind]["language"][domain]["data"]
            changes={}
            for field in ("teacher_student_KL","heldout_token_NLL"):
                z=np.asarray([x[field] for x in b])-np.asarray([x[field] for x in a])
                bootstrap=z[rng.integers(0,len(z),size=(2000,len(z)))].mean(-1)
                changes[field]={"delta_paid_minus_CB59":float(z.mean()),
                  "paired_window95":list(float(n) for n in np.quantile(bootstrap,[.025,.975]))}
            res[kind]["delta_vs_baseline"]=changes
        output["language"][domain]=res
    for task in ("sciq","boolq"):
        canonical=variants["teacher"]["QA"][task]["source"]["fixed_input_fingerprint"]
        if any(m["QA"][task]["source"]["fixed_input_fingerprint"]!=canonical for m in variants.values()):
            raise AssertionError("QA prompt and option choice source differs")
        root=variants["teacher"]["QA"][task]
        baseline=np.asarray([z["correct"] for z in variants["baseline"]["QA"][task]["items"]])
        section={"dataset":root["source"]["name"],"revision":root["source"]["revision"],
                 "source_rows":root["source"]["original_row_range"],
                 "skipped_invalid_rows":root["source"]["source_invalid_rows_uniformly_omitted"],
                 "prompt_score_protocol":root["score_method"],
                 "outcomes":{}}
        for kind in RUN_MODES:
            data=variants[kind]["QA"][task]
            labels=np.asarray([z["correct"] for z in data["items"]])
            piece={"correct":data["correct_count"],"count":data["total_valid"],
                   "accuracy":data["accuracy"],"CPU_seconds":data["CPU_forward_seconds"]}
            if kind in ("full_original","calibrated"):
                p,old,new=exact_mcnemar(baseline,labels)
                z=labels-baseline
                resamples=z[rng.integers(0,len(z),size=(3000,len(z)))].mean(-1)
                piece["paid_minus_CB59_accuracy"]=float(z.mean())
                piece["McNemar_two_sided_p"]=p
                piece["base_only_correct"]=old
                piece["paid_only_correct"]=new
                piece["paired_bootstrap95"]=list(float(z) for z in np.quantile(resamples,[.025,.975]))
            section["outcomes"][kind]=piece
        output["QA"][task]=section
    for kind in RUN_MODES:
        entry=variants[kind]["greedy_actual_token_generation"]
        output["generation"][kind]={
           "true_KV_cache_CPU_decode_tokens_per_sec":entry["tokens_per_second_one_cpu"],
           "distinct_token_ratio":entry["mean_distinct_token_ratio"],
           "top_generated_token_share":entry["mean_top_token_share"],
           "cached_post_prefill_tokens":entry["genuine_tokens_after_prefill"]}
    output["qualitative_summary_is_NOT_validated_freeform_language_ability"]=True
    (args.folder/"v78_result_audit.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"stage":"v78_actual_real_weight_repro_QA_generation_complete",
       "source_bpw":round(output["full_packed_source_bpw"],9),
       "same_runner_SHA":output["same_runner_reproducibility"]["same_runner_bitwise_identical"],
       "alpha":output["frozen_TRAIN_only_selected_alpha"],
       "QA":{t:{k:v["correct"] for k,v in output["QA"][t]["outcomes"].items()}
             for t in ("sciq","boolq")},
       "test_KL":{c:{k:round(x["KL"],5) for k,x in output["language"][c].items()}
             for c in TEST_CORPORA},
       "diversity":{k:round(v["distinct_token_ratio"],4)
             for k,v in output["generation"].items()}}),flush=True)

def smoke():
    z=zero_factors()
    assert z[0].shape==(VOCAB,RANK) and z[1].shape==(DIM,RANK)
    assert REQUIRED_PHYSICAL_BYTES<=MAX_PHYSICAL_BYTES
    assert set(WEIGHT_SCALES)=={0.,0.5,0.75,1.}
    print("v78 source complete factor geometry, fixed selection protocol OK")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("smoke","freeze","evaluate","summarize"))
    p.add_argument("--kind",choices=RUN_MODES,default="teacher")
    p.add_argument("--folder",type=Path,default=Path("results"))
    p.add_argument("--original",type=Path)
    args=p.parse_args()
    args.folder.mkdir(parents=True,exist_ok=True)
    if args.phase=="smoke":smoke();return
    if args.original is None:p.error("unmodified original physical pretrained CB59 --original needed")
    if args.phase=="freeze":freeze(args)
    elif args.phase=="evaluate":
        if args.kind not in ("teacher","baseline") and not (args.folder/"v78_frozen_training_choice.json").exists():
            raise AssertionError("trained source alpha not frozen before TEST")
        evaluate(args)
    else:summarize(args)

if __name__=="__main__":main()
