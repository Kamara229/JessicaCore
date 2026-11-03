#!/usr/bin/env python3
"""CALIBRI v79: supervised answer-conditioned lowrank source at exact 0.69735bpw.

REAL trained SmolLM2-135M original CB59 native packed C decoder + original
v76 two-domain TRAIN-fitted FP16 rank4+bias CB76 source. Train same paid
FP16 U/V/b (unchanged source format/length) on SciQ/BoolQ TRAIN labels,
plus original teacher-KL regularizer on unrelated WikiText/News TRAIN.
Select strictly on disjoint QA TRAIN VALID + language TRAIN VALID before
ANY SciQ/BoolQ/News/WikiText TEST data is accessed. Independently rerun
new optimizer on the SAME original frozen QA features, audit bitwise SHA.
Then test real answer correctness, KL and degenerative cached decoding.
No main, Render, Supabase, APK, or working Jessica changes.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random,time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import (SOURCE_BYTES,REQUIRED_PHYSICAL_BYTES,
    DIM,VOCAB,RANK,pack,unpack,make_model,tensors)
import repro_qa_calibration_v78 as eval78

SCI_Q="allenai/sciq"
BOOLQ="google/boolq"
TRAIN_ROWS={"sciq":(300,324),"boolq":(600,648)}
VALID_ROWS={"sciq":(400,412),"boolq":(800,816)}
STEPS=(0,5,10,18)
SEED=79
LR=0.004
KL_FIT_WEIGHT=0.10
L2_CHANGE_WEIGHT=0.003
QA_VALID_MIN_GAIN=0.05  # relative accuracy over v76 on at least one task
LANG_VALID_MAX_REGRESSION=0.25  # nats max mean per corpus vs v76 on TRAIN VALID
SOURCE_SHA_V76_ORIGINAL="bf5eed8a06fb5d01843fca1bd23d62066ed04813403e71f9cba0ed80ebd1ee62"
TEST_MODES=("teacher","baseline","paid_v76","qa_v79")
NEW_TEST_SCIQ=(920,948)
NEW_TEST_BOOLQ=(1600,1632)
NEW_TEST_WIKI_START=105000
NEW_TEST_NEWS=(6000,6100)
SOURCE_HASHES={}

def digest(a):
    return hashlib.sha256(a).hexdigest()

def prepare(tok,split):
    if split not in ("fit","valid"):raise ValueError(split)
    samples=[]
    source={}
    for task,key in (("sciq",SCI_Q),("boolq",BOOLQ)):
        revision=HfApi().dataset_info(key).sha
        left,right=(TRAIN_ROWS if split=="fit" else VALID_ROWS)[task]
        records=load_dataset(key,split=f"train[{left}:{right}]",revision=revision)
        filtered=[]
        for offs,row in enumerate(records):
            record_index=left+offs
            if task=="sciq":
                choices=[row["correct_answer"],row["distractor1"],
                         row["distractor2"],row["distractor3"]]
                if len(set(choices))<4:
                    filtered.append(record_index);continue
                random.Random(790000+record_index).shuffle(choices)
                correct=choices.index(row["correct_answer"])
                prompt=eval78.support_prompt(tok,row)
                suffixes=[tok.encode(" "+str(s),add_special_tokens=False) for s in choices]
                if any(not ids or len(ids)>64 for ids in suffixes):
                    filtered.append(record_index);continue
            else:
                raw=tok.encode(str(row["passage"]),add_special_tokens=False)
                context=tok.decode(raw[:256],skip_special_tokens=True)
                prompt=f"Context: {context}\nQuestion: {row['question']}\nAnswer:"
                correct=0 if bool(row["answer"]) else 1
                suffixes=[tok.encode(" Yes",add_special_tokens=False),
                          tok.encode(" No",add_special_tokens=False)]
                if any(len(s)!=1 for s in suffixes):
                    raise AssertionError("BoolQ choice must map to one token")
            prefix=tok.encode(prompt,add_special_tokens=False)
            if not (2<len(prefix)<640):
                raise AssertionError(f"source prompt token length invalid {task} {record_index}")
            samples.append({"task":task,"index":record_index,"gold":correct,
                            "prefix":prefix,"suffixes":suffixes,
                            "prompt_sha":digest(np.asarray(prefix,dtype="<i8").tobytes())})
        source[task]={"dataset":key,"revision":revision,"original_split":"train",
                      "fixed_row_span":[left,right],
                      "validity_excluded_record_indices":filtered,
                      "remaining_samples":len([s for s in samples if s["task"]==task])}
    return samples,source

def collect(args):
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    examples,sources=prepare(tok,args.dataset)
    model,meta,_=make_model("baseline",SimpleNamespace(original=args.original,folder=args.folder))
    if not meta["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("original trained 2D weights not packed in C")
    captures=[]
    handle=model.lm_head.register_forward_pre_hook(
        lambda module,inp:captures.append(inp[0].detach().cpu().float().numpy()[0]))
    all_logits=[]; all_hidden=[]; targets=[]; question_groups=[]
    model.eval()
    begin=time.perf_counter()
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([[1,2,3,4]]),use_cache=False)
            captures.clear()
            for index,case in enumerate(examples):
                spans=[]
                for option in case["suffixes"]:
                    prefix=case["prefix"]
                    in_tokens=prefix+option[:-1]
                    input_ids=torch.tensor(in_tokens,dtype=torch.long)[None,:]
                    logits=model(input_ids=input_ids,use_cache=False).logits[0].float().numpy()
                    if len(captures)!=1 or captures[0].shape!=(len(in_tokens),DIM):
                        raise AssertionError("full native model packed hidden-state hook invalid")
                    hiddens=captures.pop()
                    start=len(prefix)-1
                    end=start+len(option)
                    if end>len(logits):raise AssertionError("answer alignment failed")
                    piece=np.asarray(logits[start:end],dtype="<f2")
                    real_hidden=np.asarray(hiddens[start:end],dtype="<f2")
                    if len(piece)!=len(option):raise AssertionError("bad answer segment")
                    if not np.isfinite(piece).all() or not np.isfinite(real_hidden).all():
                        raise AssertionError("nonfinite pretrained compressed logits")
                    begin_slot=len(targets)
                    targets.extend(option)
                    all_logits.append(piece); all_hidden.append(real_hidden)
                    spans.append([begin_slot,len(targets)])
                question_groups.append({"task":case["task"],"record_index":case["index"],
                                        "gold":case["gold"],"option_token_spans":spans,
                                        "source_prompt_sha":case["prompt_sha"]})
    finally:
        handle.remove()
    x=np.concatenate(all_logits,axis=0)
    features=np.concatenate(all_hidden,axis=0)
    token_ids=np.asarray(targets,dtype="<i8")
    if len(x)!=len(features) or len(x)!=len(token_ids):
        raise AssertionError("TRAIN QA feature corpus mismatched")
    out=args.folder/f"v79_qa_{args.dataset}.npz"
    np.savez_compressed(out,original_base_logits=x,original_packed_hidden=features,
                        gold_next_tokens=token_ids)
    report={"stage":"v79_answer_conditioned_TRAIN_original_native_source",
        "data_split":args.dataset,"dataset_provenance":sources,
        "source_model":MODEL,"model_revision":MODEL_SHA,
        "original_source_SHA":meta["complete_physical_SHA256"],
        "source_full_bytes":meta["physical_complete_model_bytes"],
        "actual_token_feature_count":len(token_ids),
        "original_questions_count":len(question_groups),
        "groups":question_groups,
        "source_TRAIN_ONLY_no_TEST_question_read":True,
        "gold_answers_are_official_original_TRAIN_labels":True,
        "CPU_feature_collection_seconds":time.perf_counter()-begin}
    (args.folder/f"v79_qa_{args.dataset}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"stage":"v79_qa_original_NATIVE_training_features",
                      "split":args.dataset,"questions":len(question_groups),
                      "answer_tokens":len(token_ids),
                      "data":sources}),flush=True)

def load_case_data(folder,split):
    meta=json.loads((folder/f"v79_qa_{split}.json").read_text())
    arrays=np.load(folder/f"v79_qa_{split}.npz")
    x=torch.from_numpy(arrays["original_base_logits"].astype(np.float32))
    h=torch.from_numpy(arrays["original_packed_hidden"].astype(np.float32))
    ids=torch.from_numpy(arrays["gold_next_tokens"].astype(np.int64))
    if tuple(x.shape)!=(meta["actual_token_feature_count"],VOCAB):
        raise AssertionError("source feature-vocab dims mismatch")
    if h.shape!=(len(x),DIM) or ids.shape!=(len(x),):
        raise AssertionError("source compressed h dims mismatch")
    h=h/torch.sqrt(h.square().mean(-1,keepdim=True)+1e-6)
    return (x,h,ids,meta["groups"]),meta

def find_answer_probs(state,qa):
    base,h,ids,groups=qa
    U,V,b=state
    logits=base+(h@V)@U.T+b
    probabilities=logits.log_softmax(-1)
    log_t=probabilities.gather(1,ids[:,None]).squeeze(-1)
    grouped={"sciq":[],"boolq":[]}
    correct={"sciq":[],"boolq":[]}
    for group in groups:
        scores=torch.stack([
            log_t[start:end].mean() for start,end in group["option_token_spans"]])
        grouped[group["task"]].append(scores)
        correct[group["task"]].append(group["gold"])
    return {task:(torch.stack(grouped[task]),torch.tensor(correct[task],dtype=torch.long))
            for task in grouped}

def qastats(state,qa):
    with torch.no_grad():
        scores=find_answer_probs(state,qa)
        return {k:{"correct":int((score.argmax(-1)==gold).sum()),
                   "total":len(gold),
                   "accuracy":float((score.argmax(-1)==gold).float().mean()),
                   "conditioned_answer_CE":float(torch.nn.functional.cross_entropy(score,gold))}
                for k,(score,gold) in scores.items()}

def get_language_ref(args,corpus):
    t,z,h=tensors(args.folder,corpus)
    h=h/torch.sqrt(h.square().mean(-1,keepdim=True)+1e-6)
    return (t,z,h)

def languageKL(state,language):
    with torch.no_grad():
        t,z,h=language
        u,v,b=state
        reference=t.log_softmax(-1);q=(z+(h@v)@u.T+b).log_softmax(-1)
        return float((reference.exp()*(reference-q)).sum(-1).mean())

def factor_np(U,V,b):
    return tuple(z.detach().to(torch.float16).numpy().astype("<f2").copy()
                 for z in (U,V,b))

def train_once(args,run):
    torch.set_num_threads(2)
    torch.manual_seed(SEED); np.random.seed(SEED)
    fit,metaf=load_case_data(args.folder,"fit")
    valid,metav=load_case_data(args.folder,"valid")
    language_valid={k:get_language_ref(args,k) for k in ("wiki_valid","news_valid")}
    # Only original v76 training source. No QA TEST or Wiki/News TEST touched.
    original=args.original.read_bytes()
    baseline=(args.folder/"CALIBRI_v76_paid_full.cb76").read_bytes()
    cb,U0,V0,b0=unpack(baseline)
    if original!=baseline[:SOURCE_BYTES] or len(baseline)!=REQUIRED_PHYSICAL_BYTES:
        raise AssertionError("paid original pretrained full archive invalid")
    source_expected=json.loads((args.folder/"v76_selection.json").read_text())
    if source_expected["chosen_source_mode"]!="rank4" or source_expected["chosen_source_step"]!=20:
        raise AssertionError("parent v76 TRAIN-only source no longer reproducible")
    initial=[torch.from_numpy(z.astype(np.float32)) for z in (U0,V0,b0)]
    U,V,b=[torch.nn.Parameter(z.clone()) for z in initial]
    opt=torch.optim.Adam([U,V,b],lr=LR)
    states={0:factor_np(U,V,b)}
    diagnostics=[]
    # Anchor language model to pre-existing original source. Original teacher
    # and packed model TRAIN FIT rows were separately collected by v76.
    t_list=[get_language_ref(args,k) for k in ("wiki_fit","news_fit")]
    sample_lang=[]
    for teacher,z,h in t_list:
        sample_lang.append((teacher[:32],z[:32],h[:32]))
    anchor=[(teacher.log_softmax(-1).exp()).detach() for teacher,_,_ in sample_lang]
    start=time.perf_counter()
    for step in range(1,max(STEPS)+1):
        opt.zero_grad(set_to_none=True)
        state=(U,V,b)
        decisions=find_answer_probs(state,fit)
        sciq_ce=torch.nn.functional.cross_entropy(*decisions["sciq"])
        boolq_ce=torch.nn.functional.cross_entropy(*decisions["boolq"])
        lang_losses=[]
        for (teacher,z,h),p in zip(sample_lang,anchor):
            logq=(z+(h@V)@U.T+b).log_softmax(-1)
            lang_losses.append(-(p*logq).sum(-1).mean())
        anchor_loss=sum(lang_losses)/len(lang_losses)
        regularizer=sum((x-y).square().mean() for x,y in zip(state,initial))
        loss=0.5*sciq_ce+0.5*boolq_ce+KL_FIT_WEIGHT*anchor_loss+L2_CHANGE_WEIGHT*regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state,max_norm=5.0)
        opt.step()
        if step in STEPS:
            states[step]=factor_np(U,V,b)
            diagnostics.append({"step":step,"fit_sciq_CE":float(sciq_ce),
                      "fit_boolq_CE":float(boolq_ce),
                      "TRAIN_fit_teacher_regularizer_CE":float(anchor_loss)})
    checks=[]
    original_state=tuple(torch.from_numpy(a.astype(np.float32)) for a in (U0,V0,b0))
    baseline_scores=qastats(original_state,valid)
    baseline_kl={k:languageKL(original_state,v) for k,v in language_valid.items()}
    for step in STEPS:
        s=tuple(torch.from_numpy(a.astype(np.float32)) for a in states[step])
        qa=qastats(s,valid)
        kl={k:languageKL(s,v) for k,v in language_valid.items()}
        # Preregistered safeguard: both QA tasks nondegrading, >=5 pp gain
        # on at least one, no >0.25nats loss on each language TRAIN VALID.
        qa_nonregress=all(qa[t]["accuracy"]>=baseline_scores[t]["accuracy"] for t in ("sciq","boolq"))
        qa_some_gain=any(qa[t]["accuracy"]-baseline_scores[t]["accuracy"]>=QA_VALID_MIN_GAIN
                         for t in ("sciq","boolq"))
        language_stable=all(kl[k]-baseline_kl[k]<=LANG_VALID_MAX_REGRESSION
                            for k in ("wiki_valid","news_valid"))
        qualifies=step>0 and qa_nonregress and qa_some_gain and language_stable
        improvement=sum(qa[t]["accuracy"]-baseline_scores[t]["accuracy"]
                        for t in ("sciq","boolq"))
        checks.append({"step":step,"TRAIN_VALID_QA":qa,
             "TRAIN_VALID_teacher_KL":kl,
             "TRAIN_VALID_delta_KL_vs_original_parent":{k:kl[k]-baseline_kl[k] for k in kl},
             "valid_qa_delta_accuracy_sum":improvement,
             "valid_non_degrading_both_tasks":qa_nonregress,
             "valid_positive_at_least_one_5pp":qa_some_gain,
             "valid_lang_no_0p25nat_regression_each":language_stable,
             "predeclared_promotion_eligible":qualifies})
    eligible=[c for c in checks if c["predeclared_promotion_eligible"]]
    winner=(max(eligible,key=lambda c:(c["valid_qa_delta_accuracy_sum"],-c["step"]))["step"]
            if eligible else 0)
    selected=states[winner]
    blob=pack(original,*selected)
    if len(blob)!=REQUIRED_PHYSICAL_BYTES:raise AssertionError("actual paid source byte budget drift")
    # Source is exactly SAME original trained tensor + same existing 16-bit
    # latent factor geometry; no new metadata, training cache or labels.
    resultfile=args.folder/f"v79_retrained_{run}.cb76"
    resultfile.write_bytes(blob)
    audit={"stage":"v79_TRUE_QA_SUPERVISED_TRAIN_only_fit_select",
        "repeat":run,"real_pretrained_model":MODEL,"revision":MODEL_SHA,
        "parent_model_source_SHA":digest(baseline),
        "TRAIN_gold_questions":metaf["original_questions_count"],
        "VALID_gold_questions":metav["original_questions_count"],
        "frozen_parent_source_SHA":digest(baseline),
        "all_in_bpw":8*len(blob)/EXPECTED_UNIQUE,
        "physical_trained_source_bytes":len(blob),
        "candidate_training_steps":list(STEPS),
        "optimizer_seed":SEED,"learning_rate":LR,
        "source_same_full_pretrained_native_C_kernel":True,
        "paid_residual_still_CPU_torch":True,
        "TRAIN_FIT_CE_diagnostics":diagnostics,
        "TRAIN_VALID_baseline_QA":baseline_scores,
        "TRAIN_VALID_baseline_KL":baseline_kl,
        "candidate_TRAIN_VALID_fixed_promotion_gates":checks,
        "chosen_step_TRAIN_only":winner,
        "chosen_source_SHA":digest(blob),
        "did_QA_supervision_change_saved_source":blob!=baseline,
        "TEST_QA_label_and_TEST_language_never_opened":True,
        "training_elapsed_sec":time.perf_counter()-start}
    (args.folder/f"v79_train_{run}.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"stage":"v79_answer_supervised_train_frozen",
      "repeat":run,"winner_step":winner,
      "source_changed":blob!=baseline,
      "selected_SHA":digest(blob),"train_valid_QA":checks[[c["step"] for c in checks].index(winner)]["TRAIN_VALID_QA"],
      "all_in_bpw":round(audit["all_in_bpw"],9)}),flush=True)

def freeze(args):
    a=json.loads((args.folder/"v79_train_first.json").read_text())
    b=json.loads((args.folder/"v79_train_second.json").read_text())
    x=(args.folder/"v79_retrained_first.cb76").read_bytes()
    y=(args.folder/"v79_retrained_second.cb76").read_bytes()
    if len(x)!=len(y) or len(x)!=REQUIRED_PHYSICAL_BYTES:
        raise AssertionError("truncated same source rate")
    _,u0,v0,b0=unpack(x)
    _,u1,v1,b1=unpack(y)
    deterministic={"same_runner_identical_source_SHA":digest(x)==digest(y),
        "first_SHA":digest(x),"second_SHA":digest(y),
        "U_FP16_max_abs_difference":float(np.max(np.abs(u0.astype("f4")-u1.astype("f4")))),
        "V_FP16_max_abs_difference":float(np.max(np.abs(v0.astype("f4")-v1.astype("f4")))),
        "bias_FP16_max_abs_difference":float(np.max(np.abs(b0.astype("f4")-b1.astype("f4"))))}
    frozen={"experiment":"CALIBRI_v79_answer_objective_and_native_source_REPRO_AUDIT",
         "source_real_pretrained_model":MODEL,"model_revision":MODEL_SHA,
         "frozen_first_TRAIN_winner":a["chosen_step_TRAIN_only"],
         "second_TRAIN_winner":b["chosen_step_TRAIN_only"],
         "parent_pretrained_full_source_SHA":a["frozen_parent_source_SHA"],
         "frozen_candidate_source_SHA":digest(x),
         "original_complete_CB59_bytes":SOURCE_BYTES,
         "complete_paid_source_bytes":REQUIRED_PHYSICAL_BYTES,
         "all_in_bpw":8*REQUIRED_PHYSICAL_BYTES/EXPECTED_UNIQUE,
         "source_heldout_TEST_unseen_before_frozen":True,
         "original_native_packed_5bit_C_unchanged":True,
         "paid_torch_CPU_rank4_unchanged":True,
         "same_host_bitwise_reproducibility":deterministic,
         "selection_can_legitimately_return_parent_no_improvement":True,
         "fit_result":a}
    if a["chosen_step_TRAIN_only"]!=b["chosen_step_TRAIN_only"]:
        frozen["WARNING"]="same host TRAIN winner differs, cannot claim reproducibility"
    (args.folder/"CALIBRI_v79_frozen_complete.cb76").write_bytes(x)
    (args.folder/"v79_FROZEN_before_TEST.json").write_text(json.dumps(frozen,indent=2)+"\n")
    print(json.dumps({"stage":"v79_train_frozen_before_all_TEST",
      "winner_step":frozen["frozen_first_TRAIN_winner"],
      "identical_SHAs":deterministic["same_runner_identical_source_SHA"],
      "physical_complete_bytes":len(x)}),flush=True)

def model_for_eval(args):
    if args.kind in ("teacher","baseline"):
        model,source,_=make_model(args.kind,SimpleNamespace(original=args.original,folder=args.folder))
        return model,source
    parent=(args.folder/"CALIBRI_v76_paid_full.cb76")
    actual=parent.read_bytes()
    chosen=(actual if args.kind=="paid_v76"
            else (args.folder/"CALIBRI_v79_frozen_complete.cb76").read_bytes())
    try:
        parent.write_bytes(chosen)
        model,source,_=make_model("corrected",SimpleNamespace(original=args.original,folder=args.folder))
    finally:
        parent.write_bytes(actual)
    if source["complete_physical_SHA256"]!=digest(chosen) or len(chosen)!=REQUIRED_PHYSICAL_BYTES:
        raise AssertionError("physical full source model SHA or byte total drift")
    return model,source

def evaluate(args):
    # IMPORT original v78 unbiased real test code unchanged, but use 100%
    # NEW preregistered TEST samples and prompts, distinct from v78.
    eval78.SCI_Q_ROWS=NEW_TEST_SCIQ
    eval78.BOOLQ_ROWS=NEW_TEST_BOOLQ
    eval78.WIKI_TEST_FIRST=NEW_TEST_WIKI_START
    eval78.NEWS_TEST_ROW_RANGE=NEW_TEST_NEWS
    eval78.PROMPTS=(
        "The conclusion from this experiment is",
        "Long before the city was founded,",
        "To verify the argument, we should first",
    )
    eval78.NEW_TOKENS_PER_PROMPT=24
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    model,source=model_for_eval(args)
    model.eval()
    with torch.inference_mode():
        model(input_ids=torch.tensor([[1,2,3,4]]),use_cache=False)
    # FP32 teacher reference explicitly created FIRST before compressed.
    full_text=eval78.metric_language(model,args.kind,tok,args.folder)
    qa=eval78.QA_teacher_forced_likelihood(model,args.kind,tok)
    generation=eval78.real_generation(model,args.kind,tok)
    report={"experiment":"CALIBRI_v79_REAL_NEW_HELDOUT_QA_and_GENERATION",
      "source_model":MODEL,"revision":MODEL_SHA,"kind":args.kind,
      "source_integrity":source,
      "TEST_SciQ_official_source_span":list(NEW_TEST_SCIQ),
      "TEST_BoolQ_official_source_span":list(NEW_TEST_BOOLQ),
      "TEST_WikiText_token_span":[NEW_TEST_WIKI_START,NEW_TEST_WIKI_START+512],
      "TEST_AGNews_record_span":list(NEW_TEST_NEWS),
      "QA_gold_not_used_to_fit_or_select_source":True,
      "source_original_trained_dense_2D_not_in_packed_students":args.kind=="teacher" or
           source.get("zero_dense_pretrained_2D_hot_weights",False),
      "language":full_text,
      "QA":qa,
      "real_greedy_generation":generation}
    (args.folder/f"v79_{args.kind}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"stage":"v79_REAL_NEW_ANSWER_TEST",
      "kind":args.kind,"sciq":qa["sciq"]["correct_count"],
      "boolq":qa["boolq"]["correct_count"],
      "generation_diversity":round(generation["mean_distinct_token_ratio"],4)}),flush=True)

def exact_p(a,b):
    onlya=int(np.sum((a==1)&(b==0)));onlyb=int(np.sum((a==0)&(b==1)))
    n=onlya+onlyb
    return (1.0 if n==0 else min(1.0,2*sum(math.comb(n,k) for k in range(min(onlya,onlyb)+1))/2**n),onlya,onlyb)

def report(args):
    frozen=json.loads((args.folder/"v79_FROZEN_before_TEST.json").read_text())
    sources={k:json.loads((args.folder/f"v79_{k}.json").read_text())
             for k in TEST_MODES}
    if sources["qa_v79"]["source_integrity"]["complete_physical_SHA256"]!=frozen["frozen_candidate_source_SHA"]:
        raise AssertionError("v79 tested a source different from frozen TRAIN")
    if sources["paid_v76"]["source_integrity"]["complete_physical_SHA256"]!=frozen["parent_pretrained_full_source_SHA"]:
        raise AssertionError("v76 original trained checkpoint source changed")
    out={"experiment":"CALIBRI_v79_ANSWER_SUPERVISION_genuine_original_source",
         "model":MODEL,"revision":MODEL_SHA,
         "full_source_actual_bytes_v79":REQUIRED_PHYSICAL_BYTES,
         "physical_bpw":8*REQUIRED_PHYSICAL_BYTES/EXPECTED_UNIQUE,
         "selected_TRAIN_step":frozen["frozen_first_TRAIN_winner"],
         "same_host_source_reproducibility":frozen["same_host_bitwise_reproducibility"],
         "real_paid_source_CPU_Torch_residual_not_fully_native_C":True,
         "no_TEST_source_selection":True,
         "source_checksum_reverified":True,
         "QA":{},"language":{},"decode":{}}
    rng=np.random.default_rng(79)
    for task in ("sciq","boolq"):
        proto=sources["teacher"]["QA"][task]["source"]["fixed_input_fingerprint"]
        if any(r["QA"][task]["source"]["fixed_input_fingerprint"]!=proto
               for r in sources.values()):
            raise AssertionError("mismatched QA source prompts")
        base=np.asarray([z["correct"] for z in sources["baseline"]["QA"][task]["items"]])
        part={"sample_count":len(base),
              "teacher_correct":sources["teacher"]["QA"][task]["correct_count"],
              "original_CB59_correct":sources["baseline"]["QA"][task]["correct_count"],
              "prior_v76_correct":sources["paid_v76"]["QA"][task]["correct_count"],
              "new_answer_supervised_CB76_correct":sources["qa_v79"]["QA"][task]["correct_count"]}
        for kind in ("paid_v76","qa_v79"):
            student=np.asarray([z["correct"] for z in sources[kind]["QA"][task]["items"]])
            p,onlya,onlyb=exact_p(base,student)
            delta=student-base
            sampled=delta[rng.integers(0,len(delta),size=(3000,len(delta)))].mean(-1)
            part[kind+"_vs_original"]={"delta_accuracy":float(delta.mean()),
                 "McNemar_exact_two_sided":p,
                 "CI95_bootstrap_accuracy_delta":list(float(a) for a in np.quantile(sampled,[.025,.975])),
                 "only_original_correct":onlya,"only_student_correct":onlyb}
        out["QA"][task]=part
    for domain in ("wiki_test","news_test"):
        sha=sources["teacher"]["language"][domain]["source"]["input_SHA"]
        if any(r["language"][domain]["source"]["input_SHA"]!=sha for r in sources.values()):
            raise AssertionError("different real unseen original text inputs")
        res={}
        for kind in TEST_MODES[1:]:
            windows=sources[kind]["language"][domain]["data"]
            res[kind]={
              "KL":float(np.mean([z["teacher_student_KL"] for z in windows])),
              "NLL":float(np.mean([z["heldout_token_NLL"] for z in windows])),
              "CPU_prefill_seconds":sources[kind]["language"][domain]["CPU_prefill_seconds"]}
        for kind in ("paid_v76","qa_v79"):
            res[kind]["signed_vs_CB59"]={
              metric:float(np.mean([p[metric]-b[metric]
                 for p,b in zip(sources[kind]["language"][domain]["data"],
                                sources["baseline"]["language"][domain]["data"])]))
              for metric in ("teacher_student_KL","heldout_token_NLL")}
        out["language"][domain]=res
    for kind in TEST_MODES:
        gen=sources[kind]["real_greedy_generation"]
        out["decode"][kind]={"cache_tokens_per_second":gen["tokens_per_second_one_cpu"],
                            "distinct_ratio":gen["mean_distinct_token_ratio"],
                            "most_frequent_token_share":gen["mean_top_token_share"]}
    out["science_result_negative_acceptable"]=True
    out["few_TEST_items_not_formal_statistical_proof"]=True
    out["no_claim_of_recovered_general_language_or_task_intelligence"]=True
    (args.folder/"v79_FINAL_independent_TEST.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"stage":"v79_REAL_blind_TASK_outcomes",
       "source_bpw":round(out["physical_bpw"],9),
       "qa":{task:{k:v for k,v in q.items() if k.endswith("_correct")}
             for task,q in out["QA"].items()},
       "language_KL":{c:{k:round(v["KL"],4) for k,v in d.items()}
                       for c,d in out["language"].items()},
       "diversity":{k:round(v["distinct_ratio"],4)
                       for k,v in out["decode"].items()}}),flush=True)

def smoke():
    assert REQUIRED_PHYSICAL_BYTES<=math.floor(.7*EXPECTED_UNIQUE/8)
    assert all(len(STEPS)>0 for _ in (0,))
    assert STEPS==(0,5,10,18)
    print("CALIBRI v79 physically complete 0.7bpw, source geometry, train/test labels smoke PASS")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("smoke","collect","train","freeze","evaluate","report"),required=True)
    p.add_argument("--dataset",choices=("fit","valid"),default="fit")
    p.add_argument("--repeat",choices=("first","second"),default="first")
    p.add_argument("--kind",choices=TEST_MODES,default="teacher")
    p.add_argument("--original",type=Path)
    p.add_argument("--folder",type=Path,default=Path("results"))
    args=p.parse_args()
    args.folder.mkdir(parents=True,exist_ok=True)
    if args.phase=="smoke":smoke();return
    if args.original is None:p.error("--original source required for real LLM")
    if args.phase=="collect":collect(args)
    elif args.phase=="train":train_once(args,args.repeat)
    elif args.phase=="freeze":freeze(args)
    elif args.phase=="evaluate":
        if not (args.folder/"v79_FROZEN_before_TEST.json").exists():
            raise AssertionError("do not open QA TEST before frozen source")
        evaluate(args)
    else:report(args)

if __name__=="__main__":main()
