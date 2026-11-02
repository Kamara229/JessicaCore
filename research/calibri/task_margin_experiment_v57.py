#!/usr/bin/env python3
"""CALIBRI v57 pretrained Pythia INT6 frozen bootstrap plus paired task-answer-margin TRAIN stability.

PRE: original physically verified complete 6-bit checkpoint with MLP
calibration fitted on WikiText TRAIN A, as v46.
POST: same physically verified 6/12/24% per-tensor allocations as v51.
One frozen bootstrap NLL gate plus NEW 4-task paired gold-vs-wrong
answer margin TRAIN stability and first-use 128-item answer accuracy. Correct FP16
teacher residual rows are persisted with SHA/CRC and applied once at
cold load; full active weights still expanded into FP32 for fast inference.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np
import torch

from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from huggingface_hub import HfApi
from datasets import load_dataset
from quality import MODEL,DATASET,load_token_ids,module_selection,calibrate,ppl_eval
from all_mlp_agnews_v36 import module_identity
from pareto_fixed_highbits_v43 import trained_fixedbit
from full_checkpoint_v46 import pack_state,unpack_state,read_runtime
from sparse_recovery_v47 import encode_sparse_patch,parse_patch,apply_patch
from deterministic_budget_v49 import candidates
from tensor_cap_v51 import (plans_for_budget,smoke as cap_smoke)
from bootstrap_gate_v55 import stable_selection,smoke as gate_smoke
from task_margin_v57 import (
    TRAIN_DEFS, BLIND_DEFS, TRAIN_N, BLIND_N, TASKS,
    load_fixed_tasks, evaluate_real_task_margins,
    tasks_sha, frozen_margin_gate, smoke as task_smoke
)
from practical_int6_v46 import timing
from gradient_recovery_experiment_v47 import physical_write

A_TOKENS=2048
B_TOKENS=2048
CONTEXT=128
MAX_PHYSICAL_BPW=6.5
POST_RESERVED=15480
RATE_METHODS=("base_int6_pre","sse_cap_06pct","sse_cap_12pct","sse_cap_24pct")
SCI_Q_REPO="allenai/sciq"
BOOL_Q_REPO="super_glue"
SCI_Q_TRAIN_SLICES=((512,640),(640,768))
AG_TRAIN_SLICES=((1536,1664),(1664,1792))
BOOL_Q_TRAIN_SLICES=((256,384),(384,512))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def from_texts(tok,rows):
    if not rows:
        raise AssertionError("external evaluation was empty")
    ids=tok("\n\n".join(rows),add_special_tokens=False,
            return_tensors="pt").input_ids[0,:2048]
    if len(ids)!=2048:
        raise AssertionError("frozen first 2048 evaluation tokens missing")
    return ids


def sciq_text(rec):
    choices=(str(rec.get("correct_answer","")),
             str(rec.get("distractor1","")),
             str(rec.get("distractor2","")),
             str(rec.get("distractor3","")))
    return ("Support: "+str(rec["support"]).strip()+
            "\nQuestion: "+str(rec["question"]).strip()+
            "\nOptions: "+"; ".join(v.strip() for v in choices if v.strip()))


def boolq_text(row):
    return ("Passage: "+str(row["passage"]).strip()+
            "\nQuestion: "+str(row["question"]).strip())


def freeze_train_windows(tok,ag_rev,sciq_rev,boolq_rev,wiki_i,wiki_j):
    """8 disjoint TRAIN-only 2048-token windows, 2 in each family."""
    groups={"WikiText_TRAIN_I_16384_18432":wiki_i,
            "WikiText_TRAIN_J_18432_20480":wiki_j}
    for start,end in AG_TRAIN_SLICES:
        ds=load_dataset("fancyzhx/ag_news",revision=ag_rev,
                        split=f"train[{start}:{end}]")
        groups[f"AGNews_TRAIN_{start}_{end}"]=from_texts(tok,[
            str(row["text"]).strip() for row in ds])
    for start,end in SCI_Q_TRAIN_SLICES:
        ds=load_dataset(SCI_Q_REPO,revision=sciq_rev,
                        split=f"train[{start}:{end}]")
        groups[f"SciQ_TRAIN_{start}_{end}"]=from_texts(tok,[
            sciq_text(rec) for rec in ds])
    for start,end in BOOL_Q_TRAIN_SLICES:
        ds=load_dataset(BOOL_Q_REPO,"boolq",revision=boolq_rev,
                        split=f"train[{start}:{end}]")
        groups[f"BoolQ_TRAIN_{start}_{end}"]=from_texts(tok,[
            boolq_text(rec) for rec in ds])
    if len(groups)!=8 or len({sha(ids.numpy().tobytes())
                                for ids in groups.values()})!=8:
        raise AssertionError("source TRAIN 4 families / 8 window hash mismatch")
    return groups

def eval_ppl(model,domains,tag):
    model.eval()
    out={}
    with torch.inference_mode():
        for name,ids in domains.items():
            stats=ppl_eval(model,ids,CONTEXT)
            out[name]={"nll":float(stats["nll"]),"ppl":float(stats["ppl"])}
            print(json.dumps({"variant":tag,"corpus":name,"ppl":stats["ppl"]}),
                  flush=True)
    return out


def cold_load(cfg,base,sha_base,patches,variant):
    obj=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32)
    loader=read_runtime(obj,str(base))
    if variant!="base_int6_pre":
        loader["physical_residual_patch"]=apply_patch(
            obj,patches[variant]["path"],sha_base)
    obj.eval()
    return obj,loader


def choose_from_train(train):
    """Frozen equal-family and leave-one-family-out TRAIN-only selector."""
    return stable_selection(train)


def run(output):
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2" and
            os.getenv("DNNL_MAX_CPU_ISA")=="AVX2" and
            os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2" and
            str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("same selected AVX2 CPU model baseline required")
    torch.manual_seed(55)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cap_smoke()
    gate_smoke()
    task_smoke()
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    api=HfApi()
    source_rev=api.model_info(MODEL).sha
    wiki_rev=api.dataset_info(DATASET).sha
    ag_rev=api.dataset_info("fancyzhx/ag_news").sha
    sciq_rev=api.dataset_info(SCI_Q_REPO).sha
    boolq_rev=api.dataset_info(BOOL_Q_REPO).sha
    task_revs={name:api.dataset_info(repo).sha for name,repo,_,_ in TRAIN_DEFS}
    tok=AutoTokenizer.from_pretrained(MODEL,revision=source_rev)
    cfg=AutoConfig.from_pretrained(MODEL,revision=source_rev)
    original=AutoModelForCausalLM.from_pretrained(MODEL,revision=source_rev,
                                                  torch_dtype=torch.float32)
    original.eval()
    state={k:v.detach().cpu().contiguous().clone() for k,v in original.state_dict().items()}
    count=sum(v.numel() for v in state.values())
    source_sha=sha(b"".join(k.encode()+v.numpy().tobytes()
                           for k,v in sorted(state.items())))
    mlps=module_selection(original)
    names=sorted(mlps,key=module_identity)
    if len(names)!=12 or sum(state[n+".weight"].numel() for n in names)!=786432:
        raise AssertionError("original pretrained real Pythia architecture mismatch")
    wiki=load_token_ids(tok,"train",wiki_rev,20480)
    A=wiki[:2048]
    I=wiki[16384:18432]
    J=wiki[18432:20480]
    if (len(A)!=2048 or len(I)!=2048 or len(J)!=2048 or
        len({sha(v.numpy().tobytes()) for v in (A,I,J)})!=3):
        raise AssertionError("TRAIN-A/I/J must each contain 2048 disjoint tokens")
    with torch.inference_mode():
        captures=calibrate(original,mlps,A,CONTEXT)
    overrides={}
    for n in names:
        w=state[n+".weight"].numpy().astype(np.float32)
        x=captures[n].numpy().astype(np.float32)
        moments=np.mean(x*x,axis=0,dtype=np.float32)
        raw,adjusted,info=trained_fixedbit(w,moments,6)
        if len(raw)!=len(adjusted) or info["TRAIN_diagonal_weighted_MSE_after"]>info["TRAIN_diagonal_weighted_MSE_before"]*1.000001:
            raise AssertionError("PRE calibration failed source v46 loss+same-bytes test")
        overrides[n+".weight"]=adjusted
    blob,manifest=pack_state(state,preencoded=overrides)
    baseline_path=path.parent/"CALIBRI_v57_complete_INT6_pre.cb46"
    baseline=physical_write(baseline_path,blob)
    if baseline["bytes"]!=10803661:
        raise AssertionError("full physical pretrained Pythia 6-bit source count changed")
    decoded=unpack_state(blob,expected_keys=state.keys())
    total_max=math.floor(MAX_PHYSICAL_BPW*count/8)
    available=total_max-baseline["bytes"]-POST_RESERVED
    if available<500000:
        raise AssertionError("unexpected available trained FP16 restoration budget")

    # Global candidate priority is **solely actual source weights and
    # true physical quantization errors**, not even TRAIN-A gradients.
    # Each score is reproducible scalar math.fsum over exact FP32 inputs.
    scores,source_shapes=candidates(state,decoded)
    proposed=plans_for_budget(scores,available)
    patches={}
    expected_shapes={k:tuple(v.shape) for k,v in state.items() if v.ndim==2}
    for method in ("sse_cap_06pct","sse_cap_12pct","sse_cap_24pct"):
        pick=proposed[method]
        payload=encode_sparse_patch(state,decoded,pick["selection"],baseline["sha256"])
        patchfile=path.parent/("CALIBRI_v57_"+method+".cr47")
        evidence=physical_write(patchfile,payload)
        restored=parse_patch(patchfile.read_bytes(),baseline["sha256"],expected_shapes)
        if (len(restored)!=pick["rows"] or
            evidence["bytes"]!=pick["actual_patch_bytes_expected"] or
            baseline["bytes"]+evidence["bytes"]+POST_RESERVED>total_max):
            raise AssertionError("physical correction row count or budget failed")
        flipped=bytearray(payload);flipped[-1]^=1
        try:
            parse_patch(bytes(flipped),baseline["sha256"],expected_shapes)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted correction byte stream accepted")
        patches[method]={**evidence,
            "selected_real_rows":pick["rows"],
            "module_counts":pick["module_counts"],
            "all_source_plus_correction_bytes":baseline["bytes"]+evidence["bytes"],
            "physical_bpw_excl_reserved_extra_post":
                8*(baseline["bytes"]+evidence["bytes"])/count}
    # Four independent TRAIN corpus families, two disjoint windows each.
    # Read only AFTER base+physical CRC/SHA sidecars are finalized.
    gate_text=freeze_train_windows(tok,ag_rev,sciq_rev,boolq_rev,I,J)
    gate_metrics={}
    for name in RATE_METHODS:
        model,_=cold_load(cfg,baseline_path,baseline["sha256"],patches,name)
        gate_metrics[name]=eval_ppl(model,gate_text,name)
        del model
    # Original v55 paired four-family NLL bootstrap, unchanged and
    # computed solely on preregistered unlabeled TRAIN corpus windows.
    bootstrap_selection=choose_from_train(gate_metrics)
    # SECOND TRAIN-only accept/reject step uses continuous gold-vs-best
    # wrong logprob MARGIN (not only binary correct) for exactly 64 new
    # TRAIN records each SciQ/ARC-Easy/BoolQ/HellaSwag. No TEST data yet.
    train_tasks=load_fixed_tasks(TRAIN_DEFS,task_revs,TRAIN_N)
    train_task_audit=tasks_sha(train_tasks,TRAIN_DEFS)
    train_qa={}
    for name in RATE_METHODS:
        model,_=cold_load(cfg,baseline_path,baseline["sha256"],patches,name)
        train_qa[name]=evaluate_real_task_margins(
            model,tok,train_tasks,name+"_TRAIN_only")
        del model
    final_guard=frozen_margin_gate(bootstrap_selection,train_qa)
    selected=final_guard["chosen"]
    # The selected physical model is FROZEN. From this point onward,
    # test text and gold answer labels cannot affect any algorithm choice.
    tasks=load_fixed_tasks(BLIND_DEFS,task_revs,BLIND_N)
    task_metadata=tasks_sha(tasks,BLIND_DEFS)
    teacher=evaluate_real_task_margins(
        original,tok,tasks,"original_teacher_TEST")
    evaluated={}
    for name in RATE_METHODS:
        model,_=cold_load(cfg,baseline_path,baseline["sha256"],patches,name)
        evaluated[name]=evaluate_real_task_margins(
            model,tok,tasks,name+"_TEST")
        del model

    prompt=A[:CONTEXT].clone()
    follow=A[CONTEXT:CONTEXT+32].clone()
    before=timing(original,prompt,follow)
    del original
    model,loader=cold_load(cfg,baseline_path,baseline["sha256"],patches,selected)
    after=timing(model,prompt,follow)
    del model
    latency=after["total_prefill_and_decode32_ms"]["median"]/before[
        "total_prefill_and_decode32_ms"]["median"]
    selected_bytes=baseline["bytes"]+(
        patches[selected]["bytes"] if selected!="base_int6_pre" else 0)
    report={
        "experiment":"CALIBRI_v57_INT6_TRAIN_answer_margin_bootstrap_and_128_real_TEST",
        "real_pretrained_model":MODEL,
        "source_revision":source_rev,
        "source_checkpoint_fp32_SHA256":source_sha,
        "source_all_parameter_count":count,
        "wiki_revision":wiki_rev,"ag_news_revision":ag_rev,
        "sciq_training_revision":sciq_rev,
        "boolq_training_revision":boolq_rev,
        "external_task_source_revision_by_dataset":task_revs,
        "TRAIN_task_definition_and_record_SHA256":train_task_audit,
        "first_use_task_definition_and_record_SHA256":task_metadata,
        "WikiText_TRAIN_A_SHA256":sha(A.numpy().tobytes()),
        "eight_TRAIN_gate_token_SHA256":{name:sha(ids.numpy().tobytes()) for name,ids in gate_text.items()},
        
        "base_INT6_complete_source":baseline,
        "deterministic_row_budget":{
            "score":"fixed Python binary64 error math.fsum -> integer score -> exact row bytes",
            "balance":"one row per 2D tensor, then 6%,12%,24% of all physical correction budget per tensor, unchanged source-only SSE ranking",
            "max_total_physical_bpw":MAX_PHYSICAL_BPW,
            "reserved_other_correction_bytes":POST_RESERVED,
            "max_all_model_plus_all_correction_bytes":total_max,
            "candidate_row_count":len(scores),
            "plans":patches},
        "one_shot_train_only_gate":bootstrap_selection,
        "real_TASK_TRAIN_only_margin_gate":final_guard,
        "selected_method":selected,
        "selected_actual_byte_size":selected_bytes,
        "selected_actual_bpw":8*selected_bytes/count,
        "original_teacher_first_use_task_accuracy":teacher,
        "all_candidates_first_use_task_accuracy":evaluated,
        "actual_full_model_warm_speed":{
            "prompt_tokens":128,"autoregressive_decode_steps":32,
            "warmups":2,"timed_repeats":7,
            "original_teacher":before,"chosen":after,
            "chosen_divided_by_teacher":latency,
            "passes_plus20pct_user_limit":latency<=1.2,
            "fresh_real_physical_CB46_plus_CR47_loader":loader},
        "limitations":[
            "ALL pretrained matrices physically six-bit INT6 on disk, corrected rows separately true FP16 compressed residual sidecars, complete real stored bytes counted.",
            "All running full-model weights are reconstructed to dense FP32 at cold start: no active compressed RAM or native packed INT6 claims.",
            "Fixed bootstrap NLL on v55 TRAIN windows and NEW four-task 64-case labeled TRAIN paired answer-margin guard determine selection; no heldout TEST affects model choice.",
            "128 new HELDOUT answer items per task are scored by identical normalized conditional token likelihood; raw Pythia14m is not instruction tuned Jessica.",
            "One host may have faster absolute CPU; the measured 20%-latency cap is only within-host against original teacher.",
            "No GitHub main merge, Jessica ordinary chat test, Render, Supabase, APK, settings or secret changes."
        ]}
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "v57":"MEASURED_REAL",
        "base_SHA":baseline["sha256"],
        "int6_base_bytes":baseline["bytes"],
        "patches":{n:{"SHA":d["sha256"],"B":d["bytes"],
                      "rows":d["selected_real_rows"]} for n,d in patches.items()},
        "bootstrap_NLL_train_gate":bootstrap_selection["chosen"],
        "final_TASK_train_gate":final_guard["chosen"],
        "TASK_train_correct":{m:{k:r["correct"] for k,r in v.items()}
                              for m,v in train_qa.items()},
        "TASK_train_mean_margin":{m:{k:round(r["mean_gold_logprob_margin"],6)
                                for k,r in v.items()} for m,v in train_qa.items()},
        "train_gate_NLL":{n:{d:round(x["nll"],6) for d,x in score.items()}
                           for n,score in gate_metrics.items()},
        "selected_physical_bpw":report["selected_actual_bpw"],
        "heldout_first_use_task_accuracy":{
            "teacher":{k:{"correct":v["correct"],"n":v["n"]}
                       for k,v in teacher.items()},
            **{method:{k:{"correct":v["correct"],"n":v["n"]}
               for k,v in results.items()}
               for method,results in evaluated.items()}},
        "selected_warm_speed_ratio":latency,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="results/calibri_v57.json")
    args=ap.parse_args()
    run(args.output)
