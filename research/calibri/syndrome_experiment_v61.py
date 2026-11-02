#!/usr/bin/env python3
"""CALIBRI v61 actual 135M teacher TRAIN activation-guided CB61 residual.

One source-only supervised-free TRAIN activation calibration from
WikiText TRAIN[0:256] supplies input channel mean-square d_j. Greedy
error syndromes store the exact original-minus-CB59 FP16 residual
columns on the MOST functionally excited quantization-error channels.
All extra data fit inside a SINGLE CRC/SHA-bound physical <=0.70bpw
CB61 checkpoint. Native packed C remains the base direct GEMV, and
the paid residual channel correction is computed as low-rank x_j * U.

Original full teacher only runs in separate encode/teacher processes,
NEVER loaded into packed inference runtime. Fresh 8-prompt diagnostics
are read after source+TRAIN calibration archive has been frozen.
"""
from __future__ import annotations
import argparse,hashlib,json,math,time,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import HfApi
from quality import load_token_ids,DATASET
from shared_checkpoint_v58 import split_aliases
from direct_codebook_v59 import CB59,encode as encode_source,smoke as base_smoke
from direct_runtime_v59 import (
    NativeKernel,compile_kernel,bind_into_llama)
from syndrome_codebook_v61 import (
    CB61,pack as encode_syndrome,smoke as syndrome_smoke)
from syndrome_runtime_v61 import (
    bind_syndrome_into_model,smoke as runtime_smoke)
from sub07_experiment_v59 import (
    MODEL,EXPECTED_UNIQUE,MAX_BPW,actual_forward,setup)
from pathlib import Path

MODEL_SHA="12fd25f77366fa6b3b4b768ec3050bf629380bac"
PROMPTS=(
    "The lungs exchange oxygen and carbon dioxide because blood travels through",
    "A spacecraft must adjust its orbit before approaching the distant station",
    "During winter, animals may store energy to survive periods of",
    "In mathematics a prime number is divisible only by one and",
    "To prepare a glass of lemonade first squeeze fresh lemons into",
    "The city council voted to renovate the old bridge after engineers found",
    "We can test our scientific hypothesis by measuring the same process",
    "After the long concert the musicians carefully placed their instruments inside",
)
SOURCE_FORMAT="CB59 original source, all 2D coded K32V8 0.625 raw bpw"
TRAIN_WINDOW=256
FULL_SOURCE_CAP=math.floor(MAX_BPW*EXPECTED_UNIQUE/8)


def config(rev):
    t=AutoTokenizer.from_pretrained(MODEL,revision=rev)
    cfg=AutoConfig.from_pretrained(MODEL,revision=rev)
    return t,cfg


def prompt_ids(tok,prompt):
    ids=tok(prompt,add_special_tokens=False,return_tensors="pt").input_ids
    if ids.shape[1]<8:
        raise AssertionError("all prerecorded probe prompts must have 8 real tokens")
    return ids[:,:8].contiguous()


def stats_from_real_teacher(teacher,tok,wiki_rev,alias):
    """Actual TRAIN-only second moments of dense teacher input activations.

    Hook only original source Linear; each example contributes exactly
    128 TRUE WikiText TRAIN tokens. The tied LM head's hidden input
    energy is attributed to its canonical embedding parent name.
    No future/heldout prompt data used in the codebook or correction.
    """
    train=load_token_ids(tok,"train",wiki_rev,TRAIN_WINDOW)
    if len(train)<TRAIN_WINDOW:
        raise AssertionError("unexpected truncated WikiText TRAIN-only calibration")
    accum={}
    count={}
    handles=[]
    def makehook(key):
        def pre(mod,args):
            x=args[0].detach().to(torch.float64)
            if x.shape[-1]!=mod.in_features:
                raise AssertionError("source module TRAIN activation input is wrong")
            x=x.reshape(-1,x.shape[-1])
            a=(x*x).sum(dim=0).detach().cpu().numpy()
            accum[key]=a if key not in accum else accum[key]+a
            count[key]=count.get(key,0)+len(x)
        return pre
    for name,mod in teacher.named_modules():
        if isinstance(mod,nn.Linear):
            name_w=name+".weight"
            canonical=alias.get(name_w,name_w)
            handles.append(mod.register_forward_pre_hook(makehook(canonical)))
    if not handles:
        raise AssertionError("failed to register actual pretrained source modules")
    try:
        with torch.inference_mode():
            teacher(input_ids=train[:128].unsqueeze(0),use_cache=False)
            teacher(input_ids=train[128:256].unsqueeze(0),use_cache=False)
    finally:
        for h in handles:h.remove()
    values={k:(np.round(v/count[k],decimals=5)).astype(np.float64)
            for k,v in accum.items()}
    if not values:
        raise AssertionError("empty pretrained teacher activation statistics")
    return values,{
        "source_calibration_family":"WikiText-2 TRAIN ONLY",
        "wikitext_revision":wiki_rev,
        "token_range":[0,TRAIN_WINDOW],
        "TRAIN_token_sha256":hashlib.sha256(
             train[:TRAIN_WINDOW].numpy().tobytes()).hexdigest(),
        "trained_module_input_activation_columns":len(values),
        "fixed_second_moment_decimal_grid":1e-5,
        "no_heldout_task_data_used":True}


def stage_encode(folder,rev):
    setup()
    base_smoke()
    syndrome_smoke()
    runtime_smoke()
    tok,cfg=config(rev)
    api=HfApi()
    wiki_rev=api.dataset_info(DATASET).sha
    original=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=rev,torch_dtype=torch.float32).eval()
    weights=original.state_dict()
    unique,aliases=split_aliases(weights)
    count=sum(p.numel() for p in original.parameters())
    if (count!=EXPECTED_UNIQUE or
        sum(v.numel() for v in unique.values())!=count or
        aliases!={"lm_head.weight":"model.embed_tokens.weight"}):
        raise AssertionError("wrong actual unique pretrained SmolLM2 model")
    t=time.perf_counter()
    base,_,diagnostics=encode_source(weights)
    sec=time.perf_counter()-t
    if len(base)>FULL_SOURCE_CAP:
        raise AssertionError("immutable original CB59 already over 0.7bpw")
    real_stats,calibration=stats_from_real_teacher(
        original,tok,wiki_rev,aliases)
    t=time.perf_counter()
    corrected,manifest,detail=encode_syndrome(
        base,weights,real_stats,FULL_SOURCE_CAP)
    syndrome_seconds=time.perf_counter()-t
    cb59_file=folder/"CALIBRI_v61_original_physical_CB59.cb59"
    cb61_file=folder/"CALIBRI_v61_physical_syndrome_CB61.cb61"
    cb59_file.write_bytes(base)
    cb61_file.write_bytes(corrected)
    confirm_base=CB59(cb59_file.read_bytes(),expected_keys=weights.keys())
    confirm=CB61(cb61_file.read_bytes(),expected_keys=weights.keys())
    if (len(corrected)>FULL_SOURCE_CAP or
        confirm.base.hash()!=confirm_base.hash() or
        not confirm.meta["corrected_input_column_count"]):
        raise AssertionError("cannot recover true source-bound physical 0.7bpw syndrome")
    mut=bytearray(corrected);mut[len(mut)//2]^=1
    try:CB61(bytes(mut))
    except ValueError:pass
    else:raise AssertionError("CB61 whole file CRC/SHA corruption not caught")
    report={
        "experiment":"CALIBRI_v61_hard_rate_activation_selected_error_syndrome",
        "pretrained_source_model":MODEL,
        "source_model_revision":rev,
        "unique_pretrained_source_parameter_count":count,
        "original_tied_weight_aliases":aliases,
        "total_source_parameter_bfloat16_original_bytes":count*2,
        "hard_0_70bpw_checkpoint_max_bytes":FULL_SOURCE_CAP,
        "base_original_full_CB59_source":{
            "bytes":len(base),"sha256":confirm_base.hash(),
            "actual_effective_bpw":8*len(base)/count,
            "source_global_weight_relative_RMSE":
                diagnostics["full_source_weighted_relative_RMSE"],
            "source_only_fit_seconds":sec,
            "full_pretrained_source_unique_2D_matrices":
                diagnostics["full_source_intact_weight_matrices"]},
        "candidate_CB61_full_single_crc_sha_file":{
            **detail,"sha256":confirm.hash(),
            "actual_effective_bpw":8*len(corrected)/count,
            "source_only_syndrome_fit_seconds":syndrome_seconds},
        "correction_encoding":"one-column exact original-vs-CB59 FP16 residual per chosen source input channel, directly added as U*x_j",
        "teacher_TRAIN_input_activation_squared_moments_source":calibration,
        "predeclared_unseen_dev_prompts_sha256":
            hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest(),
        "all_syndrome_residual_is_physically_counted_not_free_parity":True,
        "no_task_answer_labels_used":True,
        "never_reconstruct_entire_dense_pretrained_weight_during_pack_inference":True,
    }
    (folder/"calibri_v61_encode.json").write_text(json.dumps(
        report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"v61":"COMPLETE_REAL_PRETRAINED_0_7_SOURCES",
        "original_bpw":report["base_original_full_CB59_source"]["actual_effective_bpw"],
        "syndrome_bpw":report["candidate_CB61_full_single_crc_sha_file"]["actual_effective_bpw"],
        "parent_bytes":len(base),"new_physical_bytes":len(corrected),
        "syndrome_columns":detail["selected_columns"],
        "TRAIN_activation_modules":calibration["trained_module_input_activation_columns"]},
        ensure_ascii=False),flush=True)


def run_probes(model,tok):
    out={}
    timings={}
    for i,prompt in enumerate(PROMPTS):
        logits,record=actual_forward(model,prompt_ids(tok,prompt))
        out["p"+str(i)]=logits
        timings["p"+str(i)]=record[
          "real_8token_full_model_prefill_and_one_KV_decode_seconds"]
    return out,timings


def stage_teacher(folder,rev):
    setup()
    tok,cfg=config(rev)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=rev,torch_dtype=torch.float32).eval()
    vals,timings=run_probes(teacher,tok)
    np.savez(folder/"calibri_v61_teacher.npz",**vals)
    (folder/"calibri_v61_teacher.json").write_text(json.dumps({
        "pretrained_source_model":MODEL,"source_SHA":rev,
        "real_original_8token_and_KV_times":timings,
        "teacher_full_unique_parameter_count":sum(p.numel() for p in teacher.parameters()),
        "never_loaded_CB59_or_CB61":True},indent=2)+"\n")
    print({"v61":"SEPARATE_8_FRESH_TEACHER_FORWARD_PROBES"},flush=True)


def stage_packed(folder,rev):
    setup()
    tok,cfg=config(rev)
    shared=NativeKernel(compile_kernel(
        Path(__file__).with_name("direct_kernel_v59.c")))
    configs=(
        ("base_CB59","CALIBRI_v61_original_physical_CB59.cb59"),
        ("CB61_activation_error_syndrome","CALIBRI_v61_physical_syndrome_CB61.cb61"),
    )
    evidence={}
    for method,filename in configs:
        data=(folder/filename).read_bytes()
        if method=="base_CB59":
            code=CB59(data)
        else:
            code=CB61(data)
        # Important: no pretrained original checkpoint, no float source
        # weights in this completely separate inference process.
        model=AutoModelForCausalLM.from_config(
            cfg,torch_dtype=torch.float32).eval()
        if method=="base_CB59":
            proof=bind_into_llama(model,code,shared)
        else:
            proof=bind_syndrome_into_model(model,code,shared)
        if any(p.ndim>=2 for p in model.parameters()):
            raise AssertionError("dense original pretrained source leaked into direct runtime")
        logits,timing=run_probes(model,tok)
        np.savez(folder/("calibri_v61_"+method+".npz"),**logits)
        evidence[method]={
            "full_real_pretrained_model_direct_and_KV_completed":True,
            "physical_checkpoint_filename":filename,
            "physical_file_SHA256":code.hash(),
            "physical_file_bytes":code.physical_bytes(),
            "pretrained_teacher_was_NOT_loaded_this_process":True,
            "hot_dense_pretrained_original_2D_weight_count":0,
            "native_base_C_5bit_GEMV_plus_optional_1D_error_syndromes":True,
            "actual_direct_inference_timing_per_fresh_dev_prompt":timing,
            "all_212_packed_source_modules":proof}
        del model,code
        gc.collect()
    (folder/"calibri_v61_packed.json").write_text(json.dumps(
        evidence,indent=2,ensure_ascii=False)+"\n")
    print({"v61":"BOTH_FULL_MODELS_NATIVE_DIRECT_WITH_8_PROMPT_KV_AND_SYNDROME",
           "control_weights_stayed_packed":True,
           "corrected_weights_stayed_packed":True},flush=True)


def metrics(x,y):
    if x.shape!=y.shape or not (
        np.isfinite(x).all() and np.isfinite(y).all()):
        raise AssertionError("invalid teacher-to-packed real logits")
    a=torch.from_numpy(x.astype(np.float64))
    b=torch.from_numpy(y.astype(np.float64))
    p=a.log_softmax(dim=-1).exp()
    kl=float(torch.sum(p*(a.log_softmax(dim=-1)-
                          b.log_softmax(dim=-1))).item())
    mse=np.mean((x.astype(np.float64)-y.astype(np.float64))**2)
    energy=np.mean(x.astype(np.float64)**2)
    return {
        "full_vocab_teacher_to_compressed_KL_nats":kl,
        "normalized_logit_RMSE":float(math.sqrt(mse/(energy+1e-18))),
        "same_teacher_top1_token":bool(np.argmax(x)==np.argmax(y)),
        "finite_full_model_outputs":True}


def stage_compare(folder,rev):
    physical=json.loads((folder/"calibri_v61_encode.json").read_text())
    runtime=json.loads((folder/"calibri_v61_packed.json").read_text())
    teacher=json.loads((folder/"calibri_v61_teacher.json").read_text())
    truth=np.load(folder/"calibri_v61_teacher.npz")
    results={}
    for name in ("base_CB59","CB61_activation_error_syndrome"):
        packed=np.load(folder/("calibri_v61_"+name+".npz"))
        per={p:metrics(truth[p],packed[p]) for p in sorted(truth.files)}
        results[name]={
            "individual_eight_new_dev_prompt_diagnostics":per,
            "mean_held_out_from_calibration_KL":
                math.fsum(t["full_vocab_teacher_to_compressed_KL_nats"]
                          for t in per.values())/len(per),
            "mean_normalized_output_logit_RMSE":
                math.fsum(t["normalized_logit_RMSE"]
                          for t in per.values())/len(per),
            "matched_teacher_next_top1_count":sum(
                x["same_teacher_top1_token"] for x in per.values())}
    delta=(results["CB61_activation_error_syndrome"]["mean_held_out_from_calibration_KL"]-
           results["base_CB59"]["mean_held_out_from_calibration_KL"])
    gate=(physical["candidate_CB61_full_single_crc_sha_file"]["physical_bytes"]<=
          physical["hard_0_70bpw_checkpoint_max_bytes"] and
          all(x["hot_dense_pretrained_original_2D_weight_count"]==0 and
              x["full_real_pretrained_model_direct_and_KV_completed"]
              for x in runtime.values()))
    report={
        "experiment":"CALIBRI_v61_activation_guided_coded_residual_sub07_direct_135m",
        "real_pretrained_model":MODEL,"pinned_revision":rev,
        "strict_full_model_rate_and_direct_inference_PASS":gate,
        "real_complete_both_source_checkpoints":physical,
        "teacher_process_source":teacher,
        "teacher_free_native_packed_model_evidence":runtime,
        "eight_predeclared_new_output_diagnostics":results,
        "CB61_minus_CB59_mean_original_output_KL":delta,
        "CB61_reduces_mean_original_output_KL":delta<0,
        "no_heldout_data_was_used_to_fit_source_corrections":True,
        "limitations":[
            "FEC analogy: lost signal cannot be recovered by checksum parity alone, selected source error columns are explicit paid residual data.",
            "Even improved output KL on eight short prompts does NOT prove instruction quality or recovery from severe sub0.7 compression loss.",
            "Both whole-trained weights files remain <0.7 physical bpw, entire original dense W never restored in packed runtime.",
            "Current native LUT GEMV and Python sparse-vector correction unoptimized; speed is scientific exploration only.",
            "No live Jessica chat, production main, Render, Supabase, APK or settings modified."
        ]}
    (folder/"calibri_v61.json").write_text(json.dumps(
        report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "v61":"ACTUAL_TRIED_HARD_ERROR_SYNDROME_0_7BPW",
        "PASS":gate,
        "actual_bpw":physical["candidate_CB61_full_single_crc_sha_file"]["actual_effective_bpw"],
        "paid_syndrome_corrections":physical["candidate_CB61_full_single_crc_sha_file"]["selected_columns"],
        "original_CB59_KL":results["base_CB59"]["mean_held_out_from_calibration_KL"],
        "new_CB61_KL":results["CB61_activation_error_syndrome"]["mean_held_out_from_calibration_KL"],
        "mean_KL_delta":delta,
        "corrected_improves_on_8":delta<0,
        "corrected_same_teacher_top1_out_of8":results[
            "CB61_activation_error_syndrome"]["matched_teacher_next_top1_count"]},
        ensure_ascii=False),flush=True)
    if not gate:
        raise AssertionError("not an actual fully packed <0.7bpw trained source")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",required=True,choices=("encode","teacher","packed","compare"))
    ap.add_argument("--output",default="results")
    ap.add_argument("--source-revision",default=MODEL_SHA)
    args=ap.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    if args.source_revision!=MODEL_SHA:
        raise AssertionError("different pretrained source version invalidates comparison")
    {"encode":stage_encode,"teacher":stage_teacher,
     "packed":stage_packed,"compare":stage_compare}[args.stage](
        dest,args.source_revision)


if __name__=="__main__":
    main()
