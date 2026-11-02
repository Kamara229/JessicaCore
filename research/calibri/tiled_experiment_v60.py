#!/usr/bin/env python3
"""CALIBRI v60 physically verified 256-row localized codebooks, real Llama.

One strictly source-only algorithmic change versus v59: each source
matrix's original shared dictionary -> per-256-row local dictionaries.
The original source and three independent runtime routes are isolated:
encode reads trained weights; teacher loads source alone; packed loads
only verified CB59/CB60 files and from_config model; compare uses JSON.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from shared_checkpoint_v58 import split_aliases
from direct_codebook_v59 import (
    CB59,encode as encode_control,smoke as control_smoke)
from tiled_codebook_v60 import (
    CB60,encode as encode_tiled,smoke as tile_smoke)
from direct_runtime_v59 import (
    compile_kernel as compile_control,
    NativeKernel as ControlKernel,bind_into_llama as bind_control)
from tiled_runtime_v60 import (
    compile_kernel as compile_tiled,
    NativeKernel as TiledKernel,bind_into_llama as bind_tiled,
    smoke as tiled_kernel_smoke)
from sub07_experiment_v59 import (
    MODEL,EXPECTED_UNIQUE,MAX_BPW,actual_forward,
    setup,physical_evidence)
from huggingface_hub import HfApi

PROMPTS=(
    "Please explain in simple words why the sky appears blue on clear days.",
    "A librarian returns books to their shelves and then helps a visitor find",
    "One plus two equals three, while seven plus five equals",
    "In a hospital, a nurse checks the patient's temperature before the doctor",
)
ORIGINAL_CB59_RMSE_REFERENCE=0.7209552288961534
TOKENS=8


def configuration(rev):
    cfg=AutoConfig.from_pretrained(MODEL,revision=rev)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev)
    return cfg,tok


def fixed_ids(tok,text):
    ids=tok(text,add_special_tokens=False,return_tensors="pt").input_ids
    if ids.shape[1]<TOKENS:
        raise AssertionError("predeclared first-use development prompt too short")
    return ids[:,:TOKENS].contiguous()


def checksum(p):
    d=p.read_bytes()
    return {"file":p.name,"sha256":hashlib.sha256(d).hexdigest(),
            "physical_bytes":len(d),"original_source_tensors_were_all_stored":True}


def stage_encode(folder,rev):
    setup()
    tile_smoke()
    control_smoke()
    tiled_kernel_smoke()
    original=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=rev,torch_dtype=torch.float32).eval()
    unique,aliases=split_aliases(original.state_dict())
    count=sum(p.numel() for p in original.parameters())
    if count!=EXPECTED_UNIQUE or sum(t.numel() for t in unique.values())!=count:
        raise AssertionError("complete unique pretrained model count altered")
    if aliases!={"lm_head.weight":"model.embed_tokens.weight"}:
        raise AssertionError("wrong tied 135M source aliases")
    t0=time.perf_counter()
    baseline,baseline_meta,baseline_rmse=encode_control(original.state_dict())
    baseline_s=time.perf_counter()-t0
    t0=time.perf_counter()
    candidate,candidate_meta,candidate_rmse=encode_tiled(original.state_dict())
    candidate_s=time.perf_counter()-t0
    control_file=folder/"CALIBRI_v60_control_unchanged_CB59.cb59"
    candidate_file=folder/"CALIBRI_v60_candidate_tiled_CB60.cb60"
    control_file.write_bytes(baseline)
    candidate_file.write_bytes(candidate)
    # Read again, do not trust in-memory source estimates.
    pcontrol=CB59(control_file.read_bytes(),expected_keys=original.state_dict().keys())
    ptiled=CB60(candidate_file.read_bytes(),expected_keys=original.state_dict().keys())
    for src,handle in ((baseline,pcontrol),(candidate,ptiled)):
        if len(src)!=handle.physical_bytes():
            raise AssertionError("corrupted physical trained source length")
        changed=bytearray(src);changed[len(changed)//2]^=1
        try:type(handle)(bytes(changed))
        except ValueError:pass
        else:raise AssertionError("source checkpoint corrupt bit accepted")
    cap=math.floor(MAX_BPW*count/8)
    if len(candidate)>cap:
        raise AssertionError(f"full actual 256-row-tiled CB60 rate fails {len(candidate)}>{cap}")
    if len(baseline)>cap:
        raise AssertionError("original verified v59 source no longer <=0.70bpw")
    info={
        "experiment":"CALIBRI_v60_row_tiled_source_codebooks_one_change",
        "trained_model":MODEL,"source_revision":rev,
        "original_unique_source_parameters":count,
        "source_tied_aliases":aliases,
        "native_5bit_indices_per_eight_original_weights":True,
        "source_only_training_no_task_labels":True,
        "predeclared_development_prompts_SHA256":hashlib.sha256(
             "\n".join(PROMPTS).encode()).hexdigest(),
        "hard_full_model_max_physical_bytes_0_70bpw":cap,
        "control_v59":{"file":checksum(control_file),
             "relative_source_RMSE":baseline_rmse["full_source_weighted_relative_RMSE"],
             "full_original_source_2D_matrix_count":baseline_rmse["full_source_intact_weight_matrices"],
             "source_5bit_code_physical_bytes":baseline_rmse["total_real_5bit_code_bytes"],
             "fit_seconds":baseline_s},
        "candidate_CB60":{"file":checksum(candidate_file),
             "relative_source_RMSE":candidate_rmse["full_source_weighted_relative_RMSE"],
             "actual_row_tile_codebook_count":candidate_rmse["total_row_tile_codebooks"],
             "full_original_source_2D_matrix_count":candidate_rmse["full_source_intact_weight_matrices"],
             "source_5bit_code_physical_bytes":candidate_rmse["total_real_5bit_code_bytes"],
             "fit_seconds":candidate_s},
        "relative_source_RMSE_improved":candidate_rmse[
            "full_source_weighted_relative_RMSE"]<baseline_rmse[
            "full_source_weighted_relative_RMSE"],
        "relative_source_RMSE_delta":candidate_rmse[
            "full_source_weighted_relative_RMSE"]-baseline_rmse[
            "full_source_weighted_relative_RMSE"],
        "candidate_actual_bpw":len(candidate)*8/count,
        "control_actual_bpw":len(baseline)*8/count,
        "source_weights_only_in_this_encode_process":True}
    (folder/"calibri_v60_encode.json").write_text(json.dumps(
        info,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"v60":"TWO_COMPLETE_REAL_PHYSICAL_SOURCES",
        "control_bpw":info["control_actual_bpw"],
        "tiled_bpw":info["candidate_actual_bpw"],
        "original_source_rmse":info["control_v59"]["relative_source_RMSE"],
        "tiled_source_rmse":info["candidate_CB60"]["relative_source_RMSE"],
        "local_codebooks":info["candidate_CB60"]["actual_row_tile_codebook_count"]},
        ensure_ascii=False),flush=True)


def evaluate(model,tok,identifier):
    vectors={}
    measurements={}
    for index,prompt in enumerate(PROMPTS):
        logits,record=actual_forward(model,fixed_ids(tok,prompt))
        vectors[f"p{index}"]=logits
        measurements[f"p{index}"]={
            "actual_forward_seconds":record[
              "real_8token_full_model_prefill_and_one_KV_decode_seconds"],
            "top1_token":record["selected_next_token"],
            "KV_cache_second_token_completed":record[
              "autoregressive_KV_cache_forward_completed"]}
    return vectors,measurements


def stage_teacher(folder,rev):
    setup()
    cfg,tok=configuration(rev)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=rev,torch_dtype=torch.float32).eval()
    logits,info=evaluate(teacher,tok,"uncompressed_teacher")
    np.savez(folder/"calibri_v60_teacher_logits.npz",**logits)
    (folder/"calibri_v60_teacher.json").write_text(json.dumps({
         "pretrained_source_model_SHA":rev,
         "measured_prompt_full_model_and_KV":info,
         "teacher_parameters":sum(p.numel() for p in teacher.parameters()),
         "never_read_compressed_source_files":True},indent=2)+"\n")
    print({"v60":"ORIGINAL_INSTRUCT_TEACHER_FOUR_FROZEN_NEW_PROMPTS"},
          flush=True)


def packed_model(cfg,codec,kernel,bind):
    m=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32)
    evidence=bind(m,codec,kernel)
    if sum(p.numel() for p in m.parameters() if p.ndim>=2):
        raise AssertionError("pretrained full 2D dense source leaked into packed runtime")
    if not evidence["tied_embed_and_output_codebook_SHARED_IN_RAM"]:
        raise AssertionError("tied pretrained 135M embeddings are duplicated in memory")
    return m,evidence


def stage_packed(folder,rev):
    setup()
    cfg,tok=configuration(rev)
    cases=(
        ("control_v59","CALIBRI_v60_control_unchanged_CB59.cb59",
          CB59,ControlKernel,compile_control,bind_control,
          Path(__file__).with_name("direct_kernel_v59.c")),
        ("candidate_CB60","CALIBRI_v60_candidate_tiled_CB60.cb60",
          CB60,TiledKernel,compile_tiled,bind_tiled,
          Path(__file__).with_name("tiled_kernel_v60.c")))
    measured={}
    for label,name,codec_type,kernel_type,compile_func,binder,src in cases:
        path=folder/name
        codec=codec_type(path.read_bytes())
        kernel=kernel_type(compile_func(src))
        model,evidence=packed_model(cfg,codec,kernel,binder)
        logits,record=evaluate(model,tok,label)
        np.savez(folder/("calibri_v60_"+label+"_logits.npz"),**logits)
        measured[label]={
            "full_trained_physical_SHA256":codec.hash(),
            "full_physical_bytes_resident_compressed":codec.physical_bytes(),
            "loaded_original_pretrained_teacher_in_this_process":False,
            "source_model_original_2D_dense_hot_weights":0,
            "native_C_5bit_direct_LUT_GEMV":True,
            "native_C_embedding_decodes_only_requested_rows":True,
            "full_pretrained_transformer_and_KV_completed":True,
            "packed_module_evidence":evidence,
            "per_frozen_prompt":record}
        del model,codec,kernel
        gc.collect()
    (folder/"calibri_v60_packed.json").write_text(json.dumps(
        measured,indent=2,ensure_ascii=False)+"\n")
    print({"v60":"BOTH_PACKED_135M_DIRECT_MODELS_COMPLETED_FOUR_PROMPTS",
           "control_modules":measured["control_v59"][
             "packed_module_evidence"]["compressed_real_2D_module_count"],
           "tiled_modules":measured["candidate_CB60"][
             "packed_module_evidence"]["compressed_real_2D_module_count"]},
          flush=True)


def output_error(x,y):
    if x.shape!=y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise AssertionError("invalid real frozen 135M model output logits")
    xt=torch.from_numpy(x.astype(np.float64))
    yt=torch.from_numpy(y.astype(np.float64))
    p=xt.log_softmax(dim=-1).exp()
    kl=float(torch.sum(p*(xt.log_softmax(dim=-1)-
                          yt.log_softmax(dim=-1))).item())
    sq=np.mean(np.square(x.astype(np.float64)-y.astype(np.float64)))
    energy=np.mean(np.square(x.astype(np.float64)))
    return {"teacher_to_packed_output_KL_nats":kl,
            "full_vocab_normalized_logit_RMSE":float(math.sqrt(sq/(energy+1e-18))),
            "same_top1_token":bool(np.argmax(x)==np.argmax(y)),
            "finite_real_logits":True}


def stage_compare(folder,rev):
    source=json.loads((folder/"calibri_v60_encode.json").read_text())
    teacher=json.loads((folder/"calibri_v60_teacher.json").read_text())
    runtime=json.loads((folder/"calibri_v60_packed.json").read_text())
    t=np.load(folder/"calibri_v60_teacher_logits.npz")
    results={}
    for label in ("control_v59","candidate_CB60"):
        compressed=np.load(folder/("calibri_v60_"+label+"_logits.npz"))
        per={}
        for i in range(len(PROMPTS)):
            key=f"p{i}"
            per[key]=output_error(t[key],compressed[key])
        mean_kl=math.fsum(row["teacher_to_packed_output_KL_nats"]
                          for row in per.values())/len(per)
        mean_rmse=math.fsum(row["full_vocab_normalized_logit_RMSE"]
                            for row in per.values())/len(per)
        correct_top1=sum(r["same_top1_token"] for r in per.values())
        results[label]={
            "all_first_use_development_prompt_outputs":per,
            "mean_full_vocab_KL":mean_kl,
            "mean_normalized_logit_RMSE":mean_rmse,
            "same_teacher_top1_count_out_of_4":correct_top1,
            "probe_passes_finite_full_transformer_KV":True}
    mean_delta=results["candidate_CB60"]["mean_full_vocab_KL"]-results[
       "control_v59"]["mean_full_vocab_KL"]
    all_rate_source=(source["candidate_CB60"]["file"]["physical_bytes"]<=
         source["hard_full_model_max_physical_bytes_0_70bpw"])
    all_runtime=all(p["full_pretrained_transformer_and_KV_completed"] and
       p["source_model_original_2D_dense_hot_weights"]==0
       for p in runtime.values())
    rt={}
    for label in ("control_v59","candidate_CB60"):
        per=runtime[label]["per_frozen_prompt"]
        ref=teacher["measured_prompt_full_model_and_KV"]
        rt[label]={k:per[k]["actual_forward_seconds"]/ref[k]["actual_forward_seconds"]
                    for k in per}
    valid=all_rate_source and all_runtime
    report={
        "experiment":"CALIBRI_v60_256row_tiled_codebooks_controlled_direct_full_model",
        "first_stage_rate_and_native_direct_execution_PASS":valid,
        "real_135M_pretrained_model":MODEL,
        "source_model_revision":rev,
        "entire_2D_trained_weight_matrices_stay_packed":True,
        "source_only_physical_codec":source,
        "two_independent_teacher_free_direct_runtime":runtime,
        "same_source_actual_teacher_four_new_dev_prompts":teacher,
        "matched_original_teacher_logit_metrics":results,
        "source_only_reconstruction_improved_v59_to_v60":
            source["relative_source_RMSE_improved"],
        "new_four_prompt_mean_KL_improved_v59_to_v60":mean_delta<0,
        "new_four_prompt_mean_KL_delta_tiled_minus_original_v59":mean_delta,
        "exploratory_direct_CPU_speed_over_teacher_by_prompt":rt,
        "limitations":[
            "Only SOURCE-trained dictionaries; no calibration on activation response.",
            "Output KL and top1 on four frozen small development prompts do not establish semantic accuracy or no fatal model error.",
            "Independent QA task correctness and 128+32 warm speed must be tested after a quality-positive format is found.",
            "Direct CPU C LUT GEMV remains unoptimized and can be much slower than dense BLAS.",
            "All unique original pretrained tensors and source aliases are counted in true physical <=0.7bpw archive.",
            "No main/Render/Supabase/APK/Jessica ordinary chat/settings/secrets modified."
        ]}
    (folder/"calibri_v60.json").write_text(json.dumps(
        report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"v60":"MEASURED_COMPLETE_DIRECT_TILED_135M_MODEL",
        "PASS":valid,
        "full_model_bpw":source["candidate_actual_bpw"],
        "control_v59_weight_RMSE":source["control_v59"]["relative_source_RMSE"],
        "new_tiled_weight_RMSE":source["candidate_CB60"]["relative_source_RMSE"],
        "control_v59_mean_KL":results["control_v59"]["mean_full_vocab_KL"],
        "tiled_mean_KL":results["candidate_CB60"]["mean_full_vocab_KL"],
        "tiled_improves_KL":mean_delta<0,
        "matched_teacher_top1_new_tiled":results["candidate_CB60"][
            "same_teacher_top1_count_out_of_4"]},ensure_ascii=False),flush=True)
    if not valid:
        raise AssertionError("source-only 0.7bpw actual or direct packed runtime failed")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",required=True,
                    choices=("encode","teacher","packed","compare"))
    ap.add_argument("--output",default="results")
    ap.add_argument("--source-revision",default=None)
    args=ap.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    rev=args.source_revision or HfApi().model_info(MODEL).sha
    {"encode":stage_encode,"teacher":stage_teacher,
     "packed":stage_packed,"compare":stage_compare}[args.stage](dest,rev)


if __name__=="__main__":
    main()
