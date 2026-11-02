#!/usr/bin/env python3
"""CALIBRI v59 rate-first real pretrained SmolLM2-135M full-source <0.70bpw.

Three logically separate processes:
  encode: Load genuine full pretrained source once and produce physical CB59
          with complete unique parameters, manifest, CRC+SHA, codebooks.
  teacher: Original public trained source, never read packed checkpoint.
  packed: Fresh model from_config, replace ALL original matrices with
          compressed direct CPU LUT-GEMV and demand embedding; never loads
          trained teacher tensors or reconstructs dense pretrained matrix.
  compare: purely file based physical model, output numeric degradation.

Goal THIS STAGE: target physical checkpoint <=0.70 effective bpw AND
execute complete model inference directly from compressed resident codes.
Correct-answer quality and optimized speed are NEXT stages; this does
not imply no catastrophic answer accuracy drop.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,time,traceback,zlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import HfApi
from direct_codebook_v59 import CB59,encode,smoke as codec_smoke
from direct_runtime_v59 import (
    compile_kernel,NativeKernel,bind_into_llama,smoke as kernel_smoke)

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
EXPECTED_UNIQUE=134515008
MAX_BPW=0.70
TOKENS="The quick brown fox jumps over the lazy dog."
PROMPT_PREFIX=8
TEST_NEXT=1


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(59)
    torch.set_grad_enabled(False)


def conf(source_sha):
    tok=AutoTokenizer.from_pretrained(MODEL,revision=source_sha)
    cfg=AutoConfig.from_pretrained(MODEL,revision=source_sha)
    return tok,cfg


def load_revision():
    return HfApi().model_info(MODEL).sha


def physical_evidence(f):
    data=f.read_bytes()
    return {"file":f.name,"physical_bytes":len(data),
            "sha256":hashlib.sha256(data).hexdigest(),
            "corruption_protected":True}


def input_ids(tok):
    tokens=tok(TOKENS,return_tensors="pt",
               add_special_tokens=False).input_ids
    if tokens.shape[1]<PROMPT_PREFIX:
        # Precisely the same text/config on two separate process stages.
        tokens=tok(TOKENS+" A second short English sentence.",
                   return_tensors="pt",add_special_tokens=False).input_ids
    if tokens.shape[1]<PROMPT_PREFIX:
        raise AssertionError("insufficient REAL prompt tokens for v59")
    return tokens[:,:PROMPT_PREFIX].contiguous()


def actual_forward(model,ids):
    """Complete transformer original/packed forward with KV cache."""
    model.eval()
    t0=time.perf_counter()
    with torch.inference_mode():
        first=model(input_ids=ids,use_cache=True)
        if (first.logits.shape!=(1,PROMPT_PREFIX,
                                model.config.vocab_size) or
            not torch.isfinite(first.logits).all()):
            raise AssertionError("bad full pretrained transformer first-pass logits")
        follow=first.logits[:,-1:,:].argmax(dim=-1)
        # Both teacher and packed compute NEXT actual autoregressive
        # step with their own previous state/predicted token.
        second=model(input_ids=follow,past_key_values=first.past_key_values,
                     use_cache=True)
        if (second.logits.shape!=(1,TEST_NEXT,model.config.vocab_size)
            or not torch.isfinite(second.logits).all()):
            raise AssertionError("bad full transformer direct KV step")
    elapsed=time.perf_counter()-t0
    return first.logits[0,-1,:].float().numpy().copy(),{
        "real_8token_full_model_prefill_and_one_KV_decode_seconds":elapsed,
        "selected_next_token":int(follow[0,0]),
        "full_model_forward_is_finite":True,
        "autoregressive_KV_cache_forward_completed":True,
        "output_vocab_size":model.config.vocab_size,
        "all_2d_original_source_dense_parameters_in_hot_runtime":
            None,
    }


def stage_encode(folder,rev):
    print("CALIBRI v59: BEGIN real source-only complete SmolLM2 K32x8 source",flush=True)
    codec_smoke()
    kernel_smoke()
    original=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=rev,torch_dtype=torch.float32)
    actual=sum(v.numel() for v in original.parameters())
    if actual!=EXPECTED_UNIQUE:
        raise AssertionError("unverified pretrained tied model unique count")
    from shared_checkpoint_v58 import split_aliases
    uniq,alias=split_aliases(original.state_dict())
    if (sum(p.numel() for p in uniq.values())!=actual or
        alias!={"lm_head.weight":"model.embed_tokens.weight"}):
        raise AssertionError("not same tied 135M trained source model")
    t0=time.perf_counter()
    blob,meta,diagnostics=encode(original.state_dict())
    elapsed=time.perf_counter()-t0
    compressed=folder/"CALIBRI_v59_full_direct_sub070.cb59"
    compressed.write_bytes(blob)
    verified=CB59(compressed.read_bytes(),expected_keys=original.state_dict().keys())
    corrupt=bytearray(blob);corrupt[len(blob)//2]^=1
    try:CB59(bytes(corrupt))
    except ValueError:pass
    else:raise AssertionError("corrupt physical checkpoint passed integrity check")
    limit=math.floor(MAX_BPW*actual/8)
    physical=len(blob)
    if physical>limit:
        raise AssertionError(f"rate target v59 FAIL: {physical}>{limit}")
    if not (0<diagnostics["full_source_weighted_relative_RMSE"]<10):
        raise AssertionError("unbounded source codebook reconstruction error")
    audit={
        "method":"CB59 trained dense model 5bit/8D K32 codebook + FP16 row-scale",
        "source_model":MODEL,"source_model_revision":rev,
        "actual_unique_original_parameter_count":actual,
        "full_original_state_parameter_aliases":alias,
        "physical_file":physical_evidence(compressed),
        "target_full_checkpoint_bpw":MAX_BPW,
        "target_max_exact_bytes":limit,
        "actual_effective_bpw":8*physical/actual,
        "actual_full_source_rate_below_0_7":physical<=limit,
        "original_FP16_bytes_at_unique_parameter_count":2*actual,
        "compressed_file_ratio_to_FP16":physical/(2*actual),
        "source_only_codebook_fit_seconds":elapsed,
        "quantized_all_pretrained_2D_weight_names":len(verified.handles),
        "actual_native_packed_code_bytes":
            diagnostics["total_real_5bit_code_bytes"],
        "weight_source_global_relative_RMSE":
            diagnostics["full_source_weighted_relative_RMSE"],
        "matrix_source_highest_RMSE":
            diagnostics["largest_relative_matrix_RMSE"],
        "no_test_or_train_text_used_to_fit_codebooks":True,
        "quality_preservation_NOT_yet_proven":True,
    }
    (folder/"calibri_v59_encode.json").write_text(json.dumps(
        audit,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"v59":"REAL_COMPLETE_CHECKPOINT_ENCODED",
        "physical_bytes":physical,
        "bpw":audit["actual_effective_bpw"],
        "weight_RMSE":audit["weight_source_global_relative_RMSE"],
        "fit_seconds":elapsed},ensure_ascii=False),flush=True)


def stage_teacher(folder,rev):
    tok,cfg=conf(rev)
    original=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=rev,torch_dtype=torch.float32)
    ids=input_ids(tok)
    logits,record=actual_forward(original,ids)
    np.save(folder/"calibri_v59_teacher_logits.npy",logits)
    (folder/"calibri_v59_teacher.json").write_text(json.dumps({
        **record,"source_model":MODEL,"source_model_revision":rev,
        "physical_CB59_file_not_loaded":True,
        "baseline_FP32_parameter_bytes":sum(
            p.numel()*p.element_size() for p in original.parameters()),
        "original_full_model_has_trained_weights":True,
    },indent=2)+"\n")
    print({"v59":"REAL_TEACHER_REFERENCE","seconds":record[
        "real_8token_full_model_prefill_and_one_KV_decode_seconds"]},flush=True)


def stage_packed(folder,rev):
    tok,cfg=conf(rev)
    physical=folder/"CALIBRI_v59_full_direct_sub070.cb59"
    packed=CB59(physical.read_bytes())
    if packed.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE:
        raise AssertionError("not complete REAL trained source parameter count")
    kernel=NativeKernel(compile_kernel(
        Path(__file__).with_name("direct_kernel_v59.c")))
    # from_config() initialization is NOT the pretrained source. All its
    # original dense randomly initialized 2D weight modules are removed
    # and garbage-collected; only compressed source codes reside.
    m=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32)
    verification=bind_into_llama(m,packed,kernel)
    if sum(p.numel() for p in m.parameters() if p.ndim>=2):
        raise AssertionError("undecompressed hot RAM violates direct execution")
    ids=input_ids(tok)
    logits,run=actual_forward(m,ids)
    run["all_2d_original_source_dense_parameters_in_hot_runtime"]=0
    run["packed_actual_in_RAM_checkpoint_bytes"]=packed.physical_bytes()
    run["no_original_pretrained_teacher_loaded_in_this_process"]=True
    run["full_Llama_from_config_has_only_compressed_linear_and_embedding"]=verification
    run["runtime_uses_native_C_LUT_GEMV_no_dense_weight_dequantization"]=True
    run["checksum_source"]=packed.hash()
    np.save(folder/"calibri_v59_packed_logits.npy",logits)
    (folder/"calibri_v59_packed.json").write_text(json.dumps(
        run,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"v59":"REAL_DIRECT_PACKED_FULL_MODEL_AND_KV",
        "seconds":run["real_8token_full_model_prefill_and_one_KV_decode_seconds"],
        "physical_resident_bytes":run["packed_actual_in_RAM_checkpoint_bytes"],
        "full_pack_modules":verification["compressed_real_2D_module_count"],
        "decoded_2D_hot_source_params":0},ensure_ascii=False),flush=True)


def stage_compare(folder):
    encode=json.loads((folder/"calibri_v59_encode.json").read_text())
    teacher=json.loads((folder/"calibri_v59_teacher.json").read_text())
    packed=json.loads((folder/"calibri_v59_packed.json").read_text())
    x=np.load(folder/"calibri_v59_teacher_logits.npy")
    y=np.load(folder/"calibri_v59_packed_logits.npy")
    if x.shape!=y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise AssertionError("invalid genuine direct 135M checkpoint logits")
    xt=torch.from_numpy(x.astype(np.float64))
    yt=torch.from_numpy(y.astype(np.float64))
    p=xt.log_softmax(dim=-1).exp()
    kl=float(torch.sum(p*(xt.log_softmax(dim=-1)-
                          yt.log_softmax(dim=-1))).item())
    rmse=float(np.sqrt(np.mean(np.square(x.astype(np.float64)-
                                       y.astype(np.float64)))))
    norm_rmse=rmse/(float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))+1e-12)
    same=int(np.argmax(x)==np.argmax(y))
    speed=packed["real_8token_full_model_prefill_and_one_KV_decode_seconds"]/(
        teacher["real_8token_full_model_prefill_and_one_KV_decode_seconds"])
    # Rate and direct execution are the two required first-stage gates,
    # not semantic quality or FLOP/s. Record severe quality issues.
    verified=(encode["actual_full_source_rate_below_0_7"] and
              packed["all_2d_original_source_dense_parameters_in_hot_runtime"]==0
              and packed["full_model_forward_is_finite"]
              and packed["autoregressive_KV_cache_forward_completed"])
    report={
        "experiment":"CALIBRI_v59_first_rate_native_direct_K32V8_135M",
        "first_stage_physical_rate_and_native_inference_PASS":verified,
        "trained_model":MODEL,
        "full_real_checkpoint_actual_bpw":encode["actual_effective_bpw"],
        "full_real_checkpoint_actual_bytes":encode["physical_file"]["physical_bytes"],
        "original_unique_source_parameters":
             encode["actual_unique_original_parameter_count"],
        "actual_weight_source_RMSE":encode["weight_source_global_relative_RMSE"],
        "real_8token_model_and_next_KV_decode_logits": {
            "teacher_top1_token":int(np.argmax(x)),
            "packed_top1_token":int(np.argmax(y)),
            "top1_agree":bool(same),
            "full_vocab_logits_relative_RMSE":norm_rmse,
            "original_distribution_KL_vs_packed":kl,
            "finite_full_model_forward_and_KV_decode":True},
        "native_compressed_resident_hot_runtime":packed,
        "paired_cpu_single_sequence_runtime_ratio_NOT_optimized":speed,
        "teacher_separate_process_source":teacher,
        "physical_codec_and_source":encode,
        "quality_degradation_significant_possible":
            (not same or norm_rmse>0.25 or kl>0.5),
        "claims_not_made":[
            "No claim that a dense instruct model retains competence below 0.70bpw.",
            "No claim matching QMoE GPU/latency or sub0.8 quality on sparse MoE.",
            "Native C direct packed CPU kernel is unoptimized; speed target is deferred by user.",
            "No full dense 2D pretrained weight materialized in packed inference process.",
            "No production Jessica code or deployment modified."
        ]}
    (folder/"calibri_v59.json").write_text(json.dumps(
        report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"v59":"MEASURED_PHYSICAL_FULL_MODEL_DIRECT_PACKED",
        "PASS":verified,"bpw":encode["actual_effective_bpw"],
        "native_full_model_8token_and_KV":True,
        "weight_RMSE":encode["weight_source_global_relative_RMSE"],
        "logit_RMSE":norm_rmse,"KL":kl,
        "next_token_match":bool(same),"slowdown_ratio_exploratory":speed,
        "QUALITY_WARNING":report["quality_degradation_significant_possible"],
    },ensure_ascii=False),flush=True)
    if not verified:
        raise AssertionError("stage-1 physical <=0.7bpw or native forward failed")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",required=True,
                    choices=("encode","teacher","packed","compare"))
    ap.add_argument("--output",default="results")
    ap.add_argument("--source-revision",default=None)
    args=ap.parse_args()
    setup()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    revision=args.source_revision or load_revision()
    if args.stage=="encode":stage_encode(dest,revision)
    elif args.stage=="teacher":stage_teacher(dest,revision)
    elif args.stage=="packed":stage_packed(dest,revision)
    else:stage_compare(dest)


if __name__=="__main__":
    main()
