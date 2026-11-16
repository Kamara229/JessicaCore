#!/usr/bin/env python3
"""S4: source-weight-only 8192-sample / 12-iteration CB59, same 0.625bpw ABI.

No EVAL labels, no G-T activation training, no FP32 student at inference.
Measure whether better weights actually restore *outputs*, not just L2.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import direct_codebook_v59 as direct
from direct_codebook_v59 import CB59, unpack5
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from v176.operators import clear
from v196.recover import build_trainable_from_paid_source
from calibri2.independent_135m.s2.rootcause import (
    PROMPTS, measure, prompt_ids,
)

SOURCE_SHA = "50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
EXPECTED_BYTES = 11229303
REFINE_SAMPLE=8192
REFINE_ITERS=12


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,
                           ensure_ascii=False)+"\n",encoding="utf-8")


def emit(stage,**values):
    print(json.dumps({"stage":stage,**values},ensure_ascii=False),flush=True)


def decoded(handle):
    m,n=handle.shape
    ids=unpack5(handle.packed.tobytes(),m*n//8).astype(np.int64)
    rows=(handle.centroids[ids].reshape(m,n).astype(np.float32)*
          handle.row_scales.astype(np.float32)[:,None])
    if not np.isfinite(rows).all():
        raise AssertionError("Corrupted original FP32 quantized source")
    return rows


def fp32_weight_error(teacher_state,base,changed):
    old_sse,new_sse,energy=0.,0.,0.
    matrix_delta={}
    for name in sorted(base.handles):
        ref=teacher_state[name].detach().cpu().numpy().astype(np.float32)
        a=decoded(base.handles[name])
        b=decoded(changed.handles[name])
        if ref.shape!=a.shape or ref.shape!=b.shape:
            raise AssertionError("Original weight structure changed")
        a_error=np.sum(np.square((a-ref).astype(np.float64)))
        b_error=np.sum(np.square((b-ref).astype(np.float64)))
        total=np.sum(np.square(ref.astype(np.float64)))
        old_sse+=float(a_error)
        new_sse+=float(b_error)
        energy+=float(total)
        matrix_delta[name]={
            "shape":list(ref.shape),
            "base_FP32_relative_RMSE":float(np.sqrt(a_error/max(total,1e-20))),
            "larger_sample_FP32_relative_RMSE":float(np.sqrt(b_error/max(total,1e-20))),
            "larger_sample_over_original_matrix_weight_sse":
                  float(b_error/max(a_error,1e-20))}
    return {
        "base_relative_RMSE":float(np.sqrt(old_sse/energy)),
        "source_only_refit_relative_RMSE":float(np.sqrt(new_sse/energy)),
        "sse_refit_ratio":float(new_sse/old_sse),
        "matrix_diagnostics":matrix_delta
    }


@torch.inference_mode()
def representative_hidden(fp32,student,src,ctx):
    clear(src)
    ori=fp32(input_ids=ctx,use_cache=False,
             output_hidden_states=True)
    actual=student(input_ids=ctx,use_cache=False,
                   output_hidden_states=True)
    results={}
    for layer in (0,1,10,20,30):
        a=ori.hidden_states[layer][0,-1].float()
        b=actual.hidden_states[layer][0,-1].float()
        results[str(layer)]={
            "cosine":float(torch.nn.functional.cosine_similarity(
                a[None],b[None]).item()),
            "relative_l2":float(torch.linalg.vector_norm(a-b)/
                                torch.linalg.vector_norm(a).clamp_min(1e-8))
        }
    af=int(ori.logits[0,-1].argmax())
    bf=int(actual.logits[0,-1].argmax())
    clear(src)
    return {"hidden_layers":results,
            "same_fp32_first_token":af==bf}


def run(source_path,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(13541117)
    torch.set_num_threads(2)
    srcraw=Path(source_path).read_bytes()
    if sha(srcraw)!=SOURCE_SHA or len(srcraw)!=EXPECTED_BYTES:
        raise AssertionError("S4 must begin from authentic original trained 135M original source")
    old=CB59(srcraw)
    before={
      "original_FP32_model_name":MODEL,
      "original_FP32_revision":MODEL_SHA,
      "baseline_source_SHA":SOURCE_SHA,
      "refit_5bit_indices_per_8vector":True,
      "refit_codebook_entries":32,
      "larger_source_vector_sample_count":REFINE_SAMPLE,
      "larger_integer_nearest_refinement_iterations":REFINE_ITERS,
      "precommitted_debug_TRAIN_only_questions":list(PROMPTS),
      "no_new_heldout_QA_or_TEST_used":True,
      "no_change_to_runtime_5bit_or_fp16_centroid_decoder":True,
      "no_NRA_and_no_FP32_teacher_in_inference":True,
    }
    save(out/"S4_SOURCE_RECODE_PRECOMMIT_BEFORE_FP32_LOAD.json",before)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    fp32=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).cpu().eval()
    state=fp32.state_dict()
    if sum(x.numel() for x in fp32.parameters())!=EXPECTED_UNIQUE:
        raise AssertionError("Wrong original pretrained 135M")
    # Exact same K32-V8 ABI. Only source-only deterministic dictionary fit
    # capacity is increased. Source-only weights do NOT see any questions.
    old_sample=direct._source_sample
    old_iterations=direct.KMEANS_ITERS
    try:
        direct.KMEANS_ITERS=REFINE_ITERS
        direct._source_sample=lambda w,scales:old_sample(
             w,scales,sample_count=REFINE_SAMPLE)
        candidate,meta,codec=direct.encode(state)
    finally:
        direct.KMEANS_ITERS=old_iterations
        direct._source_sample=old_sample
    if (len(candidate)!=EXPECTED_BYTES or
        meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE or
        not all(d["dtype"] in ("CB59_K32V8","f4") for d in
                meta["tensor_descriptors"].values())):
        raise AssertionError("Source-only refit violated CB59 exact budget/schema")
    new=CB59(candidate)
    if new.meta!=old.meta or set(new.handles)!=set(old.handles):
        raise AssertionError("Mixed metadata/alias source during 135M source-only recoding")
    if (new.handles["model.embed_tokens.weight"] is not
            new.handles["lm_head.weight"]):
        raise AssertionError("Re-encoding must retain original tied embeddings")
    for name in old.raw:
        if not np.array_equal(old.raw[name],new.raw[name]):
            raise AssertionError("Source-only improvement silently changed FP32 1D tensors")
    new_path=out/"S4_ORIGINAL_PRETRAINED_135M_SAME5BIT_RECODED.cb59"
    new_path.write_bytes(candidate)
    emit("S4_actual_original_full_source_RECODED",sha=sha(candidate),
         bytes=len(candidate),
         original_5bit_K32V8_211_matrices=len(new.handles)-1,
         original_original_unique=EXPECTED_UNIQUE)
    weight=fp32_weight_error(state,old,new)
    save(out/"S4_ALL_211_FP32_MATRIX_SOURCE_ONLY_ERRORS.json",weight)
    emit("S4_original_135M_weight_error",
         baseline_relative_RMSE=round(weight["base_relative_RMSE"],5),
         source_refit_relative_RMSE=round(weight["source_only_refit_relative_RMSE"],5),
         original_sse_ratio=round(weight["sse_refit_ratio"],6))
    baseline,base_src=build_trainable_from_paid_source(old)
    tuned,tuned_src=build_trainable_from_paid_source(new)
    baseline.cpu().eval().requires_grad_(False)
    tuned.cpu().eval().requires_grad_(False)
    rows=[]
    for i,prompt in enumerate(PROMPTS):
        ctx=prompt_ids(tok,prompt)
        ori=measure(fp32,None,tok,ctx)
        base_result=measure(baseline,base_src,tok,ctx)
        changed_result=measure(tuned,tuned_src,tok,ctx)
        base_hidden=representative_hidden(fp32,baseline,base_src,ctx)
        changed_hidden=representative_hidden(fp32,tuned,tuned_src,ctx)
        row={
          "TRAIN_debug_id":i,
          "same_prompt_sha":sha(prompt.encode()),
          "FP32_original":ori,
          "S0_CB59_original":base_result,
          "S4_CB59_8192sample_12iter":changed_result,
          "source_5bit_original_layer_hidden":base_hidden,
          "new_source_5bit_layer_hidden":changed_hidden,
          "new_first_token_matched_FP32":
               bool(ori["tokens"] and changed_result["tokens"] and
               ori["tokens"][0]==changed_result["tokens"][0]),
          "old_first_token_matched_FP32":
               bool(ori["tokens"] and base_result["tokens"] and
               ori["tokens"][0]==base_result["tokens"][0]),
        }
        rows.append(row)
        save(out/"S4_TRAIN_DEBUG_INCREMENTAL_GENERATIONS.json",rows)
        emit("S4_source_only_DEBUG_actual_output",idx=i,
             FP32=ori["text"][:70],
             old_CB59=base_result["text"][:70],
             refit_CB59=changed_result["text"][:70],
             old_final_cos=round(base_hidden["hidden_layers"]["30"]["cosine"],5),
             new_final_cos=round(changed_hidden["hidden_layers"]["30"]["cosine"],5),
             first_agree=row["new_first_token_matched_FP32"])
    result={
      **before,
      "full_source_bytes":len(candidate),
      "full_source_bpp":8*len(candidate)/EXPECTED_UNIQUE,
      "S4_model_file":new_path.name,
      "new_original_source_sha":sha(candidate),
      "base_211_2D_original_matrices":sum(1 for x in meta[
          "tensor_descriptors"].values() if x["dtype"]=="CB59_K32V8"),
      "same_codec_full_original_descriptors":True,
      "weight_errors":weight,
      "debug_generation":rows,
      "source_refit_first_matches_FP32":sum(
          x["new_first_token_matched_FP32"] for x in rows),
      "original_CB59_first_matches_FP32":sum(
          x["old_first_token_matched_FP32"] for x in rows),
      "real_FP32_135M_QA_parity_PROVEN":False,
      "native_C_speed_check_pending":True,
      "no_EVAL_used":True
    }
    save(out/"S4_FOLLOWUP_SOURCE_ONLY_REAL_135M_RECODE_RESULT.json",result)
    emit("S4_FINAL_source_only_135M_codebook_improvement",
         actual_new_source_sha=sha(candidate),
         old_original_source_sse_ratio=round(weight["sse_refit_ratio"],6),
         FP32_first_matches_old=result["original_CB59_first_matches_FP32"],
         FP32_first_matches_new=result["source_refit_first_matches_FP32"],
         same_physical_bpp=result["full_source_bpp"],
         coherent_language_PROVEN=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("original_S0_source",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    run(a.original_S0_source,a.out)
