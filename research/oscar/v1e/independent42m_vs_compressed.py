#!/usr/bin/env python3
"""Oscar v1E independent public REAL pretrained 41.6896M FP32 codec control.

Authoritative original 42M TinyStories LLaMA checkpoint from Karpathy, MIT.
Original architecture 8x512. Convert externally pretrained full FP32 state
with Oscar (no known CALIBRI priors). Compare the REAL same-original-network
FP32 against authentically repacked/reloaded K32V8 decoder and generated ids.
Never call this a 12x320 CALIBRI model or a 42→43M trained skill transfer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from huggingface_hub import HfApi

from oscar.encoder import MODE_COLD, encode_fp32,to_cb61
from oscar.v1e.audit_checkpoint import source
from oscar.v1e.safe_pretrained_llama import (
    from_checkpoint,materialize_decoded_cb59,
)
from direct_codebook_v59 import CB59,unpack5

MODEL_ORIGINAL="karpathy/tinyllamas"
TKN="hf-internal-testing/llama-tokenizer"
PRECOMMITTED_STORIES=(
    "Once upon a time, there was a little girl named Lily.",
    "In a quiet forest, the small rabbit was looking for food.",
    "The dog ran across the green field and found a red ball.",
    "A boy named Max wanted to help his friend learn to swim.",
)
EXPECTED_ACTUAL_TRAINED_UNIQUE=41689600
EXPECTED_LAYERS=8
EXPECTED_DIM=512
MAX_NEW=8
SEED=421117


def dump(path,dat):
    Path(path).write_text(json.dumps(dat,sort_keys=True,indent=2,
                          ensure_ascii=False)+"\n",encoding="utf8")


def log(**vals):
    print(json.dumps(vals,ensure_ascii=False),flush=True)


@torch.inference_mode()
def generation(model,tok,ids):
    """Real original FP32-greedy, not teacher-forced and not fake QA."""
    current=ids.clone()
    emitted=[]
    stopped=False
    for _ in range(MAX_NEW):
        logits,_=model(current)
        if not torch.isfinite(logits).all():
            raise AssertionError("Oscar real PRETRAINED FP32 logits nonfinite")
        chosen=int(logits[0,-1].argmax())
        emitted.append(chosen)
        if chosen==tok.eos_token_id:
            stopped=True
            break
        current=torch.cat((current,current.new_tensor([[chosen]])),dim=-1)
    return emitted,stopped


def evaluate(teacher,student,tok,inputs):
    rows=[]
    for idx,text in enumerate(inputs):
        source_ids=tok(text,return_tensors="pt",add_special_tokens=True).input_ids
        if (source_ids.shape[1] > 96 or
            torch.any(source_ids>=teacher.p.vocab) or
            not torch.any(source_ids!=tok.eos_token_id)):
            raise AssertionError("Independently pretrained 42M tokenizer source mismatch")
        with torch.inference_mode():
            tlog,th=teacher(source_ids,hidden=True)
            slog,sh=student(source_ids,hidden=True)
            if len(th)!=EXPECTED_LAYERS+2 or len(sh)!=len(th):
                raise AssertionError("Oscar 42M originally pretrained all 8 layer hidden comparisons not available")
            a=tlog[0,-1].float()
            b=slog[0,-1].float()
            if a.numel()!=teacher.p.vocab:
                raise AssertionError("Model original 42M pretrained vocab mismatch")
            teacher_tokens,teacher_eos=generation(teacher,tok,source_ids)
            oscar_tokens,oscar_eos=generation(student,tok,source_ids)
        original_top1=int(a.argmax())
        new_top1=int(b.argmax())
        row={
           "precommitted_index":idx,
           "precommitted_original_story_prompt_sha256":
             hashlib.sha256(text.encode()).hexdigest(),
           "prompt_text":text,
           "source_token_ids":source_ids[0].tolist(),
           "genuine_original_external_pretrained_FP32_ids":teacher_tokens,
           "source_original_pretrained_FP32_text":tok.decode(
               teacher_tokens,skip_special_tokens=True),
           "new_Oscar_K32V8_real_repacked_ids":oscar_tokens,
           "new_Oscar_real_repacked_text":tok.decode(
               oscar_tokens,skip_special_tokens=True),
           "source_original_FP32_EOS":teacher_eos,
           "Oscar_compressed_EOS":oscar_eos,
           "same_teacher_first_token":original_top1==new_top1,
           "teacher_top1_rank_in_Oscar":
             int((b>b[original_top1]).sum())+1,
           "real_model_full_8token_generation_match":teacher_tokens==oscar_tokens,
           "matched_generated_initial_token_prefix_count":next((
               k for k,(x,y) in enumerate(zip(teacher_tokens,oscar_tokens))
               if x!=y),min(len(teacher_tokens),len(oscar_tokens))),
           "real_FP32_to_5bit_first_logit_absmax":float(torch.abs(a-b).max()),
           "real_FP32_to_5bit_soft_KL_T1":float(F.kl_div(
               F.log_softmax(b[None],dim=-1),
               F.softmax(a[None],dim=-1),
               reduction="batchmean")),
           "original_FP32_logit_entropy":float(torch.distributions.Categorical(
               logits=a).entropy()),
           "new_compressed_logit_entropy":float(torch.distributions.Categorical(
               logits=b).entropy()),
           "all_original_8decoder_layer_hidden_comparison":[{
              "layer":j-1,
              "cosine":float(F.cosine_similarity(
                  u[0,-1][None],v[0,-1][None])),
              "relative_L2":float(torch.linalg.vector_norm(
                  u[0,-1]-v[0,-1])/
                  torch.linalg.vector_norm(u[0,-1]).clamp_min(1e-20))
           } for j,(u,v) in enumerate(zip(th,sh))],
        }
        rows.append(row)
        log(stage="OSCAR_V1E_INDEPENDENT_ORIGINAL_FP32_42M_VS_NEW_SOURCE",
            probe=idx+1,teacher_first=original_top1,
            compressed_first=new_top1,
            same_first=row["same_teacher_first_token"],
            complete_generated_full8_match=row["real_model_full_8token_generation_match"],
            original=tok.decode(teacher_tokens,skip_special_tokens=True)[:90],
            compressed=tok.decode(oscar_tokens,skip_special_tokens=True)[:90],
            final_hidden_cos=round(row["all_original_8decoder_layer_hidden_comparison"][-1]["cosine"],5))
    return rows


def run(out:Path):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar v1E original pretrained 42M research outputs must be initially empty")
    original_file,original_weights,source_audit=source(out)
    # Real external independent ~42M pretrained checkpoint is exact 41.6896M
    # unique, not 43.26176M originally learned CALIBRI research architecture.
    teacher=from_checkpoint(original_weights)
    orig_n=sum(p.numel() for p in teacher.parameters())
    if (orig_n!=EXPECTED_ACTUAL_TRAINED_UNIQUE or
        source_audit["model_parameters_real_unique"]!=orig_n or
        teacher.p.layers!=EXPECTED_LAYERS or teacher.p.dim!=EXPECTED_DIM):
        raise AssertionError("Oscar original independently PRETRAINED 42M FP32 shape/source changed")
    tokenizer_rev=HfApi().model_info(TKN).sha
    if not tokenizer_rev or len(tokenizer_rev)!=40:
        raise AssertionError("Model's proxy original LLaMA tokenizer is not pinned")
    tok=AutoTokenizer.from_pretrained(TKN,revision=tokenizer_rev)
    if tok.vocab_size!=teacher.p.vocab or tok.eos_token_id is None:
        raise AssertionError("Tokenizer vocab not matching original model LLaMA 42M 32k")
    plan={
      "project":"Oscar v1E real outside-PRETRAINED original FP32 42M matched-geometry codec control",
      "source_repo":MODEL_ORIGINAL,
      "source_original_HF_revision":source_audit["pinned_exact_HF_revision_sha40"],
      "pretrained_original_FP32_full_SHA256":source_audit["archive_SHA256"],
      "external_original_unique_parameters":orig_n,
      "original_pretrained_model_8decoder_layers_512_hidden":True,
      "original_pretrained_model_IS_NOT_Calibri_12layer":True,
      "pretrained_test_uses_real_teacher_FP32_first_logits":True,
      "source_only_no_pretrained_CALIBRI_codebook_prior":True,
      "fixed_authored_four_prompts_not_after_the_fact":list(PRECOMMITTED_STORIES),
      "external_tokenizer":"hf-internal-testing/llama-tokenizer",
      "external_tokenizer_PINNED_REV":tokenizer_rev,
      "tokenizer_proxy_equivalence_to_training_UNVERIFIED":True,
      "no_contaminated_pretrained_training_data_claim":True,
      "source_original_codebook_coldK32_V8_5bit":True,
      "not_yet_12layer_student_architecture_projection":True,
      "read_source_pt_with_weights_only_true":True,
      "no_unseen_J7_or_QA_used":True,
    }
    dump(out/"OSCAR_V1E_PRECOMMIT_BEFORE_COMPRESSION.json",plan)
    log(stage="OSCAR_V1E_REAL_41p6896M_SOURCE_FP32_READY",
        actual_original_FP32_unique=orig_n,tokenizer_rev=tokenizer_rev)
    start=time.monotonic()
    new=encode_fp32(teacher.state_dict(),mode=MODE_COLD)
    decoder=new.authenticated_decoder()
    if (decoder.meta["original_unique_parameter_count"]!=orig_n or
        not decoder.meta["all_weight_2d_physically_compressed"] or
        decoder.handles["tok_embeddings.weight"] is not
        decoder.handles["output.weight"]):
        raise AssertionError("Oscar independent pretrained original 42M full source missed or double counted tied weights")
    cb59=out/"OSCAR_V1E_EXTERNAL_TINYSTORIES_42M_SOURCE.cb59"
    cb59.write_bytes(new.blob)
    cb61_binary=to_cb61(new.blob)
    cb61=out/"OSCAR_V1E_EXTERNAL_TINYSTORIES_42M_SOURCE.cb61"
    cb61.write_bytes(cb61_binary)
    # Physically saved source can be reloaded completely independent of FP32
    # teacher; only experimental paired analysis uses the transient FP32 mirror.
    fresh=materialize_decoded_cb59(teacher,cb59.read_bytes())
    results=evaluate(teacher,fresh,tok,PRECOMMITTED_STORIES)
    first=sum(x["same_teacher_first_token"] for x in results)
    full=sum(x["real_model_full_8token_generation_match"] for x in results)
    log(stage="OSCAR_V1E_REAL_INDEPENDENT_PRETRAINED_42M_FP32_BENCHMARK_COMPLETE",
        paired=4,first=first,full=full,
        cold_fit_seconds=round(time.monotonic()-start,2))
    # Extra unpaid causal diagnostic after baseline actual package is saved:
    # restoring original FP32 token embedding/head to the decoded model
    # helps isolate why source-only K32V8 loses original trained generation.
    with torch.no_grad():
        fresh.tok_embeddings.weight.copy_(teacher.tok_embeddings.weight)
    embed_hybrid=evaluate(teacher,fresh,tok,PRECOMMITTED_STORIES)
    # Never ship the hybrid FP32 in a compressed model; documented UNPAID.
    hybrid={
      "unpaid_FP32_embedding_head_original_teacher_restored":True,
      "not_any_0p7_deployable_candidate":True,
      "full_greedy_comparison":embed_hybrid,
    }
    dump(out/"OSCAR_V1E_UNPAID_FP32_EMBEDDING_HYBRID_CAUSAL_DIAG.json",hybrid)
    fp32_reference_bytes=4*orig_n
    # One true CB61 inference source plus paid runtime geometry/manifest only,
    # separate CB59 kept for diagnostic and NOT simultaneously shipped.
    physical={
       "model_CB61_deployable_format_bytes":len(cb61_binary),
       "model_CB59_diagnostic_unshipped_bytes":len(new.blob),
       "real_original_FP32_teacher_reference_4byte_bytes":fp32_reference_bytes,
       "original_independently_pretrained_source_in_not_shipped":True,
       "runtime_config_json_charged_bytes":0,
       "actual_original_unique_pretrained_parameters":orig_n,
       "one_packaged_CB61_only":True,
       "model_weights_only_bpp":8*len(cb61_binary)/orig_n,
       "entire_runtime_model_plus_manifest_bpp":0.,
       "actual_0p7_weights_only_pass":8*len(cb61_binary)/orig_n<=.7,
    }
    report={
       **plan,
       "step":"REAL ORIGINAL EXTERNALLY PRETRAINED 41p6896M FP32 matched-original-architecture K32V8 Oscar v1E",
       "original_pretrained_only_Not_derived_from_Calibri":True,
       "original_external_FP32_model_original_architecture":{"n":orig_n,
          "layers":teacher.p.layers,"hidden":teacher.p.dim,
          "ffn":teacher.p.hidden,"vocab":teacher.p.vocab},
       "new_independently_compressed_CB59_sha256":hashlib.sha256(new.blob).hexdigest(),
       "new_independently_compressed_CB61_sha256":hashlib.sha256(cb61_binary).hexdigest(),
       "all_original_2D_matrices_physically_encoded":True,
       "complete_binary_aliases_crc_sha_verified":True,
       "source_only_Oscar_encoder_actual_diagnostics":new.report,
       "real_first_token_match":first,
       "real_FULL_8token_generated_sequence_match":full,
       "cases":len(results),
       "actual_paired_original_fp32_vs_CB59_NEW_source":results,
       "source_pretrained_FP32_never_at_inference":True,
       "no_full_12layer_Calibri_projection_YET":True,
       "not_full_original_teacher_43M_language_skills_restored_unless_generated_match":True,
       "unpaid_FP32_hybrid_in_runtime":False,
       "full_external_human_text_QA_never_used":True,
       "native_C_throughput_and_old_616_J7_NOT_evaluated":True,
       "new_model_deployment_accepted":False,
       "physically_measured":physical
    }
    # Inference manifest must NOT include actual training or QA/test logits;
    # small separate exact physical config is what would ship with CB61.
    manifest=out/"OSCAR_V1E_RUNTIME_MATERIALIZED_MODEL_CONFIG.json"
    live={
       "algorithm":"Oscar v1E independent public pretrained original FP32 42M",
       "original_model_sha256":source_audit["archive_SHA256"],
       "input_pretrained_HF_rev":source_audit["pinned_exact_HF_revision_sha40"],
       "original_LLaMA_layers":8,"original_hidden":512,
       "not_Calibri_12layers":True,
       "input_tokenizer_model":"LLaMA 32000; proxy external tokenizer not claimed training-equivalent",
       "format":"CB61ZL01 original K32V8 physically complete direct index source",
       "original_unique_parameters":orig_n,
       "source_only_rate_no_inference_backend_optimizations":True,
       "compressed_source_SHA256":hashlib.sha256(cb61_binary).hexdigest(),
       "original_inference_requires_LLaMA_8layer_runtime_not_original_12layer_GT":True,
       "quality_not_guaranteed_or_pretrained_skill_preserved":True,
    }
    payload=(json.dumps(live,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    manifest.write_bytes(payload)
    physical["runtime_config_json_charged_bytes"]=manifest.stat().st_size
    total=manifest.stat().st_size+len(cb61_binary)
    physical["entire_runtime_model_plus_manifest_bpp"]=8*total/orig_n
    physical["actual_0p7_full_model_plus_manifest_pass"]=8*total/orig_n<=.7
    report["physically_measured"]=physical
    dump(out/"OSCAR_V1E_FINAL_REAL_OPEN_FP32_42M_VS_NEW_INDEPENDENT_CB61_GENERATION.json",report)
    log(stage="OSCAR_V1E_FINAL_NEW_REAL_42M_FP32_TO_OSCAR_COLD",
        actual_original_pretrained_unique=orig_n,
        student_original_geometry_8x512=True,
        compressed_bytes=len(cb61_binary),
        source_fp32_bytes=fp32_reference_bytes,
        source_only_physical_full_bpp=physical["entire_runtime_model_plus_manifest_bpp"],
        first_token_equal=first,full_8token_equal=full,
        paired_n=len(results),
        original_pretrained_12layer_Calibri_conversion_PROVEN=False,
        native_CPU_C_90_not_measured=True,
        deployment_accepted=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    run(p.parse_args().out)
