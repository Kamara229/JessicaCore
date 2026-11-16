#!/usr/bin/env python3
"""S2: causal errors FP32 vs original pretrained compressed 135M.

Diagnostic hybrid FP32 input/head restoration is INTENTIONALLY UNPAID and
NEVER an inference-package candidate. It isolates why CB59 has 0 full answers.
All prompts are previously exposed TRAIN/DEBUG only, NOT sealed QA/EVAL.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from direct_codebook_v59 import CB59
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from v176.operators import clear
from v196.recover import build_trainable_from_paid_source

S0_SHA="50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
MAX_NEW=8
PROMPTS=(
    "Answer with one word: What color is fresh snow?",
    "Answer with one digit: How much is 2 plus 2?",
    "Answer with one word: What do bees make?",
    "Complete briefly: The sun rises in the ...",
)


def dump(path,v):
    Path(path).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,
                                    indent=2)+"\n",encoding="utf-8")


def log(stage,**values):
    print(json.dumps({"stage":stage,**values},ensure_ascii=False),flush=True)


def prompt_ids(tok,q):
    value=tok.apply_chat_template([{"role":"user","content":q}],
                                  tokenize=False,add_generation_prompt=True)
    return torch.tensor([tok.encode(value,add_special_tokens=False)],
                        dtype=torch.long)


@torch.inference_mode()
def measure(model,source,tok,ctx):
    ids=ctx
    result=[]
    ended=False
    t0=time.monotonic()
    for _ in range(MAX_NEW):
        if source is not None:
            clear(source)
        logits=model(input_ids=ids,use_cache=False).logits[0,-1]
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError("Nonfinite logits from original trained 135M")
        t=int(logits.argmax())
        if t==tok.eos_token_id:
            ended=True
            break
        result.append(t)
        ids=torch.cat((ids,ids.new_tensor([[t]])),1)
    if source is not None:
        clear(source)
    return {"text":tok.decode(result,skip_special_tokens=True,
             clean_up_tokenization_spaces=False),
            "tokens":result,"EOS_seen":ended,
            "wall_s":round(time.monotonic()-t0,3)}


@torch.inference_mode()
def hidden_comparison(fp32,student,sources,tok,ctx):
    clear(sources)
    original=fp32(input_ids=ctx,use_cache=False,output_hidden_states=True)
    compressed=student(input_ids=ctx,use_cache=False,output_hidden_states=True)
    if len(original.hidden_states)!=31 or len(compressed.hidden_states)!=31:
        raise AssertionError("Expected 30 original pretrained decoder layers")
    details=[]
    for i,(a,b) in enumerate(zip(original.hidden_states,
                                  compressed.hidden_states)):
        aa=a.detach().float()[:,-1].reshape(-1)
        bb=b.detach().float()[:,-1].reshape(-1)
        norm=float(torch.linalg.vector_norm(aa))
        mag=float(torch.linalg.vector_norm(bb))
        cosine=float(F.cosine_similarity(aa[None],bb[None]).item())
        relative=float(torch.linalg.vector_norm(bb-aa)/
                       torch.linalg.vector_norm(aa).clamp_min(1e-9))
        details.append({
          "after_decoder_layer":i-1,
          "is_initial_embedding_state":i==0,
          "original_norm":norm,"CB59_norm":mag,
          "norm_ratio":mag/max(norm,1e-9),
          "cosine_FP32_vs_CB59":cosine,
          "relative_FP32_CB59_hidden_error":relative
        })
    teacher_logits=original.logits[0,-1].float()
    student_logits=compressed.logits[0,-1].float()
    t=int(teacher_logits.argmax())
    q=int(student_logits.argmax())
    original_rank=int((student_logits>student_logits[t]).sum())+1
    top_t=torch.topk(teacher_logits,5)
    top_c=torch.topk(student_logits,5)
    def candidates(obj):
        return [{"id":int(i),"token":tok.decode([int(i)]),
                 "logit":round(float(v),4)}
                for i,v in zip(obj.indices,obj.values)]
    clear(sources)
    return {
        "decoder_hidden_drift":details,
        "teacher_first_token":t,
        "teacher_first_token_text":tok.decode([t]),
        "compressed_first_token":q,
        "compressed_first_token_text":tok.decode([q]),
        "FP32_first_token_rank_in_CB59":original_rank,
        "teacher_top5":candidates(top_t),
        "compressed_top5":candidates(top_c),
        "tied_input_output_weight_parent_used":True,
    }


def run(s0,out):
    torch.manual_seed(13521117)
    torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=Path(s0).read_bytes()
    if (hashlib.sha256(raw).hexdigest()!=S0_SHA or
        len(raw)!=11229303):
        raise AssertionError("Unauthenticated original pretrained S0 CB59")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE:
        raise AssertionError("135M unique denominator mismatch")
    diag={
      "stage":"S2_original_training_checkpoint_activation_diagnosis",
      "exact_model_revision":MODEL_SHA,
      "original_FP32_pretrained_unique_parameters":EXPECTED_UNIQUE,
      "source_SHA":S0_SHA,
      "original_debug_TRAIN_prompts_exposed_before_this_experiment":list(PROMPTS),
      "no_EVAL_or_QA_access":True,
      "no_new_training":True,
      "hypotheses_precommitted":["input_embedding_quantization_error",
          "output_lm_head_quantization_error","internal_layer_quantization_error"],
      "unpaid_FP32_hybrids_only_for_sensitivity_not_packaged":True,
    }
    dump(out/"S2_DIAG_PRECOMMIT_BEFORE_MODEL_LOAD.json",diag)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).cpu().eval()
    assert sum(x.numel() for x in teacher.parameters())==EXPECTED_UNIQUE
    student,sources=build_trainable_from_paid_source(archive)
    student.cpu().eval().requires_grad_(False)
    orig_input=student.model.embed_tokens
    orig_head=student.lm_head
    if orig_input.source is not orig_head.source:
        raise AssertionError("Source tied head+input embedding unexpectedly disconnected")
    fp32_emb=nn.Embedding.from_pretrained(
        teacher.model.embed_tokens.weight.detach(),freeze=True)
    fp32_head=nn.Linear(teacher.config.hidden_size,
                        teacher.config.vocab_size,bias=False)
    fp32_head.weight=nn.Parameter(teacher.lm_head.weight.detach(),
                                  requires_grad=False)
    rows=[]
    try:
        for i,prompt in enumerate(PROMPTS):
            ctx=prompt_ids(tok,prompt)
            baseline=measure(teacher,None,tok,ctx)
            student.model.embed_tokens=orig_input
            student.lm_head=orig_head
            compressed=measure(student,sources,tok,ctx)
            diagnostic=hidden_comparison(teacher,student,sources,tok,ctx)
            student.model.embed_tokens=fp32_emb
            student.lm_head=orig_head
            embed_only=measure(student,sources,tok,ctx)
            student.model.embed_tokens=orig_input
            student.lm_head=fp32_head
            head_only=measure(student,sources,tok,ctx)
            student.model.embed_tokens=fp32_emb
            student.lm_head=fp32_head
            both=measure(student,sources,tok,ctx)
            student.model.embed_tokens=orig_input
            student.lm_head=orig_head
            row={
              "TRAIN_DEBUG_index":i,
              "question":prompt,
              "FP32_original_trained":baseline,
              "CB59_direct":compressed,
              "hybrid_FP32_INPUT_only_UNPAID":embed_only,
              "hybrid_FP32_OUTPUT_only_UNPAID":head_only,
              "hybrid_FP32_INPUT_AND_OUTPUT_UNPAID":both,
              "full_original_vs_compressed_hidden":diagnostic,
              "same_first_token_B_vs_FP32":bool(
                  baseline["tokens"] and compressed["tokens"] and
                  baseline["tokens"][0]==compressed["tokens"][0]),
            }
            rows.append(row)
            dump(out/"S2_CAUSAL_LOCALIZATION_INCREMENTAL.json",rows)
            log("S2_causal_localization",idx=i,
                FP32=baseline["text"][:80],
                CB59=compressed["text"][:80],
                FP32_input_restored=embed_only["text"][:80],
                FP32_output_restored=head_only["text"][:80],
                BOTH_restored=both["text"][:80],
                hidden_start_cos=round(diagnostic["decoder_hidden_drift"][0][
                    "cosine_FP32_vs_CB59"],4),
                hidden_end_cos=round(diagnostic["decoder_hidden_drift"][-1][
                    "cosine_FP32_vs_CB59"],4),
                original_top1_rank_in_CB59=diagnostic["FP32_first_token_rank_in_CB59"])
    finally:
        student.model.embed_tokens=orig_input
        student.lm_head=orig_head
        clear(sources)
    log("S2_causal_localization_COMPLETE",cases=len(rows),
        first_tokens_equal=sum(r["same_first_token_B_vs_FP32"] for r in rows),
        original_FP32_was_loaded_only_for_DIAGNOSTIC=True)
    dump(out/"S2_FINAL_ORIGINAL_FP32_VS_COMPRESSED_LAYER_ERROR.json",{
        **diag,"completed":True,
        "full_probe_records":rows,
        "all_30_layers_diagnosed":True,
        "teacher_FP32_never_in_packed_inference":True,
        "full_original_bpp_0p7_not_claimed_for_hybrids":True,
        "new_unseen_heldout_EVAL_not_used":True
    })
    del teacher
    gc.collect()


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("S0_full_source",type=Path)
    a.add_argument("out",type=Path)
    args=a.parse_args()
    run(args.S0_full_source,args.out)
