"""CALIBRI 2.0 S0: real smallest-viable 5bit packed Llama from RANDOM initialization.

The only original checkpoint resources are the public architecture CONFIG
and tokenizer. Never loads FP32 pretrained weights, v228, v231 or teacher.
Candidates in fixed ascending order are selected by FULL CB59 serialized
rate, NEVER by answers or QA. Original 49152-vocabulary stays unchanged.
Check full native C inference correctness before any training updates.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59,encode,unpack5
from direct_runtime_v59 import compile_kernel,NativeKernel,bind_into_llama
from v174.run import save

SEED=2021
BPP_CEILING=0.700
# This is a fixed objective order. 256x6 may fail the physical rate cap;
# larger architecture distributes overhead over more neural parameters.
CANDIDATES=(
    ("tiny_256x6",256,768,6,4,2),
    ("small_320x6",320,896,6,5,1),
    ("small_384x6",384,1024,6,6,2),
)
VOCAB=49152
HEAD_DIM=64

def build_config(spec):
    label,hidden,ffn,layers,heads,kv=spec
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    if cfg.model_type!="llama" or int(cfg.vocab_size)!=VOCAB:
        raise AssertionError("Precommitted SmolLM2 architecture/tokenizer mismatch")
    cfg.hidden_size=hidden
    cfg.intermediate_size=ffn
    cfg.num_hidden_layers=layers
    cfg.num_attention_heads=heads
    cfg.num_key_value_heads=kv
    cfg.tie_word_embeddings=True
    cfg._attn_implementation="eager"
    if hidden%heads or heads%kv or hidden//heads!=HEAD_DIM:
        raise AssertionError("Invalid RoPE/GQA geometry")
    # Some LlamaConfig revisions expose explicit head_dim. Keep it stable.
    if hasattr(cfg,"head_dim") and cfg.head_dim is not None:
        cfg.head_dim=HEAD_DIM
    return cfg

def candidate_archive(spec):
    torch.manual_seed(SEED)
    cfg=build_config(spec)
    fresh=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).cpu().eval()
    n=sum(int(p.numel()) for p in fresh.parameters())
    if n<1_000_000 or n>=134_515_008:
        raise AssertionError("S0 candidate not truly smaller than 135M")
    weights=fresh.state_dict()
    if weights["lm_head.weight"].data_ptr()!=weights["model.embed_tokens.weight"].data_ptr():
        raise AssertionError("S0 did not preserve shared embedding/head")
    blob,meta,diag=encode(weights)
    del weights,fresh
    gc.collect()
    archive=CB59(blob)
    if archive.meta!=meta or archive.meta["original_unique_parameter_count"]!=n:
        raise AssertionError("S0 lost exact physical dimensions")
    required=1+7*spec[3]
    actual=sum(d["dtype"]=="CB59_K32V8" for d in
               archive.meta["tensor_descriptors"].values())
    if actual!=required or archive.handles["lm_head.weight"] is not archive.handles["model.embed_tokens.weight"]:
        raise AssertionError(f"Missing S0 operators or tied embeddings: {actual}/{required}")
    physical=len(blob)
    bpp=8*physical/n
    report={"candidate":spec[0],"config":{
                "model_type":cfg.model_type,"hidden_size":spec[1],
                "intermediate_size":spec[2],"num_hidden_layers":spec[3],
                "num_attention_heads":spec[4],"num_key_value_heads":spec[5],
                "head_dim":HEAD_DIM,"vocab_size":VOCAB,
                "tie_word_embeddings":True},
            "unique_original_architecture_parameters":n,
            "complete_archive_bytes":physical,"physical_bpp":bpp,
            "within_0_7_bpp":bpp<=BPP_CEILING,
            "physical_sha256":hashlib.sha256(blob).hexdigest(),
            "real_paid_5bit_matrices":actual,
            "initial_source_weight_RMSE_diagnostic_only":
                diag["full_source_weighted_relative_RMSE"],
            "used_pretrained_model_weights":False,
            "used_pretrained_tokenizer_and_architecture_only":True,
            "same_49152_original_tokenizer_vocabulary":True,
            "no_QA_or_teacher_used_to_choose_geometry":True}
    print(json.dumps({"stage":"CALIBRI2_S0_PHYSICAL_CANDIDATE",**report}),flush=True)
    return archive,report

@torch.inference_mode()
def native_forward_test(archive,cfg,out):
    cfile=Path(__file__).resolve().parents[2]/"direct_kernel_v59.c"
    dylib=compile_kernel(cfile,out/"libcalibri2_s0_native.so")
    kernel=NativeKernel(dylib)
    handles=("model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.mlp.down_proj.weight",
             "model.embed_tokens.weight")
    error=[]
    for name in handles:
        h=archive.handles[name]
        m,n=h.shape
        ids=unpack5(h.packed.tobytes(),m*n//8)
        x=(np.arange(n,dtype=np.float32)[None,:]/max(n-1,1)-.5).astype(np.float32)
        decoded=h.centroids[ids].reshape(m,n)*h.row_scales[:,None]
        ref=x@decoded.T
        actual=kernel.linear(h,x)
        difference=float(np.max(np.abs(ref-actual)))
        if not np.isfinite(difference) or difference>2e-3:
            raise AssertionError(f"Native C kernel wrong: {name}: {difference}")
        error.append({"operator":name,"max_abs_C_vs_float_decoder":difference})
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    evidence=bind_into_llama(model,archive,kernel)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    text=tok.apply_chat_template(
        [{"role":"user","content":"Write a short greeting."}],
        tokenize=False,add_generation_prompt=True)
    ids=tok.encode(text,add_special_tokens=False)
    outputs=model(input_ids=torch.tensor([ids],dtype=torch.long),
                  use_cache=False).logits
    if outputs.shape!=(1,len(ids),VOCAB) or not torch.isfinite(outputs).all():
        raise AssertionError("Not a complete finite tiny Llama inference pass")
    actual_token=int(outputs[0,-1].argmax())
    if evidence["runtime_dense_multi_dim_source_parameters"]!=0:
        raise AssertionError("Dense inference leaked")
    del model
    gc.collect()
    return {"real_native_C_complete_transformer_forward":True,
        "last_token_argmax_on_random_untrained_model":actual_token,
        "full_packed_original_matrices":evidence["compressed_real_2D_module_count"],
        "dense_original_weight_matrices_at_runtime":0,
        "C_numeric_checks":error,"not_language_quality_or_speed_certification":True}

def run(folder):
    folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    records=[]
    selected=None
    for spec in CANDIDATES:
        archive,metrics=candidate_archive(spec)
        records.append(metrics)
        if metrics["within_0_7_bpp"]:
            selected=(spec,archive,metrics)
            break
        del archive
        gc.collect()
    save(folder/"ARCHITECTURE_RATE_CANDIDATES.json",records)
    if selected is None:
        raise AssertionError("No precommitted small S0 architecture within FULL <=0.7 bpp rate")
    spec,archive,metrics=selected
    fname=folder/"CALIBRI2_S0_RANDOM_INITIAL.cb59"
    fname.write_bytes(archive.blob)
    readback=CB59(fname.read_bytes())
    if readback.hash()!=metrics["physical_sha256"]:
        raise AssertionError("Serialized S0 starting source mutated")
    cfg=build_config(spec)
    proof=native_forward_test(readback,cfg,folder)
    save(folder/"ARCHITECTURE_SELECTED.json",metrics)
    save(folder/"NATIVE_INFERENCE_CONTRACT.json",proof)
    print(json.dumps({"stage":"CALIBRI2_S0_READY_RANDOM_PAID_MODEL",
        "selected":metrics["candidate"],
        "parameters":metrics["unique_original_architecture_parameters"],
        "physical_bpp":metrics["physical_bpp"],
        "SHA256":metrics["physical_sha256"],
        "native_full_transformer":True,"is_trained_language_model":False}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    arguments=parser.parse_args()
    run(arguments.output)
