#!/usr/bin/env python3
"""CALIBRI v68: fixed-rate GQA V/O gauge experiment on REAL pinned 135M weights.

Research ONLY. For two pretrained Llama GQA attention layers, compare frozen
CB59 5bit K32V8 quantization against an exact FP32 function-preserving diagonal
V/O reparameterization followed by THE SAME CB59 codec and physical source budget.

The measured object is teacher-attention-conditioned local attention output, not
whole-model language performance. No live Jessica, Render, Supabase, APK changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

from direct_codebook_v59 import CB59, HEADER, compress_matrix, unpack5
from quality import DATASET, load_token_ids
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE

TEST_START=1024
TEST_TOKENS=128
WINDOW=64
TARGET_LAYERS=(0,15)

def scales_for_pair(v, o, hq, hk, hd):
    """Source-only L2 balancing of each V coordinate against its GQA O copies."""
    assert hq % hk == 0
    repeat = hq//hk
    assert v.shape[0] == hk*hd and o.shape[1] == hq*hd
    vs=np.sqrt(np.sum(v.astype(np.float64)**2,axis=1)).reshape(hk,hd)
    os=o.reshape(o.shape[0],hk,repeat,hd).astype(np.float64)
    on=np.sqrt(np.sum(os**2,axis=(0,2)))
    d=np.sqrt(np.maximum(on,1e-30)/np.maximum(vs,1e-30))
    return np.clip(d,1/8,8).astype(np.float32).reshape(-1)

def gauge(v, o, d, hq, hk, hd):
    repeated=np.repeat(d.reshape(hk,hd),hq//hk,axis=0).reshape(-1)
    return v*d[:,None],o/repeated[None,:]

def aggregate(x, prob, v, o, hq, hk, hd):
    """Fixed original pretrained attention probabilities; explicit GQA repetition."""
    if x.ndim!=2 or prob.shape!=(hq,len(x),len(x)):
        raise AssertionError("unexpected attention activation/probability shape")
    values=x@v.T
    values=values.reshape(len(x),hk,hd).transpose(1,0,2)
    values=np.repeat(values,hq//hk,axis=0)
    out=np.einsum("hts,hsd->htd",prob,values,optimize=True)
    inner=out.transpose(1,0,2).reshape(len(x),hq*hd)
    return inner@o.T,inner

def quantized_from_existing(h):
    m,n=h.shape
    codes=unpack5(h.packed.tobytes(),m*n//8)
    return (h.centroids[codes].reshape(m,n)*h.row_scales[:,None]).astype(np.float32)

def quantized_from_encoded(w, enc):
    m,n=w.shape
    codes=unpack5(enc["packed"],m*n//8)
    c=np.frombuffer(enc["centroids"],dtype="<f2").astype(np.float32).reshape(32,8)
    s=np.frombuffer(enc["row_scales"],dtype="<f2").astype(np.float32)
    return (c[codes].reshape(m,n)*s[:,None]).astype(np.float32)

def relative(y,target):
    return float(np.linalg.norm((y-target).astype(np.float64))/
                 max(np.linalg.norm(target.astype(np.float64)),1e-30))

def collect(model, token_ids, target_layers):
    capture={i:{"x":[],"original_h":[],"probs":[]} for i in target_layers}
    handles=[]
    for i in target_layers:
        module=model.model.layers[i].self_attn
        def vin(m, args, *, layer=i):
            capture[layer]["x"].append(args[0].detach().cpu().float().numpy()[0])
        def oin(m, args, *, layer=i):
            capture[layer]["original_h"].append(args[0].detach().cpu().float().numpy()[0])
        handles.append(module.v_proj.register_forward_pre_hook(vin))
        handles.append(module.o_proj.register_forward_pre_hook(oin))
    try:
        with torch.inference_mode():
            for offset in range(0,len(token_ids),WINDOW):
                ans=model(input_ids=token_ids[offset:offset+WINDOW][None],
                          output_attentions=True,use_cache=False,return_dict=True)
                if ans.attentions is None:
                    raise RuntimeError("original eager model did not return attention probabilities")
                for i in target_layers:
                    capture[i]["probs"].append(
                        ans.attentions[i].detach().cpu().float().numpy()[0])
    finally:
        for handle in handles: handle.remove()
    for i in target_layers:
        if not (len(capture[i]["x"])==len(capture[i]["probs"])==
                len(capture[i]["original_h"])==len(token_ids)//WINDOW):
            raise AssertionError("teacher forward hooks missing a window")
    return capture

def patch_pair(original, original_cb, replacements, expected_n):
    # Preserve EXACT original full 135M archive length / manifest / all aliases.
    _,_,mlen,_=HEADER.unpack_from(original)
    body_offset=HEADER.size+mlen
    patched=bytearray(original)
    for name,fields in replacements.items():
        target=original_cb.meta["tensor_descriptors"][name]
        assert target["dtype"]=="CB59_K32V8"
        for part in ("packed","centroids","row_scales"):
            off,size=target["parts"][part]
            if len(fields[part])!=size:
                raise AssertionError("source byte cost changed "+name+":"+part)
            patched[body_offset+off:body_offset+off+size]=fields[part]
    raw=bytes(patched[:-36])
    with_crc=raw+struct.pack("<I",zlib.crc32(raw))
    full=with_crc+hashlib.sha256(with_crc).digest()
    cb=CB59(full)
    if len(full)!=len(original) or cb.meta["original_unique_parameter_count"]!=expected_n:
        raise AssertionError("whole model source not physically complete/fixed")
    if set(cb.handles)!=set(original_cb.handles):
        raise AssertionError("lost trained source/alias tensors")
    return full,cb

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    begun=time.monotonic()
    original_blob=args.source.read_bytes()
    original_cb=CB59(original_blob)
    if original_cb.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE:
        raise AssertionError("unexpected pinned physical CB59 model")
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg=model.config
    hq=int(cfg.num_attention_heads)
    hk=int(cfg.num_key_value_heads)
    hd=int(getattr(cfg,"head_dim",cfg.hidden_size//hq))
    assert hq==9 and hk==3 and hd==64 and int(cfg.hidden_size)==576
    ds_sha=HfApi().dataset_info(DATASET).sha
    tokens=load_token_ids(tokenizer,"test",ds_sha,TEST_START+TEST_TOKENS)
    test=tokens[TEST_START:]
    assert len(test)==TEST_TOKENS
    source=collect(model,test,TARGET_LAYERS)
    experiments=[]
    paired_gauge_encoded={}
    for layer in TARGET_LAYERS:
        prefix=f"model.layers.{layer}.self_attn"
        vk=prefix+".v_proj.weight"
        ok=prefix+".o_proj.weight"
        V=model.model.layers[layer].self_attn.v_proj.weight.detach().cpu().numpy().astype(np.float32)
        O=model.model.layers[layer].self_attn.o_proj.weight.detach().cpu().numpy().astype(np.float32)
        hv=original_cb.handles[vk]
        ho=original_cb.handles[ok]
        assert V.shape==(hk*hd,576) and O.shape==(576,hq*hd)
        assert hv.shape==V.shape and ho.shape==O.shape
        # Identity-source refit must EXACTLY reproduce the original CB59
        # compiler's physical segments, or comparisons are not codec-controlled.
        if layer==TARGET_LAYERS[0]:
            baseline_recode=(compress_matrix(V),compress_matrix(O))
            for key,handle,enc in ((vk,hv,baseline_recode[0]),(ok,ho,baseline_recode[1])):
                if not (enc["packed"]==handle.packed.tobytes() and
                        np.array_equal(np.frombuffer(enc["row_scales"],dtype="<f2"),
                                       handle.row_scales.astype("<f2")) and
                        np.array_equal(np.frombuffer(enc["centroids"],dtype="<f2").reshape(32,8),
                                       handle.centroids.astype("<f2"))):
                    raise AssertionError("same-source CB59 re-encoding did not match for "+key)
        base_v=quantized_from_existing(hv)
        base_o=quantized_from_existing(ho)
        d=scales_for_pair(V,O,hq,hk,hd)
        vg,og=gauge(V,O,d,hq,hk,hd)
        assert np.isfinite(vg).all() and np.isfinite(og).all()
        # Exactly same format/bytes, learned from source weights only.
        enc_v=compress_matrix(vg)
        enc_o=compress_matrix(og)
        for key,handle,enc in ((vk,hv,enc_v),(ok,ho,enc_o)):
            if (len(enc["packed"])!=len(handle.packed) or
                len(enc["row_scales"])!=2*handle.shape[0] or
                len(enc["centroids"])!=32*8*2):
                raise AssertionError("rate-matched encoded pair not equal original source")
        qv=quantized_from_encoded(vg,enc_v)
        qo=quantized_from_encoded(og,enc_o)
        paired_gauge_encoded[vk]=enc_v
        paired_gauge_encoded[ok]=enc_o
        baseline_errors=[]
        balanced_errors=[]
        full_exact_differences=[]
        hook_differences=[]
        for x,prob,real_inner in zip(source[layer]["x"],
                                     source[layer]["probs"],
                                     source[layer]["original_h"]):
            target,inner=aggregate(x,prob,V,O,hq,hk,hd)
            # Verify this exact manual GQA equation equals HF's attention
            # pre-projection value input for the same pinned real window.
            hook_differences.append(relative(inner,real_inner))
            unquantized_gauged,_=aggregate(x,prob,vg,og,hq,hk,hd)
            full_exact_differences.append(relative(unquantized_gauged,target))
            baseline,_=aggregate(x,prob,base_v,base_o,hq,hk,hd)
            fixed,_=aggregate(x,prob,qv,qo,hq,hk,hd)
            baseline_errors.append(np.linalg.norm((baseline-target).astype(np.float64))**2)
            balanced_errors.append(np.linalg.norm((fixed-target).astype(np.float64))**2)
        if max(hook_differences)>5e-4 or max(full_exact_differences)>1e-4:
            raise AssertionError("REAL GQA paired gauge does not preserve source function")
        base_mse=math.sqrt(sum(baseline_errors))
        equal_mse=math.sqrt(sum(balanced_errors))
        record={"layer":layer,
                "v_shape":list(V.shape),"o_shape":list(O.shape),
                "source_only_d_min":float(d.min()),"source_only_d_max":float(d.max()),
                "quantizer":"original CB59 integer lattice K32V8 5bits/8 coefficients",
                "GQA_repeat_per_KV_head":hq//hk,
                "windows":len(source[layer]["probs"]),
                "original_eager_hook_relative_discrepancy":max(hook_differences),
                "exact_FP32_gauge_relative_discrepancy":max(full_exact_differences),
                "original_CB59_layer_attention_output_error_L2":base_mse,
                "balanced_reencoded_layer_attention_output_error_L2":equal_mse,
                "balanced_to_original_CB59_response_error_ratio":equal_mse/max(base_mse,1e-30),
                "original_pair_source_payload_bytes":int(
                    hv.packed.nbytes+ho.packed.nbytes+2*(hv.shape[0]+ho.shape[0])+2*(2*32*8)),
                "candidate_pair_same_physical_bytes":True}
        experiments.append(record)
        print(json.dumps({"layer":layer,"GQA_exact_relative":round(max(full_exact_differences),8),
                          "CB59_attention_error":round(base_mse,5),
                          "gauge_attention_error":round(equal_mse,5),
                          "ratio":round(record["balanced_to_original_CB59_response_error_ratio"],6)}),flush=True)
    new_blob,new_cb=patch_pair(original_blob,original_cb,paired_gauge_encoded,EXPECTED_UNIQUE)
    audit={"experiment":"CALIBRI v68 real pinned pretrained GQA V/O mathematical gauge vs source codec",
           "model":MODEL,"revision":MODEL_SHA,"dataset_revision":ds_sha,
           "TEST_heldout_token_span":[TEST_START,TEST_START+TEST_TOKENS],
           "TEST_token_sha256":hashlib.sha256(test.numpy().tobytes()).hexdigest(),
           "source_only_analytical_gauge_no_TEST_fitting":True,
           "original_CB59_SHA256":hashlib.sha256(original_blob).hexdigest(),
           "candidate_SHA256":hashlib.sha256(new_blob).hexdigest(),
           "original_physical_full_model_bytes":len(original_blob),
           "candidate_physical_full_model_bytes":len(new_blob),
           "max_full_model_bytes_at_0_700":math.floor(.700*EXPECTED_UNIQUE/8),
           "candidate_full_model_bpw":8*len(new_blob)/EXPECTED_UNIQUE,
           "all_source_tensors_and_aliases_preserved":True,
           "full_pretrained_2D_weights_absent_from_archive":True,
           "ABl_same_CB59_K32V8_direct_kernel_format":True,
           "all_paired_quantized_checksums_valid":True,
           "tested_real_source_layers":experiments,
           "whole_model_language_quality_NOT_measured":True,
           "native_cpu_packed_kernel_speed_NOT_measured":True,
           "seconds":time.monotonic()-begun}
    assert len(new_blob)==len(original_blob)==11229303
    assert 8*len(new_blob)/EXPECTED_UNIQUE<=.7
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"layers":len(experiments),
                      "full_source_bytes":len(new_blob),
                      "physical_bpw":round(audit["candidate_full_model_bpw"],9),
                      "real_gqa_invariance_pass":True,
                      "whole_model_quality_measured":False}),flush=True)

if __name__=="__main__": main()
