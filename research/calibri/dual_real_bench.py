"""CALIBRI 11: REAL pre-trained Switch MoE weights, dual-stage dictionary codec.

An expert FFN has *paired* matrices Wi[3072,768], Wo[768,3072].
We quantize these once (simple imposed .885 zeros, NOT GPTQ), and
compare EXACT lossless formats for those very same ternary symbols.

Experiments:
 A QMoE's official STATIC 65536-word dictionary construction, GREEDY CPU
   row encoding, uint16 words + uint32 row offsets + fp16 row scales.
 B Stage1 -> Stage2 independently seekable grouped raw/zlib/2-D byteplanes.
 C Equivalent hidden ReLU neuron reordering P.Wi + Wo.P^T before dictionary
   (no runtime permutation needed, both matrices replaced atomically).
 D Negative-control random permutation, row-addressability, CPU decode
   latency. No CUDA, no actual published QMoE quantizer, no full-model PPL.
"""
import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics
import time
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from dual_codec import (
    LUT_BYTES,StaticPairDictionary,original_qmoe_like_bytes,
    encode_grouped,decode_grouped,read_one_row,grouped_total_bytes,
    stable_fp64_ffn_equivalence
)
from index import encode as cl8_encode,decode as cl8_decode

MODEL="google/switch-base-8"
CKPT="pytorch_model.bin"
PREFIX="decoder.block.1.layer.2.mlp.experts.expert_0."
TENSORS=("wi.weight","wo.weight")

def row_ternarize(weight,zero=0.885):
    v=weight.detach().float().numpy().copy()
    threshold=np.quantile(np.abs(v),zero,axis=1,keepdims=True)
    q=np.where(np.abs(v)>threshold,np.sign(v),0).astype(np.int8)
    scales=np.empty((len(q),2),dtype='<f2')
    for i in range(len(q)):
        neg=v[i,q[i]<0];pos=v[i,q[i]>0]
        scales[i,0]=np.float16(neg.mean()) if len(neg) else np.float16(0)
        scales[i,1]=np.float16(pos.mean()) if len(pos) else np.float16(0)
    return q,scales

def reconstructed(q,scales):
    level=scales.astype(np.float64)
    return np.where(q<0,level[:,0:1],np.where(q>0,level[:,1:2],0.))

def read_latency(blob,rows,n=64):
    rng=np.random.default_rng(613)
    indices=rng.integers(0,rows,size=n)
    samples=[]
    for r in indices:
        start=time.perf_counter_ns()
        _=read_one_row(blob,int(r))
        samples.append((time.perf_counter_ns()-start)/1e6)
    return {"cpu_python_random_row_p50_ms":float(np.percentile(samples,50)),
            "cpu_python_random_row_p95_ms":float(np.percentile(samples,95)),
            "rows_sampled":len(samples)}

def encode_variants(d,q,group_list=(16,64),details=False):
    start=time.perf_counter()
    words,off=d.encode(q)
    t1=time.perf_counter()-start
    if not np.array_equal(q,d.decode(words,off,q.shape[1])):
        raise AssertionError("Stage1 dictionary roundtrip failed")
    base=original_qmoe_like_bytes(words,off,len(q))
    scales=4*len(q)
    variants={}
    for group in group_list:
        for kind in ("raw","zlib","planes","tier8","adaptive"):
            t=time.perf_counter()
            blob,modes=encode_grouped(words,off,q.shape[1],group=group,selector=kind)
            seconds=time.perf_counter()-t
            recovered,roff,width=decode_grouped(blob)
            if width!=q.shape[1] or not np.array_equal(roff,off) or not np.array_equal(words,recovered):
                raise AssertionError("Stage2 dictionary words NOT lossless")
            if not np.array_equal(q,d.decode(recovered,roff,width)):
                raise AssertionError("Stage1+Stage2 symbol mismatch")
            key=f"{kind}_g{group}"
            variants[key]={"bytes_with_scales":int(len(blob)+scales),
                           "seconds_stage2_encode":round(seconds,4),
                           "mode_counts":{str(k):int(v) for k,v in modes.items()}}
            if details and kind=="adaptive":
                variants[key].update(read_latency(blob,len(q)))
    cl8=cl8_encode(q)
    if not np.array_equal(q,cl8_decode(cl8)):raise AssertionError("CL8 symbols lost")
    return {"rows":int(q.shape[0]),"cols":int(q.shape[1]),
            "weights":int(q.size),"true_zero_fraction":float(np.mean(q==0)),
            "qmoe_style_dictionary_words":int(len(words)),
            "qmoe_style_stage1_bytes_with_scales":int(base),
            "cl8_bytes_with_scales":int(len(cl8)+scales),
            "stage1_cpu_python_seconds":round(t1,3),
            "variants":variants}

def candidate_permutations(qwi,qwo):
    n=qwi.shape[0]
    if n!=qwo.shape[1]:raise AssertionError("mismatched FFN")
    variants={"identity":np.arange(n,dtype=np.int32)}
    # Organize similar support density in adjacent FFN hidden neurons.
    cnt=np.count_nonzero(qwo,axis=0)
    variants["graph_degree"]=np.argsort(cnt,kind="stable").astype(np.int32)
    # Sort hidden neurons by a short 2-D signature across output directions:
    # geometric feature sorting of sparse connectivity, not claimed novel.
    rows=[0,3,17,71,131,257,411,701]
    variants["graph_signature"]=np.lexsort(tuple(
        qwo[i,:].astype(np.int16) for i in rows)).astype(np.int32)
    rng=np.random.default_rng(420)
    variants["random_control"]=rng.permutation(n).astype(np.int32)
    return variants

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="results/calibri_v11_real_dual_codec.json")
    a=ap.parse_args()
    rev=HfApi().model_info(MODEL).sha
    path=hf_hub_download(repo_id=MODEL,filename=CKPT,revision=rev)
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    state=torch.load(path,map_location="cpu",weights_only=True)
    keys=[PREFIX+x for x in TENSORS]
    if not all(k in state for k in keys):raise ValueError("no expected paired FFN weights")
    # Explicitly verify no hidden neuron biases have been missed.
    potential=[k for k in state if k.startswith(PREFIX)
               and (k.endswith("bias") or ".layer_norm" in k)]
    if potential:raise AssertionError("unhandled paired biases: "+str(potential[:5]))
    wio=state[keys[0]];woo=state[keys[1]]
    if tuple(wio.shape)!=(3072,768) or tuple(woo.shape)!=(768,3072):
        raise ValueError("unexpected trained expert shape")
    qwi,swi=row_ternarize(wio)
    qwo,swo=row_ternarize(woo)
    del state
    d=StaticPairDictionary()
    wi_base=encode_variants(d,qwi,details=True)
    wo_base=encode_variants(d,qwo,details=True)
    scales_total=4*(len(qwi)+len(qwo))
    raw_base=wi_base["qmoe_style_stage1_bytes_with_scales"]+wo_base["qmoe_style_stage1_bytes_with_scales"]
    cl8_base=wi_base["cl8_bytes_with_scales"]+wo_base["cl8_bytes_with_scales"]
    baseline_stage2=wi_base["variants"]["adaptive_g16"]["bytes_with_scales"]+wo_base["variants"]["adaptive_g16"]["bytes_with_scales"]
    variants=[]
    wi_actual=reconstructed(qwi,swi);wo_actual=reconstructed(qwo,swo)
    for label,p in candidate_permutations(qwi,qwo).items():
        func_rel,func_abs=stable_fp64_ffn_equivalence(wi_actual,wo_actual,p)
        if func_rel>1e-11:
            raise AssertionError(f"not function preserving: {label}={func_rel}")
        # Preprocessing permutation of a paired, exact ReLU FFN. Importantly,
        # a matching permutation of WI row scales is implicit.
        new_wi=qwi[p];new_wo=qwo[:,p]
        wi_enc=wi_base if label=="identity" else encode_variants(d,new_wi,group_list=(16,))
        wo_enc=wo_base if label=="identity" else encode_variants(d,new_wo,group_list=(16,))
        first=wi_enc["qmoe_style_stage1_bytes_with_scales"]+wo_enc["qmoe_style_stage1_bytes_with_scales"]
        second=wi_enc["variants"]["adaptive_g16"]["bytes_with_scales"]+wo_enc["variants"]["adaptive_g16"]["bytes_with_scales"]
        variants.append({
            "hidden_ordering":label,
            "paired_ReLU_function_relative_error":func_rel,
            "paired_ReLU_function_max_abs_error":func_abs,
            "first_dictionary_stage_bytes":first,
            "dual_stage_bytes":second,
            "change_vs_original_dual_bytes_pct":100*(second/baseline_stage2-1),
            "two_stage_gain_after_dictionary_pct":100*(1-second/first),
            "permutation_metadata_bytes_in_transformed_model":0,
            "caveat":"Requires BOTH FFN matrices and all paired biases/scales to be committed together"
        })
    report={
        "run":"CALIBRI_v0.11_real_weight_dictionary_then_geometry",
        "checkpoint_source":MODEL,"revision_sha":rev,"weights_file_sha256":h.hexdigest(),
        "trained_matrices":keys,"trained_weight_count":int(qwi.size+qwo.size),
        "quantization":"naive independently row-thresholded ternary with forced ~88.5% zeros; NOT QMoE GPTQ",
        "stage1":"precise QMoE sub1.py 65,536 pair-sequence dictionary construction + greedy CPU packing",
        "stage2":"short independently seekable blocks, raw/zlib/2D low-high byte planes, best per group",
        "dictionary_lut_bytes_shared_model_wide":LUT_BYTES,
        "dictionary_lut_amortization_warning":"512KiB shared LUT must be charged once per model; pair-only numbers exclude it, and are NOT whole-checkpoint ratios",
        "baseline_pair_stage1_bytes_excluding_shared_lut":raw_base,
        "baseline_pair_cl8_bytes":cl8_base,
        "baseline_pair_two_stage_adaptive_g16_bytes_excluding_shared_lut":baseline_stage2,
        "pair_gain_stage2_vs_stage1_percent":100*(1-baseline_stage2/raw_base),
        "wi":wi_base,"wo":wo_base,
        "functional_graph_ordering_candidates":variants,
        "limitations":["Model-level quality and CUDA/ARM execution NOT tested",
                       "No official QMoE compressed weights used in this experiment",
                       "zlib stage2 requires decompression, so storage gain may cost runtime",
                       "False claims about sub-0.8 bpw avoided by counting scales and indexes",
                       "No model bytes delivered to the Android phone"],
    }
    p=pathlib.Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"dictionary_lut_bytes":LUT_BYTES,
                      "trained_weight_count":report["trained_weight_count"],
                      "stage1_pair_bytes":raw_base,
                      "stage2_pair_bytes":baseline_stage2,
                      "stage2_saving_pct":report["pair_gain_stage2_vs_stage1_percent"],
                      "cl8_pair_bytes":cl8_base,
                      "permutations":variants},indent=2),flush=True)

if __name__=="__main__":main()
