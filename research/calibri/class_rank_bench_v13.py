"""CALIBRI v0.13 reproducible pilot/holdout real MoE expert codec study.

Select 8/16/32/64 occupancy class-rank BLOCK SIZE on expert_0 only.
Evaluate on UNSEEN expert_1..expert_7 using exactly the selected choice.
Report all genuine physical bytes incl row-scale FP16 and indexes.
Other codecs: CL8-I and C12 MASK-HUFF. 88.5% zeros are IMPOSED by
simple naive rowwise quantization: NOT QMoE GPTQ, not model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from huggingface_hub import HfApi,hf_hub_download

from index import encode as cl8_encode
from huffman_masks_v12 import encode as mask_encode
from class_rank_v13 import encode as rank_encode,decode as rank_decode,read_row,direct_matvec
from hybrid_v13 import encode as hybrid_encode,decode as hybrid_decode
from dual_real_bench import MODEL,CKPT,row_ternarize,reconstructed

KEY_PATTERN=re.compile(r"decoder\.block\.1\.layer\.2\.mlp\.experts\.expert_(\d+)\.(wi|wo)\.weight$")
BLOCKS=(8,16,32,64)
GROUP=64

def encode_options(q,options,run_math=False,seed=42):
    scales=4*q.shape[0]
    cl8=len(cl8_encode(q))+scales
    mask=len(mask_encode(q,group=GROUP,index_mode='auto'))+scales
    sizes={}
    for block in options:
        encoded=rank_encode(q,block=block,group=GROUP,prior='auto')
        size=len(encoded)+scales
        if not np.array_equal(q,rank_decode(encoded)):
            raise AssertionError(f"CLASS-RANK roundtrip error at {block}")
        sizes[str(block)]=size
    hybrid_blob,hybrid_info=hybrid_encode(q,group=GROUP)
    if not np.array_equal(q,hybrid_decode(hybrid_blob)):
        raise AssertionError("v13 hybrid lost coefficients")
    return {"hybrid_codec":hybrid_info["selection"],
            "HYBRID_bytes_with_fp16_levels":len(hybrid_blob)+scales,
            "hybrid_compared_sizes":hybrid_info["bytes_by_codec"],
            "shape":list(q.shape),
            "parameters":int(q.size),
            "zero_fraction":float(np.mean(q==0)),
            "CL8_bytes_with_fp16_levels":cl8,
            "MASK_HUFF_bytes_with_fp16_levels":mask,
            "CLASS_RANK_bytes_with_fp16_levels":sizes}

def test_matvec(q,levels,block,seed=42):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=q.shape[1])
    stream=rank_encode(q,block=block,group=GROUP)
    start=time.perf_counter()
    streamed=direct_matvec(stream,x,levels)
    elapsed=time.perf_counter()-start
    expected=reconstructed(q,levels)@x
    residual=streamed-expected
    rel=float(np.linalg.norm(residual)/max(np.linalg.norm(expected),1e-20))
    if rel>1e-12:raise AssertionError(f"class rank matvec failed {rel}")
    choices=rng.integers(0,len(q),size=48)
    timings=[]
    for i in choices:
        t=time.perf_counter_ns()
        row=read_row(stream,int(i))
        timings.append((time.perf_counter_ns()-t)/1e6)
        if not np.array_equal(row,q[int(i)]):
            raise AssertionError(f"class rank row fail {i}")
    return {"cpu_python_matvec_seconds":elapsed,
            "fp64_relative_error_vs_same_ternary":rel,
            "fp64_max_abs_error_vs_same_ternary":float(np.max(np.abs(residual))),
            "cpu_python_row_read_p50_ms":float(np.percentile(timings,50)),
            "cpu_python_row_read_p95_ms":float(np.percentile(timings,95)),
            "random_rows_checked":len(choices)}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--output",default="results/calibri_v13_real_class_rank.json")
    args=a.parse_args()
    revision=HfApi().model_info(MODEL).sha
    ckpt=hf_hub_download(repo_id=MODEL,filename=CKPT,revision=revision)
    digest=hashlib.sha256()
    with open(ckpt,"rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""):
            digest.update(chunk)
    model=torch.load(ckpt,map_location="cpu",weights_only=True)
    keys={}
    for key in model:
        match=KEY_PATTERN.fullmatch(key)
        if match:
            e=int(match.group(1));kind=match.group(2)
            keys[(e,kind)]=key
    if len(keys)!=16 or any((e,t) not in keys for e in range(8) for t in ("wi","wo")):
        raise RuntimeError(f"expected 8 paired experts / 16 tensors; got {keys}")
    records=[];pilot=[];candidates=[]
    for sparsity in (0.5,0.75,0.885):
        for kind in ("wi","wo"):
            key=keys[(0,kind)]
            q,levels=row_ternarize(model[key],sparsity)
            res=encode_options(q,BLOCKS)
            res.update({"expert":0,"name":kind,"sparsity_target":sparsity,"set":"pilot"})
            pilot.append(res)
    # Only the .885 pilot is allowed to tune block size for the .885 holdout
    for block in BLOCKS:
        n=sum(r["CLASS_RANK_bytes_with_fp16_levels"][str(block)] for r in pilot
              if abs(r["sparsity_target"]-0.885)<1e-8)
        candidates.append({"block":block,"pilot_bytes":n})
    chosen=min(candidates,key=lambda x:(x["pilot_bytes"],x["block"]))["block"]
    heldout=[]
    for expert in range(1,8):
        for kind in ("wi","wo"):
            key=keys[(expert,kind)]
            q,levels=row_ternarize(model[key],0.885)
            res=encode_options(q,(chosen,))
            res.update({"expert":expert,"name":kind,"sparsity_target":0.885,"set":"holdout"})
            heldout.append(res)
            # No direct-matvec claim from a subset of arbitrary flattened bits:
            # independently verify each entire row and all coefficients.
            if expert==1 and kind=="wi":
                res["operator_check"]=test_matvec(q,levels,chosen,seed=20261101)
    count=sum(t["parameters"] for t in heldout)
    summaries={
        "weight_count":count,
        "CL8_bytes":sum(t["CL8_bytes_with_fp16_levels"] for t in heldout),
        "MASK_HUFF_bytes":sum(t["MASK_HUFF_bytes_with_fp16_levels"] for t in heldout),
        "CLASS_RANK_bytes":sum(t["CLASS_RANK_bytes_with_fp16_levels"][str(chosen)] for t in heldout),
        "HYBRID_bytes":sum(t["HYBRID_bytes_with_fp16_levels"] for t in heldout),
        "HYBRID_choice_histogram":{name:sum(t["hybrid_codec"]==name for t in heldout)
            for name in ("CL8","MASK_HUFF","CLASS_RANK16","CLASS_RANK32","CLASS_RANK64")}
    }
    summaries["CLASS_RANK_bpw"]=8*summaries["CLASS_RANK_bytes"]/count
    summaries["HYBRID_bpw"]=8*summaries["HYBRID_bytes"]/count
    summaries["HYBRID_vs_CL8_saved_pct"]=100*(1-summaries["HYBRID_bytes"]/summaries["CL8_bytes"])
    summaries["HYBRID_vs_MASK_HUFF_saved_pct"]=100*(1-summaries["HYBRID_bytes"]/summaries["MASK_HUFF_bytes"])
    summaries["vs_CL8_saved_pct"]=100*(1-summaries["CLASS_RANK_bytes"]/summaries["CL8_bytes"])
    summaries["vs_MASK_HUFF_saved_pct"]=100*(1-summaries["CLASS_RANK_bytes"]/summaries["MASK_HUFF_bytes"])
    report={
        "experiment":"CALIBRI_v13_real_pretrained_MoE_CLASS_RANK",
        "model":MODEL,"model_revision_sha":revision,
        "checkpoint_sha256":digest.hexdigest(),
        "quantization":"naive per-row threshold 50/75/88.5% sparsity on PILOT; 88.5% on heldout; NO QMoE GPTQ",
        "method":"fixed-prior Huffman cardinality + combinatorial support rank + sign bits",
        "prior_options":[.05,.115,.25,.50],
        "pilot_expert":0,
        "heldout_experts":[1,2,3,4,5,6,7],
        "pilot_measured":pilot,
        "pilot_block_candidates":candidates,
        "frozen_selected_block":chosen,
        "holdout_measured":heldout,
        "heldout_aggregate":summaries,
        "caveats":["No full-model task-quality, model perplexity or QMoE GPTQ check",
                   "Compression uses simple forced sparsity and exactly reproduces it",
                   "Huffman/count/combinatorial code are established techniques, novelty not proven",
                   "No ARM/GPU fused kernel measured; Python CPU timings only",
                   "No model was installed on user phone, no production changes"]
    }
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"pilot":candidates,"selected_block":chosen,
                      "heldout":summaries},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
