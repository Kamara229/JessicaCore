"""CALIBRI S37 physical 36M native C CPU audit, paired against exact S26.

One closed-output check of the newly trained physical CB59 archive (same C
direct-resident kernel, same architecture, same prompts, 8 generated tokens).
No neural QA/TEST labels used, no checkpoint selection by runtime speed.
No Jessica chat test or LIVE deployment. This is native CPU throughput
relative to S26 physical runtime, NOT a 135M FP32 quality reference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from direct_runtime_v59 import compile_kernel, NativeKernel, bind_into_llama
from calibri2.s0.bootstrap import build_config
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC
from calibri2.s30.matched_dense_fp32_train import SOURCE_SHA as SOURCE_S26_SHA
from v174.run import save

SOURCE_S37_SHA = "e7f8b549eebb117039e2953de6270c605b05b27e6e45fe44c882d68b0f829a0a"
TOKEN_CAP = 8
QUESTIONS = (
    "Is 321 greater than 312? Reply with only Yes or No.",
    "In a story, Nara has a red folder. What two words describe it?",
    "What are the two secret words 'quiet harbor'?",
)
REPEATS = 3


def resident_model(path, expected, cfg, kernel):
    raw = path.read_bytes()
    if len(raw)!=SOURCE_BYTES or hashlib.sha256(raw).hexdigest()!=expected:
        raise AssertionError("S37 native C audit source archive hash drift")
    cb=CB59(raw)
    bpp = 8*len(raw)/cb.meta["original_unique_parameter_count"]
    if cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS or bpp>.7:
        raise AssertionError("S37 incorrect unique original 36M physical budget")
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    info=bind_into_llama(model,cb,kernel)
    if info["runtime_dense_multi_dim_source_parameters"]!=0:
        raise AssertionError("S37 native CPU leaked dense multi-dimensional matrix")
    return model,info


@torch.inference_mode()
def fixed_tokens(model, ids):
    context = torch.tensor([ids],dtype=torch.long)
    out=[]
    started=time.perf_counter()
    for _ in range(TOKEN_CAP):
        logits=model(input_ids=context,use_cache=False).logits[0,-1]
        if not torch.isfinite(logits).all():
            raise AssertionError("S37 nonfinite native C CPU logits")
        nxt=int(logits.argmax())
        out.append(nxt)
        context=torch.cat((context,context.new_tensor([[nxt]])),dim=1)
    return time.perf_counter()-started,out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("s26",type=Path)
    p.add_argument("s37",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    config=build_config(MODEL_SPEC)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    ids=[tok.encode(tok.apply_chat_template(
        [{"role":"user","content":q}],tokenize=False,
        add_generation_prompt=True),add_special_tokens=False)
        for q in QUESTIONS]
    if not all(2<len(v)<120 for v in ids):
        raise AssertionError("S37 CPU prompts malformed")
    binary=Path(__file__).resolve().parents[2]/"calibri"/"direct_kernel_v59.c"
    kernel=NativeKernel(compile_kernel(binary,a.out/"libs37_native.so"))
    original,orig_info=resident_model(a.s26,SOURCE_S26_SHA,config,kernel)
    trained,new_info=resident_model(a.s37,SOURCE_S37_SHA,config,kernel)
    for m in (original,trained):
        fixed_tokens(m,ids[0]) # same preload warmup, timed separately.
    samples=[]
    for rep in range(REPEATS):
        for i,prompt in enumerate(ids):
            order=("S26","S37") if (rep+i)%2==0 else ("S37","S26")
            results={}
            seq={}
            for name in order:
                model=original if name=="S26" else trained
                results[name],seq[name]=fixed_tokens(model,prompt)
            if results["S26"]<=0 or results["S37"]<=0:
                raise AssertionError("Invalid native runtime measurements")
            ratio=results["S26"]/results["S37"]
            samples.append({"repetition":rep,"prompt":i,
                            "seconds":results,"ratio_S37_to_S26":ratio,
                            "generated_tokens":seq})
    median=statistics.median(x["ratio_S37_to_S26"] for x in samples)
    report={
      "stage":"S37_NATIVE_C_PHYSICAL_CPU_PAIRED_AGAINST_S26",
      "S26_sha256":SOURCE_S26_SHA,
      "S37_sha256":SOURCE_S37_SHA,
      "parameters":SOURCE_PARAMS,
      "all_model_bytes":SOURCE_BYTES,
      "complete_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
      "token_count_per_sample":TOKEN_CAP,
      "samples":len(samples),
      "median_S37_over_S26_native_CPU_speed":median,
      "S37_cpu_speed_at_least_90pct_S26":median>=.90,
      "same_cpu_threads":2,
      "same_model_architecture":True,
      "same_C_lookup_kernel":True,
      "S26_native_binding":orig_info,
      "S37_native_binding":new_info,
      "native_full_2D_dense_FP32_source_absent":True,
      "not_135M_FP32_quality_baseline":True,
      "speed_not_model_quality":True,
      "Jessica_chat_test_never_run":True,
      "LIVE_unmodified":True,
      "paired_timings":samples,
    }
    save(a.out/"S37_ACTUAL_PHYSICAL_NATIVE_CPU_SPEED.json",report)
    print(json.dumps({"event":"S37_NATIVE_C_CPU_VERIFIED",
                      "median_S37_over_S26":median,
                      "meets_90pct":median>=.90,
                      "physical_bpp":report["complete_bpp"]}),flush=True)


if __name__=="__main__":
    main()
