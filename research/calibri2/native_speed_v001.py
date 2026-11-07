"""Read-only matched native C CPU throughput: CALIBRI2 packed vs v228 CB59.

Uses identical architecture, same ABI, same native C kernel, same CPU,
same fixed prompts and EXACTLY 8 greedy tokens. Only TESTING physical
trained source; original pretrained compressed archive used ONLY for
speed baseline, never for CALIBRI 2.0 initialization or training.
The model is not yet of usable response quality: speed is independent.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from direct_runtime_v59 import compile_kernel,NativeKernel,bind_into_llama
from v174.run import save

C2_SHA="a93b28ed577b536f544fd0ec168aab9b2dd74c0e0660d174a4175b7b954d5703"
OLD_SHA="0db01518534fb61e57a903f1c5155233993664ca34c4e7f9010fd9a1c8901483"
TOKEN_CAP=8
PROMPTS=(
    "Is 431 greater than 428? Reply with only Yes or No.",
    "In a story, Nara carries a green notebook and Teba carries a blue notebook. What color is Nara's notebook?",
    "What is 114 plus 6? Reply with only the number.",
)
def model_from_packed(path,sha,kernel):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:
        raise AssertionError("Not the exact complete physically trained source")
    archive=CB59(raw)
    if (archive.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE or
        8*len(raw)/EXPECTED_UNIQUE>0.700):
        raise AssertionError("Complete source geometry/physical budget wrong")
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(
            cfg,torch_dtype=torch.float32).cpu().eval()
    contract=bind_into_llama(model,archive,kernel)
    if contract["runtime_dense_multi_dim_source_parameters"]!=0:
        raise AssertionError("Unpaid dense FP32 source in native inference")
    return model,archive,contract

@torch.inference_mode()
def generate(model,ids):
    context=torch.tensor(ids,dtype=torch.long)[None]
    t0=time.perf_counter()
    tokens=[]
    for _ in range(TOKEN_CAP):
        logits=model(input_ids=context,use_cache=False).logits[0,-1]
        if not torch.isfinite(logits).all():
            raise AssertionError("Failed model-native logits")
        token=int(logits.argmax())
        tokens.append(token)
        context=torch.cat((context,context.new_tensor([[token]])),dim=1)
    return time.perf_counter()-t0,tokens

def main():
    p=argparse.ArgumentParser()
    p.add_argument("calibri2",type=Path)
    p.add_argument("v228",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    input_ids=[]
    for question in PROMPTS:
        text=tok.apply_chat_template(
            [{"role":"user","content":question}],
            tokenize=False,add_generation_prompt=True)
        ids=tok.encode(text,add_special_tokens=False)
        if not 2<len(ids)<128:
            raise AssertionError("Frozen common speed prompt length")
        input_ids.append(ids)
    protocol={
      "two_exact_sources":[{"kind":"CALIBRI2_v001_from_zero","SHA":C2_SHA},
                           {"kind":"CALIBRI_v228_original_compressed","SHA":OLD_SHA}],
      "one_C_native_packed_binary_kernel":True,"token_cap":TOKEN_CAP,
      "prompt_sha256":[hashlib.sha256(
              bytes(str(ids),"utf8")).hexdigest() for ids in input_ids],
      "n_frozen_prompt_templates":len(PROMPTS),
      "same_CPU_threads":2,"same_CPU_runtime_input_and_generation":True,
      "C2_trained_before_loading_reference_archive":True,
      "no_pretrained_FP32_weights_ever_used_to_train_C2":True,
      "performance_only_NOT_quality_parity":True,
    }
    save(a.out/"calibri2_v001_native_speed_protocol.json",protocol)
    kernel=NativeKernel(compile_kernel(
        Path(__file__).resolve().parents[1]/"calibri"/"direct_kernel_v59.c",
        a.out/"libcalibri2_native_speed.so"))
    models={}
    meta={}
    for name,path,sha in (("CB59_v228",a.v228,OLD_SHA),
                          ("CALIBRI2",a.calibri2,C2_SHA)):
        model,archive,info=model_from_packed(path,sha,kernel)
        models[name]=model
        meta[name]={
            "native_contract":info,"physically_paid_bytes":len(archive.blob),
            "complete_bpp":8*len(archive.blob)/EXPECTED_UNIQUE,
            "physically_source_sha256":archive.hash()}
    # Each model warms on exactly identical prompt length and 8-token work.
    for name in models:
        generate(models[name],input_ids[0])
    trials=[]
    for repeat in range(2):
        for k,ids in enumerate(input_ids):
            order=(("CB59_v228","CALIBRI2")
                   if (repeat+k)%2==0 else ("CALIBRI2","CB59_v228"))
            duration={}; generated={}
            for name in order:
                sec,pred=generate(models[name],ids)
                duration[name]=sec
                generated[name]=pred
            ratio=duration["CB59_v228"]/duration["CALIBRI2"]
            trials.append({"repeat":repeat,"case":k,"timings_seconds":duration,
                "ratio_CALIBRI2_vs_CB59":ratio,
                "both_generated_eight_real_tokens":all(
                     len(x)==TOKEN_CAP for x in generated.values()),
                "different_predictions_are_expected_not_a_speed_bias":True})
            print(json.dumps({"stage":"CALIBRI2_native_CPU_8tokens_pair",
                "repeat":repeat,"case":k,"seconds":duration,
                "speed_ratio_CALIBRI2_vs_old_CB59":round(ratio,4)}),flush=True)
    ratios=[x["ratio_CALIBRI2_vs_CB59"] for x in trials]
    if any(not x["both_generated_eight_real_tokens"] for x in trials):
        raise AssertionError("Unmatched token count")
    result={"complete":True,"compared":meta,"protocol":protocol,
        "paired_trials":trials,
        "median_native_inference_token_throughput_ratio_vs_CB59":
             statistics.median(ratios),
        "min_paired_ratio":min(ratios),
        "native_CPU_speed_at_least_90pct_CB59":statistics.median(ratios)>=.90,
        "same_native_kernel_and_compressed_operators":True,
        "does_NOT_evaluate_language_answer_quality":True}
    save(a.out/"ACTUAL_CALIBRI2_v001_native_CPU_speed_vs_v228.json",result)
    print(json.dumps({"stage":"CALIBRI2_v001_native_CPU_FINAL",
        "paired_trials":len(trials),
        "median_ratio":round(result["median_native_inference_token_throughput_ratio_vs_CB59"],5),
        "requirement_90pct_pass":result["native_CPU_speed_at_least_90pct_CB59"]}),flush=True)

if __name__=="__main__":
    main()
