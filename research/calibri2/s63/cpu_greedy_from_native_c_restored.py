"""CALIBRI S63: native-C decoded model -> actual unchanged CPU greedy responses.

S62 compiled native C restores exact CB59 bytes. This script then instantiates
S56's original PyTorch CPU inference engine from those *C-restored bytes* and
compares to the immutable original source on the same TRAIN prompts. No new
weights, no sealed QA, no Jessica test, no GPU.

Crucial distinction: native C performs decoding only. Native C neural
generation is NOT implemented by this experiment and must not be claimed.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s61.cpu_native_greedy_parity import build
from calibri2.s61.exact_entropy_cb59 import verify_cb59, sha, SIDE_SHA
from v186.pairs import build_pairs

def make_samples(tok):
    s30.TOK=tok
    original=s30.make_training_pairs(tok)
    augmented=s54.build_new_train(tok)
    historical=build_pairs(tok,"TRAIN")
    assert len(original)==32 and len(augmented)==32 and len(historical)==24
    selection=(("original",original,(0,7,15)),
               ("augmented",augmented,(3,6,18)),
               ("historical",historical,(0,19)))
    prompts=[]
    for group,pairs,ids in selection:
        for i in ids:
            for side in ("left","right"):
                row=pairs[i][side]
                expected=tok.encode(row["answer"],add_special_tokens=False)+[tok.eos_token_id]
                prompts.append((group,f"{i}_{side}",row["question"],expected))
    assert len(prompts)==16
    return prompts

def run(state,tok,q):
    model,sources,first,later,head=state
    t0=time.perf_counter()
    actual=s56.greedy_first_active(model,sources,first,later,head,tok,q)
    return actual,time.perf_counter()-t0

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_original",type=Path)
    p.add_argument("S62_native_C_restored_CB59",type=Path)
    p.add_argument("S56_controller",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(630063)
    src=a.S45_original.read_bytes()
    decoded=a.S62_native_C_restored_CB59.read_bytes()
    verify_cb59(src)
    verify_cb59(decoded)
    assert decoded==src, "The C decoder's physical model bytes changed"
    tunnel=a.S56_controller.read_bytes()
    assert sha(tunnel)==SIDE_SHA
    sidecar=load_file(str(a.S56_controller))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    prompts=make_samples(tok)
    states={"S56_original":build(src,sidecar,tok),
            "S61_CB61_C_restored":build(decoded,sidecar,tok)}
    records={k:[] for k in states}
    correctness={k:[] for k in states}
    all_same=True
    for index,(category,key,q,expected) in enumerate(prompts):
        # Alternate execution order to reduce systematic cache/thermal bias.
        order=list(states) if index%2==0 else list(reversed(states))
        produced={}
        for label in order:
            actual,duration=run(states[label],tok,q)
            records[label].append({"group":category,"row":key,
                "generated_tokens":len(actual),"time_seconds":duration})
            correctness[label].append(actual==expected)
            produced[label]=actual
        if produced["S56_original"]!=produced["S61_CB61_C_restored"]:
            all_same=False
            raise AssertionError(f"Actual greedy response changed: {category}/{key}")
    t1=[item["time_seconds"] for item in records["S56_original"]]
    t2=[item["time_seconds"] for item in records["S61_CB61_C_restored"]]
    ratio=statistics.median(t1)/statistics.median(t2)
    by_group={}
    for group in ("original","augmented","historical"):
        indexes=[i for i,prompt in enumerate(prompts) if prompt[0]==group]
        by_group[group]={
            "sample_count":len(indexes),
            "S56_exact_answer_plus_EOS":sum(correctness["S56_original"][i] for i in indexes),
            "C_restored_exact_answer_plus_EOS":sum(correctness["S61_CB61_C_restored"][i] for i in indexes)}
    report={
        "stage":"CALIBRI_S63_NATIVE_C_PHYSICAL_RESTORE_AND_CPU_GREEDY_PARITY",
        "C_restored_SHA256":hashlib.sha256(decoded).hexdigest(),
        "all_16_full_responses_exact_token_parity":all_same,
        "original_train_samples":16,
        "correct_answers_by_group":by_group,
        "cpu_seconds_per_prompt":records,
        "cpu_total_seconds_source":sum(t1),
        "cpu_total_seconds_C_restored":sum(t2),
        "cpu_median_prompt_time_ratio":ratio,
        "cpu_ratio_exploratory":True,
        "native_C_decodes_weights_only":True,
        "native_C_neural_generation_benchmark_measured":False,
        "identical_binary_weights_guarantee_same_deterministic_algorithm_not_FP32_parity":True,
        "no_test_QA_S26_S29_S53_S55_S57":True,
        "no_GPU_no_Jessica_chat_test_no_LIVE_mutations":True,
    }
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({"stage":report["stage"],
        "full_response_parity":all_same,
        "correct":by_group,
        "cpu_median_time_ratio":ratio,
        "native_C_neural_generation_measured":False}),flush=True)

if __name__=="__main__":main()
