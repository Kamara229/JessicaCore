"""CALIBRI S64 CPU-only rank128 initialization parity and speed.

No model training, no independent QA, no Jessica chat, no GPU.
The candidate still needs fresh blind-quality evaluation after any training.
S56 rank12 is the baseline; S64 rank128 is a new physical CPU-deployed
candidate with first-rank2 and the original later rank12 subspace frozen.
"""
from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s61.exact_entropy_cb59 import verify_cb59,sha,SIDE_SHA
from calibri2.s63.cpu_greedy_from_native_c_restored import make_samples

def instantiate(raw,weights,rank):
    model,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    model.requires_grad_(False);model.eval()
    head=model.lm_head.source().detach().cpu().contiguous()
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,rank)
    first.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("later.")})
    first.eval();later.eval()
    return model,sources,first,later,head

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_original_CB59",type=Path)
    p.add_argument("S56_rank12_controller",type=Path)
    p.add_argument("S64_rank128_controller",type=Path)
    p.add_argument("S64_paid_manifest",type=Path)
    p.add_argument("output_json",type=Path)
    a=p.parse_args()
    a.output_json.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(640128);torch.set_num_threads(2)
    raw=a.S45_original_CB59.read_bytes()
    verify_cb59(raw)
    old=a.S56_rank12_controller.read_bytes()
    if sha(old)!=SIDE_SHA:raise ValueError("baseline SHA mismatch")
    state0=load_file(str(a.S56_rank12_controller))
    state1=load_file(str(a.S64_rank128_controller))
    paid=json.loads(a.S64_paid_manifest.read_text())
    if paid["controller_sha256"]!=sha(a.S64_rank128_controller.read_bytes()) or float(paid["all_inclusive_bpp"])>.7:
        raise ValueError("new physical candidate not budget sealed")
    for k,v in state0.items():
        if k=="later.down.weight":
            if not torch.equal(v,state1[k][:12]):raise AssertionError("old down changed")
        elif k=="later.up.weight":
            if not torch.equal(v,state1[k][:,:12]):raise AssertionError("old up changed")
        elif not torch.equal(v,state1[k]):raise AssertionError("old S56 gate/first changed: "+k)
    assert torch.count_nonzero(state1["later.up.weight"][:,12:]).item()==0
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    samples=make_samples(tok)
    models={"S56_rank12":instantiate(raw,state0,12),
            "S64_rank128_init":instantiate(raw,state1,128)}
    timings={name:[] for name in models}
    full_correct={name:[] for name in models}
    outputs_match=True
    for i,(category,key,question,expected) in enumerate(samples):
        observed={}
        order=list(models) if i%2==0 else list(reversed(models))
        for k in order:
            model,sources,first,later,head=models[k]
            start=time.perf_counter()
            got=s56.greedy_first_active(model,sources,first,later,head,tok,question)
            elapsed=time.perf_counter()-start
            observed[k]=got
            timings[k].append(elapsed)
            full_correct[k].append(got==expected)
        if observed["S56_rank12"]!=observed["S64_rank128_init"]:
            outputs_match=False
    t0=timings["S56_rank12"]
    t1=timings["S64_rank128_init"]
    medratio=statistics.median(t0)/statistics.median(t1)
    report={
        "stage":"CALIBRI_S64_PAID_RANK128_CPU_GREEDY_INIT_PARITY",
        "sample_count":len(samples),
        "all_full_native_greedy_outputs_identical":outputs_match,
        "original_rank12_correct_full_EOS":sum(full_correct["S56_rank12"]),
        "rank128_initialization_correct_full_EOS":sum(full_correct["S64_rank128_init"]),
        "cpu_seconds_per_prompt":timings,
        "cpu_median_prompt_speed_ratio_new_vs_baseline":medratio,
        "cpu_90_percent_target_met_on_frozen_TRAIN_sample":medratio>=.90,
        "paid_all_inclusive_bpp":paid["all_inclusive_bpp"],
        "rank128_controller_SHA":paid["controller_sha256"],
        "original_rank12_first_and_later_embedding_unchanged":True,
        "rank128_training_performed":False,
        "no_heldout_QA_or_FP32_parity_measured":True,
        "not_promoted_until_quality_and_speed_after_training":True,
        "no_GPU_no_Jessica_chat_test_no_LIVE_changes":True,
    }
    a.output_json.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="cpu_seconds_per_prompt"}),flush=True)
    if not outputs_match:
        raise AssertionError("expansion changed full native S56 answers")

if __name__=="__main__":main()
