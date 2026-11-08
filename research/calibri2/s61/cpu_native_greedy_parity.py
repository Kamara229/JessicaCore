"""S61 measured free-generation CPU timing and full-token parity for CB61.

Freeze exact S45 CB59 / S56 two-gate Green Tunnel and compare native greedy
entire answer token sequences after an *actual* CB61 file decode. This is a
TRAIN-only compatibility probe, not independent generalization/FP32 parity.
CPU speed measured on small frozen TRAIN sample is exploratory, not a final
independent CPU throughput certification. No GPU or Jessica chat involved.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, time
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
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s61.exact_entropy_cb59 import decode,verify_cb59,sha, SOURCE_SHA, SIDE_SHA
from v186.pairs import build_pairs

def build(archive, controller, tok):
    model,sources=create_small_paid_model(CB59(archive),build_config(MODEL_SPEC))
    model.requires_grad_(False)
    model.eval()
    head=model.lm_head.source().detach().cpu().contiguous()
    first=s47.ActiveGreenTunnel(320,2)
    after=s47.ActiveGreenTunnel(320,12)
    first.load_state_dict({k[6:]:v for k,v in controller.items() if k.startswith("first.")})
    after.load_state_dict({k[6:]:v for k,v in controller.items() if k.startswith("later.")})
    first.eval(); after.eval()
    return model,sources,first,after,head

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_original",type=Path)
    p.add_argument("S61_physical",type=Path)
    p.add_argument("S56_controller",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(6161)
    raw=a.S45_original.read_bytes()
    verify_cb59(raw)
    t0=time.perf_counter()
    recovered=decode(a.S61_physical.read_bytes())
    startup_decode_ms=(time.perf_counter()-t0)*1000
    if recovered!=raw:
        raise ValueError("physical entropy decode modified CB59 model")
    side=a.S56_controller.read_bytes()
    if sha(side)!=SIDE_SHA:raise ValueError("S56 physically paid two-gate controller altered")
    controller=load_file(str(a.S56_controller))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    original=s30.make_training_pairs(tok)
    aug=s54.build_new_train(tok)
    old=build_pairs(tok,"TRAIN")
    # Fixed TRAIN-only prompts from original, expanded and historical skills.
    samples=[
        ("original_left",original[0]["left"]["question"]),
        ("original_right",original[7]["right"]["question"]),
        ("augmented_left",aug[3]["left"]["question"]),
        ("augmented_right",aug[6]["right"]["question"]),
        ("historical_yesno",old[0]["left"]["question"]),
        ("historical_binding",old[19]["right"]["question"]),
    ]
    # Two separate model instances, exact immutable physical weights.
    baseline=build(raw,controller,tok)
    restored=build(recovered,controller,tok)
    timings={"original":[],"entropy_recovered":[]}
    sequences={"original":[],"entropy_recovered":[]}
    # Interleave timings by question rather than benchmarking separate
    # model runs with different CPU thermal/cache periods.
    for label,q in samples:
        pair=[]
        for tag,state in (("original",baseline),("entropy_recovered",restored)):
            model,sources,first,later,head=state
            t1=time.perf_counter()
            ids=s56.greedy_first_active(model,sources,first,later,head,tok,q)
            elapsed=time.perf_counter()-t1
            timings[tag].append(elapsed)
            sequences[tag].append(ids)
            pair.append(ids)
        if pair[0]!=pair[1]:
            raise AssertionError("S61 CB61 archive changed actual autoregressive token sequence")
    median_raw=statistics.median(timings["original"])
    median_new=statistics.median(timings["entropy_recovered"])
    speed_ratio=median_raw/median_new
    report={
        "stage":"CALIBRI_S61_SAME_FULL_GREEDY_NEURAL_RESPONSE_CPU",
        "physical_restored_S45_SHA256":hashlib.sha256(recovered).hexdigest(),
        "physical_S56_SHA256":hashlib.sha256(side).hexdigest(),
        "actual_full_free_answer_tokens_exactly_equal":True,
        "frozen_TRAIN_questions_sampled":len(samples),
        "native_model_cpu_engine":"PyTorch CB59 differentiable surrogate used by S56 (not original C direct kernel)",
        "CPU_original_seconds_per_question":timings["original"],
        "CPU_restored_seconds_per_question":timings["entropy_recovered"],
        "CPU_median_speed_ratio":speed_ratio,
        "exploratory_90_percent_speed_gate_pass":speed_ratio>=.9,
        "cold_CPU_decompression_onetime_ms":startup_decode_ms,
        "steady_state_extra_entropy_decode_ops_per_token":0,
        "not_independent_CPU_throughput_or_FP32_parity":True,
        "no_QA_TEST_DEV_S26_S29_S53_S55_S57":True,
        "no_GPU_no_Jessica_test_or_LIVE_changes":True,
    }
    a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()
