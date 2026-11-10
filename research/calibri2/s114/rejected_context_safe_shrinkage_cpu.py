"""S114 TRAIN-only physically guarded shrinkage/masking of REJECTED S112 gates.

S112 actually trained 340 prompt-conditioned shared GT coefficients on MPS.
Actual original S101 456-TRAIN CPU check found +1 new full/EOS but -7 old,
and the S112 atomic transaction rolled back. S114 is NOT a re-training.
Load the immutable rejected S112 *artifact* only for read-only ablations:
  - 0.25, 0.50, 0.75 times all 340 trained coefficients;
  - only donor GT3 at original gain;
  - only donor GT9 at original gain.
Each candidate has real physically saved FP16 sidecar, genuine original
S95/S98 greedy KV CPU outputs for ALL 456 original authorized TRAIN
questions, the original 353 full/EOS and 387 first-token retention guard,
0.700 bpp and paired same-host CPU speed >=90% of S101.
Do not promote or merge any candidate; independent QA remains sealed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s99.guard_neural_copy_456_cpu import load_frozen_numeric_weights
from calibri2.s102.hybrid_retention import strict_train_retention
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import (
    cohort, load_cpu_original, all_456_cached_cpu,
    verified_interleaved_cpu_speed,
)

# Hypotheses fixed in advance, never selected via once-inspected QA.
VARIANTS = (
    ("all_quarter", 0.25, (0,1)),
    ("all_half", 0.50, (0,1)),
    ("all_three_quarters", 0.75, (0,1)),
    ("donor3_only", 1.0, (0,)),
    ("donor9_only", 1.0, (1,)),
)
KEY = "conditional_dual_bridge.coefficients"


def main() -> None:
    p=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S101_full456_report","S111_parity_report",
        "S112_report","S112_rejected_sidecar","S112_runtime_config",
        "S87_report","S91_report","S93_report",
    ):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(1140114)
    torch.set_num_threads(2)

    p99=json.loads(a.S99_full456_report.read_text())
    p101=json.loads(a.S101_full456_report.read_text())
    p111=json.loads(a.S111_parity_report.read_text())
    p112=json.loads(a.S112_report.read_text())
    if (p101.get("cached_full_EOS_correct")!=353
            or p101.get("new_first_token_correct")!=387
            or p99.get("S99_new_correct_full_EOS")!=353
            or p111.get("stage")!="S111_PROMPT_CONDITIONED_ALL_12_ZERO_GAIN_PHYSICAL_CPU_TRAIN_456"
            or p111.get("full_456_exact_parity") is not True
            or p111.get("candidate_full_EOS_correct")!=353
            or p111.get("candidate_first_token_correct")!=387
            or p111.get("physical",{}).get("physically_below_0_7_bpp") is not True
            or p112.get("stage")!="S112_REAL_MPS_CONDITIONAL_340_FP16_456_CPU_TRANSACTION"
            or p112.get("real_MPS_steps")!=3
            or p112.get("trainable_parameter_count")!=340
            or p112.get("baseline_full")!=353
            or p112.get("candidate_full")!=347
            or p112.get("guard",{}).get("lost_full")!=7
            or p112.get("guard",{}).get("gained_full")!=1
            or p112.get("guard",{}).get("lost_first")!=0
            or p112.get("rollback",{}).get("committed") is not False
            or p112.get("model_accepted") is not False):
        raise AssertionError("S114 requires original rejected S112 full456 actual CPU outcome")

    for path,digest in (
        (a.S93_CB61,p99["immutable_S95_source_sha256"]),
        (a.S94_bank,p99["S94_weightbank_sha256"]),
        (a.S95_router,p99["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net,p99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes())!=digest:
            raise AssertionError("S101 physical CB61/profile/router/copy source drift")
    if (sha(a.S93_CB61.read_bytes())!=p112["S101_original_CB61_sha256"]
            or sha(a.S112_rejected_sidecar.read_bytes())!=p112["S112_sidecar_SHA256"]):
        raise AssertionError("S112 rejected candidate artifact provenance drift")

    old_side=load_file(str(a.S93_sidecar))
    rejected=load_file(str(a.S112_rejected_sidecar))
    if set(rejected)!=set(old_side)|{KEY}:
        raise AssertionError("Unexpected S112 extra physical tensors")
    for name in old_side:
        if not torch.equal(rejected[name],old_side[name]):
            raise AssertionError("S112 altered forbidden original source tensor")
    coeff=rejected[KEY]
    if (coeff.shape!=(2,10,17) or coeff.dtype!=torch.float16
            or not bool(torch.isfinite(coeff).all())
            or not bool(torch.count_nonzero(coeff).item())):
        raise AssertionError("S112 actual FP16 prompt condition coefficients absent")
    runtime=json.loads(a.S112_runtime_config.read_text())
    if (runtime.get("stage")!="S112_UNACCEPTED_PROMPT_CONDITIONED_340_FP16_COEFFS"
            or runtime.get("enable_only_route")!="S93"
            or runtime.get("sidecar_key")!=KEY
            or runtime.get("donor_layers")!=[3,9]):
        raise AssertionError("S112 physical conditional runtime descriptor drift")

    _,_,_,bank,router,scorer=load_frozen_numeric_weights(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    reports={p:freeze_report(getattr(a,p+"_report"))
             for p in ("S87","S91","S93")}
    rows,keys,routes=cohort(tok,reports,router)
    expected={key:rows[key]["expected"] for key in keys}
    state=load_cpu_original(
        s87.open_envelope(a.S93_CB61.read_bytes()),old_side)
    original=None
    measures=[]
    chosen=None
    try:
        original,_=all_456_cached_cpu(
            state,rows,keys,routes,tok,bank,scorer,bridge=None)
        old_full=sum(original[k]==expected[k] for k in keys)
        old_first=sum(bool(original[k]) and original[k][0]==expected[k][0]
                      for k in keys)
        if old_full!=353 or old_first!=387:
            raise AssertionError("Frozen actual S101 reference full/first quality drift")

        bridge=PromptConditionedDualTunnel(
            state[6],total_layers=12,donor_layers=(3,9)).eval().install(state[0])
        try:
            for label,alpha,donors in VARIANTS:
                new_coeff=torch.zeros_like(coeff)
                for d in donors:
                    new_coeff[d]=coeff[d].float().mul(alpha).to(torch.float16)
                candidate_side=dict(old_side)
                candidate_side[KEY]=new_coeff.contiguous()
                candidate_path=a.out/f"S114_UNACCEPTED_{label}_S93_SIDECAR.safetensors"
                save_file(candidate_side,str(candidate_path))
                roundtrip=load_file(str(candidate_path))
                if (set(roundtrip)!=set(candidate_side)
                        or any(not torch.equal(roundtrip[k],candidate_side[k])
                               for k in candidate_side)):
                    raise AssertionError("S114 physical FP16 sidecar roundtrip mismatch")
                bridge.load_physical_gain({KEY:roundtrip[KEY]})
                extra_bytes=(candidate_path.stat().st_size
                             -a.S93_sidecar.stat().st_size
                             +a.S112_runtime_config.stat().st_size)
                bpp=float(p101["S99_all_inclusive_physical_bpp"])+(
                    8*extra_bytes/SOURCE_PARAMS)
                if (not math.isfinite(bpp) or bpp>0.7
                        or abs(bpp-float(p112["bpp"]))>1e-8):
                    raise AssertionError("S114 physical all-inclusive bpp regression")

                proposed,_=all_456_cached_cpu(
                    state,rows,keys,routes,tok,bank,scorer,bridge=bridge)
                speed=verified_interleaved_cpu_speed(
                    state,rows,keys,routes,tok,bank,scorer,bridge,
                    original,proposed)
                guard=strict_train_retention(
                    expected,original,proposed,physical_bpp=bpp,
                    cpu_ratio=speed["ratio"],speed_required=True)
                case={
                    "name":label,
                    "original_gains_scaled_by":alpha,
                    "original_two_donors_enabled":list(donors),
                    "correct_full_EOS":sum(proposed[k]==expected[k] for k in keys),
                    "correct_first_token":sum(
                        bool(proposed[k]) and proposed[k][0]==expected[k][0]
                        for k in keys),
                    "lost_full":guard["lost_full"],
                    "lost_first":guard["lost_first"],
                    "gained_full":guard["gained_full"],
                    "all_original_353_full_and_387_first_protected":
                        guard["lost_full"]==0 and guard["lost_first"]==0,
                    "all_456_frozen_TRAIN_actual_CPU_verified":
                        guard["all_456_verified"],
                    "physical_bpp":bpp,
                    "added_physical_bytes":extra_bytes,
                    "paired_CPU_S101_ratio":speed["ratio"],
                    "paired_CPU_details":speed,
                    "safe_train_research_candidate":
                        guard["transaction_can_commit"] and guard["gained_full"]>0,
                    "new_physical_sha256":sha(candidate_path.read_bytes()),
                    "gained_full_identity_SHA256":[sha(x.encode()) for x in
                                                    guard["gained_full_identities"]],
                    "lost_full_identity_SHA256":[sha(x.encode()) for x in
                                                  guard["lost_full_identities"]],
                    "NO_model_accepted":True,
                }
                measures.append(case)
                print(json.dumps({
                    "event":"S114_CONDITIONAL_FP16_PHYSICAL_456_CPU_LINESEARCH",
                    "mask":label,"scale":alpha,"full":case["correct_full_EOS"],
                    "lost":case["lost_full"],"first_lost":case["lost_first"],
                    "gained":case["gained_full"],
                    "guard":guard["transaction_can_commit"],
                    "ratio":speed["ratio"],"physical_bpp":bpp
                }),flush=True)
                if case["safe_train_research_candidate"] and (
                    chosen is None or
                    (case["gained_full"],case["paired_CPU_S101_ratio"])>
                    (chosen["gained_full"],chosen["paired_CPU_S101_ratio"])
                ):
                    chosen=case
        finally:
            bridge.uninstall()
    finally:
        state[6].uninstall()

    report={
        "stage":"S114_REJECTED_S112_SHRINKAGE_AND_DONOR_MASK_FULL456_CPU_TRAIN",
        "predeclared_candidates":[x[0] for x in VARIANTS],
        "original_S101_full_353_first_387_confirmed":True,
        "original_S112_model_rejected_plus1_minus7":True,
        "results":measures,
        "safe_with_new_full_EOS_identity":None if chosen is None else chosen["name"],
        "max_safe_full_EOS_gained":0 if chosen is None else chosen["gained_full"],
        "no_optimizer_steps_or_new_training":True,
        "not_independent_QA_validation":True,
        "previously_inspected_QA_never_opened":True,
        "all_models_unaccepted":True,
        "LIVE_main_Render_Supabase_APK_untouched":True,
    }
    (a.out/"S114_PROTECTED_S112_CONTEXT_SHRINKAGE_REPORT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
    print(json.dumps({
        "event":"S114_CONTEXT_FP16_SHRINKAGE_END",
        "safe_candidate":report["safe_with_new_full_EOS_identity"],
        "max_safe_full_gained":report["max_safe_full_EOS_gained"],
        "model_accepted":False,
    }),flush=True)


if __name__=="__main__":
    main()
