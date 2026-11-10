"""S115: real frozen S101 456 TRAIN decoder step mask evaluation of S112.

S112 real MPS trained 340 prompt-conditioned coefficients; original S101
autogreedy showed one NEW full answer and seven LOST old full answers, but
a first-token improvement 387->388 with ZERO previously correct first
tokens forgotten. All S112 changes were therefore rejected and rolled back.
Test whether disabling the rejected prompt-conditioned GT at later KV-cache
steps preserves its first-token benefit while preventing late/EOS damage.

NO learning or QA/TEST. Load original actual S101 immutable CB61/S94/S95/S98,
and exactly the REJECTED, immutable physical S112 FP16 sidecar. Compare
five predeclared decode-phase scopes (all, prefill-only, prefill+second,
later-decode only, second-only). Re-generate all 456 original authorized
TRAIN full+EOS greedy output, enforce 353 full +387 first retention,
physical all-in <=.7 bpp and paired original S101 CPU >=.9 speed.
Never call any model accepted or merge to main/LIVE.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s95.train_and_audit_router import prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    apply_profile,load_frozen_numeric_weights)
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s102.hybrid_retention import strict_train_retention
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import (
    cohort,load_cpu_original,all_456_cached_cpu)
from calibri2.s115.cached_phase_gate import VALID_PHASES,frozen_cpu_phase_scope

KEY="conditional_dual_bridge.coefficients"
SPEED_SAMPLE=48


def candidate_actual_CPU(state,rows,keys,routes,tok,bank,scorer,bridge,phase):
    model,sources,first,later,pointer,head,tunnel,centroids,old_tunnel=state
    chosen=None
    out={}
    for key in keys:
        profile=routes[key]
        if profile!=chosen:
            apply_profile(sources,tunnel,profile,bank,centroids,old_tunnel)
            chosen=profile
        question=rows[key]["question"]
        features=None
        if profile=="S93":
            features=torch.as_tensor(
                prompt_only_vector(tok,question),dtype=torch.float32)
        with torch.inference_mode():
            with frozen_cpu_phase_scope(
                    model,bridge,profile,features,phase=phase):
                tokens,_=cached_full_cpu_with_neural_copy(
                    model,sources,first,later,head,tok,question,pointer,scorer)
        out[key]=tokens
    return out


def paired_speed_original_vs_phase(
    state,rows,keys,routes,tok,bank,scorer,bridge,phase,original,proposed,
):
    """48 strictly paired S101 full autoregressive CPU measurements.

    Alternate order across prompts to prevent the S109 cold-vs-warm
    timing anomaly. Recheck exact IDs in both original and masked passes.
    """
    model,sources,first,later,pointer,head,tunnel,centroids,old_tunnel=state
    selected=None
    before=[]
    after=[]
    for index,key in enumerate(keys[:SPEED_SAMPLE]):
        profile=routes[key]
        if selected!=profile:
            apply_profile(sources,tunnel,profile,bank,centroids,old_tunnel)
            selected=profile
        question=rows[key]["question"]
        features=None
        if profile=="S93":
            features=torch.as_tensor(
                prompt_only_vector(tok,question),dtype=torch.float32)
        def original_once():
            bridge.uninstall()
            try:
                start=time.perf_counter()
                answer,_=cached_full_cpu_with_neural_copy(
                    model,sources,first,later,head,tok,question,pointer,scorer)
                elapsed=time.perf_counter()-start
            finally:
                bridge.install(model)
            if answer!=original[key]:
                raise AssertionError("S115 original S101 paired speed token drift")
            before.append(elapsed)
        def candidate_once():
            with torch.inference_mode():
                with frozen_cpu_phase_scope(
                        model,bridge,profile,features,phase=phase):
                    start=time.perf_counter()
                    answer,_=cached_full_cpu_with_neural_copy(
                        model,sources,first,later,head,tok,
                        question,pointer,scorer)
                    elapsed=time.perf_counter()-start
            if answer!=proposed[key]:
                raise AssertionError("S115 masked S101 paired speed token drift")
            after.append(elapsed)
        if index%2:
            candidate_once()
            original_once()
        else:
            original_once()
            candidate_once()
    if len(before)!=SPEED_SAMPLE or len(after)!=SPEED_SAMPLE:
        raise AssertionError("S115 absent paired CPU speed samples")
    return {
        "ratio":statistics.median(before)/statistics.median(after),
        "baseline_median_seconds":statistics.median(before),
        "mask_median_seconds":statistics.median(after),
        "paired_examples":SPEED_SAMPLE,
        "alternate_order":True,
    }


def main():
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
            or p112.get("candidate_first")!=388
            or p112.get("guard",{}).get("lost_full")!=7
            or p112.get("guard",{}).get("lost_first")!=0
            or p112.get("guard",{}).get("gained_full")!=1
            or p112.get("rollback",{}).get("committed") is not False
            or p112.get("model_accepted") is not False):
        raise AssertionError("S115 must start from genuine rejected S112 +1/-7 and first+1 evidence")

    for path,digest in (
        (a.S93_CB61,p99["immutable_S95_source_sha256"]),
        (a.S94_bank,p99["S94_weightbank_sha256"]),
        (a.S95_router,p99["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net,p99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes())!=digest:
            raise AssertionError("S115 physical S101 source artifact SHA drift")
    if (sha(a.S93_CB61.read_bytes())!=p112["S101_original_CB61_sha256"]
            or sha(a.S112_rejected_sidecar.read_bytes())!=p112["S112_sidecar_SHA256"]):
        raise AssertionError("S115 S112 rejected candidate artifact SHA drift")

    original_side=load_file(str(a.S93_sidecar))
    rejected=load_file(str(a.S112_rejected_sidecar))
    if set(rejected)!=set(original_side)|{KEY}:
        raise AssertionError("S115 rejects extra/unexpected model weights")
    for key in original_side:
        if not torch.equal(rejected[key],original_side[key]):
            raise AssertionError("S115 forbidden S101 base weight changed")
    if (rejected[KEY].shape!=(2,10,17)
            or rejected[KEY].dtype!=torch.float16
            or not bool(torch.isfinite(rejected[KEY]).all())):
        raise AssertionError("S115 original 340 FP16 coefficients invalid")
    config112=json.loads(a.S112_runtime_config.read_text())
    if (config112.get("enable_only_route")!="S93"
            or config112.get("sidecar_key")!=KEY):
        raise AssertionError("S112 frozen runtime ABI lost")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    reports={p:freeze_report(getattr(a,p+"_report")) for p in ("S87","S91","S93")}
    _,_,_,bank,router,scorer=load_frozen_numeric_weights(a)
    rows,keys,routes=cohort(tok,reports,router)
    expected={key:rows[key]["expected"] for key in keys}
    state=load_cpu_original(
        s87.open_envelope(a.S93_CB61.read_bytes()),original_side)
    full=[]
    winner=None
    try:
        base,_=all_456_cached_cpu(
            state,rows,keys,routes,tok,bank,scorer,bridge=None)
        if (sum(base[k]==expected[k] for k in keys)!=353
                or sum(bool(base[k]) and base[k][0]==expected[k][0]
                       for k in keys)!=387):
            raise AssertionError("S115 original S101 TRAIN baseline changed")

        bridge=PromptConditionedDualTunnel(
            state[6],total_layers=12,donor_layers=(3,9)).eval()
        bridge.load_physical_gain({KEY:rejected[KEY]})
        bridge.install(state[0])
        try:
            for phase in VALID_PHASES:
                # Every phase is a separate hypothetical deployment bundle
                # with same single physical FP16 coefficients plus ONE
                # phase-specific counted runtime descriptor.
                target=a.out/f"S115_UNACCEPTED_{phase}_S93_SIDECAR.safetensors"
                save_file(dict(rejected),str(target))
                roundtrip=load_file(str(target))
                if (set(roundtrip)!=set(rejected) or
                        any(not torch.equal(roundtrip[k],rejected[k]) for k in rejected)):
                    raise AssertionError("S115 learned FP16 sidecar physical roundtrip failed")
                cfg=dict(config112)
                cfg["stage"]="S115_UNACCEPTED_TEMPORAL_GREEN_TUNNEL"
                cfg["green_tunnel_kv_phase"]=phase
                cfg_path=a.out/f"S115_UNACCEPTED_{phase}_RUNTIME_CONFIG.json"
                cfg_path.write_text(json.dumps(
                    cfg,sort_keys=True,separators=(",",":")))
                physical_bytes=(target.stat().st_size
                                -a.S93_sidecar.stat().st_size
                                +cfg_path.stat().st_size)
                bpp=float(p101["S99_all_inclusive_physical_bpp"])+(
                    8*physical_bytes/SOURCE_PARAMS)
                if not math.isfinite(bpp) or bpp>0.7:
                    raise AssertionError("S115 physical all-inclusive 0.7 bpp exceeded")
                bridge.load_physical_gain({KEY:roundtrip[KEY]})
                proposed=candidate_actual_CPU(
                    state,rows,keys,routes,tok,bank,scorer,bridge,phase)
                speed=paired_speed_original_vs_phase(
                    state,rows,keys,routes,tok,bank,scorer,bridge,
                    phase,base,proposed)
                guard=strict_train_retention(
                    expected,base,proposed,physical_bpp=bpp,
                    cpu_ratio=speed["ratio"],speed_required=True)
                record={
                    "phase":phase,
                    "full_correct":sum(proposed[k]==expected[k] for k in keys),
                    "first_correct":sum(
                        bool(proposed[k]) and proposed[k][0]==expected[k][0]
                        for k in keys),
                    "lost_full":guard["lost_full"],
                    "lost_first":guard["lost_first"],
                    "gained_full":guard["gained_full"],
                    "gained_first_vs_original":sum(
                        bool(proposed[k]) and proposed[k][0]==expected[k][0]
                        for k in keys)-387,
                    "guard_all_456":guard["all_456_verified"],
                    "train_retention_safe":guard["transaction_can_commit"],
                    "physical_bpp":bpp,
                    "physical_bytes_added":physical_bytes,
                    "CPU_relative_original_S101":speed["ratio"],
                    "paired_CPU_speed_details":speed,
                    "physical_FP16_sidecar_SHA256":sha(target.read_bytes()),
                    "physical_phase_config_SHA256":sha(cfg_path.read_bytes()),
                    "gained_full_TRAIN_hashes":[sha(x.encode()) for x in
                                                guard["gained_full_identities"]],
                    "lost_full_TRAIN_hashes":[sha(x.encode()) for x in
                                              guard["lost_full_identities"]],
                    "promoted":False,
                }
                full.append(record)
                print(json.dumps({
                    "event":"S115_TEMPORAL_GREEN_TUNNEL_456_REAL_CPU",
                    "phase":phase,
                    "correct":record["full_correct"],
                    "first_correct":record["first_correct"],
                    "lost_full":record["lost_full"],
                    "lost_first":record["lost_first"],
                    "gained_full":record["gained_full"],
                    "safe":record["train_retention_safe"],
                    "physical_bpp":bpp,
                    "CPU_S101_ratio":speed["ratio"],
                }),flush=True)
                if (guard["transaction_can_commit"]
                        and guard["gained_full"]>=1
                        and (winner is None or
                             (record["gained_full"],record["CPU_relative_original_S101"])>
                             (winner["gained_full"],winner["CPU_relative_original_S101"]))):
                    winner=record
        finally:
            bridge.uninstall()
    finally:
        state[6].uninstall()

    report={
        "stage":"S115_REJECTED_S112_TEMPORAL_KV_PHASE_REAL_456_TRAIN_CPU",
        "original_S101_353_full_387_first_protected":True,
        "S112_rejected_347_full_388_first_plus1_minus7":True,
        "precommitted_phase_masks":list(VALID_PHASES),
        "results":full,
        "safe_with_real_new_full_answers":None if winner is None else winner["phase"],
        "max_safe_new_full_EOS_answers":0 if winner is None else winner["gained_full"],
        "model_accepted":False,
        "no_new_weights_or_optimizer_steps":True,
        "QA_TEST_never_opened":True,
        "LIVE_main_Render_Supabase_APK_untouched":True,
    }
    (a.out/"S115_REJECTED_S112_TEMPORAL_GREEN_TUNNEL_REPORT.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False))
    print(json.dumps({
        "event":"S115_KV_PHASE_ABLATION_COMPLETE",
        "best_train_safe_phase":report["safe_with_real_new_full_answers"],
        "max_safe_train_gained":report["max_safe_new_full_EOS_answers"],
        "model_accepted":False,
    }),flush=True)


if __name__=="__main__":
    main()
