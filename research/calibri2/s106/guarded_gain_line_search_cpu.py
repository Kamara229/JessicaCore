"""S106 — physically guarded TRAIN-only line search of S104 all-layer gains.

S104 learned a genuine ten-dimensional gain vector (two real MPS steps) and
preserved 353/456 full+EOS and all 387 first tokens, but did not acquire a
new correct answer. This read-only training-data line search explores whether
scaling that same vector changes actual S101 autoregressive outcomes without
breaking any previous skill. No optimizer gradients or new QA are used.

Each independent alpha candidate is physically serialized, reloaded and
re-evaluated on ALL 456 original authorized TRAIN prompts using the actual
S101 KV-cache + frozen S95 profile router + S98 second-token copy policy.
No accepted model, no LIVE mutations, no merges to main.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import freeze_report, sha
from calibri2.s99.guard_neural_copy_456_cpu import load_frozen_numeric_weights
from calibri2.s102.hybrid_retention import strict_train_retention
from calibri2.s103.shared_all_layer_green_bridge import SharedInterLayerGreenBridge
from calibri2.s104.gain_only_mps_transaction import (
    all_456_cached_cpu, cohort, load_cpu_original,
)

ALPHAS = (1, 2, 4, 8)
MIN_BASE = 353
FIRST_BASE = 387


def main():
    p=argparse.ArgumentParser()
    for key in (
        "S98_copy_net", "S98_report", "S95_router", "S95_report",
        "S94_bank", "S94_report", "S93_CB61", "S93_sidecar",
        "S99_full456_report", "S101_full456_report", "S103_parity_report",
        "S104_report", "S104_learned_sidecar", "S104_runtime_config",
        "S87_report", "S91_report", "S93_report",
    ):
        p.add_argument(key,type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)

    p99=json.loads(args.S99_full456_report.read_text())
    p101=json.loads(args.S101_full456_report.read_text())
    p103=json.loads(args.S103_parity_report.read_text())
    p104=json.loads(args.S104_report.read_text())
    if (p101.get("cached_full_EOS_correct")!=MIN_BASE
            or p101.get("new_first_token_correct")!=FIRST_BASE
            or p103.get("full_456_exact_parity") is not True
            or p103.get("candidate_full_EOS_correct")!=MIN_BASE
            or p104.get("stage")!="S104_REAL_MPS_TEN_GAINS_456_CPU_TRANSACTION"
            or p104.get("real_MPS_steps")!=2
            or p104.get("trainable_parameter_count")!=10
            or p104.get("baseline_full")!=MIN_BASE
            or p104.get("candidate_full")!=MIN_BASE
            or p104.get("rollback",{}).get("committed") is not True
            or p104.get("guard",{}).get("lost_full")!=0
            or p104.get("guard",{}).get("lost_first")!=0
            or p104.get("guard",{}).get("gained_full")!=0
            or p104.get("model_accepted") is not False):
        raise AssertionError("S106 requires authentic accepted-for-research S104 no-gain evidence")

    for path, digest in (
        (args.S93_CB61, p99["immutable_S95_source_sha256"]),
        (args.S94_bank, p99["S94_weightbank_sha256"]),
        (args.S95_router, p99["S95_neural_profile_selector_sha256"]),
        (args.S98_copy_net, p99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes()) != digest:
            raise AssertionError("S101 immutable physical artifact mismatch")
    if (sha(args.S104_learned_sidecar.read_bytes()) != p104["S104_sidecar_SHA256"]
            or sha(args.S93_CB61.read_bytes()) != p104["S101_original_CB61_sha256"]):
        raise AssertionError("S104 physical gain/source provenance mismatch")
    runtime=json.loads(args.S104_runtime_config.read_text())
    if (runtime.get("stage")!="S104_UNACCEPTED_TEN_GAINS"
            or runtime.get("enable_only_route")!="S93"
            or runtime.get("donor_layer")!=3
            or runtime.get("existing_layers")!=[3,9]):
        raise AssertionError("S104 physical bridge deployment descriptor drift")

    old_side=load_file(str(args.S93_sidecar))
    s104_side=load_file(str(args.S104_learned_sidecar))
    gain_key="all_layer_bridge.layer_gains"
    if set(s104_side)!=set(old_side)|{gain_key}:
        raise AssertionError("S104 sidecar contains unexpected tensors")
    for key in old_side:
        if not torch.equal(old_side[key],s104_side[key]):
            raise AssertionError("S104 trained a forbidden original tensor")
    learned=s104_side[gain_key]
    if (learned.shape!=(10,) or learned.dtype!=torch.float32
            or not bool(torch.isfinite(learned).all())
            or not bool(torch.count_nonzero(learned).item())):
        raise AssertionError("S104 no real learned shared gain vector")

    tok=AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK=tok
    report={key:freeze_report(getattr(args,key+"_report"))
            for key in ("S87","S91","S93")}
    _,_,_,bank,router,copy_net=load_frozen_numeric_weights(args)
    rows,keys,routed=cohort(tok,report,router)
    expected={key:rows[key]["expected"] for key in keys}
    state=load_cpu_original(
        s87.open_envelope(args.S93_CB61.read_bytes()),old_side)
    core=state[0]
    bridge=SharedInterLayerGreenBridge(state[6]).eval().install(core)
    try:
        # Identical fully reconstructed old S101, all bridges disabled.
        baseline,sample_base=all_456_cached_cpu(
            state,rows,keys,routed,tok,bank,copy_net,bridge=None)
        original_full=sum(baseline[k]==expected[k] for k in keys)
        original_first=sum(bool(baseline[k]) and baseline[k][0]==expected[k][0]
                           for k in keys)
        if original_full!=MIN_BASE or original_first!=FIRST_BASE:
            raise AssertionError("Frozen original S101 baseline failed reconstitution")

        # Explicitly remove empty bridge hooks for a genuine S101 speed baseline.
        bridge.uninstall()
        baseline_no_hooks,sample_unhooked=all_456_cached_cpu(
            state,rows,keys,routed,tok,bank,copy_net,bridge=None)
        if baseline_no_hooks!=baseline:
            raise AssertionError("Zero-gain hooks changed frozen S101 source outputs")
        bridge.install(core)
        outcomes=[]
        winning=None
        for alpha in ALPHAS:
            new_gain=learned * float(alpha)
            if not torch.isfinite(new_gain).all() or float(new_gain.abs().max())>=1.0:
                raise AssertionError("S106 proposed gain outside safe bounded range")
            candidate_side=dict(old_side)
            candidate_side[gain_key]=new_gain.cpu().contiguous()
            model_path=args.out/f"S106_UNACCEPTED_ALPHA_{alpha}_S93_SIDECAR.safetensors"
            save_file(candidate_side,str(model_path))
            physically_loaded=load_file(str(model_path))
            if (set(physically_loaded)!=set(candidate_side)
                    or any(not torch.equal(v,physically_loaded[k])
                           for k,v in candidate_side.items())):
                raise AssertionError("S106 candidate sidecar failed byte-real reload")
            bridge.load_bridge_state({gain_key:physically_loaded[gain_key]})
            packed_delta=model_path.stat().st_size-args.S93_sidecar.stat().st_size
            # Only one of these alternative archives would be selected,
            # NEVER add up all research-side alternatives as a deployed model.
            add_bytes=packed_delta+args.S104_runtime_config.stat().st_size
            physical_bpp=float(p101["S99_all_inclusive_physical_bpp"])+(
                8*add_bytes/SOURCE_PARAMS)
            if abs(physical_bpp-float(p104["bpp"]))>1e-8:
                raise AssertionError("S106 gain scaling unexpectedly changed physical bytes")

            proposed,samples=all_456_cached_cpu(
                state,rows,keys,routed,tok,bank,copy_net,bridge=bridge)
            ratio=statistics.median(sample_unhooked)/statistics.median(samples)
            guard=strict_train_retention(
                expected,baseline_no_hooks,proposed,
                physical_bpp=physical_bpp,
                cpu_ratio=ratio,speed_required=True)
            summary={
                "scale":alpha,
                "full_correct":sum(proposed[k]==expected[k] for k in keys),
                "first_correct":sum(bool(proposed[k]) and
                                     proposed[k][0]==expected[k][0] for k in keys),
                "lost_full":guard["lost_full"],
                "lost_first":guard["lost_first"],
                "gained_full":guard["gained_full"],
                "guard_passed":guard["transaction_can_commit"],
                "S101_CPU_ratio":ratio,
                "physical_bpp":physical_bpp,
                "new_sidecar_sha256":sha(model_path.read_bytes()),
                "gained_identity_SHA256":[sha(x.encode())
                                         for x in guard["gained_full_identities"]],
                "lost_identity_SHA256":[sha(x.encode())
                                       for x in guard["lost_full_identities"]],
            }
            outcomes.append(summary)
            print(json.dumps({
                "event":"S106_REAL_CPU_GAIN_SCALE_GUARD",
                "scale":alpha,"correct":summary["full_correct"],
                "old_lost":summary["lost_full"],"first_lost":summary["lost_first"],
                "new_gained":summary["gained_full"],"pass":summary["guard_passed"],
                "physical_bpp":physical_bpp,"cpu_relative_S101":ratio,
            }),flush=True)
            if (summary["guard_passed"] and summary["gained_full"]>=1
                    and (winning is None
                         or (summary["gained_full"],summary["S101_CPU_ratio"])>
                            (winning["gained_full"],winning["S101_CPU_ratio"]))):
                winning=summary
    finally:
        bridge.uninstall()
        state[6].uninstall()
    results={
        "stage":"S106_SAFE_GAIN_SCALING_REAL_S101_CPU_456_TRAIN_ONLY",
        "provenance":"unchanged original S101 CB61 and original S104 learned gains",
        "frozen_353_full_correct":MIN_BASE,
        "frozen_387_first_correct":FIRST_BASE,
        "evaluated_train_identities":len(keys),
        "candidate_scales":list(ALPHAS),
        "results":outcomes,
        "first_scientifically_eligible_scale":None if winning is None
            else winning["scale"],
        "research_gain":0 if winning is None else winning["gained_full"],
        "new_heldout_QA_not_opened":True,
        "native_C_CPU_vs_S82_and_MPS_parity_still_pending":True,
        "model_accepted":False,
        "LIVE_main_Render_Supabase_APK_untouched":True,
    }
    (args.out/"S106_SAFE_GAIN_ALPHA_LINE_SEARCH_TRAIN_REPORT.json").write_text(
        json.dumps(results,indent=2,sort_keys=True,ensure_ascii=False))
    print(json.dumps({
        "event":"S106_GAIN_LINE_SEARCH_END",
        "selected_alpha":results["first_scientifically_eligible_scale"],
        "max_SAFE_train_gained":results["research_gain"],
        "candidate_accepted":False,
    }),flush=True)


if __name__=="__main__":
    main()
