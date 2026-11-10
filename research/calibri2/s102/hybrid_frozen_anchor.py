"""S102 hybrid feasibility: freeze the REAL S101 353/456 TRAIN quality anchor.

Scientific scope only. Build an exact 456 full+EOS reference from physically
serialized S93 CB61 + S94 3-profile Green Tunnel bank + frozen S95 router +
S98 neural noun copier + S101 incremental KV decoder. This dataset is
authorized TRAIN, NEVER S100/S96 inspected one-shot unseen/QA. A future
J7-style MPS transaction may modify only EXISTING S93 core centroids and
internal-tunnel tensors. Every proposed checkpoint must pass the same S101
full+EOS CPU policy on all 353 previously correct TRAIN examples, preserve
previously correct first tokens, and remain below physical 0.7 bpp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s94.physical_profile_bank import CHANGED, sha, freeze_report
from calibri2.s95.train_and_audit_router import CHOICES, prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    apply_profile, load_frozen_numeric_weights)
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s102.hybrid_retention import (
    numerical_rollback_selftest, strict_train_retention)


def main():
    ap = argparse.ArgumentParser()
    for name in (
        "S98_copy_net", "S98_report", "S95_router", "S95_report",
        "S94_bank", "S94_report", "S93_CB61", "S93_sidecar",
        "S99_full456_report", "S101_full456_report",
        "S87_report", "S91_report", "S93_report"):
        ap.add_argument(name, type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    prior = json.loads(a.S99_full456_report.read_text())
    cached = json.loads(a.S101_full456_report.read_text())
    if (cached.get("stage") != "S101_FROZEN_S99_CPU_KV_CACHE_EXACT_456_TRAIN_ONLY"
        or cached.get("cached_full_EOS_correct") != 353
        or cached.get("original_no_cache_full_EOS_correct") != 353
        or cached.get("full_EOS_token_mismatch_count") != 0
        or cached.get("legacy_old_union_282_full_lost") != 0
        or cached.get("legacy_old_union_first_lost") != 0
        or cached.get("candidate_accepted") is not False):
        raise AssertionError("S102 requires immutable exact S101 456 CPU baseline")
    if (prior.get("S99_new_correct_full_EOS") != 353 or
        prior.get("candidate_accepted") is not False):
        raise AssertionError("S102 requires frozen S99 353 correct TRAIN baseline")
    for path, digest in (
        (a.S93_CB61, prior["immutable_S95_source_sha256"]),
        (a.S94_bank, prior["S94_weightbank_sha256"]),
        (a.S95_router, prior["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net, prior["S98_neural_copy_selector_sha256"])):
        if sha(path.read_bytes()) != digest:
            raise AssertionError("S102 frozen S101 physical weights were changed")
    original_raw = s87.open_envelope(a.S93_CB61.read_bytes())
    side = load_file(str(a.S93_sidecar))
    if any(key not in side for key in (
        "internal_tunnel.blocks.3.down.weight",
        "internal_tunnel.blocks.9.down.weight")):
        raise AssertionError("S93 2-layer Green Tunnel ABI changed")
    _,_,_,bank,router,copy_net = load_frozen_numeric_weights(a)
    tok = AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK = tok
    reports={p:freeze_report(getattr(a,p+"_report")) for p in ("S87","S91","S93")}
    questions={}
    for family,pairs in s87.families(tok).items():
        for pair in pairs:
            for side_name in ("left","right"):
                key=f'{family}/{pair["pair_id"]}/{side_name}'
                questions[key] = pair[side_name]["question"]
    ids=sorted(questions)
    if len(ids)!=456 or set(ids)!=set(reports["S87"]):
        raise AssertionError("Immutable S101 TRAIN question identities changed")
    expected={key: reports["S87"][key]["target_ids"] for key in ids}
    for key in ids:
        if sha(questions[key].encode())!=reports["S87"][key]["question_sha256"]:
            raise AssertionError("Frozen TRAIN question SHA drift")
    x=torch.as_tensor(
        np.stack([prompt_only_vector(tok, questions[key]) for key in ids]),
        dtype=torch.float32)
    with torch.inference_mode():
        choices=router(x).argmax(dim=1).tolist()
    routed={key: CHOICES[int(choice)] for key,choice in zip(ids,choices)}
    model, sources, first, later, ptr, head, inner=s87.build_cpu(
        original_raw, side, True)
    if any(key not in sources for key in CHANGED):
        raise AssertionError("Two existing paid S93 core dictionaries missing")
    if install_verified_fastpath(inner)!=("3","9"):
        raise AssertionError("S101 exact frozen two-layer Green Tunnel fastpath missing")
    original_centroids={
        key:sources[key].centroids.detach().clone() for key in CHANGED}
    original_tunnel={
        k.removeprefix("internal_tunnel."):v for k,v in side.items()
        if k.startswith("internal_tunnel.")}
    actual={}
    selected=None
    try:
        for idx,key in enumerate(ids):
            profile=routed[key]
            if selected!=profile:
                apply_profile(sources,inner,profile,bank,original_centroids,
                              original_tunnel)
                selected=profile
            actual[key],_=cached_full_cpu_with_neural_copy(
                model,sources,first,later,head,tok,questions[key],ptr,copy_net)
    finally:
        inner.uninstall()
        del model
    bpp=float(cached["S99_all_inclusive_physical_bpp"])
    speed=float(cached["cached_CPU_speed_vs_original_S82"])
    guard=strict_train_retention(
        expected,actual,actual,physical_bpp=bpp,cpu_ratio=speed,
        speed_required=True)
    if (not guard["transaction_can_commit"] or
        guard["old_full_correct_protected"]!=353 or
        sum(bool(actual[k] and actual[k][0]==expected[k][0])
            for k in ids)!=cached["new_first_token_correct"]):
        raise AssertionError("S102 actual S101 456 full+EOS or first-token anchor drift")
    distribution=Counter()
    for key in ids:
        distribution[(routed[key],actual[key]==expected[key])]+=1
    # Physical headroom is NOT permission to allocate dense extra adapters.
    # Train only S93 2 existing paid FP16 core dictionaries + existing tunnel.
    headroom=(0.7-bpp)*SOURCE_PARAMS/8
    if headroom<0:
        raise AssertionError("S101 source has no physical compression headroom")
    synthetic=numerical_rollback_selftest()
    anchor={
        "event":"S102_FROZEN_S101_HYBRID_J7_BRIDGE",
        "stage":"S102_HYBRID_IMMUTABLE_S101_ANCHOR_FULL456_CPU",
        "research_branch_only":True,
        "S101_true_cached_CPU_full_EOS_correct":353,
        "source_profiles_S87_S91_S93":list(("S87","S91","S93")),
        "physical_S93_CB61_sha256":sha(a.S93_CB61.read_bytes()),
        "frozen_S94_bank_sha256":sha(a.S94_bank.read_bytes()),
        "frozen_S95_router_sha256":sha(a.S95_router.read_bytes()),
        "frozen_S98_copy_net_sha256":sha(a.S98_copy_net.read_bytes()),
        "all_inclusive_bpp":bpp,
        "all_inclusive_bpp_budget":0.7,
        "physical_budget_headroom_bytes_approx":headroom,
        "CPU_speed_ratio_vs_S82_frozen_S101":speed,
        "S101_has_three_Green_Tunnel_profiles":True,
        "S101_residual_integration_layers":[3,9],
        "S101_frozen_router_copy_and_S82_controller_must_stay_frozen":True,
        "J7_algorithm_to_port":"transactional trainable existing S93 centroids + internal 3/9 tunnel, not incompatible J7 S82 learned tensors",
        "paid_mutation_scope":list(CHANGED)+[
            "internal_tunnel.blocks.3.*","internal_tunnel.blocks.9.*"],
        "full_protected_TRAIN_answers":353,
        "first_protected_TRAIN_answers":guard["old_first_correct_protected"],
        "protected_TRAIN_identity_hashes":[hashlib.sha256(k.encode()).hexdigest()
             for k in ids if actual[k]==expected[k]],
        "all_456_original_TRAIN_identity_digest":sha(
            "\n".join(ids).encode()),
        "profile_counts":{profile:{
            "rows":sum(routed[k]==profile for k in ids),
            "full_EOS_correct":distribution[(profile,True)]}
            for profile in ("S87","S91","S93")},
        "synthetic_J7_optimizer_rollback_verified":synthetic,
        "S100_S96_and_previously_inspected_QA_unseen_never_imported":True,
        "current_prototype_trainable_weights_added":0,
        "new_inference_weights_added":0,
        "full_model_joint_MPS_training_not_yet_started":True,
        "candidate_accepted":False,
        "LIVE_untouched":True,
    }
    target=a.out/"S102_FROZEN_S101_J7_HYBRID_353_TRAIN_ANCHOR.json"
    target.write_text(json.dumps(anchor,indent=2,sort_keys=True,ensure_ascii=False))
    print(json.dumps({
        "event":"S102_HYBRID_PHYSICAL_ABI_AND_353_TRAIN_COMPATIBILITY_PASS",
        "old_correct_full_EOS":353,
        "protected_first":guard["old_first_correct_protected"],
        "profiles":anchor["profile_counts"],
        "bpp":bpp,"headroom_bytes":headroom,
        "CPU_ratio":speed,"rollback_synthetic":True,
        "no_new_inference_weights":True,
        "ready_for_J7_safeguarded_MPS_training":True,
        "candidate_accepted":False,
    }),flush=True)


if __name__=="__main__":
    main()
