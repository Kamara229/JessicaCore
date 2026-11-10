"""S102 stage-B: first real MPS S101/S93 joint-TRAIN transaction + J7 guard.

One 8-step transaction, no retraining frozen S95/S98, no QA/TEST.
MPS updates EXISTING S93 paid FP16 MLP codebooks (layers 3/9) and
EXISTING internal Green Tunnel blocks (layers 3/9). Source 5-bit codes,
S87/S91 alternatives, neural prompt-only router and copy scorer remain
frozen. Evaluate original + proposal on 456 real PHYSICAL S101 cached
CPU full+EOS replies, rollback both model parameters AND complete Adam
state if old correct or first-token skills lost or physical budget breached.
This is a feasibility trial, NEVER production model acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s87.internal_green_tunnel import (
    IntegratedGreenTunnel,select_paid_core_codebooks)
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s91 import train_joint_quality_rehearsal as s91
from calibri2.s94.physical_profile_bank import CHANGED,freeze_report,sha
from calibri2.s95.train_and_audit_router import CHOICES,prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    load_frozen_numeric_weights,apply_profile)
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s102.hybrid_retention import (
    AtomicProtectedUpdate,strict_train_retention)
from calibri2.s102.v_legacy_quality_diagnostics import (
    sequence_diagnostics,synthetic_regressions)
from v176.operators import clear
from v185.codec import replace_discrete

STEPS=8
SEED=1020102
MPS_BATCH_SIZE=6
PHYSICAL_BPP_LIMIT=0.7


def training_cohort(tok, reports, frozen_router, protected_hashes):
    rows={}
    for family,pairs in s87.families(tok).items():
        for pair in pairs:
            for side_name in ("left","right"):
                key=f'{family}/{pair["pair_id"]}/{side_name}'
                row=pair[side_name]
                if sha(row["question"].encode())!=reports["S87"][key]["question_sha256"]:
                    raise AssertionError("Frozen authorized TRAIN question drift")
                rows[key]={"id":key,"question":row["question"],
                           "expected":reports["S87"][key]["target_ids"]}
    ids=sorted(rows)
    if len(ids)!=456 or set(ids)!=set(reports["S87"]):
        raise AssertionError("Exactly 456 immutable TRAIN required")
    features=torch.tensor(np.stack([
        prompt_only_vector(tok,rows[k]["question"]) for k in ids]),
        dtype=torch.float32)
    with torch.inference_mode():
        choices=frozen_router(features).argmax(dim=1).tolist()
    routed={k:CHOICES[int(v)] for k,v in zip(ids,choices)}
    protected={k for k in ids if sha(k.encode()) in protected_hashes}
    if len(protected)!=353:
        raise AssertionError("S102 immutable S101 353 protected hashes changed")
    wrong=[rows[k] for k in ids if routed[k]=="S93" and k not in protected]
    replay=[rows[k] for k in ids if routed[k]=="S93" and k in protected]
    return rows,ids,routed,wrong,replay


def build_frozen_s101_cpu(raw,side,bank,copy_net):
    model,sources,first,later,ptr,head,inner=s87.build_cpu(
        raw,side,True)
    if install_verified_fastpath(inner)!=("3","9"):
        raise AssertionError("S101 verified exact zero-gate inference ABI missing")
    base_centroids={name:sources[name].centroids.detach().clone()
                    for name in CHANGED}
    base_tunnel={
        k.removeprefix("internal_tunnel."):v for k,v in side.items()
        if k.startswith("internal_tunnel.")}
    return (model,sources,first,later,ptr,head,inner,
            base_centroids,base_tunnel)


def compare_full456_serialized(
    rows, ids, routed, tok, bank, copy_net,
    old_raw,old_side,new_raw,new_side):
    original=build_frozen_s101_cpu(old_raw,old_side,bank,copy_net)
    proposal=build_frozen_s101_cpu(new_raw,new_side,bank,copy_net)
    samples={"old":[],"proposed":[]}
    results={"old":{},"proposed":{}}
    copy_notes={}
    try:
        selected={"old":None,"proposed":None}
        for i,ident in enumerate(ids):
            route=routed[ident]
            note={}
            for label,state in (("old",original),("proposed",proposal)):
                (model,sources,first,later,ptr,head,inner,
                 centroids,tunnel)=state
                if selected[label]!=route:
                    apply_profile(sources,inner,route,bank,centroids,tunnel)
                    selected[label]=route
                t=time.perf_counter()
                ids_out,policy_note=cached_full_cpu_with_neural_copy(
                    model,sources,first,later,head,tok,
                    rows[ident]["question"],ptr,copy_net)
                elapsed=time.perf_counter()-t
                results[label][ident]=ids_out
                note["old" if label=="old" else "new"]=policy_note
                if i<48:
                    samples[label].append(elapsed)
            copy_notes[ident]=note
    finally:
        original[6].uninstall()
        proposal[6].uninstall()
    ratio=statistics.median(samples["old"])/statistics.median(
        samples["proposed"])
    return results["old"],results["proposed"],ratio,copy_notes


def main():
    ap=argparse.ArgumentParser()
    for name in ("S98_copy_net","S98_report","S95_router","S95_report",
                 "S94_bank","S94_report","S93_CB61","S93_sidecar",
                 "S99_full456_report","S101_full456_report","S102_anchor",
                 "S87_report","S91_report","S93_report"):
        ap.add_argument(name,type=Path)
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S102 requires REAL Apple Silicon MPS GPU gradients")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    print(json.dumps(synthetic_regressions()),flush=True)
    frozen=json.loads(args.S102_anchor.read_text())
    baseline=json.loads(args.S101_full456_report.read_text())
    old_s99=json.loads(args.S99_full456_report.read_text())
    if (frozen.get("stage")!="S102_HYBRID_IMMUTABLE_S101_ANCHOR_FULL456_CPU"
        or frozen.get("full_protected_TRAIN_answers")!=353
        or frozen.get("candidate_accepted") is not False
        or baseline.get("cached_full_EOS_correct")!=353
        or baseline.get("full_EOS_token_mismatch_count")!=0
        or old_s99.get("S99_new_correct_full_EOS")!=353):
        raise AssertionError("Stage B requires verified frozen S101/S102")
    for path,name in ((args.S93_CB61,"physical_S93_CB61_sha256"),
                      (args.S94_bank,"frozen_S94_bank_sha256"),
                      (args.S95_router,"frozen_S95_router_sha256"),
                      (args.S98_copy_net,"frozen_S98_copy_net_sha256")):
        if sha(path.read_bytes())!=frozen[name]:
            raise AssertionError("Frozen S101 physical input changed: "+name)
    old_bytes=args.S93_CB61.read_bytes()
    raw_original=s87.open_envelope(old_bytes)
    original_side=load_file(str(args.S93_sidecar))
    _,_,_,bank,router,copy_net=load_frozen_numeric_weights(args)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    reports={p:freeze_report(getattr(args,p+"_report"))
             for p in ("S87","S91","S93")}
    protected_hashes=set(frozen["protected_TRAIN_identity_hashes"])
    rows,ids,routed,wrong,replay=training_cohort(
        tok,reports,router,protected_hashes)
    if len(wrong)<1 or len(replay)<1:
        raise AssertionError("No S93-routed, wrong/correct TRAIN contrast to improve")
    # Only S93 is modified. Other profiles are exact frozen physical bank
    # weights, hence a candidate can alter only S93-routed inferences.
    student,sources=create_small_paid_model(
        CB59(raw_original),build_config(MODEL_SPEC))
    student.to("mps").eval().requires_grad_(False)
    inner=IntegratedGreenTunnel(layers=(3,9),rank=8).to("mps").eval()
    inner.load_state_dict({
        k.removeprefix("internal_tunnel."):v for k,v in original_side.items()
        if k.startswith("internal_tunnel.")},strict=True)
    # S101's zero-gate CPU fastpath is mandatory. J7 optimizer may not
    # train gates because that would silently invalidate S101 inference.
    for block in inner.blocks.values():
        block.gate.requires_grad_(False)
    install_verified_fastpath(inner)
    inner.install(student)
    core=select_paid_core_codebooks(sources,(3,9))
    if set(core)!=set(CHANGED):
        raise AssertionError("S102 may train only S93 existing two core dictionaries")
    core_params=[p.centroids for p in core.values()]
    tunnel_params=[p for p in inner.parameters() if p.requires_grad]
    params=core_params+tunnel_params
    if not tunnel_params:
        raise AssertionError("S102 core and Green Tunnel must train together")
    optimizer=torch.optim.Adam(
        [{"params":core_params,"lr":0.00003},
         {"params":tunnel_params,"lr":0.00007}])
    first,later,ptr=s91.frozen_controllers(original_side)
    head=student.lm_head.source().detach()
    tx=AtomicProtectedUpdate(params,optimizer)
    steps=[]
    print(json.dumps({
        "event":"S102_HYBRID_REAL_MPS_JOINT_S93_TRAIN_START",
        "total_TRAIN":456,"old_S101_protected_full":353,
        "S93_routed_wrong_TRAIN":len(wrong),
        "S93_routed_correct_TRAIN":len(replay),
        "trainable_existing_codebooks":list(core),
        "trainable_existing_tunnel_params":sum(p.numel() for p in tunnel_params),
        "max_steps":STEPS,"no_S100_QA_used":True,
        "original_bpp":baseline["S99_all_inclusive_physical_bpp"]
    }),flush=True)
    for step in range(STEPS):
        # S93-routed subset only; 2 wrong and 4 S101-correct protected
        # authorized TRAIN rows each step. No actual S100/QA here.
        chosen=[
            wrong[(2*step+j)%len(wrong)] for j in range(2)
        ]+[
            replay[(4*step+j)%len(replay)] for j in range(4)
        ]
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for row in chosen:
            clear(sources)
            l=s91.policy_loss(student,tok,row,first,later,ptr,head,
                              protected=row in replay)
            if not torch.isfinite(l):
                raise AssertionError("S102 non-finite aligned S82 training objective")
            losses.append(l)
        value=torch.stack(losses).mean()
        value.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in params):
            raise AssertionError("S102 missing or nonfinite MPS core/tunnel gradients")
        core_nonzero=sum(bool(torch.count_nonzero(p.grad).item())
                         for p in core_params)
        tunnel_nonzero=sum(bool(torch.count_nonzero(p.grad).item())
                           for p in tunnel_params)
        if core_nonzero!=2 or not tunnel_nonzero:
            raise AssertionError("S102 not a real joint gradient update")
        torch.nn.utils.clip_grad_norm_(params,1.0)
        optimizer.step()
        with torch.no_grad():
            for p in core_params:p.clamp_(-16,16)
        steps.append({"step":step+1,"loss":float(value.detach()),
                      "nonzero_core_gradients":core_nonzero,
                      "nonzero_tunnel_gradients":tunnel_nonzero})
        print(json.dumps({"event":"S102_MPS_HYBRID_JOINT_STEP",**steps[-1]}),flush=True)
        clear(sources)
    # Serialize changed ACTUAL physical S93, never dense pretrained weights.
    new_raw,new_side,new_cb61,new_side_path,mutation=s91.materialize_candidate(
        raw_original,sources,inner,original_side,args.out,STEPS)
    for k in original_side:
        if (not k.startswith("internal_tunnel.") and
            not torch.equal(original_side[k],new_side[k])):
            raise AssertionError("S102 illegally changed frozen S82 policy")
    # Source S94 alternatives, router, copy and controllers are never changed.
    orig_bpp=float(baseline["S99_all_inclusive_physical_bpp"])
    new_bpp=orig_bpp+8*(new_cb61.stat().st_size-len(old_bytes)+
                        new_side_path.stat().st_size-args.S93_sidecar.stat().st_size
                        )/SOURCE_PARAMS
    required={"bpp":new_bpp,"base_bpp":orig_bpp,
              "proposed_bytes_delta":(
                  new_cb61.stat().st_size-len(old_bytes)+
                  new_side_path.stat().st_size-args.S93_sidecar.stat().st_size),
              "changed_FP16_centroid_cells":mutation["actually_changed_FP16_codebook_cells"]}
    if any(not math.isfinite(float(v)) for v in
           (new_bpp,orig_bpp)):
        raise AssertionError("S102 physical accounting invalid")
    # The full CPU quality experiment is valuable even if compressed
    # entropy grew beyond 0.7: this remains a REJECTED scientific prototype.
    original,outcome,ratio,copy_notes=compare_full456_serialized(
        rows,ids,routed,tok,bank,copy_net,
        raw_original,original_side,new_raw,new_side)
    expected={k:rows[k]["expected"] for k in ids}
    legacy_diag=sequence_diagnostics(
        expected,original,outcome,routed,tok=tok,copy_notes=copy_notes)
    (args.out/"S102_V_LEGACY_FULL_SEQUENCE_DIAGNOSTICS.json").write_text(
        json.dumps(legacy_diag,indent=2,sort_keys=True,ensure_ascii=False))
    print(json.dumps({key:legacy_diag[key] for key in (
        "event","checked_TRAIN","lost_full_count","gained_full_count",
        "lost_first_divergence_position","lost_full_by_frozen_route",
        "pairs_old_both_correct","pairs_new_both_correct",
        "spurious_adjacent_repeats_old","spurious_adjacent_repeats_new",
        "wrong_early_EOS_old","wrong_early_EOS_new","copy_policy")}),flush=True)
    guard=strict_train_retention(
        expected,original,outcome,physical_bpp=new_bpp,
        cpu_ratio=ratio,speed_required=False)
    if any(sha(k.encode()) not in protected_hashes
           for k in ids if original[k]==expected[k]):
        raise AssertionError("S102 old 353 S101 source differs from stage A")
    if (len([k for k in ids if original[k]==expected[k]])!=353
        or ratio<=0 or not math.isfinite(ratio)):
        raise AssertionError("S102 baseline/paired CPU diagnostics invalid")
    evidence=tx.finalize(guard)
    rollback_source_sha_verified=None
    if not evidence["committed"]:
        # Recover the exact original paid CB59 and sidecar from GPU
        # parameters after reverting BOTH weights and Adam state.
        restore_raw,_=replace_discrete(CB59(raw_original),sources,strict=False)
        restored_side=dict(original_side)
        for name,tensor in inner.state_dict().items():
            restored_side["internal_tunnel."+name]=tensor.detach().cpu().contiguous()
        rollback_source_sha_verified=sha(restore_raw)==sha(raw_original)
        if (not rollback_source_sha_verified or
            any(not torch.equal(restored_side[k],original_side[k])
                for k in original_side)):
            raise AssertionError("S102 GPU rollback failed exact S93 physical weights")
    inner.uninstall()
    record={
        "event":"S102_HYBRID_S101_J7_MPS_PHYSICAL_TRANSACTION_END",
        "stage":"S102_ONE_REAL_MPS_JOINT_TRANSACTION_UNACCEPTED",
        "GPU_training_8_steps":True,
        "J7_rollback_algorithm_ported_not_incompatible_S82_weights":True,
        "S101_original_full_correct":353,
        "S101_protected_first_correct":guard["old_first_correct_protected"],
        "S101_new_full_correct":sum(outcome[k]==expected[k] for k in ids),
        "S101_full_correct_gained":guard["gained_full"],
        "S101_full_correct_lost":guard["lost_full"],
        "V_LEGACY_GOLD_AWARE_FULL_SEQUENCE_CAUSAL_DIAGNOSTIC":legacy_diag,
        "S101_first_correct_lost":guard["lost_first"],
        "regressed_identity_sha256":[sha(k.encode()) for k in
                                     guard["lost_full_identities"]],
        "actual_post_training_physical":required,
        "true_compressed_CPU_S101_candidate_to_original_speed_ratio":ratio,
        "CPU_vs_S82_speed_verified_on_new_candidate":False,
        "transaction":evidence,
        "rollback_serialized_S93_identical_to_original":rollback_source_sha_verified,
        "mps_steps":steps,
        "S100_S96_and_all_inspected_QA_never_read":True,
        "new_inference_weights_added":0,
        "frozen_S95_router_S98_copy_S87_S91_profiles":True,
        "must_gain_full_train_for_development_candidate":
            guard["gained_full"]>=1 and evidence["committed"],
        "native_C_and_MPS_CPU_parity_pending":True,
        "fresh_unseen_QA_pending":True,
        "candidate_accepted":False,
        "LIVE_untouched":True
    }
    (args.out/"S102_HYBRID_FIRST_MPS_TRANSACTION_REPORT.json").write_text(
        json.dumps(record,indent=2,sort_keys=True,ensure_ascii=False))
    print(json.dumps({k:record[k] for k in (
        "event","S101_original_full_correct","S101_new_full_correct",
        "S101_full_correct_gained","S101_full_correct_lost",
        "S101_first_correct_lost","actual_post_training_physical",
        "true_compressed_CPU_S101_candidate_to_original_speed_ratio",
        "transaction","must_gain_full_train_for_development_candidate",
        "candidate_accepted")}),flush=True)


if __name__=="__main__":
    main()
