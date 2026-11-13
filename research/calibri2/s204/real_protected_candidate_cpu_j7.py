"""S204 strict original 456 + new128 + novel32 native CPU J7 for genuine NRA FP16.

S199's original S190/S187/S168 inference and S176 rejection gate are
UNMODIFIED, except physically loading the immutable S204 unaccepted FP16
TRAIN artifact. Original S142 baseline only disables NRA in its own leg.
All 616 native generated answers are scored *after* inference, not routed
by hash or expected tokens. A failed guard stays rejected. No QA/TEST/LIVE.
"""
from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
import sys
import zlib

import torch
from safetensors.torch import load

from calibri2.s132.progressive_core import WEIGHTS_KEY
from calibri2.s195_neuro_area.representation_area import PREFIX
from calibri2.s196.core_growth_budget import NEXT_TOTAL_PARAMS, PREVIOUS_PARAMS
from calibri2.s201 import real_s200_snapshots_full_j7 as s201
from calibri2.s158 import targeted_retention_gpu as s158

STAGE = "S204_ACTUAL_NATIVE_616_CPU_J7_PROTECTED_NRA_CANDIDATE"
PINNED_S204_TRAIN_RUN = 43388797067
CHOSEN_STEP = 1
SOURCE_FILES = (
    "research/calibri2/s198/one_step_real_nra.py",
    "research/calibri2/s199/real_trained_nra_cpu_j7.py",
    "research/calibri2/s200/multistep_real_nra_train.py",
    "research/calibri2/s201/real_s200_snapshots_full_j7.py",
    "research/calibri2/s204/first_token_grad_guard.py",
    "research/calibri2/s204/guarded_nra_optimizer.py",
    "research/calibri2/s204/protected_s73_train.py",
    "research/calibri2/s204/original_train_source_scope.py",
    "research/calibri2/s204/real_s73_guarded_step.py",
    "research/calibri2/s204/real_native_s73_train.py",
    "research/calibri2/s204/real_protected_candidate_cpu_j7.py",
)

def sha(b):
    return hashlib.sha256(b).hexdigest()

def load_exact_s204(*, step, parent_zlib, parent_report, output_path):
    """Require the exact independent S204 real CPU TRAIN physical FP16 proof."""
    if step != CHOSEN_STEP:
        raise AssertionError("S204 only one predeclared genuine CPU protected step")
    root=Path("inputs/s204")
    report=json.loads((root/"S204_PROTECTED_S73_REAL_CPU_TRAIN_ONLY.json").read_text())
    frozen=json.loads(parent_report.read_text())
    base=parent_zlib.read_bytes()
    filename="S204_UNACCEPTED_PROTECTED_NRA_ONE_STEP_FP16.safetensors.lzma"
    packed=(root/filename).read_bytes()
    if (report.get("stage")!="S204_REAL_PROTECTED_S73_CPU_NRA_STEP_UNACCEPTED"
        or report.get("S197_immutable_parent_CI_run")!=43369398297
        or report.get("model_accepted") is not False
        or report.get("source_original_456_full_EOS")!=456
        or report.get("source_new128_full_EOS")!=128
        or report.get("genuine_old_token_gradients")!=4
        or report.get("frozen_non_NRA_tensors_equal") is not True
        or report.get("physical_roundtrip_verified") is not True
        or report.get("s204_train_audit",{}).get("actual_nra_adam_step") is not True
        or report.get("s204_train_audit",{}).get("fp16_candidate_feasible_first_order_only") is not True
        or report.get("s204_train_audit",{}).get("model_accepted") is not False
        or report.get("candidate_filename")!=filename
        or report.get("candidate_lzma_bytes")!=len(packed)
        or report.get("physical_candidate_sha256")!=sha(packed)
        or report.get("parent_package_sha256")!=sha(base)
        or frozen.get("stage")!="S197_ACTUAL_PHYSICAL_46M_NRA_REAL_CPU_616_J7"
        or frozen.get("exact_protected_584_generated_ids_equal_S194") is not True
        or frozen.get("checkpoint_runtime_physical",{}).get("runtime_packed_sha256")!=sha(base)):
        raise AssertionError("S204 immutable real protected TRAIN checkpoint provenance invalid")
    parent=load(zlib.decompress(base))
    candidate=load(lzma.decompress(packed))
    if (set(candidate)!=set(parent)
        or candidate[WEIGHTS_KEY].shape!=(12,320,72)
        or any(not torch.equal(parent[k],candidate[k])
               for k in parent if not k.startswith(PREFIX))
        or not any(not torch.equal(parent[k],candidate[k])
                   for k in parent if k.startswith(PREFIX))):
        raise AssertionError("S204 S197 original frozen 46M/GT not fully retained")
    if output_path.exists():
        raise FileExistsError("S204 immutable J7 physical load cannot overwrite")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_bytes(packed)
    if sha(output_path.read_bytes())!=sha(packed):
        raise AssertionError("S204 physically reloaded CPU package corrupted")
    physical={
        "parent_package_bytes":len(base),
        "actual_single_runtime_lzma_bytes":len(packed),
        "actual_runtime_lzma_SHA256":sha(packed),
        "raw_all_FP16_bytes":len(lzma.decompress(packed)),
        "actual_runtime_FP16_sha256":sha(lzma.decompress(packed)),
        "new_NRA_FP16_train_changed":{
            k:int(torch.count_nonzero(parent[k]!=candidate[k]))
            for k in candidate if k.startswith(PREFIX)},
        "S204_pinned_CPU_real_train_run":PINNED_S204_TRAIN_RUN,
    }
    return frozen,{"actual_CPU_train_steps":1,
        "train_NRA_Fp16_changed_elements":physical["new_NRA_FP16_train_changed"]},candidate,physical

def physical_invoice(*,parent_report,physical,manifest_bytes,source_paths):
    del source_paths
    old=float(parent_report["all_in_physical_bpp"]["all_in_paid_bits"])
    parent=int(physical["parent_package_bytes"])
    current=int(physical["actual_single_runtime_lzma_bytes"])
    sources=sum(Path(p).stat().st_size for p in SOURCE_FILES)
    if old<=0 or parent<=0 or current<=0 or manifest_bytes<0:
        raise AssertionError("S204 invalid source-inclusive physical budget")
    total=old+8*(current-parent+sources+manifest_bytes)
    return {
        "all_in_bits":total,
        "architectural_bpp":total/NEXT_TOTAL_PARAMS,
        "conservative_bpp_excluding_zero_structural_growth":total/(PREVIOUS_PARAMS+21_900),
        "replaced_S197_package_bytes":parent,
        "persisted_lzma_runtime_bytes":current,
        "S198_to_S204_source_bytes_charged":sources,
        "this_manifest_bytes_charged":manifest_bytes,
    }

def main():
    if len(sys.argv)<17 or sys.argv[1] != "1":
        raise SystemExit("S204: 1 then exact original S201 CPU J7 CLI, output")
    out=Path(sys.argv[-1])
    old_load=s201.load_precommitted_s200
    old_invoice=s201.physical_invoice
    old_steps=s201.STEPS
    old_stage=s201.STAGE
    original_retention=s158.retention
    checkpoint={"scored":False}
    def monitor(parent_old,candidate_old,old,parent_new,candidate_new,new):
        stats=original_retention(parent_old,candidate_old,old,parent_new,candidate_new,new)
        if len(old)==456 and len(new)==128:
            if checkpoint["scored"]:
                raise AssertionError("S204 duplicate real 456+128 CPU cohort scoring")
            checkpoint["scored"]=True
            checkpoint["protected_456_full_EOS"]=int(stats["old_full_after"])
            checkpoint["protected_128_full_EOS"]=int(stats["new_full_after"])
            checkpoint["lost_original_456_full_ids"]=list(stats["lost_old_full_ids"])
            checkpoint["lost_original_456_first_ids"]=list(stats["lost_old_first_ids"])
            checkpoint["lost_new128_full_ids"]=list(stats["lost_new_full_ids"])
            checkpoint["lost_new128_first_ids"]=list(stats["lost_new_first_ids"])
            checkpoint["S73_actual_generated_ids"]={
                k:list(candidate_old[k]) for k in sorted(old) if
                k in ("S73/73012/left","S73/73012/right",
                      "S73/73015/left","S73/73015/right")}
            checkpoint["real_CPU_inference_completed_before_gold_scoring"]=True
        return stats
    s201.load_precommitted_s200=load_exact_s204
    s201.physical_invoice=physical_invoice
    s201.STEPS=(CHOSEN_STEP,)
    s201.STAGE=STAGE
    s158.retention=monitor
    rejected_guard=False
    try:
        try:
            s201.main()
        except AssertionError as e:
            if str(e)!="S176 rejected: S172 cohort regression or bpp/CPU/protected gate failed":
                raise
            rejected_guard=True
    finally:
        s201.load_precommitted_s200=old_load
        s201.physical_invoice=old_invoice
        s201.STEPS=old_steps
        s201.STAGE=old_stage
        s158.retention=original_retention
    if not checkpoint["scored"]:
        raise AssertionError("S204 no genuine full original 456+128 J7 predictions")
    finished=out/"S201_STEP_1_FULL_616_REAL_CPU_J7.json"
    if finished.exists() and not rejected_guard:
        a=json.loads(finished.read_text())
        a["stage"]=STAGE
        a["source_trained_candidate_unaccepted"]=True
        a["S204_native_CPU_456_128_source_hook"]=checkpoint
        a["model_accepted"]=False
        a["candidate_pending_heldout_review"]=True
        size=0
        for _ in range(40):
            a["total_physical_budget"]=physical_invoice(
                parent_report=json.loads(Path("inputs/s197/S197_REAL_46M_NRA_616_CPU_J7.json").read_text()),
                physical=a["physical_checkpoint"],manifest_bytes=size,source_paths=())
            a["J7_acceptance_guard"]["conservative_bpp_no_free_zeros_under_07"]=(
                a["total_physical_budget"]["conservative_bpp_excluding_zero_structural_growth"]<=.7)
            a["J7_acceptance_guard"]["full_physical_bpp_architecture_under_07"]=(
                a["total_physical_budget"]["architectural_bpp"]<=.7)
            payload=json.dumps(a,indent=2,sort_keys=True)
            if len(payload.encode())==size:
                break
            size=len(payload.encode())
        else:
            raise AssertionError("S204 all-source report size accounting failed")
        finished.unlink()
        (out/"S204_REAL_616_CPU_J7.json").write_text(payload)
        print(json.dumps({
            "stage":STAGE,
            "protected_original456_full_EOS":a["full_456_correct_EOS"],
            "protected_new128_full_EOS":a["full_128_correct_EOS"],
            "novel32_full_EOS":a["32_NEW_TRAIN_full_EOS_correct"],
            "conservative_bpp":a["total_physical_budget"]["conservative_bpp_excluding_zero_structural_growth"],
            "CPU_ratio":a["same_CPU_original_S101_relative_ratio"],
            "J7_guard":a["J7_acceptance_guard"],
            "model_accepted":False},sort_keys=True),flush=True)
    else:
        checkpoint.update({
            "stage":STAGE,"S176_unmodified_guard_rejected":True,
            "J7_candidate_rejected":True,"model_accepted":False,
            "no_QA_TEST_or_LIVE":True,
        })
        (out/"S204_REAL_J7_REJECTION_DIAGNOSTIC.json").write_text(
            json.dumps(checkpoint,indent=2,sort_keys=True))
        print(json.dumps({
            "event":"S204_REJECTED_BY_STRICT_ORIGINAL_S176_GUARD",
            "old456":checkpoint["protected_456_full_EOS"],
            "new128":checkpoint["protected_128_full_EOS"],
            "lost_old":checkpoint["lost_original_456_full_ids"],
            "lost_new":checkpoint["lost_new128_full_ids"],
            "model_accepted":False},sort_keys=True),flush=True)
        # The diagnostic was correctly captured but strict model gate failed.
        raise SystemExit("S204 protected candidate rejected; details in J7 artifact")


if __name__=="__main__":
    main()
