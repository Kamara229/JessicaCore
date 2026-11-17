"""S28: provenance-locked single-host source of S27 FP16 SHA instability.

S27 same-host replays differed (both 2-thread and 1-thread CPU). Before
another architecture/optimizer change, determine whether identical raw338,
identical seed and frozen S19d parent create identical initial weights,
and at which optimizer step count physically paid FP16 weights diverge.
Only 248 fold33 synthetic TRAIN-side rows: 8 preposed REJECTED rows are
NEVER part of a loss or gradient in this diagnostic. No EVAL/LIVE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_train9_real43m_graph import (
    fixed_multitask_loss,LR,
)
from research.nra3_s19.s19h_true43m_train9_loo_diagnostics import take
from research.nra3_s20.s22d_train256_grouped40_true43m_graph import (
    load_physical_training_source,new_graph_model,SEED,S19D_FULL_PHYSICAL_SHA,
)
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    splits,FROZEN_S22_TRAIN256_SHA256,
)

FOLD=33
OBSERVED_HISTORIC_BASE_FOLD_SHA="097b4471b3049e738bdf84b32b4066408d6f27fc878dee878dc0b0e26922a67a"
PROBE_STEPS=(0,1,5,40,240)


def check(ok,message):
    if not ok:
        raise ValueError("S28 frozen 43M+GT CPU TRAIN-only probe: "+message)


def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()



def canonical_paid_semantic_fp16_sha(model):
    """Stable digest of numerically paid FP16 tensors, NOT the serializer."""
    h=hashlib.sha256(b"NRA3_CANONICAL_PAID_FP16_TENSORS_V1")
    with torch.no_grad():
        for name,param in sorted(model.named_parameters(),key=lambda pair:pair[0]):
            values=param.detach().cpu().half().contiguous().numpy().astype("<f2",copy=False).tobytes()
            meta=json.dumps({"name":name,"shape":list(param.shape),
                             "paid_dtype":"float16"},sort_keys=True,
                            separators=(",",":")).encode("utf-8")
            h.update(len(meta).to_bytes(4,"little"))
            h.update(meta)
            h.update(len(values).to_bytes(8,"little"))
            h.update(values)
    return h.hexdigest()


def train_exact_prefix(physical_parent,raw,valid,labels,steps):
    """Same graph, seed, optimizer and TRAIN sequence as S22d, stopped early."""
    frozen=PaidContextualEventScope.from_physical(physical_parent)
    check(all(not p.requires_grad for p in frozen.parameters()),
          "original S19d frozen parent not actually frozen")
    model=new_graph_model(frozen,SEED+FOLD)
    initial=model.head_physical()
    initial_physical=model.physical_bytes()
    initial_canonical=canonical_paid_semantic_fp16_sha(model)
    parent_bytes1=model.parent.physical_bytes()
    parent_bytes2=model.parent.physical_bytes()
    optimizer=torch.optim.AdamW(list(model.semantic_params()),
                                lr=LR,weight_decay=.001)
    losses=[]
    for iteration in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output=model(raw,valid) # only original frozen43M hidden + raw18; no gold here
        loss=fixed_multitask_loss(output,labels,valid)
        check(bool(torch.isfinite(loss)),"nonfinite TRAIN-only objective")
        loss.backward()
        check(all(p.grad is None for p in model.parent.parameters()),
              "frozen parent received gradient")
        torch.nn.utils.clip_grad_norm_(list(model.semantic_params()),1.)
        optimizer.step()
        if iteration in (0,steps-1):
            losses.append(float(loss.detach()))
    paid=model.physical_bytes()
    post_head=model.head_physical()
    post_canonical=canonical_paid_semantic_fp16_sha(model)
    param=[p.detach().cpu().clone() for p in model.semantic_params()]
    return {
        "initial_head_sha256":sha(initial),
        "initial_full_fp16_sha256":sha(initial_physical),
        "initial_canonical_paid_FP16_SHA256":initial_canonical,
        "same_model_two_frozen_parent_serializations_identical":parent_bytes1==parent_bytes2,
        "parent_payload_first_sha":sha(parent_bytes1),
        "parent_payload_second_sha":sha(parent_bytes2),
        "paid_prefix_SHA256":sha(paid),
        "post_head_FP16_SHA256":sha(post_head),
        "post_canonical_paid_FP16_SHA256":post_canonical,
        "first_last_logged_train_losses":losses,
        "actual_prefix_steps":steps,
    },param


def compare_fp32_exact(left,right):
    check(len(left)==len(right),"semantic param count changed between refits")
    equal=all(torch.equal(a,b) for a,b in zip(left,right))
    mismatch=[{
        "parameter_index":i,
        "shape":list(a.shape),
        "max_absolute_difference":float((a-b).abs().max()),
        "different_float32_scalars":int((a!=b).sum()),
    } for i,(a,b) in enumerate(zip(left,right)) if not torch.equal(a,b)]
    return {"fp32_all_equal":equal,
            "number_of_differing_parameters":len(mismatch),
            "first_differing_parameters":mismatch[:5]}


def main():
    p=argparse.ArgumentParser()
    for name in ("features","gold","manifest","parent","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    check(not a.out.exists(),"do not overwrite existing artifacts")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    check(torch.are_deterministic_algorithms_enabled(),"torch deterministic mode not enabled")
    check(torch.get_num_threads()==1,"torch CPU threads not isolated")
    raw,valid,y,m=load_physical_training_source(a.features,a.gold,a.manifest)
    check(m["S22_sha_locked_synthetic_TRAIN256_graph_gold"]==
          FROZEN_S22_TRAIN256_SHA256,"original exact 256 TRAIN graph source changed")
    physical_parent=a.parent.read_bytes()
    check(sha(physical_parent)==S19D_FULL_PHYSICAL_SHA,"original 9332 FP16 S19d SHA changed")
    axis,group,tr,held=splits()[FOLD]
    check(axis=="CONSTRUCTION" and group==25 and len(held)==8 and len(tr)==248
          and set(tr).isdisjoint(held),
          "S27 known single inversion fold geometry drift")
    raw,valid,labels=take(raw,tr),take(valid,tr),take(y,tr)
    report={
        "stage":"NRA3_S28_REAL43M_GT_SINGLE_HOST_INITIAL_AND_OPTIMIZER_PREFIX_SHA_PROBE",
        "source_s22c_train256_SHA256":FROZEN_S22_TRAIN256_SHA256,
        "frozen_real_9332_FP16_S19d_SHA256":S19D_FULL_PHYSICAL_SHA,
        "same_training_seed":SEED+FOLD,
        "fixed_predeclared_fold":FOLD,
        "TRAIN_side_rows":248,
        "completely_held_synthetic_TRAIN_rows_NOT_used_in_training":8,
        "torch_num_threads":torch.get_num_threads(),
        "torch_deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "torch_version":torch.__version__,
        "historic_real_S22d_fold33_SHA_original":OBSERVED_HISTORIC_BASE_FOLD_SHA,
        "tests":[],
        "no_independent_EVAL_or_original_616_J7_replay":True,
        "no_original43M_GT_or_live_model_training":True,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
    for steps in PROBE_STEPS:
        one,pone=train_exact_prefix(physical_parent,raw,valid,labels,steps)
        two,ptwo=train_exact_prefix(physical_parent,raw,valid,labels,steps)
        fp=compare_fp32_exact(pone,ptwo)
        entry={
            "prefix_steps":steps,
            "first":one,
            "repeat":two,
            "same_runner_initial_head_equal":one["initial_head_sha256"]==two["initial_head_sha256"],
            "same_runner_initial_full_physical_equal":one["initial_full_fp16_sha256"]==two["initial_full_fp16_sha256"],
            "same_runner_post_steps_full_FP16_equal":one["paid_prefix_SHA256"]==two["paid_prefix_SHA256"],
            "same_runner_post_steps_head_FP16_equal":one["post_head_FP16_SHA256"]==two["post_head_FP16_SHA256"],
            "same_runner_post_steps_canonical_paid_FP16_equal":one["post_canonical_paid_FP16_SHA256"]==two["post_canonical_paid_FP16_SHA256"],
            "same_runner_initial_canonical_paid_FP16_equal":one["initial_canonical_paid_FP16_SHA256"]==two["initial_canonical_paid_FP16_SHA256"],
            "post_steps_fp32_difference":fp,
        }
        check(fp["fp32_all_equal"] and entry["same_runner_post_steps_head_FP16_equal"]
              and entry["same_runner_post_steps_canonical_paid_FP16_equal"],
              "actual paid FP16 tensors/FP32 model weights changed, NOT merely physical serializer")
        report["tests"].append(entry)
        print("S28_REAL_SINGLE_CPU_THREAD_PREFIX_COMPARE",
              json.dumps({k:v for k,v in entry.items()
                          if k not in ("first","repeat","post_steps_fp32_difference")},sort_keys=True),
              "FP32_PARAM_DIFFERENCES",fp["number_of_differing_parameters"],
              flush=True)
    a.out.mkdir(parents=True,exist_ok=False)
    target=a.out/"NRA3_S28_ACTUAL43M_GT_FOLD33_SEEDED_PREFIX_FP16_REPRODUCIBILITY.json"
    target.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                      encoding="utf-8")
    print("S28_SOURCE_AND_FROZEN_PARENT_SHA_LOCKED_NO_TRAIN_HELD8",flush=True)


if __name__=="__main__":
    main()
