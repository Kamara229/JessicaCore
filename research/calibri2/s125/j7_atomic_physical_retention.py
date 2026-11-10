"""S125: transplant S87-J7's ATOMIC optimizer/parameter rollback into S124.

This is a transactional primitive, not a model acceptance or inference
algorithm. The J7 lesson is *not* transfer of S82 weights; the exact
old+new, FP16 physical original S101 autogreedy decoder must be invoked
by the caller after every bounded block BEFORE its updates can commit.

S87-J7 ran 64 MPS updates in 8-step blocks: 6 committed, 2 rolled back
(32 and 48) after the original 196 previously correct full+EOS examples
were verified on physically serialized compressed CPU. It gained ZERO
new correct answers. S102 already tried J7's algorithm on S101 and
rejected +0/-5 full answer proposal after 8 GPU steps. The transferable
idea is the atomic transaction only, not the original 196 threshold.
S125 protects ALL 353 S101 old correct full and 387 old first-token
identities, and counts inclusive physical <=.7bpp & CPU speed>=.9.

No QA/TEST reads, model acceptance, production writes or auto merging.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class PhysicalRetentionCheck:
    # Values computed by callback after actual persisted-and-reloaded
    # S101 FP16 physical + phase-specific runtime-config CPU inference.
    original_456_checked: int
    original_full_correct: int
    original_first_correct: int
    lost_original_full: int
    lost_original_first: int
    newly_correct_original_full: int
    new_S122_train_full: int
    new_S122_train_both_correct_pairs: int
    physical_all_inclusive_bpp: float
    paired_original_S101_CPU_speed_ratio: float
    physical_FP16_reload_verified: bool
    immutable_original_S101_bytes_verified: bool
    S95_S98_unchanged: bool

    def valid(self) -> bool:
        return (
            self.original_456_checked == 456
            and self.original_full_correct == 353
            and self.original_first_correct == 387
            and self.lost_original_full == 0
            and self.lost_original_first == 0
            and self.physical_FP16_reload_verified
            and self.immutable_original_S101_bytes_verified
            and self.S95_S98_unchanged
            and math.isfinite(self.physical_all_inclusive_bpp)
            and self.physical_all_inclusive_bpp <= 0.7
            and math.isfinite(self.paired_original_S101_CPU_speed_ratio)
            and self.paired_original_S101_CPU_speed_ratio >= 0.9
            and 0 <= self.new_S122_train_full <= 128
            and 0 <= self.new_S122_train_both_correct_pairs <= 64
            and self.newly_correct_original_full >= 0
        )


def identical_tree(a,b)->bool:
    """Exact tensor-by-tensor optimizer/weight state verification."""
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        return a.dtype==b.dtype and a.shape==b.shape and bool(torch.equal(a,b))
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(identical_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) and isinstance(b,type(a)):
        return len(a)==len(b) and all(identical_tree(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


@dataclass(frozen=True)
class TransactionOutcome:
    block_steps: int
    committed: bool
    rollback_verified: bool
    learning_rate_after: float
    guard: PhysicalRetentionCheck
    model_accepted: bool = False


def atomic_train_block(
    trainable_params:list[torch.nn.Parameter],
    optimizer:torch.optim.Optimizer,
    perform_steps:Callable[[],int],
    persisted_actual_cpu_guard:Callable[[],PhysicalRetentionCheck],
    *,
    reduction_factor:float=0.5,
)->TransactionOutcome:
    """Execute bounded steps; first failure restores params AND full Adam.

    The physical callback MUST reserialize & re-open candidate FP16 source
    sidecar+config and decode genuine frozen S101 original 456 + S122 128
    with the actual S95/S98 policy before a commit decision is possible.
    If callback throws or reports invalid safety, restore optimizer and
    parameters atomically; verify BOTH exact bytewise tensor states;
    only then adjust future LR. The callback is invoked again after
    rollback, proving physical old answers are restored.
    """
    if (not trainable_params or not 0<reduction_factor<1
            or not optimizer.param_groups or
            any(p not in [q for group in optimizer.param_groups
                          for q in group["params"]] for p in [])):
        raise ValueError("Invalid transaction configuration")
    original_params=[x.detach().clone() for x in trainable_params]
    original_opt=copy.deepcopy(optimizer.state_dict())
    original_group_lrs=[float(g["lr"]) for g in optimizer.param_groups]
    if any(not math.isfinite(x) or x<=0 for x in original_group_lrs):
        raise ValueError("Invalid optimizer learning rate")
    try:
        steps=perform_steps()
        if not isinstance(steps,int) or steps<1 or steps>8:
            raise AssertionError("S125 only bounded 1–8 update transactions")
        result=persisted_actual_cpu_guard()
        if not isinstance(result,PhysicalRetentionCheck):
            raise TypeError("Physical CPU guard must return documented evidence")
        if result.valid():
            return TransactionOutcome(
                block_steps=steps,committed=True,rollback_verified=False,
                learning_rate_after=float(optimizer.param_groups[0]["lr"]),
                guard=result,
            )
        failed=result
    except Exception:
        # When actual physical evaluation throws, never silently keep
        # candidate weights; ensure exact rollback and re-raise.
        with torch.no_grad():
            for p,saved in zip(trainable_params,original_params):
                p.copy_(saved)
        optimizer.load_state_dict(original_opt)
        if (not all(torch.equal(p.detach(),original_params[i])
                    for i,p in enumerate(trainable_params))
                or not identical_tree(optimizer.state_dict(),original_opt)):
            raise AssertionError("S125 fail-closed rollback verification FAILED")
        raise

    with torch.no_grad():
        for p,saved in zip(trainable_params,original_params):
            p.copy_(saved)
    optimizer.load_state_dict(original_opt)
    if (not all(torch.equal(p.detach(),original_params[i])
                for i,p in enumerate(trainable_params))
            or not identical_tree(optimizer.state_dict(),original_opt)):
        raise AssertionError("S125 rejected transaction failed exact Adam/weight rollback")
    restored=persisted_actual_cpu_guard()
    if not isinstance(restored,PhysicalRetentionCheck) or not restored.valid():
        raise AssertionError("S125 rollback physically failed original 456 old-skill guard")
    for group,lr in zip(optimizer.param_groups,original_group_lrs):
        group["lr"]=lr*reduction_factor
    return TransactionOutcome(
        block_steps=steps,committed=False,rollback_verified=True,
        learning_rate_after=float(optimizer.param_groups[0]["lr"]),
        guard=failed,
    )
