"""S134: safe bootstrap for the NEXT five-cycle 43.26M CALIBRI stage.

Only verifies S132 physically grown + five successful J7 audits and prepares
newly activated 448:472 adapter channels for training; it does NOT train a
model or claim 43.26M parameters are learned. Continuing S132-learned first
24 channels are kept immutable throughout stage-2 optimizer updates.
No QA/TEST/S118, LIVE Jessica, Render, Supabase, APK, or main writes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from calibri2.s132.growth_schedule import architecture_params,MAX_BPP,MIN_CPU_RATIO
from calibri2.s132.progressive_core import (
    ProgressiveRouteCore,IDS_KEY,WEIGHTS_KEY,initial_active_indices,
)
from calibri2.s133.verify_stage2_entry import verified_s132_stage2_entry

STAGE2_RANK=896
LOCKED_CHANNELS=tuple(range(24))
NEW_CHANNELS=tuple(range(448,472))
STAGE2_TRAINING_CYCLES=5
STEPS_PER_CYCLE=2
STAGE2_TOTAL_CORE=architecture_params(STAGE2_RANK)
assert STAGE2_TOTAL_CORE==43_261_760


@dataclass(frozen=True)
class Stage2Parent:
    full_archive_path:Path
    s131_source_path:Path
    report_path:Path
    expected_sha256:str
    actual_rank:int
    new_parameter_count:int


def prepare_stage2(
        *,s132_report:Path,s132_physical:Path,s131_physical:Path,
        device:str="cpu",
)->tuple[ProgressiveRouteCore,Stage2Parent,torch.Tensor]:
    """Verify source at every invocation; caller owns real MPS + J7 trainer."""
    manifest=verified_s132_stage2_entry(
        s132_report,s132_physical,s131_physical)
    if (manifest["stage2_rank"]!=STAGE2_RANK
        or manifest["stage2_architecture_params"]!=STAGE2_TOTAL_CORE
        or manifest["completed_s132_j7_cycles"]!=5
        or manifest["min_paired_cpu_ratio"]!=MIN_CPU_RATIO
        or manifest["max_physical_bpp"]!=MAX_BPP):
        raise AssertionError("S134 wrong verified S132 parent or growth schedule")
    physical=load_file(str(s132_physical))
    core=ProgressiveRouteCore(rank=STAGE2_RANK).to(device)
    core.load_active_fp16({
        IDS_KEY:physical[IDS_KEY],
        WEIGHTS_KEY:physical[WEIGHTS_KEY],
    })
    immutable=core.up.detach().cpu().clone()
    if (not tuple(core.active_indices)==initial_active_indices(STAGE2_RANK)
        or not bool(torch.isfinite(core.up).all())
        or core.architectural_new_parameter_count()!=6_881_280):
        raise AssertionError("S134 actual S132 growth geometry invalid")
    parent=Stage2Parent(
        full_archive_path=s132_physical,
        s131_source_path=s131_physical,
        report_path=s132_report,
        expected_sha256=manifest["physical_source_sha256"],
        actual_rank=STAGE2_RANK,
        new_parameter_count=core.architectural_new_parameter_count(),
    )
    return core,parent,immutable


def mask_gradient_to_new_channels_only(core:ProgressiveRouteCore)->dict:
    """Bound S134 learning to *new* 448..471 channels, freeze S132 + all else."""
    if (core.rank!=STAGE2_RANK
        or tuple(core.active_indices)!=initial_active_indices(STAGE2_RANK)
        or core.up.grad is None
        or core.up.grad.shape!=core.up.shape
        or not bool(torch.isfinite(core.up.grad).all())):
        raise AssertionError("S134 missing real correctly shaped finite new parameter gradient")
    mask=torch.zeros(core.rank,dtype=core.up.grad.dtype,device=core.up.device)
    mask[list(NEW_CHANNELS)]=1
    core.up.grad.mul_(mask.view(1,1,STAGE2_RANK))
    active_grad=core.up.grad[:,:,list(NEW_CHANNELS)]
    if torch.count_nonzero(active_grad).item()==0:
        raise AssertionError("S134 no genuine gradient in newly added channels")
    return {
        "trainable_new_channels":len(NEW_CHANNELS),
        "nonzero_new_gradient_coefficients":int(torch.count_nonzero(active_grad).item()),
        "locked_original_growth_channels":len(LOCKED_CHANNELS),
        "stage2_params":STAGE2_TOTAL_CORE,
    }


def verify_frozen_inherited(
        core:ProgressiveRouteCore,original_before:torch.Tensor)->None:
    """Exact inherited tensor identity; no net old skill claim without J7."""
    if (core.rank!=STAGE2_RANK
        or original_before.shape!=(12,320,STAGE2_RANK)
        or not torch.equal(
            core.up.detach().cpu()[:,:,:448],original_before[:,:,:448])):
        raise AssertionError("S134 altered inherited first-stage learned tensor")
    core.assert_inactive_zero()
