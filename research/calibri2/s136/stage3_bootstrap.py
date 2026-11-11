"""CALIBRI S136: stage-3 46.70M research bootstrap, not accepted training.

Stage-3 keeps every inherited FP16 coefficient from S132 and S135 frozen.
Only newly activated 896..919 rank-channel coefficients may receive a
gradient, after verifying all five *real* S135 J7 cycles and the physical
46.70M source archive. S136 never accesses QA/S118 or LIVE services.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from calibri2.s132.growth_schedule import architecture_params
from calibri2.s132.progressive_core import (
    ProgressiveRouteCore, IDS_KEY, WEIGHTS_KEY, initial_active_indices,
)
from calibri2.s136.verify_stage3_entry import (
    verified_s135_stage3_entry, STAGE3_RANK, TARGET_PARAM_COUNT,
)

FROZEN_STAGE1_CHANNELS=tuple(range(24))
FROZEN_STAGE2_CHANNELS=tuple(range(448,472))
STAGE3_NEW_CHANNELS=tuple(range(896,920))
PREVIOUS_RANK=896
STAGE3_TRAINING_CYCLES=5
STEPS_PER_J7_CYCLE=2

assert architecture_params(STAGE3_RANK)==46_702_400
assert len(initial_active_indices(STAGE3_RANK))==72


@dataclass(frozen=True)
class Stage3Parent:
    grown_source_archive:Path
    s132_source_archive:Path
    s135_report_path:Path
    verified_sha256:str
    stage3_rank:int
    all_inclusive_physical_bpp:float
    previous_cpu_speed_ratio:float
    old_full_correct:int
    new_full_correct:int


def prepare_stage3(
    *, s135_report:Path, s135_physical:Path, s132_physical:Path,
    device:str="cpu",
)->tuple[ProgressiveRouteCore,Stage3Parent,torch.Tensor]:
    """Fail closed on real S135 physical lineage before materializing weights."""
    evidence=verified_s135_stage3_entry(
        s135_report,s135_physical,s132_physical)
    if (
        evidence["stage3_rank"]!=STAGE3_RANK
        or evidence["stage3_architecture_params"]!=TARGET_PARAM_COUNT
        or evidence["completed_s135_J7_cycles"]!=STAGE3_TRAINING_CYCLES
        or evidence["stage3_new_24_coefficients_exact_zero"] is not True
        or evidence["original_S101_Q2_tensors_immutable"] is not True
        or evidence["model_accepted"] is not False
    ):
        raise AssertionError("S136 stage3 requires real protected five-J7 S135 proof")

    archive=load_file(str(s135_physical))
    core=ProgressiveRouteCore(rank=STAGE3_RANK).to(device)
    core.load_active_fp16({
        IDS_KEY:archive[IDS_KEY], WEIGHTS_KEY:archive[WEIGHTS_KEY],
    })
    core.assert_inactive_zero()
    if (
        tuple(core.active_indices)!=initial_active_indices(STAGE3_RANK)
        or core.architectural_new_parameter_count()!=
           TARGET_PARAM_COUNT-36_380_480
        or int(torch.count_nonzero(core.up[:,:,STAGE3_NEW_CHANNELS]).item())!=0
        or not bool(torch.isfinite(core.up).all())
    ):
        raise AssertionError("S136 stage3 materialized coefficients invalid")
    immutable=core.up.detach().cpu().clone()
    parent=Stage3Parent(
        grown_source_archive=s135_physical,
        s132_source_archive=s132_physical,
        s135_report_path=s135_report,
        verified_sha256=evidence["physical_sha256"],
        stage3_rank=STAGE3_RANK,
        all_inclusive_physical_bpp=evidence["physical_bpp"],
        previous_cpu_speed_ratio=evidence["CPU_speed_ratio"],
        old_full_correct=evidence["old_456_full"],
        new_full_correct=evidence["new_128_full"],
    )
    return core,parent,immutable


def mask_gradient_to_stage3_new_only(core:ProgressiveRouteCore)->dict:
    """Mask every inherited and inactive coefficient, fail on zero new gradient.

    Never reuse an optimizer with old momentum: caller must initialize a fresh
    optimizer on this core after the verified stage3 physical reload.
    """
    if (
        core.rank!=STAGE3_RANK
        or tuple(core.active_indices)!=initial_active_indices(STAGE3_RANK)
        or core.up.grad is None
        or core.up.grad.shape!=core.up.shape
        or not bool(torch.isfinite(core.up.grad).all())
    ):
        raise AssertionError("S136 missing finite gradient for new rank1344 channels")
    mask=torch.zeros((STAGE3_RANK,),dtype=core.up.grad.dtype,device=core.up.device)
    mask[list(STAGE3_NEW_CHANNELS)]=1
    core.up.grad.mul_(mask.view(1,1,STAGE3_RANK))
    fresh=core.up.grad[:,:,list(STAGE3_NEW_CHANNELS)]
    if int(torch.count_nonzero(fresh).item())==0:
        raise AssertionError("S136 stage3 has no genuine gradient in new channels")
    if int(torch.count_nonzero(core.up.grad[:,:,:PREVIOUS_RANK]).item())!=0:
        raise AssertionError("S136 inherited stage1/stage2 gradient not zero")
    return {
        "stage3_rank":STAGE3_RANK,
        "new_trainable_channels":len(STAGE3_NEW_CHANNELS),
        "frozen_stage1_channels":len(FROZEN_STAGE1_CHANNELS),
        "frozen_stage2_channels":len(FROZEN_STAGE2_CHANNELS),
        "nonzero_fresh_gradient_coefficients":
            int(torch.count_nonzero(fresh).item()),
        "research_only":True,
    }


def verify_frozen_stage1_stage2(
    core:ProgressiveRouteCore, immutable_parent:torch.Tensor,
)->None:
    """Exact inherited FP32 tensor identity; J7 still decides actual skill safety."""
    if (
        core.rank!=STAGE3_RANK
        or immutable_parent.shape!=(12,320,STAGE3_RANK)
        or not torch.equal(
            core.up.detach().cpu()[:,:,:PREVIOUS_RANK],
            immutable_parent[:,:,:PREVIOUS_RANK],
        )
    ):
        raise AssertionError("S136 changed S132/S135 inherited 48 learned channels")
    core.assert_inactive_zero()
