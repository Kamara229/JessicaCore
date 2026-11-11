"""S136: fail-closed physical entry gate for stage3 (46.70M after S135).

S135 must have five actual J7 cycles, a physically persisted 46.70M
zero-grown checkpoint, unchanged S132/S101/Q2 lineage, and no forgotten
old/new full+EOS answers. This is research validation, not model training,
approval, heldout QA, or changes to Jessica LIVE/Render/Supabase/APK.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from calibri2.s132.growth_schedule import (
    ActualJ7Cycle,FIVE_CYCLES,MAX_BPP,MIN_CPU_RATIO,
    architecture_params,authorize_growth,
)
from calibri2.s132.progressive_core import (
    IDS_KEY,WEIGHTS_KEY,ProgressiveRouteCore,initial_active_indices,
)

STAGE2_RANK=896
STAGE3_RANK=1344
PRIOR_456_FULL=361
PRIOR_456_FIRST=389
PRIOR_128_FULL=17
SOURCE_PARAM_COUNT=architecture_params(STAGE2_RANK)
TARGET_PARAM_COUNT=architecture_params(STAGE3_RANK)
OLD_ACTIVE_COUNT=len(initial_active_indices(STAGE2_RANK))
EXPECTED_ACTIVE_INDICES=initial_active_indices(STAGE3_RANK)
assert SOURCE_PARAM_COUNT==43_261_760
assert TARGET_PARAM_COUNT==46_702_400
assert OLD_ACTIVE_COUNT==48 and len(EXPECTED_ACTIVE_INDICES)==72


def sha256_file(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_s135_stage3_entry(
    s135_report:Path,s135_physical:Path,s132_physical:Path,
)->dict[str,Any]:
    """Validate real S135 evidence; synthetic tests are NOT proof of results."""
    report=json.loads(s135_report.read_text())
    if (
        report.get("stage")!="S135_STAGE2_FIVE_CYCLES_THEN_ZERO_GROWTH"
        or report.get("previous_core_architecture")!=SOURCE_PARAM_COUNT
        or report.get("next_core_architecture")!=TARGET_PARAM_COUNT
        or report.get("next_rank")!=STAGE3_RANK
        or report.get("new_channels_trained") is not False
        or report.get("new_architecture_initial_up_exact_zero") is not True
        or report.get("old_456_exact_EOS_parity") is not True
        or report.get("new_128_exact_EOS_parity") is not True
        or report.get("old_456_full_correct",0)<PRIOR_456_FULL
        or report.get("new_128_full_correct",0)<PRIOR_128_FULL
        or report.get("additional_structural_core_params")!=
           TARGET_PARAM_COUNT-SOURCE_PARAM_COUNT
        or report.get("newly_learned_params_from_structural_growth")!=0
        or report.get("physically_persisted_active_coefficients")!=12*320*72
        or report.get("inherited_trainable_channel_count")!=24
        or report.get("stage2_trainable_new_channel_count")!=24
        or report.get("not_promoted_to_Jessica_or_MAIN") is not True
        or report.get("physical_sha256")!=sha256_file(s135_physical)
        or not math.isfinite(report.get("rate_bpp_all_runtime_included",math.inf))
        or report["rate_bpp_all_runtime_included"]>MAX_BPP
        or not math.isfinite(report.get("CPU_speed_ratio",math.nan))
        or report["CPU_speed_ratio"]<MIN_CPU_RATIO
    ):
        raise AssertionError("S136 S135 checkpoint lacks verified 46.70M physical five-cycle evidence")

    raw=report.get("five_J7_cycles_verified")
    if not isinstance(raw,list) or len(raw)!=FIVE_CYCLES:
        raise AssertionError("S136 requires all five ordered actual J7 cycle records")
    cycles=[]
    fields=set(ActualJ7Cycle.__dataclass_fields__)
    for n,entry in enumerate(raw,1):
        if not isinstance(entry,dict) or not fields.issubset(entry):
            raise AssertionError("S136 malformed J7 physical evidence")
        cycle=ActualJ7Cycle(**{name:entry[name] for name in fields})
        if (
            cycle.cycle!=n
            or not cycle.safely_accounted()
            or cycle.old_full_count<PRIOR_456_FULL
            or cycle.old_first_count<PRIOR_456_FIRST
            or cycle.new_full_count<PRIOR_128_FULL
        ):
            raise AssertionError("S136 forbidden skill loss or invalid real rollback")
        cycles.append(cycle)
    if authorize_growth(cycles,STAGE2_RANK)!=STAGE3_RANK:
        raise AssertionError("S136 no authorized five-cycle 46.70M growth")

    previous=load_file(str(s132_physical))
    grown=load_file(str(s135_physical))
    if (
        set(previous)!=set(grown)
        or {IDS_KEY,WEIGHTS_KEY}.difference(grown)
        or any(not torch.equal(previous[k],grown[k])
               for k in previous if k not in (IDS_KEY,WEIGHTS_KEY))
    ):
        raise AssertionError("S136 S101/Q2 inherited frozen physical tensors changed")
    old_ids=previous[IDS_KEY]
    old_weights=previous[WEIGHTS_KEY]
    indices=grown[IDS_KEY]
    weights=grown[WEIGHTS_KEY]
    if (
        old_ids.dtype!=torch.int32
        or tuple(old_ids.tolist())!=initial_active_indices(STAGE2_RANK)
        or old_weights.dtype!=torch.float16
        or old_weights.shape!=(12,320,OLD_ACTIVE_COUNT)
        or indices.dtype!=torch.int32
        or tuple(indices.tolist())!=EXPECTED_ACTIVE_INDICES
        or weights.dtype!=torch.float16
        or weights.shape!=(12,320,len(EXPECTED_ACTIVE_INDICES))
        or not bool(torch.isfinite(weights).all())
        or not torch.equal(weights[:,:,:24],old_weights[:,:,:24])
        or int(torch.count_nonzero(weights[:,:,OLD_ACTIVE_COUNT:]))!=0
    ):
        raise AssertionError("S136 inherited learned stage1 channels changed or new stage3 not zero")

    decoded=ProgressiveRouteCore(rank=STAGE3_RANK).eval()
    decoded.load_active_fp16({IDS_KEY:indices,WEIGHTS_KEY:weights})
    if (
        decoded.architectural_new_parameter_count()!=
        TARGET_PARAM_COUNT-36_380_480
        or not torch.equal(decoded.export_active_fp16()[WEIGHTS_KEY],weights)
    ):
        raise AssertionError("S136 46.70M real loaded model ABI differs")
    return {
        "stage":"S136_VERIFIED_STAGE3_46_702_400_RESEARCH_ONLY",
        "stage3_rank":STAGE3_RANK,
        "stage3_architecture_params":TARGET_PARAM_COUNT,
        "completed_s135_J7_cycles":FIVE_CYCLES,
        "old_456_full":report["old_456_full_correct"],
        "new_128_full":report["new_128_full_correct"],
        "physical_bpp":report["rate_bpp_all_runtime_included"],
        "CPU_speed_ratio":report["CPU_speed_ratio"],
        "stage1_frozen_24_coefficients_unchanged":True,
        "stage2_new_24_coefficients_retained":True,
        "stage3_new_24_coefficients_exact_zero":True,
        "original_S101_Q2_tensors_immutable":True,
        "physical_sha256":report["physical_sha256"],
        "model_accepted":False,
        "no_QA_no_LIVE":True,
    }
