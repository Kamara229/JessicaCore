"""S133 stage-2 preflight: verify S132 produced REAL five-cycle 43.26M growth.

Fails closed unless physical safetensors ZIP-extracted bytes, original
S131 frozen sidecar, five strict J7 cycle records, exact output parity,
CPU >=90%, and bpp <=0.7 all agree. Validation is the ONLY allowed entry
to further stage-2 training (and does not itself train/promote a model).
Green Tunnel rank expansion remains optional after positive quality evidence.
No heldout QA/TEST, S118, Jessica LIVE, Render/Supabase or APK.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from calibri2.s129.additive_core_expansion import UP_KEY
from calibri2.s132.growth_schedule import (
    FIVE_CYCLES, MAX_BPP, MIN_CPU_RATIO, START_RANK, architecture_params,
    planned_stages, authorize_growth, ActualJ7Cycle,
)
from calibri2.s132.progressive_core import (
    IDS_KEY, WEIGHTS_KEY, ProgressiveRouteCore, initial_active_indices,
)

PREVIOUS_FULL_OLD=358
PREVIOUS_FIRST_OLD=388
PREVIOUS_FULL_NEW=14
OLD_456=456
NEW_128=128
STAGE2_RANK=896
EXPECTED_ACTIVE_INDICES=initial_active_indices(STAGE2_RANK)


def sha256_file(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_s132_stage2_entry(
    report_path:Path,
    grown_physical_path:Path,
    s131_physical_path:Path,
)->dict[str,Any]:
    report=json.loads(report_path.read_text())
    if (report.get("stage")!="S132_FIVE_CYCLES_THEN_ZERO_GROWTH"
            or report.get("next_rank")!=STAGE2_RANK
            or report.get("previous_core_architecture")!=architecture_params(START_RANK)
            or report.get("next_core_architecture")!=architecture_params(STAGE2_RANK)
            or report.get("new_channels_trained") is not False
            or report.get("new_architecture_initial_up_exact_zero") is not True
            or report.get("old_456_exact_EOS_parity") is not True
            or report.get("new_128_exact_EOS_parity") is not True
            or report.get("old_456_full_correct",0)<PREVIOUS_FULL_OLD
            or report.get("new_128_full_correct",0)<PREVIOUS_FULL_NEW
            or report.get("not_promoted_to_Jessica_or_MAIN") is not True
            or report.get("CPU_speed_ratio",0)<MIN_CPU_RATIO
            or report.get("rate_bpp_all_runtime_included",9)>MAX_BPP
            or report.get("physical_sha256")!=sha256_file(grown_physical_path)
            or not bool(report.get("five_J7_cycles_verified"))):
        raise AssertionError("S133 stage-2 entry requires true audited S132 physical growth")
    cycles=report["five_J7_cycles_verified"]
    if len(cycles)!=FIVE_CYCLES:
        raise AssertionError("S133 missing one of five guarded training cycles")
    actual=[]
    for index,c in enumerate(cycles,1):
        required=set(ActualJ7Cycle.__dataclass_fields__)
        if not required.issubset(c):
            raise AssertionError("S133 incomplete physical J7 cycle evidence")
        stage=ActualJ7Cycle(**{k:c[k] for k in required})
        if stage.cycle!=index:
            raise AssertionError("S133 out-of-order J7 cycle")
        if stage.old_full_count<PREVIOUS_FULL_OLD or stage.old_first_count<PREVIOUS_FIRST_OLD:
            raise AssertionError("S133 lost an S131 original full/first correct count")
        if stage.new_full_count<PREVIOUS_FULL_NEW:
            raise AssertionError("S133 lost full correct TRAIN answers from S131")
        if not stage.safely_accounted():
            raise AssertionError("S133 missing J7 physical rollback/retention proof")
        actual.append(stage)
    if authorize_growth(actual,START_RANK)!=STAGE2_RANK:
        raise AssertionError("S133 growth was not authorized by guarded five cycles")
    new=load_file(str(grown_physical_path))
    old=load_file(str(s131_physical_path))
    new_fields={IDS_KEY,WEIGHTS_KEY}
    if (UP_KEY not in old
            or UP_KEY in new
            or not new_fields.issubset(new)
            or set(new)-new_fields!=set(old)-{UP_KEY}
            or any(not torch.equal(old[key],new[key])
                   for key in old if key!=UP_KEY)):
        raise AssertionError("S133 original compressed/Q2 tensor bytes altered")
    indices=new[IDS_KEY]
    weights=new[WEIGHTS_KEY]
    inherited=old[UP_KEY]
    if (indices.dtype!=torch.int32
        or tuple(indices.tolist())!=EXPECTED_ACTIVE_INDICES
        or weights.dtype!=torch.float16
        or weights.shape!=(12,320,len(EXPECTED_ACTIVE_INDICES))
        or inherited.dtype!=torch.float16
        or inherited.shape!=(12,320,24)
        or not bool(torch.isfinite(weights).all())
        or torch.count_nonzero(weights[:,:,24:]).item()!=0):
        # Learned first-24 weights are allowed to improve during S132's
        # five *committed and physically guarded* J7 cycles; requiring
        # equality to S131 would wrongly forbid any continued learning.
        raise AssertionError("S133 growth did not preserve S131 or appended nonzero untrained weights")
    # Actual architecture physically reconstructed and decoded; inactive
    # newly counted parameters are not represented as learned FP32.
    decoder=ProgressiveRouteCore(rank=STAGE2_RANK)
    decoder.load_active_fp16({IDS_KEY:indices,WEIGHTS_KEY:weights})
    if (decoder.architectural_new_parameter_count()!=
        architecture_params(STAGE2_RANK)-36_380_480
        or not torch.equal(decoder.export_active_fp16()[WEIGHTS_KEY],weights)):
        raise AssertionError("S133 actual expanded learned FP16 reload mismatch")
    return {
        "stage":"S133_READY_FOR_FIVE_J7_CYCLES_ON_43_261_760_CORE",
        "previous_39M_full_456":PREVIOUS_FULL_OLD,
        "verified_43M_full_456":report["old_456_full_correct"],
        "verified_43M_full_new128":report["new_128_full_correct"],
        "stage2_architecture_params":architecture_params(STAGE2_RANK),
        "stage2_rank":STAGE2_RANK,
        "stage2_active_indices":list(EXPECTED_ACTIVE_INDICES),
        "completed_s132_j7_cycles":FIVE_CYCLES,
        "max_physical_bpp":MAX_BPP,
        "min_paired_cpu_ratio":MIN_CPU_RATIO,
        "green_tunnel_enlarged":False,
        "expected_next_rank_after_five_more_cycles":1344,
        "original_S101_and_Q2_immutable":True,
        "S131_trained_source_shape_and_originals_verified":True,
        "trained_new_growth_channels":0,
        "no_QA_or_LIVE":True,
        "physical_source_sha256":report["physical_sha256"],
    }
