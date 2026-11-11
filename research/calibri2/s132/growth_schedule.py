"""CALIBRI S132+ progression contract: FIVE J7 cycles, THEN new capacity.

This is a research scheduler and no-training policy, not a claim that
untrained/newly-zero parameters are learned intelligence. Every stage
requires a complete 456 old + 128 TRAIN physical J7 record, true current
full/EOS identities, bpp<=0.7 including runtime and physical sidecars, and
CPU speed >=90% of paired S101 baseline. No heldout QA or S118.

A 'cycle' is exactly one J7 atomic two-MPS-step transaction with a
physically serialized candidate and exhaustive 456+128 CPU assessment;
failed cycles are rolled back. After exactly five fully audited cycles,
advance the structural rank by at most 448 per layer. This expansion is
zero-initialized and not counted as newly learned. Old successful weights
carry forward unaltered. Growth never promotes a model to LIVE.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

BASE=36_380_480
DEPTH=12
WIDTH=320
RANK_STEP=448
START_RANK=448
TARGET=135_000_000
PER_RANK=2*DEPTH*WIDTH
# Exactly 134,999,360, 640 params below 135.0M target (no overshoot).
MAX_RANK=(TARGET-BASE)//PER_RANK
FIVE_CYCLES=5
STEPS_PER_CYCLE=2
MAX_BPP=0.7
MIN_CPU_RATIO=0.9
GT_GROWTH_RULE="quality_and_budget_only"


def architecture_params(rank:int)->int:
    if not isinstance(rank,int) or rank<START_RANK:
        raise ValueError("S132 rank must preserve 448-channel S129 origin")
    return BASE+PER_RANK*rank


def next_rank(rank:int,*,completed_cycles:int)->int:
    if completed_cycles!=FIVE_CYCLES:
        raise ValueError("S132 forbids capacity growth before five J7 cycles")
    if rank>MAX_RANK:
        raise ValueError("S132 architecture already overshot target")
    return min(MAX_RANK,rank+RANK_STEP)


def planned_stages():
    rank=START_RANK
    stages=[]
    while rank<MAX_RANK:
        new_rank=next_rank(rank,completed_cycles=FIVE_CYCLES)
        stages.append({
            "stage_index":len(stages),
            "training_rank":rank,
            "training_core_params":architecture_params(rank),
            "j7_cycles_before_growth":FIVE_CYCLES,
            "two_MPS_steps_per_cycle":STEPS_PER_CYCLE,
            "next_rank":new_rank,
            "next_core_params":architecture_params(new_rank),
            "added_core_params":architecture_params(new_rank)-architecture_params(rank),
        })
        if new_rank<=rank:
            raise AssertionError("S132 nonadvancing structural stage")
        rank=new_rank
    if architecture_params(rank)!=134_999_360 or len(stages)!=28:
        raise AssertionError("S132 planned 39.82M to 135M schedule changed")
    return stages


@dataclass(frozen=True)
class ActualJ7Cycle:
    cycle:int
    old_456_checked:int
    lost_previously_correct_full:int
    lost_previously_correct_first:int
    physical_bpp:float
    paired_cpu_ratio:float
    physical_reloaded:bool
    source_immutable:bool
    old_full_count:int
    old_first_count:int
    new_full_count:int
    new_pairs_both_count:int
    transaction_committed:bool
    transaction_rollback_verified:bool

    def safely_accounted(self)->bool:
        return (self.old_456_checked==456
                and self.lost_previously_correct_full==0
                and self.lost_previously_correct_first==0
                and self.old_full_count>=353
                and self.old_first_count>=387
                and math.isfinite(self.physical_bpp)
                and self.physical_bpp<=MAX_BPP
                and math.isfinite(self.paired_cpu_ratio)
                and self.paired_cpu_ratio>=MIN_CPU_RATIO
                and self.physical_reloaded and self.source_immutable
                and (self.transaction_committed or self.transaction_rollback_verified)
                and 0<=self.new_full_count<=128
                and 0<=self.new_pairs_both_count<=64)


def authorize_growth(cycles:list[ActualJ7Cycle],current_rank:int)->int:
    if len(cycles)!=FIVE_CYCLES or [c.cycle for c in cycles]!=list(range(1,FIVE_CYCLES+1)):
        raise AssertionError("S132 requires five ordered auditable training cycles")
    if not all(c.safely_accounted() for c in cycles):
        raise AssertionError("S132 forbids growth after missing/unsafe physical J7 evidence")
    return next_rank(current_rank,completed_cycles=len(cycles))


def may_grow_green_tunnel(*,new_correct_full:int,forgotten:int,
                          extra_serialized_bytes:int,projected_bpp:float,
                          paired_cpu_ratio:float)->bool:
    """GT capacity changes require measured gain; never automatic per stage."""
    return (new_correct_full>=1 and forgotten==0
            and extra_serialized_bytes>0
            and math.isfinite(projected_bpp) and projected_bpp<=MAX_BPP
            and math.isfinite(paired_cpu_ratio) and paired_cpu_ratio>=MIN_CPU_RATIO)
