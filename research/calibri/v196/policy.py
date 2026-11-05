"""Reusable Green Tunnel recovery policy, independent of 135M tensor geometry.

A policy is a hypothesis about a suitable recovery budget, NOT a claim
that cycles scale linearly or restoration is guaranteed on any LLM.
All model-specific binding is in run.py; no extra run-time weights added.
"""
from dataclasses import dataclass
from math import ceil

@dataclass(frozen=True)
class RecoveryPolicy:
    cycles: int = 2
    steps_per_cycle: int = 96
    learning_rate: float = 0.0015
    gradient_clip: float = 1.0
    fp32_centroid_guard: float = 0.0005
    teacher_soft_temperature: float = 2.0
    pair_every_n_steps: int = 4
    max_5bit_swaps_per_step: int = 512
    exact_physical_rate_ceiling_bpp: float = 0.700
    def __post_init__(self):
        if not (1 <= self.cycles <= 16 and 1 <= self.steps_per_cycle <= 256):
            raise ValueError("Invalid finite Green Tunnel cycle budget")
        if not (0 < self.learning_rate < .1 and 0 < self.gradient_clip < 100):
            raise ValueError("Invalid paid source optimizer budget")
        if not (0 < self.fp32_centroid_guard < 1 and 1 <= self.pair_every_n_steps <= 16):
            raise ValueError("Invalid centroid / contrast budget")
        if self.max_5bit_swaps_per_step < 0 or self.exact_physical_rate_ceiling_bpp != .7:
            raise ValueError("All source updates must remain physically paid at <=0.700 bpp")

def suggested_cycles(unique_parameters, reference=134_515_008, base_cycles=2):
    """Non-executing model-agnostic *planning* heuristic, unproven at scale."""
    if unique_parameters<=0 or reference<=0 or base_cycles<=0:
        raise ValueError("Invalid original unique-parameter scale")
    return max(base_cycles,min(16,ceil(base_cycles*(unique_parameters/reference)**0.5)))

POLICY=RecoveryPolicy()
