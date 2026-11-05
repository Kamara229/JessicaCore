"""Codec-agnostic Green Tunnel recovered-source orchestration.

A model family requires an explicit physical codec adapter; no universal
quality guarantee is implied. All source learning occurs in paid fields,
and teacher/optimizer are dropped before compact inference.
"""
from .contract import PhysicalSource, verify_physical_source, require_paid_update
from .schedule import Operator, plan_cycles
from .parity import compare_machine_parity
