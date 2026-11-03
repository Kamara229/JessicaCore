"""Pure numerical aggregation for paired teacher-versus-packed measurements.

No weights, filesystem, datasets, or TEST access. Sum squared errors before
computing global ratios; averaging windowwise relative errors is misleading.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class Moment:
    n: int = 0
    ref_sq: float = 0.0
    alt_sq: float = 0.0
    cross: float = 0.0
    error_sq: float = 0.0
    error_sum: float = 0.0
    observations: int = 0

    def add(self, reference: torch.Tensor, alternative: torch.Tensor) -> None:
        if reference.shape != alternative.shape or not reference.numel():
            raise AssertionError("paired tensors must have equal, nonempty shape")
        a = reference.detach().to(dtype=torch.float64).reshape(-1)
        b = alternative.detach().to(dtype=torch.float64).reshape(-1)
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise AssertionError("nonfinite original/packed operator outputs")
        e = b - a
        self.n += a.numel()
        self.ref_sq += float(a.square().sum())
        self.alt_sq += float(b.square().sum())
        self.cross += float((a * b).sum())
        self.error_sq += float(e.square().sum())
        self.error_sum += float(e.sum())
        self.observations += 1

    def merge(self, other: "Moment") -> None:
        for name in ("n", "ref_sq", "alt_sq", "cross", "error_sq",
                     "error_sum", "observations"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict:
        if not self.n or self.ref_sq <= 0 or self.alt_sq <= 0:
            raise AssertionError("empty or degenerate paired observations")
        return {
            "scalar_count": self.n,
            "observations": self.observations,
            "relative_L2": math.sqrt(self.error_sq / self.ref_sq),
            "cosine": self.cross / math.sqrt(self.ref_sq * self.alt_sq),
            "teacher_RMS": math.sqrt(self.ref_sq / self.n),
            "absolute_RMSE": math.sqrt(self.error_sq / self.n),
            "signed_bias": self.error_sum / self.n,
            "teacher_squared_energy": self.ref_sq,
            "squared_error_energy": self.error_sq,
        }


class MetricBook:
    """Named groups of cumulative moments; categories remain disjoint."""

    def __init__(self) -> None:
        self.groups: dict[str, dict[str, Moment]] = {}

    def observe(self, category: str, key: str,
                teacher: torch.Tensor, packed: torch.Tensor) -> None:
        self.groups.setdefault(category, {}).setdefault(key, Moment()).add(
            teacher, packed)

    def snapshots(self) -> dict:
        return {category: {name: metric.to_dict() for name, metric in group.items()}
                for category, group in self.groups.items()}
