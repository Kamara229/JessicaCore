"""S145 research-only neural skills: learnable hidden-state transforms, never answer lookup.

Stand-alone CPU interface prototype. NOT connected to the real S142 decode path;
does NOT claim better language quality, physical acceptance, or sparse CPU speed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NeuralSkillSpec:
    skill_id: str
    layers: tuple[int, ...]
    rank: int
    version: int = 1

    def __post_init__(self):
        if not self.skill_id or not self.skill_id.replace("_", "").isalnum():
            raise ValueError("skill_id must be a safe stable identifier")
        if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 0:
            raise ValueError("layers must be distinct nonnegative integers")
        if self.rank < 1 or self.version < 1:
            raise ValueError("rank and version must be positive")


class NeuralSkillBlock(nn.Module):
    """Low-rank residual: zero 'up' ensures exact parent identity on insertion."""

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        self.rank = rank

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(hidden))) / self.rank


class NeuralSkillLayer(nn.Module):
    """Independent skill residuals and gates. Each token is routed CAUSALLY.

    Router never reads future sequence tokens, prompt IDs, answers or exact text.
    Frozen old modules cannot change weights; their decoded answers can STILL
    change after adding a new module. A physical full+EOS J7 check is essential.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleDict()
        self.routers = nn.ModuleDict()

    def register_skill(self, skill_id: str, rank: int) -> None:
        if skill_id in self.blocks:
            raise ValueError(f"duplicate neural skill: {skill_id}")
        self.blocks[skill_id] = NeuralSkillBlock(self.hidden_size, rank)
        self.routers[skill_id] = nn.Linear(self.hidden_size, 1)
        nn.init.zeros_(self.routers[skill_id].weight)
        nn.init.constant_(self.routers[skill_id].bias, -2.0)

    def freeze_skill(self, skill_id: str) -> None:
        if skill_id not in self.blocks:
            raise KeyError(skill_id)
        for module in (self.blocks[skill_id], self.routers[skill_id]):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, hidden: torch.Tensor, *,
                disabled: frozenset[str] = frozenset()) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.size(-1) != self.hidden_size:
            raise ValueError("expected [batch, sequence, hidden_size]")
        result = hidden
        # Additive independent paths; frozen routes survive disabling new ones.
        for skill_id, block in self.blocks.items():
            if skill_id in disabled:
                continue
            # Token-local gate: no future leakage, stable across KV chunk sizes.
            gate = torch.sigmoid(self.routers[skill_id](hidden))
            result = result + gate * block(hidden)
        return result


class LayerwiseNeuralSkillBank(nn.Module):
    """Create versioned trainable skills without mutating the original core."""

    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.layers = nn.ModuleDict({
            str(i): NeuralSkillLayer(hidden_size) for i in range(num_layers)
        })
        self.specs: dict[str, NeuralSkillSpec] = {}

    def add_skill(self, spec: NeuralSkillSpec) -> None:
        if spec.skill_id in self.specs:
            raise ValueError(f"duplicate neural skill: {spec.skill_id}")
        if max(spec.layers) >= self.num_layers:
            raise ValueError("nonexistent transformer layer")
        for layer in spec.layers:
            self.layers[str(layer)].register_skill(spec.skill_id, spec.rank)
        self.specs[spec.skill_id] = spec

    def freeze_skill(self, skill_id: str) -> None:
        spec = self.specs[skill_id]
        for layer in spec.layers:
            self.layers[str(layer)].freeze_skill(skill_id)

    def select_new_skill_for_training(self, skill_id: str) -> list[nn.Parameter]:
        """Strictly train selected module: freeze every other neural skill."""
        if skill_id not in self.specs:
            raise KeyError(skill_id)
        for name, spec in self.specs.items():
            for layer in spec.layers:
                layer_bank = self.layers[str(layer)]
                for module in (layer_bank.blocks[name], layer_bank.routers[name]):
                    module.requires_grad_(name == skill_id)
        return [p for p in self.parameters() if p.requires_grad]

    def forward_at_layer(self, layer_index: int, hidden: torch.Tensor, *,
                         disabled: frozenset[str] = frozenset()) -> torch.Tensor:
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(layer_index)
        return self.layers[str(layer_index)](hidden, disabled=disabled)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def physical_bytes_in_current_dtype(self) -> int:
        """CPU estimate only. Actual acceptance must use physical files+manifest."""
        return sum(t.numel() * t.element_size() for t in self.state_dict().values())

    def canonical_manifest(self) -> bytes:
        specs = [
            {"skill_id": s.skill_id, "layers": list(s.layers),
             "rank": s.rank, "version": s.version}
            for s in sorted(self.specs.values(), key=lambda s: s.skill_id)
        ]
        return json.dumps(
            {"format": "S145-neural-skill-research-v1", "skills": specs},
            sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_manifest()).hexdigest()

    def provisional_bpp(self, *, parent_parameters: int,
                        parent_physical_bytes: int) -> float:
        """Count every new parameter and every physical byte, not just active.

        Conservative test uses the IMMUTABLE parent denominator as well as
        an expanded denominator. Neither substitutes for real packed files.
        """
        if parent_parameters <= 0 or parent_physical_bytes < 0:
            raise ValueError("invalid parent physical accounting")
        bytes_total = (parent_physical_bytes +
                       self.physical_bytes_in_current_dtype() +
                       len(self.canonical_manifest()))
        return 8 * bytes_total / (parent_parameters + self.parameter_count())

    def fixed_parent_reference_bpp(self, *, parent_parameters: int,
                                   parent_physical_bytes: int) -> float:
        if parent_parameters <= 0 or parent_physical_bytes < 0:
            raise ValueError("invalid parent physical accounting")
        return 8 * (parent_physical_bytes +
                    self.physical_bytes_in_current_dtype() +
                    len(self.canonical_manifest())) / parent_parameters
