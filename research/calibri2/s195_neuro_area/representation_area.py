"""S195-NRA: learnable neural representation area inside the 12-layer Calibri 2.0 core.

Research-only. This is an information-representation bottleneck, not an answer
table, gold-label router, second language model, or replacement for Green Tunnel.
Each transformer layer sends its contextual hidden state into ONE shared
learnable semantic-slot area; the area sends a residual representation back to
the same layer. It has 21,900 distinct trainable weights, counted and serialized
independently of the existing 43,261,760-parameter architecture.

The decoder/readout starts EXACTLY at zero, so enabling the untrained area is
bit-for-bit identical to the parent model. New competence is never claimed until
real TRAIN gradients, physical reload, native CPU J7, zero forgetting, speed,
and an all-inclusive <=0.7 bpp gate independently succeed.

Per-token representation makes cached autoregressive operation causal: no
future tokens or answer labels are consulted; the existing transformer hidden
states already hold prompt context. Research only; no QA/TEST or LIVE.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 320
DEPTH = 12
LATENT = 32
SLOTS = 16
PARENT_PARAMS = 43_261_760
EXPECTED_AREA_PARAMS = 21_900
PREFIX = "neuro_representation_area."


class NeuroRepresentationArea(nn.Module):
    """Shared trainable concept-slot bottleneck with per-layer readout control."""

    def __init__(self, width: int = WIDTH, depth: int = DEPTH,
                 latent: int = LATENT, slots: int = SLOTS):
        super().__init__()
        if (width, depth, latent, slots) != (WIDTH, DEPTH, LATENT, SLOTS):
            raise ValueError("S195 exact original 320x12 neural ABI required")
        self.encoder = nn.Linear(width, latent, bias=False)
        self.slot_keys = nn.Parameter(torch.empty(slots, latent))
        self.slot_values = nn.Parameter(torch.empty(slots, latent))
        self.decoder = nn.Linear(latent, width, bias=False)
        self.layer_context = nn.Parameter(torch.zeros(depth, latent))
        self.layer_gate = nn.Parameter(torch.zeros(depth))
        nn.init.normal_(self.encoder.weight, std=.02)
        nn.init.normal_(self.slot_keys, std=.15)
        nn.init.normal_(self.slot_values, std=.15)
        nn.init.zeros_(self.decoder.weight)
        if self.parameter_count() != EXPECTED_AREA_PARAMS:
            raise AssertionError("S195 counted neural region capacity changed")
        self._handles = []
        self._active = False
        self._visited = []
        self.last_coverage = ()
        self.forward_calls = 0

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def concepts(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        if not isinstance(h, torch.Tensor) or h.ndim != 3 or h.shape[-1] != WIDTH:
            raise AssertionError("S195 expects [batch, tokens, 320] contextual states")
        if not 0 <= layer < DEPTH:
            raise IndexError("S195 wrong decoder layer")
        # The transformer has already performed causal contextual attention.
        # This region reorganizes those contextual activations into concepts.
        x = F.layer_norm(h, (WIDTH,))
        query = F.gelu(self.encoder(x))
        logits = 4.0 * (F.normalize(query, dim=-1) @
                        F.normalize(self.slot_keys, dim=-1).T)
        slot_read = logits.softmax(dim=-1) @ self.slot_values
        return query + slot_read + self.layer_context[layer]

    def inject(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        if h.device != self.encoder.weight.device:
            raise AssertionError("S195 area weights and core hidden states differ in device")
        represented = self.concepts(h, layer)
        # Zero readout makes the ENTIRE initial 43M decoder output unchanged.
        return h + self.decoder(represented) * torch.sigmoid(self.layer_gate[layer])

    def install(self, model: nn.Module) -> "NeuroRepresentationArea":
        if self._handles:
            raise RuntimeError("S195 area already installed")
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise AssertionError("S195 requires actual decoder.model.layers")
        if len(model.model.layers) != DEPTH:
            raise AssertionError("S195 must integrate across all 12 decoder layers")

        def start(_module, _args):
            if self._active:
                self._visited = []

        def finish(_module, _args, output):
            if self._active:
                if self._visited != list(range(DEPTH)):
                    raise AssertionError("S195 missing or reordered decoder integration")
                self.last_coverage = tuple(self._visited)
                self.forward_calls += 1
            return output

        self._handles.append(model.register_forward_pre_hook(start))
        for layer_index, decoder_layer in enumerate(model.model.layers):
            def hook(_layer, _args, output, *, i=layer_index):
                if not self._active:
                    return output
                if len(self._visited) != i:
                    raise AssertionError("S195 decoder layer order mismatched")
                self._visited.append(i)
                if isinstance(output, tuple):
                    return (self.inject(output[0], i), *output[1:])
                if isinstance(output, torch.Tensor):
                    return self.inject(output, i)
                raise TypeError("S195 unrecognized transformer decoder output")
            self._handles.append(decoder_layer.register_forward_hook(hook))
        self._handles.append(model.register_forward_hook(finish, always_call=True))
        return self

    @contextmanager
    def activated(self) -> Iterator["NeuroRepresentationArea"]:
        if not self._handles or self._active:
            raise RuntimeError("S195 area must be installed and not already activated")
        self._active = True
        try:
            yield self
        finally:
            self._active = False
            self._visited = []

    def uninstall(self) -> None:
        if self._active:
            raise RuntimeError("Cannot uninstall an active S195 area")
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._visited = []

    def fp16_state(self) -> dict[str, torch.Tensor]:
        """Physical payload: every trainable value, including zeros, with no omissions."""
        output = {}
        for name, tensor in self.state_dict().items():
            if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all():
                raise AssertionError("S195 non-finite/neither-float trainable tensor")
            output[PREFIX + name] = tensor.detach().cpu().to(torch.float16).contiguous()
        if sum(v.numel() for v in output.values()) != self.parameter_count():
            raise AssertionError("S195 unaccounted trainable region weights")
        return output

    def save_fp16(self, path: str | Path) -> Path:
        from safetensors.torch import save_file
        path = Path(path)
        if path.exists():
            raise FileExistsError("S195 physical area must not overwrite a checkpoint")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.fp16_state(), str(path))
        return path

    def load_fp16(self, path: str | Path) -> None:
        from safetensors.torch import load_file
        physical = load_file(str(path), device="cpu")
        expected = self.fp16_state()
        if set(physical) != set(expected):
            raise ValueError("S195 missing or extra learnable region tensor")
        for name in expected:
            found = physical[name]
            if (found.dtype != torch.float16 or found.shape != expected[name].shape
                    or not bool(torch.isfinite(found).all())):
                raise ValueError("S195 bad physical FP16 region tensor")
        self.load_state_dict(
            {k.removeprefix(PREFIX): v.float() for k, v in physical.items()},
            strict=True)

    @staticmethod
    def all_in_bpp(*, parent_bpp: float, physical_area_bytes: int,
                   additional_source_bytes: int = 0,
                   parent_params: int = PARENT_PARAMS) -> float:
        """No free zeros: charge every actual physical area/source byte."""
        import math
        if (not math.isfinite(parent_bpp) or parent_bpp < 0
                or type(physical_area_bytes) is not int or physical_area_bytes < 0
                or type(additional_source_bytes) is not int or additional_source_bytes < 0
                or parent_params != PARENT_PARAMS):
            raise ValueError("S195 invalid physical-accounting provenance")
        return ((parent_bpp * parent_params +
                 8 * (physical_area_bytes + additional_source_bytes)) /
                (parent_params + EXPECTED_AREA_PARAMS))


def trainable_region_parameters(area: NeuroRepresentationArea):
    """Freeze the old 43M/GT; ONLY this region may receive new optimizer updates."""
    return [p for p in area.parameters() if p.requires_grad]
