"""NRA2 3.0 research-only Calibri mid-layer read port.

Reads contextual states after decoder block 8 of 12, performs unified NRA3
computation and returns EXACT original hidden object. Output gate hard zero.
No optimizer, no model loader patch, no LIVE or held-out EVAL use.
Caching semantic state across incremental decoder calls is NOT implemented;
never activate an unverified read/write path.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn
from .neural_area import NRA3NeuralArea


class NRA3MidLayerPort:
    TARGET_LAYER = 7
    DECODER_LAYERS = 12

    def __init__(self, area: NRA3NeuralArea):
        if not isinstance(area, NRA3NeuralArea):
            raise TypeError("NRA3 requires an integrated neural area")
        self.area = area
        self._handle = None
        self._active = False
        self.calls = 0
        self.last_shape = None
        self.last_features = None
        self.output_gate = 0.0
        self.candidate_accepted = False

    def set_output_gate(self, value: float):
        if value != 0.0:
            raise RuntimeError("NRA3 no active write path before J7, CPU and bpp gates")
        self.output_gate = 0.0

    def install(self, decoder: nn.Module):
        if self._handle is not None:
            raise RuntimeError("NRA3 duplicate hook")
        layers = getattr(getattr(decoder, "model", None), "layers", None)
        if layers is None or len(layers) != self.DECODER_LAYERS:
            raise ValueError("NRA3 requires actual 12-layer Calibri decoder ABI")

        def hook(_layer, _args, output):
            if not self._active:
                return output
            if self.output_gate != 0.0 or self.candidate_accepted:
                raise RuntimeError("NRA3 research-only output contract violated")
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor) or h.ndim != 3 or h.shape[-1] != 320:
                raise ValueError("NRA3 invalid layer8 hidden ABI")
            # No attention-mask argument available in this hook: restrict to
            # unpadded, single-item read-only diagnostics until mask-aware ABI.
            if h.shape[0] != 1:
                raise ValueError("NRA3 shadow hook requires unpadded batch=1")
            with torch.no_grad():
                result = self.area(h)
            self.calls += 1
            self.last_shape = tuple(h.shape)
            self.last_features = {
                "upos": tuple(result.upos_logits.shape),
                "roles": tuple(result.semantic_role_logits.shape),
                "relation": tuple(result.relations.shape),
            }
            return output  # Same object and numerical identity (gate=0)

        self._handle = layers[self.TARGET_LAYER].register_forward_hook(hook)
        return self

    @contextmanager
    def activated(self):
        if self._handle is None or self._active:
            raise RuntimeError("NRA3 uninstalled or already active")
        self._active = True
        try:
            yield self
        finally:
            self._active = False

    def uninstall(self):
        if self._active:
            raise RuntimeError("NRA3 cannot uninstall during active forward")
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
