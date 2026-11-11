"""S145 isolated transformer-layer hook bridge (not loaded on real S142 yet).

Uses the native CALIBRI model.model.layers ABI demonstrated by S87/S126/S129.
Never overwrites existing hooks, base weights, cache tuples or model classes.
"""
from __future__ import annotations

import torch
from torch import nn

from .neural_skill_blocks import LayerwiseNeuralSkillBank


class NeuralSkillHookBridge:
    def __init__(self, bank: LayerwiseNeuralSkillBank):
        self.bank = bank
        self._handles = []

    def install(self, model: nn.Module) -> "NeuralSkillHookBridge":
        if self._handles:
            raise RuntimeError("S145 skill hook already installed")
        decoder = getattr(getattr(model, "model", None), "layers", None)
        if decoder is None or len(decoder) != self.bank.num_layers:
            raise AssertionError("S145 model decoder layer count mismatch")
        selected = sorted({i for s in self.bank.specs.values() for i in s.layers})
        for index in selected:
            def hook(_module, _inputs, output, *, layer_index=index):
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                if (not isinstance(h, torch.Tensor) or h.ndim != 3 or
                        h.shape[-1] != self.bank.layers[str(layer_index)].hidden_size):
                    raise AssertionError("S145 hidden-state tensor ABI mismatch")
                params = list(self.bank.layers[str(layer_index)].parameters())
                if params and any(p.device != h.device for p in params):
                    raise AssertionError("S145 neural skill and core device mismatch")
                updated = self.bank.forward_at_layer(layer_index, h)
                return (updated, *output[1:]) if is_tuple else updated

            self._handles.append(decoder[index].register_forward_hook(hook))
        return self

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
