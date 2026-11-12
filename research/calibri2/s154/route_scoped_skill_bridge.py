"""S154: prompt-route-scoped Neural Skill hooks on immutable CALIBRI S142.

A frozen S95 router chooses S93 versus other original profiles BEFORE decode.
Only S93-scoped prompt context enables this newly trained S154 language skill.
The original S87/S91/S82 routes remain mathematically untouched: no prompt-ID,
answer-ID, gold token, or historical failure identity is consulted.

The frozen S111 bridge's _features is non-None for the entire S93 prompt scope,
even when _active_profile is cleared during cached autoregressive decode.
"""

from __future__ import annotations

import torch
from torch import nn

from calibri2.s145.neural_skill_blocks import LayerwiseNeuralSkillBank


class S93ScopedNeuralSkillHookBridge:
    def __init__(self, bank: LayerwiseNeuralSkillBank, frozen_route_bridge):
        self.bank = bank
        self.frozen_route_bridge = frozen_route_bridge
        self._handles = []

    def install(self, model: nn.Module):
        if self._handles:
            raise RuntimeError("S154 duplicate neural-skill hooks")
        decoder = getattr(getattr(model, "model", None), "layers", None)
        if decoder is None or len(decoder) != self.bank.num_layers:
            raise AssertionError("S154 unexpected original transformer layer ABI")
        if not hasattr(self.frozen_route_bridge, "_features"):
            raise AssertionError("S154 requires frozen prompt-route scope")
        selected = sorted({i for s in self.bank.specs.values() for i in s.layers})
        for index in selected:
            def hook(_module, _inputs, output, *, layer_index=index):
                # Route isolation: original non-S93 families use the original
                # frozen path, even after the S154 new skill was trained.
                # _active_profile is NOT enough: it is None on S93 decode steps.
                if self.frozen_route_bridge._features is None:
                    return output
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                if (not isinstance(h, torch.Tensor) or h.ndim != 3 or
                        h.shape[-1] != self.bank.layers[str(layer_index)].hidden_size):
                    raise AssertionError("S154 invalid hidden-state hook ABI")
                params = self.bank.layers[str(layer_index)].parameters()
                if any(p.device != h.device for p in params):
                    raise AssertionError("S154 neural skill and core device mismatch")
                updated = self.bank.forward_at_layer(layer_index, h)
                return (updated, *output[1:]) if is_tuple else updated
            self._handles.append(decoder[index].register_forward_hook(hook))
        return self

    def uninstall(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
