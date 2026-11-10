"""S103: shared Green Tunnel at every transformer layer boundary.

RESEARCH ONLY. This does not modify the validated S101 inference ABI, does
not train or evaluate a candidate, and does not relax the <=0.700 bpp budget.

S101 already has independent S93 internal Green Tunnel residuals on layers
3 and 9 (12-layer, width-320 model). This module adds connections on the
remaining ten boundaries by REUSING the existing layer-3 adapter rather than
allocating ten new rank-8 networks. The only new learned values are ten
per-boundary gain scalars. Gains are zero at construction, so the S101
forward pass is byte-for-byte unaffected until they are trained.

For an additional boundary i:
    h_(i+1) = h_i + tanh(gain_i) * GT_S93_layer3(h_i)
The ordinary next decoder layer receives the result as its input.
The existing GT residuals at layers 3 and 9 remain unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import weakref
from typing import Iterator

import torch
from torch import nn


class SharedInterLayerGreenBridge(nn.Module):
    """Share an existing low-rank Green Tunnel across ALL layer boundaries.

    The existing IntegratedGreenTunnel must already be installed on the
    same model. We hold only a weak reference to its blocks, so its weights
    are not re-registered, re-counted, or copied in the sidecar.

    The bridge is route-scoped: it is OFF by default and may run only when
    the S95 policy selects the mutable S93 profile. S87/S91 stay untouched.
    Use one model/bridge per worker; set_active_profile is not thread-safe.
    """

    def __init__(
        self,
        existing_tunnel: nn.Module,
        *,
        total_layers: int = 12,
        shared_from_layer: int = 3,
        allowed_profile: str = "S93",
    ) -> None:
        super().__init__()
        current_layers = tuple(existing_tunnel.layers)
        if not current_layers or shared_from_layer not in current_layers:
            raise ValueError("Shared Green Tunnel donor must be an existing layer")
        if len(set(current_layers)) != len(current_layers):
            raise ValueError("Duplicate existing Green Tunnel layers")
        if total_layers < 2 or any(x < 0 or x >= total_layers for x in current_layers):
            raise ValueError("Invalid transformer depth or existing layer indices")
        if str(shared_from_layer) not in existing_tunnel.blocks:
            raise ValueError("Donor adapter is absent")
        self.total_layers = total_layers
        self.original_layers = current_layers
        self.shared_from_layer = shared_from_layer
        self.additional_layers = tuple(
            i for i in range(total_layers) if i not in current_layers
        )
        self.allowed_profile = allowed_profile
        # weakref prevents accidental duplicate weights and physical accounting.
        self._existing_tunnel = weakref.ref(existing_tunnel)
        # 10 FP32 values for S101: 40 raw parameter bytes + serialized metadata.
        # The ACTUAL all-inclusive physical archive must still be audited.
        self.layer_gains = nn.Parameter(
            torch.zeros(len(self.additional_layers), dtype=torch.float32)
        )
        self._handles: list = []
        self._active_profile: str | None = None

    def _donor(self) -> nn.Module:
        old = self._existing_tunnel()
        if old is None:
            raise RuntimeError("Underlying Green Tunnel was destroyed")
        return old.blocks[str(self.shared_from_layer)]

    @property
    def connected_layers(self) -> tuple[int, ...]:
        return tuple(range(self.total_layers))

    def set_active_profile(self, profile: str | None) -> None:
        """S93 only; None, S87, S91 and any other profile mean fully disabled."""
        self._active_profile = profile

    @contextmanager
    def profile(self, selected: str | None) -> Iterator[None]:
        """Fail-safe route scope, returning to the previous route on exit."""
        previous = self._active_profile
        self.set_active_profile(selected)
        try:
            yield
        finally:
            self.set_active_profile(previous)

    def install(self, model: nn.Module) -> "SharedInterLayerGreenBridge":
        if self._handles:
            raise RuntimeError("All-layer Green Tunnel already installed")
        old = self._existing_tunnel()
        if old is None or not getattr(old, "_handles", None):
            raise RuntimeError("Install existing layer-3/9 Green Tunnel first")
        decoder_layers = model.model.layers
        if len(decoder_layers) != self.total_layers:
            raise AssertionError("Model depth changed from S101")
        if set(self.original_layers) | set(self.additional_layers) != set(
            range(self.total_layers)
        ):
            raise AssertionError("Not every transformer layer is connected")

        for parameter_index, layer_index in enumerate(self.additional_layers):
            def boundary_hook(_module, _args, output, *, position=parameter_index):
                # Frozen alternate bank profiles NEVER receive S93-specific edits.
                if self._active_profile != self.allowed_profile:
                    return output

                if isinstance(output, tuple):
                    h = output[0]
                elif isinstance(output, torch.Tensor):
                    h = output
                else:
                    raise TypeError("Unexpected transformer decoder layer output")
                if not isinstance(h, torch.Tensor):
                    raise TypeError("Decoder hidden state must be a tensor")
                if self.layer_gains.device != h.device:
                    raise AssertionError("Bridge gains and decoder have different devices")

                gain = self.layer_gains[position]
                # Exact zero-cost path for frozen S101. During TRAIN we cannot
                # skip it: gradients must reach initially zero layer gains.
                if not torch.is_grad_enabled() and gain.detach().item() == 0.0:
                    return output

                adapted = self._donor()(h)
                if adapted.shape != h.shape:
                    raise AssertionError("Green Tunnel changed hidden-state shape")
                updated = h + adapted * torch.tanh(gain).to(dtype=h.dtype)
                if isinstance(output, tuple):
                    return (updated, *output[1:])
                return updated

            self._handles.append(
                decoder_layers[layer_index].register_forward_hook(boundary_hook)
            )
        return self

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active_profile = None

    def trainable_added_parameter_count(self) -> int:
        return self.layer_gains.numel()

    def get_bridge_state(self) -> dict[str, torch.Tensor]:
        """Sidecar payload; do NOT include repeated layer-3/9 donor weights."""
        return {
            "all_layer_bridge.layer_gains": self.layer_gains.detach().cpu().clone()
        }

    def load_bridge_state(self, payload: dict[str, torch.Tensor]) -> None:
        key = "all_layer_bridge.layer_gains"
        if set(payload) != {key}:
            raise ValueError("Unexpected Green Tunnel bridge sidecar keys")
        values = payload[key]
        if values.shape != self.layer_gains.shape:
            raise ValueError("Green Tunnel bridge depth mismatch")
        if not torch.isfinite(values).all().item():
            raise ValueError("Non-finite bridge gains")
        with torch.no_grad():
            self.layer_gains.copy_(
                values.to(device=self.layer_gains.device, dtype=self.layer_gains.dtype)
            )
