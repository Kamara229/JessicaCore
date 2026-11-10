"""S108: research-only 12-layer shared Green Tunnel with TWO frozen donors.

The S103 / S104 single-donor bridge shares S93 layer-3 Green Tunnel over
all ten missing layer boundaries. S105 found 46/96 new-vs-old gradient
conflicts; no causal attribution to donor placement has yet been proven.

S108 tests a higher-capacity but still low-byte alternate hypothesis:
reuse BOTH previously paid S93 3/9 Green Tunnel blocks across the ten
missing layer boundaries, with two zero-initialized gains per boundary.
Do not duplicate or update the donor weights or compressed model.

At an additional boundary:
   new_h = h + tanh(gain[0,i]) * original_GT3(h)
             + tanh(gain[1,i]) * original_GT9(h)

Original GT3 and GT9 layers are unmodified. S95 alt profiles S87/S91
are disabled; the bridge is enabled only on S93. At exactly zero gains
this module is an exact inference identity, while 20 gain gradients exist
under full backprop. All-in physical bytes and actual CPU cached runtime
must be measured before any training or acceptance.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
import weakref

import torch
from torch import nn


class DualDonorInterlayerGreenBridge(nn.Module):
    """Share existing S93 Green Tunnel blocks across twelve transformer layers.

    Donor tensors remain registered only on the original tunnel, not in
    this bridge. The bridge contains exactly 20 new FP32 gain scalars
    (for 12 total decoder layers with two dedicated donor layers).

    Not thread-safe: route-scoped controls are for a single research worker.
    """

    SIDECAR_KEY = "dual_bridge.layer_gains"

    def __init__(
        self,
        existing_tunnel: nn.Module,
        *,
        total_layers: int = 12,
        donor_layers: tuple[int, int] = (3, 9),
        enabled_profile: str = "S93",
    ):
        super().__init__()
        donor_layers = tuple(donor_layers)
        originals = tuple(existing_tunnel.layers)
        if (len(donor_layers) != 2 or len(set(donor_layers)) != 2
                or not set(donor_layers).issubset(set(originals))):
            raise ValueError("Require two different originally installed donors")
        if (total_layers < 3 or not all(
                isinstance(i, int) and 0 <= i < total_layers for i in originals
        ) or len(set(originals)) != len(originals)):
            raise ValueError("Invalid transformer depth or existing donor indices")
        if any(str(i) not in existing_tunnel.blocks for i in donor_layers):
            raise ValueError("Both pre-existing trained Green Tunnel donors required")
        self.total_layers = total_layers
        self.original_layers = originals
        self.donor_layers = donor_layers
        self.additional_layers = tuple(
            i for i in range(total_layers) if i not in originals
        )
        self.enabled_profile = enabled_profile
        self._old_tunnel = weakref.ref(existing_tunnel)
        self.layer_gains = nn.Parameter(torch.zeros(
            2, len(self.additional_layers), dtype=torch.float32
        ))
        self._handles = []
        self._active_profile: str | None = None

    def _donors(self):
        existing = self._old_tunnel()
        if existing is None:
            raise RuntimeError("Original Green Tunnel has been destroyed")
        return tuple(existing.blocks[str(i)] for i in self.donor_layers)

    def install(self, model: nn.Module) -> "DualDonorInterlayerGreenBridge":
        if self._handles:
            raise RuntimeError("Dual Green Tunnel bridge already installed")
        old = self._old_tunnel()
        if old is None or not getattr(old, "_handles", None):
            raise RuntimeError("Install original 3/9 Green Tunnel first")
        decoder = model.model.layers
        if len(decoder) != self.total_layers:
            raise AssertionError("S101 decoder depth mismatch")
        if (set(self.additional_layers) | set(self.original_layers)
                != set(range(self.total_layers))):
            raise AssertionError("All twelve transformer layers must be connected")

        for position, layer_index in enumerate(self.additional_layers):
            def hook(_layer, _args, output, *, at=position):
                if self._active_profile != self.enabled_profile:
                    return output
                if isinstance(output, tuple):
                    hidden = output[0]
                elif isinstance(output, torch.Tensor):
                    hidden = output
                else:
                    raise TypeError("Unsupported decoder layer return type")
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("Unexpected non-tensor decoder hidden state")
                if hidden.device != self.layer_gains.device:
                    raise AssertionError("Bridge and decoder must share a device")

                gains = self.layer_gains[:, at]
                training_gradient = torch.is_grad_enabled()
                # Keep exact S101 at zero gains, including no unnecessary
                # calls to the shared blocks when inference is disabled.
                if (not training_gradient and
                        torch.count_nonzero(gains.detach()).item() == 0):
                    return output
                changed = hidden
                for offset, donor in enumerate(self._donors()):
                    scalar = gains[offset]
                    if (not training_gradient and scalar.detach().item() == 0):
                        continue
                    correction = donor(hidden)
                    if correction.shape != hidden.shape:
                        raise AssertionError("Donor residual changed hidden shape")
                    changed = changed + torch.tanh(scalar).to(hidden.dtype) * correction
                if isinstance(output, tuple):
                    return (changed, *output[1:])
                return changed

            self._handles.append(decoder[layer_index].register_forward_hook(hook))
        return self

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active_profile = None

    @property
    def connected_layers(self) -> tuple[int, ...]:
        return tuple(range(self.total_layers))

    def set_active_profile(self, route: str | None) -> None:
        self._active_profile = route

    @contextmanager
    def profile(self, route: str | None) -> Iterator[None]:
        prev = self._active_profile
        self.set_active_profile(route)
        try:
            yield
        finally:
            self._active_profile = prev

    def trainable_new_parameters(self) -> int:
        return self.layer_gains.numel()

    def physical_sidecar_payload(self) -> dict[str, torch.Tensor]:
        return {self.SIDECAR_KEY: self.layer_gains.detach().cpu().clone()}

    def load_physical_sidecar_payload(self, state: dict[str, torch.Tensor]) -> None:
        if set(state) != {self.SIDECAR_KEY}:
            raise ValueError("Unexpected dual Green Tunnel sidecar keys")
        candidate = state[self.SIDECAR_KEY]
        if (candidate.shape != self.layer_gains.shape
                or candidate.dtype != self.layer_gains.dtype
                or not bool(torch.isfinite(candidate).all())):
            raise ValueError("Invalid layer gains in sidecar")
        with torch.no_grad():
            self.layer_gains.copy_(candidate.to(self.layer_gains.device))
