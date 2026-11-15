"""S1: composability contract with frozen Calibri Green Tunnel hooks.

Reuses the original GT module, its weights and optional S103 shared bridge.
Install order: existing GT -> optional all-layer bridge -> NRA3 shadow.
No duplicate GT parameters, no GT optimizer writes, no Calibri output edits.
This is a read-only compatibility mechanism, NOT active full integration.
"""
from __future__ import annotations

import hashlib
import weakref
import torch
from torch import nn

from research.nra3_s0.integration import NRA3MidLayerPort
from research.nra3_s0.neural_area import NRA3NeuralArea


def module_fingerprint(module: nn.Module) -> str:
    """Stable hash of all physically loaded tensor values, not stored weights."""
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError("GT non-tensor state")
        if not torch.isfinite(value.detach().float()).all():
            raise ValueError("nonfinite GT state")
        t = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(bytes(str(tuple(t.shape)), "ascii"))
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class NRA3WithExistingGreenTunnel:
    """Keep GT exact, install NRA3 after its layer-7 residual in hook order."""
    def __init__(self, area: NRA3NeuralArea, *, green_tunnel: nn.Module,
                 shared_bridge: nn.Module | None = None):
        self.area = area
        self._green_tunnel = weakref.ref(green_tunnel)
        self._bridge = weakref.ref(shared_bridge) if shared_bridge is not None else None
        self.port = NRA3MidLayerPort(area)
        self.gt_sha_before = module_fingerprint(green_tunnel)
        self.bridge_sha_before = (module_fingerprint(shared_bridge)
                                  if shared_bridge is not None else None)
        self.bound = False

    def install(self, decoder: nn.Module):
        if self.bound:
            raise RuntimeError("NRA3 GT compatibility duplicate binding")
        gt = self._green_tunnel()
        if (gt is None or not getattr(gt, "_handles", None) or
                not hasattr(gt, "blocks") or
                not hasattr(gt, "layers") or len(gt.layers)<1):
            raise ValueError("Install an existing Green Tunnel before NRA3")
        if any(k not in range(12) for k in gt.layers):
            raise ValueError("Invalid Green Tunnel layer index")
        if self._bridge is not None:
            bridge = self._bridge()
            if (bridge is None or not getattr(bridge, "_handles", None) or
                    not hasattr(bridge, "connected_layers") or
                    tuple(bridge.connected_layers)!=tuple(range(12))):
                raise ValueError("S103 shared bridge must be installed first")
            if module_fingerprint(bridge) != self.bridge_sha_before:
                raise ValueError("Existing S103 bridge unexpectedly modified")
        if module_fingerprint(gt)!=self.gt_sha_before:
            raise ValueError("Existing GT unexpectedly modified before readout")
        self.port.install(decoder)
        self.bound=True
        return self

    def integrity(self):
        gt=self._green_tunnel()
        if gt is None or module_fingerprint(gt)!=self.gt_sha_before:
            raise RuntimeError("GT weights changed during NRA3 shadow observation")
        if self._bridge is not None:
            bridge=self._bridge()
            if bridge is None or module_fingerprint(bridge)!=self.bridge_sha_before:
                raise RuntimeError("All-layer GT bridge weights changed")
        if self.port.output_gate!=0.0 or self.port.candidate_accepted:
            raise RuntimeError("Active NRA3 connection unauthorized")
        return {
            "existing_green_tunnel_sha256":self.gt_sha_before,
            "existing_green_tunnel_unchanged":True,
            "shared_all_layer_gt_unchanged":True if self._bridge is not None else None,
            "nra_output_gate":0,
            "nra_candidate_accepted":False,
            "green_tunnel_weights_not_duplicated_in_nra3":True,
        }

    def uninstall(self):
        self.port.uninstall()
        self.bound=False
        self.integrity()
