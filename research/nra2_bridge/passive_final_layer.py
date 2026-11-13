"""NRA2 stage-2: exactly one read-only Calibri decoder read port (layer 12).

The independently trained relational core has a graph ABI. Calibri supplies
320-dimensional hidden states, NOT graph edges. We MUST NOT pretend those
inputs are interchangeable. This prototype loads the real standalone FP16
weights, but only taps the final decoder state. No output injection is enabled
until a *separately trained* hidden-state-to-relations adapter exists.

Research only. Never alter pretrained core, Green Tunnel, S195 NRA, J7,
selector, token predictions or any production component.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import torch
from torch import nn

from research.nra2_standalone.model import RelationalNRA

WIDTH=320
DEPTH=12


class NRA2PassiveFinalLayer:
    def __init__(self, checkpoint: str | Path, expected_sha256: str):
        checkpoint=Path(checkpoint)
        if not checkpoint.is_file() or len(expected_sha256)!=64:
            raise FileNotFoundError("Pinned physical independent NRA2 FP16 missing")
        actual_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual_sha256!=expected_sha256:
            raise AssertionError("Standalone NRA2 physical SHA drift")
        source=torch.load(checkpoint,map_location="cpu",weights_only=True)
        self.area=RelationalNRA(width=16,hops=5).eval()
        if set(source)!=set(self.area.state_dict()):
            raise AssertionError("Missing or unexpected learned NRA2 weight")
        if any(t.dtype!=torch.float16 or not torch.isfinite(t).all()
               for t in source.values()):
            raise AssertionError("Standalone NRA2 must consist of finite FP16 weights")
        self.area.load_state_dict(source, strict=True)
        self.area.requires_grad_(False)
        self.physical_sha256=actual_sha256
        self.active=False
        self.forward_calls=0
        self.last_shape=None
        self.read_port_layer=11
        self.output_gate=0.0
        self.read_count=0
        self._handle=None

    def install(self, model: nn.Module):
        if self._handle is not None:
            raise RuntimeError("Duplicate NRA2 bridge installation")
        if (not hasattr(model,"model") or not hasattr(model.model,"layers")
                or len(model.model.layers)!=DEPTH):
            raise AssertionError("NRA2 expected original 12-layer decoder ABI")
        self.active=True
        def read_only(_layer,_args,output):
            if not self.active:
                return None
            if self.output_gate != 0.0:
                raise AssertionError("NRA2 learned hidden-state adapter absent: nonzero output prohibited")
            h=output[0] if isinstance(output,tuple) else output
            if not isinstance(h,torch.Tensor) or h.ndim!=3 or h.shape[-1]!=WIDTH:
                raise AssertionError("NRA2 requires actual contextual [B,T,320]")
            if not torch.isfinite(h).all():
                raise AssertionError("Nonfinite final Calibri features")
            self.forward_calls+=1
            self.last_shape=tuple(h.shape)
            self.read_count+=int(h.shape[0]*h.shape[1])
            # CRITICAL: returning None from a forward hook preserves the
            # original output object bit-for-bit. No graph inference, no gold.
            return None
        self._handle=model.model.layers[-1].register_forward_hook(read_only)
        return self

    def uninstall(self):
        self.active=False
        if self._handle is not None:
            self._handle.remove()
            self._handle=None

    @property
    def report(self):
        return {"stage":"NRA2_CALIBRI_FINAL_LAYER_READ_ONLY",
                "read_layer_index":self.read_port_layer,
                "connected_decoder_layers":1,
                "real_cpu_forward_calls":self.forward_calls,
                "observed_token_states":self.read_count,
                "last_hidden_shape":self.last_shape,
                "standalone_fp16_sha256":self.physical_sha256,
                "standalone_trainable_parameters":self.area.parameter_count(),
                "standalone_weights_frozen":all(not p.requires_grad for p in self.area.parameters()),
                "feature_to_graph_adapter_trained":False,
                "relation_inference_enabled":False,
                "output_gate":self.output_gate,
                "calibri_weights_modified":False,
                "nra2_production_accepted":False}
