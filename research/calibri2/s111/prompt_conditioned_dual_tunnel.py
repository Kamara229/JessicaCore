"""CALIBRI S111: prompt-conditioned dual-donor Green Tunnel across all 12 layers.

MOTIVATION. S110's original, unmodified physical S101 produced 103 wrong
TRAIN full answers: 69 first-token, 18 second-token, 16 EOS. S105 found
46/96 opposing wrong-vs-protected gradients. Global gains S104/S109 could
not repair a correct answer. S111 adds **prompt-dependent** donor mixing:
different questions may route information through the 3/9 donor weights
differently, reducing the damage from globally shared gain coefficients.

Frozen S95 prompt_only_vector(question) creates 512 signed hash features
from user prompt tokens and character trigrams (NO target labels). Collapse
them deterministically to 16 numbers (no trained weights), append one
constant bias and compute a scalar gate for each 2x10 donor/layer position.
The only newly paid model tensor is 2x10x17 FP16 coefficients = 680 raw
bytes plus actual safetensors header and deployment config. Donor weights
and compressed core are never copied or updated.

    e(x) = normalize(reshape(S95_prompt_features, 16, 32).sum(axis=1))
    a_{d,l}(x) = tanh(dot(learned_coefficients[d,l,:], [1, e(x)]))
    h_{l+1} = h_l + a_{3,l}(x)*GT3(h_l) + a_{9,l}(x)*GT9(h_l)

Original paid GT layers 3/9 remain exactly unchanged. Activations and KV
cache are conserved; S95 frozen route S87/S91 bypasses this module.
At all-zero coefficients the exact original S101 outputs are identical.
Physical bpp, speed and actual 456 autoregressive TRAIN outputs must be
measured before learning. This research hook is single-worker, NOT threadsafe.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
import weakref
import torch
from torch import nn


class PromptConditionedDualTunnel(nn.Module):
    """20 prompt-specific dual-donor gates, each with 17 coefficients."""

    KEY="conditional_dual_bridge.coefficients"
    HASHED_DIM=512
    CONTEXT_DIM=16
    DEPLOY_DTYPE=torch.float16

    def __init__(self, existing_tunnel: nn.Module, *, total_layers:int=12,
                 donor_layers:tuple[int,int]=(3,9),
                 allowed_profile:str="S93"):
        super().__init__()
        old_layers=tuple(existing_tunnel.layers)
        if (len(donor_layers)!=2 or len(set(donor_layers))!=2
                or not set(donor_layers).issubset(old_layers)):
            raise ValueError("Two distinct existing Green Tunnel donors needed")
        if (total_layers<3 or len(set(old_layers))!=len(old_layers)
                or any(i<0 or i>=total_layers for i in old_layers)
                or any(str(i) not in existing_tunnel.blocks for i in donor_layers)):
            raise ValueError("Original donor model depth/blocks changed")
        self.donor_layers=tuple(donor_layers)
        self.original_layers=old_layers
        self.additional_layers=tuple(i for i in range(total_layers)
                                     if i not in old_layers)
        self.total_layers=total_layers
        self.allowed_profile=allowed_profile
        # Keep old trained block weights in their only owner, not this module.
        self._existing=weakref.ref(existing_tunnel)
        self.coefficients=nn.Parameter(torch.zeros(
            (2,len(self.additional_layers),1+self.CONTEXT_DIM),
            dtype=torch.float32))
        self._handles=[]
        self._active_profile=None
        self._features=None

    @classmethod
    def prepare_prompt(cls, vector:torch.Tensor) -> torch.Tensor:
        """Compress original S95 512D feature vector; no learned projection.

        This module MUST receive the exact S95 prompt-only vector for the
        SAME question that is being generated. No answer- or identity-input.
        """
        if (not isinstance(vector,torch.Tensor) or
                vector.ndim!=1 or vector.numel()!=cls.HASHED_DIM or
                not bool(torch.isfinite(vector).all())):
            raise ValueError("S111 needs one finite frozen S95 prompt feature vector")
        group=vector.float().reshape(cls.CONTEXT_DIM,
                                      cls.HASHED_DIM//cls.CONTEXT_DIM).sum(1)
        group=group/group.square().sum().sqrt().clamp_min(1.e-8)
        return torch.cat([torch.ones_like(group[:1]), group])

    @contextmanager
    def prompt_scope(self, profile:str|None,
                     frozen_s95_features:torch.Tensor|None) -> Iterator[None]:
        old_route,old_features=self._active_profile,self._features
        if profile==self.allowed_profile:
            if frozen_s95_features is None:
                raise ValueError("S93 inference requires prompt context")
            prepared=self.prepare_prompt(frozen_s95_features)
            self._features=prepared.to(self.coefficients.device)
        else:
            self._features=None
        self._active_profile=profile
        try:
            yield
        finally:
            self._active_profile,self._features=old_route,old_features

    @property
    def connected_layers(self):
        return tuple(range(self.total_layers))

    def install(self, model:nn.Module) -> "PromptConditionedDualTunnel":
        if self._handles:
            raise RuntimeError("S111 conditional bridge already installed")
        old=self._existing()
        if old is None or not getattr(old,"_handles",None):
            raise RuntimeError("Original Green Tunnel must be installed first")
        decoder=model.model.layers
        if len(decoder)!=self.total_layers or (
            set(self.additional_layers)|set(self.original_layers)
        )!=set(range(self.total_layers)):
            raise AssertionError("S101 12-layer decoder connectivity violated")

        for position,layer_number in enumerate(self.additional_layers):
            def interlayer_hook(_layer,_inputs,output,*,index=position):
                if self._active_profile!=self.allowed_profile:
                    return output
                if self._features is None:
                    raise AssertionError("Missing active frozen S95 prompt features")
                if isinstance(output,tuple):
                    h=output[0]
                elif isinstance(output,torch.Tensor):
                    h=output
                else:
                    raise TypeError("Unsupported decoder output/hidden tuple")
                if h.device!=self.coefficients.device:
                    raise AssertionError("Layer and bridge on different devices")
                local=self.coefficients[:,index,:] @ self._features
                if (not torch.is_grad_enabled()
                        and not bool(torch.count_nonzero(local.detach()).item())):
                    return output
                gates=local.tanh()
                updated=h
                old_tunnel=self._existing()
                if old_tunnel is None:
                    raise RuntimeError("Frozen Green Tunnel reference lost")
                for donor_i,layer_i in enumerate(self.donor_layers):
                    scalar=gates[donor_i]
                    if not torch.is_grad_enabled() and scalar.detach().item()==0:
                        continue
                    residual=old_tunnel.blocks[str(layer_i)](h)
                    if residual.shape!=h.shape:
                        raise AssertionError("Donor hidden representation mismatch")
                    updated=updated+residual*scalar.to(h.dtype)
                return (updated,*output[1:]) if isinstance(output,tuple) else updated
            self._handles.append(
                decoder[layer_number].register_forward_hook(interlayer_hook))
        return self

    def uninstall(self):
        for handle in self._handles: handle.remove()
        self._handles.clear()
        self._active_profile=None
        self._features=None

    def export_physical_gain(self):
        # The actual serialized bytes must still be measured independently.
        return {self.KEY:self.coefficients.detach().cpu().to(
            self.DEPLOY_DTYPE).contiguous()}

    def load_physical_gain(self,tensors):
        if set(tensors)!={self.KEY}:
            raise ValueError("Unexpected S111 physical sidecar fields")
        value=tensors[self.KEY]
        if (value.shape!=self.coefficients.shape or
                value.dtype!=self.DEPLOY_DTYPE or
                not bool(torch.isfinite(value).all())):
            raise ValueError("Incorrect S111 quantized gate coefficient payload")
        with torch.no_grad():
            self.coefficients.copy_(value.to(self.coefficients.device,dtype=torch.float32))

    def raw_weight_bytes(self):
        return self.coefficients.numel()*torch.empty((),dtype=self.DEPLOY_DTYPE).element_size()
