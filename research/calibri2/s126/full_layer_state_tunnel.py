"""S126: full-depth recurrent Green Tunnel, RESEARCH ONLY.

Complements (does not replace) immutable S87 GT3/GT9 and S111's
prompt-conditioned donor hooks. One SHARED rank-8 recurrent state passes
through all 12 decoder layers in their actual order, not merely two
donor blocks broadcast to other layers. A newly learned scalar gate at
each layer starts at ZERO, giving strictly identical baseline outputs.

The state is ephemeral PER forward call (KV prefill or continuation);
no cross-user, cross-request or cross-token-step state is retained.
No pretrained weights, S118 QA, production files or S125 weights are
loaded or modified by this module. It is deliberately single-worker.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
from typing import Iterator
import torch
from torch import nn
from torch.nn import functional as F


class FullLayerStateTunnel(nn.Module):
    """One compact shared projection with 12 trainable layer gates.

    At layer i, a rank-r summary is blended with the carried summary
    from layer i-1, then fed back into the current hidden state:
       z_i = (1-sigmoid(m_i))*encode(h_i) + sigmoid(m_i)*z_(i-1)
       h_i' = h_i + tanh(g_i)*decode(z_i)
    The next decoder layer consumes h_i' and the explicit z_i state.
    Original CB61 core, GT3/9 blocks and S111 bridge are left intact.
    """
    PREFIX="full_layer_tunnel."
    def __init__(self, *, width:int=320, depth:int=12, rank:int=8):
        super().__init__()
        if width<1 or depth!=12 or not (1<=rank<=width):
            raise ValueError("S126 requires the original 12-layer positive-rank model")
        self.width,self.depth,self.rank=width,depth,rank
        self.encoder=nn.Parameter(torch.empty(rank,width))
        self.decoder=nn.Parameter(torch.empty(width,rank))
        self.gates=nn.Parameter(torch.zeros(depth))
        self.mix_logits=nn.Parameter(torch.zeros(depth))
        nn.init.normal_(self.encoder,std=0.02)
        nn.init.normal_(self.decoder,std=0.02)
        self._handles=[]
        self._carry=None
        self._active=False
        self._visited=[]
        self.last_forward_coverage=()

    @contextmanager
    def scope(self, enabled:bool=True) -> Iterator[None]:
        if not isinstance(enabled,bool):
            raise ValueError("S126 scope requires a boolean")
        prior=self._active
        self._active=enabled
        try:
            yield
        finally:
            self._active=prior
            self._carry=None
            self._visited=[]

    def _begin(self,_model,_inputs):
        self._carry=None
        self._visited=[]

    def _end(self,_model,_inputs,output):
        if self._active:
            if self._visited!=list(range(self.depth)):
                raise AssertionError("S126 full 12-layer sequential coverage missing")
            self.last_forward_coverage=tuple(self._visited)
        self._carry=None
        self._visited=[]
        return output

    def _layer_hook(self,index:int):
        def hook(_layer,_inputs,output):
            if not self._active:
                return output
            if isinstance(output,tuple):
                h=output[0]
            elif isinstance(output,torch.Tensor):
                h=output
            else:
                raise TypeError("S126 unsupported decoder output")
            if h.ndim!=3 or h.shape[-1]!=self.width:
                raise AssertionError("S126 original 320 hidden ABI changed")
            if h.device!=self.encoder.device:
                raise AssertionError("S126 shared state and core are on different devices")
            if len(self._visited)!=index:
                raise AssertionError("S126 decoder layers not in sequential order")
            self._visited.append(index)
            encoded=torch.tanh(F.linear(F.layer_norm(h,(self.width,)),self.encoder))
            if self._carry is None:
                carried=encoded
            else:
                if self._carry.shape!=encoded.shape:
                    raise AssertionError("S126 recurrent state shape mismatch")
                mix=torch.sigmoid(self.mix_logits[index])
                carried=mix*self._carry+(1.0-mix)*encoded
            self._carry=carried
            gain=torch.tanh(self.gates[index])
            updated=h+gain*F.linear(carried,self.decoder)
            return (updated,*output[1:]) if isinstance(output,tuple) else updated
        return hook

    def install(self,model:nn.Module):
        if self._handles:
            raise RuntimeError("S126 already installed")
        decoder=model.model.layers
        if len(decoder)!=self.depth:
            raise AssertionError("S126 stage-0 must attach to all original 12 layers")
        self._handles.append(model.register_forward_pre_hook(self._begin))
        for index,layer in enumerate(decoder):
            self._handles.append(layer.register_forward_hook(self._layer_hook(index)))
        self._handles.append(model.register_forward_hook(self._end))
        return self

    def uninstall(self):
        for handle in self._handles:handle.remove()
        self._handles=[]
        self._carry=None
        self._active=False
        self._visited=[]

    def new_parameter_count(self):
        return sum(int(p.numel()) for p in self.parameters())

    def export_fp16(self):
        """Actual sidecar must be serialized and bpp audited by caller."""
        return {self.PREFIX+name:t.detach().cpu().to(torch.float16).contiguous()
                for name,t in self.named_parameters()}

    def load_fp16(self,weights):
        expected={self.PREFIX+name for name,_ in self.named_parameters()}
        if set(weights)!=expected:
            raise ValueError("S126 sidecar field list changed")
        with torch.no_grad():
            for name,param in self.named_parameters():
                tensor=weights[self.PREFIX+name]
                if (tensor.dtype!=torch.float16 or tensor.shape!=param.shape
                        or not bool(torch.isfinite(tensor).all())):
                    raise ValueError("Invalid S126 FP16 physical sidecar")
                param.copy_(tensor.to(device=param.device,dtype=param.dtype))

    def assert_pretrained_unchanged(self,old_weights,new_weights):
        """Exact compare for all original tensor identities, not just aggregate."""
        if not set(old_weights).issubset(new_weights):
            raise AssertionError("Original sidecar keys missing")
        if not all(torch.equal(v,new_weights[k]) for k,v in old_weights.items()):
            raise AssertionError("S126 altered original learned GT/core tensors")


def physical_bpp(original_complete_bytes:int,extra_sidecar_bytes:int,
                 runtime_config_bytes:int,original_neural_params:int,
                 additional_neural_params:int)->float:
    """Strict FULL on-disk budget (caller supplies actual serialized sizes).

    Required includes old bank, selector, neural controllers, old GT, new
    S126 physical sidecar AND runtime config. Never count raw FP16 alone.
    """
    counts=(original_complete_bytes,extra_sidecar_bytes,
            runtime_config_bytes,original_neural_params,
            additional_neural_params)
    if any(not isinstance(x,int) or x<0 for x in counts):
        raise ValueError("Physical counts must be non-negative integers")
    if original_neural_params<=0:
        raise ValueError("Empty original architecture")
    return 8.0*(original_complete_bytes+extra_sidecar_bytes+
                runtime_config_bytes)/(original_neural_params+additional_neural_params)
