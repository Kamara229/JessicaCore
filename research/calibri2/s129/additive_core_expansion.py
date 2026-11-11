"""S129 additive 39.82M core expansion. Research only; never changes original CB61.

Old 36,380,480 original parameters stay byte-identical. Twelve *independent*
residual adapters (320 -> 448 -> 320) append 3,440,640 explicit parameters:
new architectural total 39,821,120 (without pre-existing small GT sidecars).

Only an initial 24 out of 448 new channels per decoder layer are active
during S129's first J7-protected TRAIN transaction. The remaining new
weights are allocated, exactly zero in the up projection, and can be unlocked
by later stages, not claimed as trained or as useful learning capacity yet.

The immutable down projection is GENERATED from a physical integer seed,
so its complete fixed numerical values do not have to be stored. New up
weights start at zero (exact old-output parity), and only active FP16 up
weights are physically serialized. No missing learned FP32/FP16 tensor may
be silently omitted: inactive up values MUST be identically zero. Seed,
active rank, dimensions, and exact persisted learned weights are included
in the full on-disk bpp accounting by the caller.

No QA/TEST access, no prod writes or promotion. Each prefill or continuation
is isolated: active only for prefill with explicit KV-none.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
import torch
from torch import nn
from torch.nn import functional as F

WIDTH=320
DEPTH=12
EXPANSION_RANK=448
ACTIVE_RANK=24
SEED=1290129
ORIGINAL_CORE_PARAMS=36_380_480
EXTRA_CORE_PARAMS=DEPTH * 2 * WIDTH * EXPANSION_RANK
EXPANDED_CORE_PARAMS=ORIGINAL_CORE_PARAMS + EXTRA_CORE_PARAMS
assert EXTRA_CORE_PARAMS==3_440_640 and EXPANDED_CORE_PARAMS==39_821_120
PREFIX="s129_added_core."
UP_KEY=PREFIX+"active_up_fp16"


class AdditiveCoreExpansion(nn.Module):
    def __init__(self,*,active_rank:int=ACTIVE_RANK):
        super().__init__()
        if not 1<=active_rank<=EXPANSION_RANK:
            raise ValueError("Invalid enabled expansion rank")
        self.active_rank=int(active_rank)
        generator=torch.Generator(device="cpu").manual_seed(SEED)
        down=torch.randn((DEPTH,EXPANSION_RANK,WIDTH),generator=generator)*0.02
        self.down=nn.Parameter(down,requires_grad=False)
        self.up=nn.Parameter(torch.zeros((DEPTH,WIDTH,EXPANSION_RANK)))
        self._handles=[]
        self._active=False
        self._visited=[]
        self.last_prefill_coverage=()

    def architectural_new_parameter_count(self):
        return sum(int(p.numel()) for p in self.parameters())

    def trainable_initial_parameters(self):
        if self.down.requires_grad or not self.up.requires_grad:
            raise AssertionError("S129 only new up tensor trainable initially")
        if self.architectural_new_parameter_count()!=EXTRA_CORE_PARAMS:
            raise AssertionError("S129 39.82M architecture changed")
        return [self.up]

    def assert_inactive_zero(self):
        if torch.count_nonzero(self.up.detach()[:,:,self.active_rank:]).item()!=0:
            raise AssertionError("S129 unpersisted expansion coefficients nonzero")
        if not bool(torch.isfinite(self.up.detach()).all()):
            raise AssertionError("Nonfinite S129 new weights")

    def _begin(self,_model,_args):
        self._visited=[]

    def _end(self,_model,_args,output):
        if self._active:
            if self._visited!=list(range(DEPTH)):
                raise AssertionError("S129 not all twelve decoder layers visited in order")
            self.last_prefill_coverage=tuple(self._visited)
        self._visited=[]
        return output

    def _layer_hook(self,index:int):
        def hook(_layer,_args,output):
            if not self._active:
                return output
            h=output[0] if isinstance(output,tuple) else output
            if not isinstance(h,torch.Tensor) or h.ndim!=3 or h.shape[-1]!=WIDTH:
                raise AssertionError("S129 original hidden state ABI changed")
            if h.device!=self.up.device or h.device!=self.down.device:
                raise AssertionError("S129 expansion and core on different devices")
            if len(self._visited)!=index:
                raise AssertionError("S129 decoder order changed")
            self._visited.append(index)
            z=F.gelu(F.linear(F.layer_norm(h,(WIDTH,)),self.down[index,:self.active_rank,:]))
            delta=F.linear(z,self.up[index,:,:self.active_rank])
            updated=h+delta
            return (updated,*output[1:]) if isinstance(output,tuple) else updated
        return hook

    def install(self,model:nn.Module):
        if self._handles:
            raise RuntimeError("S129 duplicate install")
        if not hasattr(model,"model") or len(model.model.layers)!=DEPTH:
            raise AssertionError("S129 expects original twelve decoder layers")
        self._handles.append(model.register_forward_pre_hook(self._begin))
        for i,layer in enumerate(model.model.layers):
            self._handles.append(layer.register_forward_hook(self._layer_hook(i)))
        self._handles.append(model.register_forward_hook(self._end))
        return self

    def uninstall(self):
        for handle in self._handles:
            handle.remove()
        self._handles=[]
        self._active=False
        self._visited=[]

    @contextmanager
    def prefill_only(self,model:nn.Module)->Iterator[None]:
        if not self._handles or self._active:
            raise RuntimeError("S129 install before prefill-only inference")
        def pre(_model,args,kwargs):
            if "past_key_values" not in kwargs:
                raise AssertionError("S129 explicit causal KV ABI required")
            if kwargs.get("use_cache") is False:
                raise AssertionError("S129 requires causal cached decoding")
            self._active=kwargs["past_key_values"] is None
        def post(_model,args,kwargs,result):
            self._active=False
            return result
        hpre=model.register_forward_pre_hook(pre,with_kwargs=True)
        hpost=model.register_forward_hook(post,with_kwargs=True,always_call=True)
        try:
            yield
        finally:
            hpost.remove()
            hpre.remove()
            self._active=False
            self._visited=[]

    def export_active_fp16(self):
        self.assert_inactive_zero()
        return {UP_KEY:self.up.detach().cpu()[:,:,:self.active_rank].half().contiguous()}

    def load_active_fp16(self,sidecar):
        if set(sidecar)!={UP_KEY}:
            raise ValueError("S129 unexpected physical learned-weight keys")
        x=sidecar[UP_KEY]
        if (x.shape!=(DEPTH,WIDTH,self.active_rank)
                or x.dtype!=torch.float16 or not bool(torch.isfinite(x).all())):
            raise ValueError("S129 physically learned FP16 shape/type invalid")
        with torch.no_grad():
            self.up.zero_()
            self.up[:,:,:self.active_rank].copy_(
                x.to(device=self.up.device,dtype=self.up.dtype))
        self.assert_inactive_zero()
