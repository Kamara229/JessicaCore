"""Oscar v1F: real causal interlayer rank32 Green Tunnel plus paid rank8 head.

Source pretrained 8x512 decoder weights are always FROZEN K32V8 Oscar-encoded.
This architecture is NOT the CALIBRI S142 12x320 trained Green Tunnel.
New full8 recurrent state + separate output logits correction are physically
charged, source itself never receives gradient/FP32 restoration at inference.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

DEPTH=8
WIDTH=512
RANK=32
HEAD_RANK=8
VOCAB=32000
PREFIX="oscar_v1f."
EPS=1e-6


class FullLayerGreenTunnel(nn.Module):
    """Explicit rank32 z_i carried through ALL original eight decoder layers.

    z_i=(1-mix_i)*encode(h_i) + mix_i*z_{i-1}, 
    h_i'=h_i + (0.5 + 0.5*tanh(g_i))*decode_i(z_i).
    The previous corrected h_i' is consumed by the NEXT normal decoder.
    Carry resets per model.forward, never crosses prompts or token positions.
    """
    def __init__(self):
        super().__init__()
        gen=torch.Generator(device="cpu").manual_seed(421932)
        self.down=nn.Parameter(
            torch.randn((DEPTH,RANK,WIDTH),generator=gen)*0.018)
        self.up=nn.Parameter(torch.zeros((DEPTH,WIDTH,RANK)))
        self.gain=nn.Parameter(torch.zeros(DEPTH))
        self.mix_logits=nn.Parameter(torch.zeros(DEPTH))
        self.enabled=True
        self._hooks=[]
        self._carry=None
        self._visited=[]
        self._layers=[]
        self.latest_layers=[]
        self.latest_norm=None
        self.last_coverage=()

    def install(self,model):
        if self._hooks:
            raise RuntimeError("Oscar v1F Green Tunnel already installed")
        if len(model.layers)!=DEPTH:
            raise AssertionError("Oscar v1F physically expects original pretrained 8layer/512 checkpoint")
        def begin(_model,_args):
            self._carry=None
            self._visited=[]
            self._layers=[]
            self.latest_layers=[]
            self.latest_norm=None
        def hook_for(i):
            def h(_module,_args,output):
                if len(self._visited)!=i:
                    raise AssertionError("Oscar genuine original 8layer causal GT coverage order violated")
                if not isinstance(output,torch.Tensor):
                    raise AssertionError("Original pretrained 42M layer must output real hidden Tensor")
                if output.ndim!=3 or output.shape[-1]!=WIDTH:
                    raise AssertionError("Oscar v1F original 42M 8layer hidden width corrupted")
                self._visited.append(i)
                if self.enabled:
                    z=torch.tanh(F.linear(
                        F.layer_norm(output,(WIDTH,)),self.down[i]))
                    if self._carry is not None:
                        if self._carry.shape!=z.shape:
                            raise AssertionError("Oscar v1F GT recurrent carry seq changed")
                        mix=torch.sigmoid(self.mix_logits[i])
                        z=(1-mix)*z+mix*self._carry
                    self._carry=z
                    scale=0.5*(1+torch.tanh(self.gain[i]))
                    amended=output+scale*F.linear(z,self.up[i])
                else:
                    amended=output
                    self._carry=None
                self._layers.append(amended)
                return amended
            return h
        def norm(_module,_args,output):
            if output.shape[-1]!=WIDTH:
                raise AssertionError("Original pretrained final RMS norm width changed")
            self.latest_norm=output
            return output
        def finish(_model,_args,output):
            if self._visited!=list(range(DEPTH)) or len(self._layers)!=DEPTH:
                raise AssertionError("Oscar full interlayer Green Tunnel failed actual all8 decoder hooks")
            if self.latest_norm is None:
                raise AssertionError("Oscar v1F original final prehead norm hook never ran")
            self.latest_layers=list(self._layers)
            self.last_coverage=tuple(self._visited)
            self._layers=[]
            self._visited=[]
            self._carry=None
            return output
        self._hooks.append(model.register_forward_pre_hook(begin))
        for i,layer in enumerate(model.layers):
            self._hooks.append(layer.register_forward_hook(hook_for(i)))
        self._hooks.append(model.norm.register_forward_hook(norm))
        self._hooks.append(model.register_forward_hook(finish))
        return self

    def uninstall(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks=[]
        self._carry=None
        self._layers=[]
        self._visited=[]
        self.latest_layers=[]
        self.latest_norm=None

    @contextmanager
    def scope(self,enabled:bool=True)->Iterator[None]:
        prev=self.enabled
        if not isinstance(enabled,bool):
            raise TypeError("Green Tunnel scope must be boolean")
        self.enabled=enabled
        try:
            yield
        finally:
            self.enabled=prev
            self._carry=None
            self._visited=[]

    def source_paid_tensors(self):
        return {
          PREFIX+"down_fp16":self.down.detach().cpu().half().contiguous(),
          PREFIX+"up_fp16":self.up.detach().cpu().half().contiguous(),
          PREFIX+"gains_fp16":self.gain.detach().cpu().half().contiguous(),
          PREFIX+"mix_fp16":self.mix_logits.detach().cpu().half().contiguous(),
        }

    def restore_paid(self,payload):
        expected=self.source_paid_tensors()
        if set(payload)!=set(expected):
            raise AssertionError("Oscar full8 physically paid GT missing/unexpected tensor")
        with torch.no_grad():
            for key,dest in [
                (PREFIX+"down_fp16",self.down),
                (PREFIX+"up_fp16",self.up),
                (PREFIX+"gains_fp16",self.gain),
                (PREFIX+"mix_fp16",self.mix_logits)]:
                x=payload[key]
                if (x.dtype!=torch.float16 or x.shape!=dest.shape
                    or not torch.isfinite(x).all()):
                    raise AssertionError("Oscar authentic FP16 Green Tunnel shape/dtype invalid")
                dest.copy_(x.float())


class PaidHeadCorrection(nn.Module):
    """8-channel FP16 learned correction of original compressed 32k LM head.

    Corrected logits = original physically decoded K32V8 tied head logits
        + U( GELU(D( layer_norm(final_hidden) )) )
    U zero at init; correction trainable and completely accounted physically.
    """
    def __init__(self):
        super().__init__()
        gen=torch.Generator(device="cpu").manual_seed(421933)
        self.down=nn.Parameter(
            torch.randn((HEAD_RANK,WIDTH),generator=gen)*.02)
        self.up=nn.Parameter(torch.zeros((VOCAB,HEAD_RANK)))
        self.enabled=True

    def forward(self,final_hidden):
        if not self.enabled:
            return final_hidden.new_zeros(
                (*final_hidden.shape[:-1],VOCAB))
        x=F.layer_norm(final_hidden,(WIDTH,))
        features=F.gelu(F.linear(x,self.down))
        return F.linear(features,self.up)

    @contextmanager
    def scope(self,enabled:bool=True):
        prior=self.enabled
        self.enabled=enabled
        try:
            yield
        finally:
            self.enabled=prior

    def source_paid_tensors(self):
        return {
           PREFIX+"head_down_fp16":self.down.detach().cpu().half().contiguous(),
           PREFIX+"head_up_fp16":self.up.detach().cpu().half().contiguous()
        }

    def restore_paid(self,payload):
        expected=self.source_paid_tensors()
        if set(payload)!=set(expected):
            raise AssertionError("Oscar v1F paid 32k output correction invalid source keys")
        with torch.no_grad():
            for name,p in [
                (PREFIX+"head_down_fp16",self.down),
                (PREFIX+"head_up_fp16",self.up)]:
                val=payload[name]
                if (val.dtype!=torch.float16 or val.shape!=p.shape or
                    not torch.isfinite(val).all()):
                    raise AssertionError("Oscar v1F original output paid FP16 invalid")
                p.copy_(val.float())


def merge_paid(green:FullLayerGreenTunnel,head:PaidHeadCorrection):
    whole=green.source_paid_tensors()
    whole.update(head.source_paid_tensors())
    if len(whole)!=6 or len(set(whole))!=6:
        raise AssertionError("Oscar actual full 8layer GT+head missing physical tensors")
    return whole


def restore_paid(green,head,loaded):
    gt={k:v for k,v in loaded.items() if k in green.source_paid_tensors()}
    hd={k:v for k,v in loaded.items() if k in head.source_paid_tensors()}
    if len(gt)+len(hd)!=len(loaded):
        raise AssertionError("Oscar v1F additional hidden FP16 weights unaccounted")
    green.restore_paid(gt)
    head.restore_paid(hd)


def actual_full_inference(model,green,head,input_ids):
    """Use real unmodified original pretrained model + physically paid GT/head."""
    scores,_=model(input_ids)
    if green.last_coverage!=tuple(range(DEPTH)):
        raise AssertionError("Oscar original full 8 decoder Green Tunnel wasn't actually executed")
    if green.latest_norm is None or len(green.latest_layers)!=DEPTH:
        raise AssertionError("Oscar original student hidden activations not measured")
    corrected=scores+head(green.latest_norm)
    if corrected.shape!=scores.shape or not bool(torch.isfinite(corrected).all()):
        raise AssertionError("Oscar genuine compressed same-network output invalid")
    return corrected
