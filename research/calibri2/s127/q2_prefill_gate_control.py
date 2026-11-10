"""S127 research-only: strict prefill-only activation for packed S126-Q GT.

Install the tunnel on the model first. This opt-in scope then activates it
ONLY when a model.forward has an explicit past_key_values=None; bypasses
all cached token continuation with a non-None cache. Never persist state
between forwards, users or callers. Missing explicit KV ABI fails closed.

This module has NO access to Jessica, live services, QA or original S101.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator
import torch
from torch import nn
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel


@contextmanager
def prefill_only_tunnel(
    model:nn.Module,tunnel:FullLayerStateTunnel
)->Iterator[None]:
    if not isinstance(tunnel,FullLayerStateTunnel) or not tunnel._handles:
        raise ValueError("Install S126-Q full GT12 first")
    if tunnel._active:
        raise RuntimeError("Do not nest inside a full activation scope")
    enabled=[]

    def pre(_model,args,kwargs):
        if "past_key_values" not in kwargs:
            raise AssertionError("S127 requires an explicit KV-cache contract")
        if "input_ids" not in kwargs and "inputs_embeds" not in kwargs:
            # Synthetic tests pass x as the first positional input.
            if not args:
                raise AssertionError("S127 missing model input")
        if "use_cache" in kwargs and kwargs["use_cache"] is False:
            raise AssertionError("S127 requires causal cached decoding")
        if tunnel._active:
            raise RuntimeError("S127 previous forward scope leaked active state")
        is_prefill=kwargs["past_key_values"] is None
        tunnel._active=is_prefill
        enabled.append(is_prefill)

    def post(_model,args,kwargs,result):
        tunnel._active=False
        return result

    # These are registered AFTER tunnel.install(model): its prehook _begin
    # clears recurrent state before this hook enables the current prefill;
    # its posthook _end validates all 12-layer visits before ours disables.
    hpre=model.register_forward_pre_hook(pre,with_kwargs=True)
    hpost=model.register_forward_hook(post,with_kwargs=True,always_call=True)
    try:
        yield enabled
    finally:
        hpost.remove()
        hpre.remove()
        tunnel._active=False
        tunnel._carry=None
        tunnel._visited=[]


def restrict_to_gate_only(tunnel:FullLayerStateTunnel):
    """Freeze both Q2 projections and mixing state: only 12 FP16-paid gates."""
    if (tunnel.width,tunnel.depth,tunnel.rank)!=(320,12,4):
        raise ValueError("S127 must start with original Q2-rank4 12-layer model")
    tunnel.encoder.requires_grad_(False)
    tunnel.decoder.requires_grad_(False)
    tunnel.mix_logits.requires_grad_(False)
    tunnel.gates.requires_grad_(True)
    if [name for name,p in tunnel.named_parameters() if p.requires_grad]!=["gates"]:
        raise AssertionError("S127 attempted unapproved dense FP32 training")
    return [tunnel.gates]
