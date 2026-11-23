"""Z18 preregister cheap 12-layer h320 readout consensus BEFORE any new DEV scoring.

Read only THREE selected true original S142 decoder intermediate states
(L3,L7,L11, zero-based indices2,6,10), plus the SAME original43M
model's already-computed final lm_head input h320. Frozen layer mix
weights = .25 each, all layer_norm per vector. No extra original43M
forward. Contrast final h320 ONLY vs fixed multi-layer 320-d mean
under identical top64 candidates, rank2/1280 params, learning
hyperparameters, TRAIN-only BPE2 labels, and A/B original weights.
No new neural region, no retained layer hooks, no DEV gradients.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z18_fresh28train14dev_multilayer_source import (
    MULTI_LAYER_ZERO_BASED,MULTI_LAYER_MIX_WEIGHTS,
)

HIDDEN=320
DEPTH=12

def check(v,msg):
    if not v:
        raise ValueError("Z18 physical multi-layer consensus h320 contract: "+msg)

@contextmanager
def capture_original_selected_layers(model):
    """Exactly one ORIGINAL full decoder forward; 3 real intermediates, CPU detached."""
    check(hasattr(model,"model")
          and hasattr(model.model,"layers")
          and len(model.model.layers)==DEPTH
          and not model.training
          and MULTI_LAYER_ZERO_BASED==(2,6,10),
          "actual original S142 12 layers and fixed selected indices")
    old={k:set(model.model.layers[k]._forward_hooks)
         for k in MULTI_LAYER_ZERO_BASED}
    states={}
    handles=[]
    def hook(k):
        def audit(_module,_args,out):
            x=out[0] if isinstance(out,tuple) else out
            check(k not in states and isinstance(x,torch.Tensor)
                  and x.ndim==3 and x.shape[0]==1 and x.shape[-1]==HIDDEN
                  and bool(torch.isfinite(x).all()),
                  "one real original layer output tensor, same model pass")
            states[k]=x[:,-1,:].detach().cpu().float().clone()
            return None
        return audit
    try:
        handles=[model.model.layers[k].register_forward_hook(hook(k))
                 for k in MULTI_LAYER_ZERO_BASED]
        yield states
        check(set(states)==set(MULTI_LAYER_ZERO_BASED)
              and all(v.shape==(1,HIDDEN) and not v.requires_grad
                      for v in states.values()),
              "all L3 L7 L11 actual original 43M intermediate states captured")
    finally:
        for h in handles:h.remove()
        check(all(set(model.model.layers[k]._forward_hooks)==old[k]
                  for k in MULTI_LAYER_ZERO_BASED),
              "all intermediate hooks FULLY removed, never production")


def fixed_multi_layer_h320(layer_states,final_lm_h320):
    """No learned layer routing: fixed preregistered equal 4-way 320 mix."""
    check(set(layer_states)==set(MULTI_LAYER_ZERO_BASED)
          and isinstance(final_lm_h320,torch.Tensor)
          and tuple(final_lm_h320.shape)==(1,HIDDEN)
          and MULTI_LAYER_MIX_WEIGHTS==(.25,.25,.25,.25),
          "precommitted 3 layers and actual native last h320 required")
    values=[layer_states[k] for k in MULTI_LAYER_ZERO_BASED]+[final_lm_h320]
    check(all(isinstance(x,torch.Tensor)
              and x.shape==(1,HIDDEN) and torch.isfinite(x).all()
              for x in values),
          "all four finite original S142 h320 readouts")
    with torch.no_grad():
        normalized=[F.layer_norm(x.detach().float(),(HIDDEN,))
                    for x in values]
        out=sum(v*w for v,w in zip(normalized,MULTI_LAYER_MIX_WEIGHTS))
    check(out.shape==(1,HIDDEN) and not out.requires_grad
          and torch.isfinite(out).all(),
          "frozen 3 intermediate + final native head vector, no trainable mix")
    return out
