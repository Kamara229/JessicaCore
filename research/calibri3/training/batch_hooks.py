"""Batched core with per-row legacy NRA3; no neural weights or math changed."""
from contextlib import contextmanager
import torch
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import build_early_causal_view

@contextmanager
def causal_batch(st,l6,balance,n):
    model,_,_,_,_,child,_=st
    seen={};handles=[];witness={"core_mid_calls":0,"nra3_row_calls":0}
    def capture(i):
        def hook(m,a,o):
            h=o[0] if isinstance(o,tuple) else o
            assert h.ndim==3 and h.shape[0]==n
            seen[i]=h.detach()
        return hook
    def bounded_l6(m,a,o):
        h=a[0];return h+balance.bounded(h,o-h,0)
    def mid(m,a,o):
        h=o[0] if isinstance(o,tuple) else o
        seen[5]=h.detach()
        assert set(seen)==set(range(6))
        # Preserve old NRA3 B=1 implementation exactly for each row.
        # The 12-layer core, MLP, attention, GT, and heads remain batched.
        d=[]
        for i in range(n):
            view=build_early_causal_view({k:v[i:i+1] for k,v in seen.items()})
            delta=child(view,contextual=True)
            assert delta.shape==(1,320) and delta.requires_grad
            d.append(balance.bounded(h[i:i+1,-1,:],delta,1))
            witness["nra3_row_calls"]+=1
        delta=torch.cat(d,0)
        updated=h+torch.cat((torch.zeros_like(h[:,:-1,:]),delta[:,None,:]),1)
        witness["core_mid_calls"]+=1
        return (updated,*o[1:]) if isinstance(o,tuple) else updated
    handles.append(l6.register_forward_hook(bounded_l6))
    for i in range(5):handles.append(model.model.layers[i].register_forward_hook(capture(i)))
    handles.append(model.model.layers[5].register_forward_hook(mid))
    try:yield witness
    finally:
        for h in handles:h.remove()
