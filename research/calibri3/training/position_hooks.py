"""V005 causal position injection preflight. Not an adopted model path."""
from contextlib import contextmanager
import torch
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import build_early_causal_view

@contextmanager
def causal_positions(st,l6,balance,positions):
    model,_,_,_,_,child,_=st
    positions=tuple(positions)
    assert positions and tuple(sorted(set(positions)))==positions and positions[0]>=1
    seen={};handles=[];witness={"core_mid_calls":0,"causal_prefix_calls":0}
    def capture(i):
        def hook(m,a,o):
            h=o[0] if isinstance(o,tuple) else o
            assert h.ndim==3 and h.shape[0]==1
            seen[i]=h.detach()
        return hook
    def bounded_l6(m,a,o):
        h=a[0];return h+balance.bounded(h,o-h,0)
    def mid(m,a,o):
        h=o[0] if isinstance(o,tuple) else o
        seen[5]=h.detach()
        assert set(seen)==set(range(6)) and positions[-1]<h.shape[1]
        deltas=[]
        for pos in positions:
            # Never expose future teacher-forced tokens to NRA3.
            prefix={k:v[:,:pos+1,:] for k,v in seen.items()}
            view=build_early_causal_view(prefix)
            delta=child(view,contextual=True)
            assert delta.shape==(1,320) and delta.requires_grad
            deltas.append(balance.bounded(h[:,pos,:],delta,1))
            witness["causal_prefix_calls"]+=1
        index=torch.tensor(positions,device=h.device)
        patch=torch.zeros_like(h).index_copy(1,index,torch.stack(deltas,1))
        updated=h+patch
        witness["core_mid_calls"]+=1
        return (updated,*o[1:]) if isinstance(o,tuple) else updated
    handles.append(l6.register_forward_hook(bounded_l6))
    for i in range(5):handles.append(model.model.layers[i].register_forward_hook(capture(i)))
    handles.append(model.model.layers[5].register_forward_hook(mid))
    try:yield witness
    finally:
        for h in handles:h.remove()
