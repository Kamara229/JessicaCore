"""Matched real optimizer step with original G44 objective and all five groups."""
from contextlib import ExitStack
from time import perf_counter
import torch
from v176.operators import TrainWeight
from ..weights.freeze import all_sources,freeze_sources
from ..weights.cache import frozen_cache
from ..weights.architecture import replace_selected
from .last_head import last_position_head
from .batch_native import native_batch
from .batch_probe import objective
from research.calibri2.fusion_growth.g35_full_mps_preflight import native
from research.calibri2.fusion_growth.g41_balanced_joint import balance_scope

def selected_parameters(st,l6,slot):
    _,sources,_,full,_,child,_=st
    keys=[f"model.layers.{i}.{k}.weight" for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    selected={"head."+k:p for k,p in child.language_head.named_parameters()}|{"slot."+k:p for k,p in slot.named_parameters()}|{"l6."+k:p for k,p in l6.named_parameters()}|{"core."+k:sources[k].centroids for k in keys}|{"GT.gates":full.gates}
    for p in selected.values():p.requires_grad_(True)
    return selected,[sources[k] for k in keys]

def optimizer(selected):
    return torch.optim.AdamW([{"params":[p],"lr":.0002 if k.startswith(("head.","slot.")) else .00003} for k,p in selected.items()],foreach=False,weight_decay=0.)

def reset(selected,initial,sources):
    with torch.no_grad():
        for k,p in selected.items():p.copy_(initial[k]);p.grad=None
    for s in sources:s.clear()

def step(st,l6,slot,balance,rows,selected,opt,size,legacy_checks=False):
    ps=tuple(selected.values());acc=[torch.zeros_like(p) for p in ps]
    opt.zero_grad(set_to_none=True);outputs=[];losses=[];calls=0
    denom=sum(len(r["g"]) for r in rows)
    for pos in range(2):
        for j in range(0,len(rows),size):
            rr=rows[j:j+size]
            pp=[r["p"]+r["g"][:pos] for r in rr];ff=[r["features"] for r in rr]
            if len(rr)==1:
                with balance_scope(st,l6,balance,"balanced"):
                    z,z2=native(st,pp[0],ff[0],slot)
            else:z,z2=native_batch(st,pp,ff,slot,l6,balance)
            l=objective(z,[r["g"][pos] for r in rr])
            if pos==0:l=l+objective(z2,[r["g"][1] for r in rr])
            if legacy_checks:float(l.detach().sum())
            gg=torch.autograd.grad(l.sum()/denom,ps,allow_unused=True)
            for i,g in enumerate(gg):
                if g is not None:
                    if legacy_checks:
                        assert torch.isfinite(g).all()
                        float(g.abs().sum())
                    acc[i]+=g.detach()
            outputs.append(torch.stack((z.detach(),z2.detach()),1));losses.append(l.detach());calls+=1
    for p,g in zip(ps,acc):p.grad=g.clone()
    torch.nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True)
    opt.step()
    torch.mps.synchronize()
    return torch.cat(outputs),torch.cat(losses),acc,calls

def policy(stack,restored,selected,chosen,optimized):
    if not optimized:return None
    model=restored[0][0];sources=all_sources(restored)
    stack.enter_context(replace_selected(chosen))
    stack.enter_context(freeze_sources(sources,selected.values(),True))
    stats=stack.enter_context(frozen_cache(TrainWeight,sources))
    stack.enter_context(last_position_head(model))
    return stats
