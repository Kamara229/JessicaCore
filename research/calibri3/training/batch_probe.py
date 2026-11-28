"""Matched same-token-budget forward/backward test of B=1 versus B=2/4."""
import json
from time import perf_counter
import torch
from torch.nn import functional as F
from v176.operators import TrainWeight
from ..weights.cache import frozen_cache
from ..weights.freeze import all_sources,freeze_sources
from ..weights.architecture import replace_selected,error
from research.calibri2.fusion_growth.g35_full_mps_preflight import native
from research.calibri2.fusion_growth.g41_balanced_joint import balance_scope
from .batch_native import native_batch

def select_rows(rows):
    buckets={}
    for r in rows:
        assert len(r["g"])==2
        buckets.setdefault(len(r["p"]),[]).append(r)
    valid=[v for v in buckets.values() if len(v)>=4 and len({r["answer"] for r in v[:4]})==2]
    assert valid,[(k,len(v)) for k,v in buckets.items()]
    return valid[0][:4]

def objective(z,targets):
    y=torch.tensor(targets,device=z.device)
    gold=z.gather(1,y[:,None])[:,0]
    wrong=z.scatter(1,y[:,None],-torch.inf).max(1).values
    return F.cross_entropy(z,y,reduction="none")+F.softplus(2.+wrong-gold)

def run_probe(restored,l6,slot,balance,rows):
    st=restored[0];model,sources,_,full,_,child,_=st
    rows=select_rows(rows)
    keys=[f"model.layers.{i}.{k}.weight" for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    chosen=[sources[k] for k in keys]
    selected={"core."+k:sources[k].centroids for k in keys}
    selected.update({"head."+k:p for k,p in child.language_head.named_parameters()})
    selected.update({"slot."+k:p for k,p in slot.named_parameters()})
    selected["GT.gates"]=full.gates
    for p in selected.values():p.requires_grad_(True)
    ps=tuple(selected.values());allsrc=all_sources(restored)
    def once(size):
        logits=[];losses=[];grads=[torch.zeros_like(p) for p in ps];calls=0
        for pos in range(2):
            for j in range(0,len(rows),size):
                rr=rows[j:j+size]
                pp=[r["p"]+r["g"][:pos] for r in rr];ff=[r["features"] for r in rr]
                if size==1:
                    with balance_scope(st,l6,balance,"balanced"):
                        z,z2=native(st,pp[0],ff[0],slot)
                else:z,z2=native_batch(st,pp,ff,slot,l6,balance)
                calls+=1
                l=objective(z,[r["g"][pos] for r in rr])
                if pos==0:l=l+objective(z2,[r["g"][1] for r in rr])
                gg=torch.autograd.grad(l.sum()/8,ps,allow_unused=True)
                for i,g in enumerate(gg):
                    if g is not None:grads[i]+=g.detach()
                logits.append(torch.stack((z.detach(),z2.detach()),1));losses.append(l.detach())
        return torch.cat(logits),torch.cat(losses),grads,calls
    arms={};ref=None
    # All arms use the verified V003 freeze/cache policy; only batch changes.
    with replace_selected(chosen) as counts,freeze_sources(allsrc,ps,True),frozen_cache(TrainWeight,allsrc) as stats:
        for size in (1,2,4):
            seconds=[]
            for trial in range(4):
                torch.mps.synchronize();start=perf_counter()
                a,b,c,calls=once(size)
                torch.mps.synchronize()
                elapsed=perf_counter()-start
                if trial:seconds.append(elapsed)
            actual=(a.cpu(),b.cpu(),[g.cpu() for g in c])
            if ref is None:ref=actual
            parity={"logits":error(actual[0],ref[0]),"loss":error(actual[1],ref[1]),
                    "top1_equal":bool(torch.equal(actual[0].argmax(-1),ref[0].argmax(-1))),
                    "gradients":{k:error(x,y) for k,x,y in zip(selected,actual[2],ref[2])}}
            gate=parity["top1_equal"] and max(parity["logits"]["relative_l2"],parity["loss"]["relative_l2"],*(v["relative_l2"] for v in parity["gradients"].values()))<.002
            arms[str(size)]={"seconds":seconds,"median_seconds":sorted(seconds)[1],"core_calls":calls,"parity":parity,"gate":gate}
            arms[str(size)]["speed_ratio"]=arms["1"]["median_seconds"]/arms[str(size)]["median_seconds"]
            print("V004_ARM",size,json.dumps(arms[str(size)]),flush=True)
        snapshot=dict(stats)
    return {"arms":arms,"rows":[{"q":r["q"],"answer":r["answer"],"prefix_length":len(r["p"])} for r in rows],
            "cache":snapshot,"compact_calls":counts["compact_calls"],"batch_gate":all(a["gate"] for a in arms.values())}
