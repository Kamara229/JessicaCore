"""Matched real MPS loss/gradient and timing; never update physical weights."""
import torch
from torch.nn import functional as F
from time import perf_counter
from research.calibri2.fusion_growth.g35_full_mps_preflight import native
from ..weights.freeze import freeze_sources
from ..weights.architecture import error
from ..weights.profile import profile_legacy

def probe(st,slot,rows,all_sources,selected,enabled):
    ps=tuple(selected.values());result=[];timings=[]
    with freeze_sources(all_sources,ps,enabled) as flags:
        for trial in range(4):
            # identical warmup then three measured rounds; synchronize once per round
            torch.mps.synchronize();start=perf_counter()
            with profile_legacy() as counts:
                vals=[];grads=[]
                for r in rows:
                    for pos,t in enumerate(r["g"]):
                        z,z2=native(st,r["p"]+r["g"][:pos],r["features"],slot)
                        def objective(logits,target):
                            wrong=logits[0].clone();wrong[target]=-torch.inf
                            return F.cross_entropy(logits,torch.tensor([target],device=logits.device))+F.softplus(2.+wrong.max()-logits[0,target])
                        loss=objective(z,t)
                        if pos+1<len(r["g"]):loss=loss+objective(z2,r["g"][pos+1])
                        g=torch.autograd.grad(loss,ps,allow_unused=True)
                        vals.append((z.detach(),z2.detach(),loss.detach()))
                        grads.append(tuple(torch.zeros_like(p) if v is None else v.detach() for p,v in zip(ps,g)))
                torch.mps.synchronize()
            elapsed=perf_counter()-start
            if trial>0:timings.append(elapsed)
            if trial==3:
                result=[(tuple(v.cpu() for v in vv),tuple(g.cpu() for g in gg)) for vv,gg in zip(vals,grads)]
        return result,{"flags":flags,"seconds":timings,"median_seconds":sorted(timings)[1],"profile":counts}

def compare(a,b,selected):
    checks={}
    for k in ("logits","slot","loss"):
        i=("logits","slot","loss").index(k)
        checks[k]=max(error(y[0][i],x[0][i])["relative_l2"] for x,y in zip(a,b))
    checks["top1_equal"]=all(torch.equal(x[0][i].argmax(),y[0][i].argmax()) for x,y in zip(a,b) for i in (0,1))
    checks["gradient_by_tensor"]={k:max(error(y[1][i],x[1][i])["relative_l2"] for x,y in zip(a,b)) for i,k in enumerate(selected)}
    return checks
