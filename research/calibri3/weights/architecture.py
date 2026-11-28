"""Local weight-engine composition: real operator parity, gradients and timing."""
from contextlib import contextmanager
from time import perf_counter
import torch
from v176.operators import TrainLinear
from .compact import compact_linear,compact_source,dense_reference

def error(a,b):
    d=(a-b).float()
    return {"max_abs":float(d.abs().max()),"relative_l2":float(d.norm()/b.float().norm().clamp_min(1e-8))}

def sync(device):
    if device=="mps":torch.mps.synchronize()

def check_matrix(source,x,device):
    c,s=source.fields()
    c=c.detach().to(device);s=s.detach().to(device);ids=source.codes.to(device)
    x=x.detach().to(device)
    # Actual restored codebooks; same FP16-rounded values in both paths.
    shape=source.shape
    def run(fn):
        xx=x.clone().requires_grad_(True);cc=c.clone().requires_grad_(True)
        ss=s.clone().requires_grad_(True)
        y=fn(xx,cc,ss,ids,shape)
        target=torch.linspace(-1.,1.,y.numel(),device=device).reshape_as(y)
        g=torch.autograd.grad((y*target).sum()/y.numel(),(xx,cc,ss))
        return y.detach(),[v.detach() for v in g]
    a,ga=run(dense_reference);b,gb=run(compact_linear)
    report={"shape":list(shape),"input_shape":list(x.shape),"device":device,
            "forward":error(b,a),"grad_input":error(gb[0],ga[0]),
            "grad_centroids":error(gb[1],ga[1]),"grad_scales":error(gb[2],ga[2])}
    assert all(report[k]["relative_l2"]<.002 for k in ("forward","grad_input","grad_centroids","grad_scales")),report
    for name,fn in (("dense",dense_reference),("compact",compact_linear)):
        run(fn);sync(device);ts=[]
        for _ in range(3):
            t=perf_counter();run(fn);sync(device);ts.append(perf_counter()-t)
        report[name+"_seconds"]=sorted(ts)[1]
    report["speed_ratio"]=report["dense_seconds"]/report["compact_seconds"]
    return report

@contextmanager
def replace_selected(sources):
    chosen={id(s) for s in sources};original=TrainLinear.forward
    counts={"compact_calls":0}
    def forward(self,x):
        if id(self.source) in chosen:
            counts["compact_calls"]+=1
            return compact_source(x,self.source)
        return original(self,x)
    TrainLinear.forward=forward
    try:yield counts
    finally:TrainLinear.forward=original
