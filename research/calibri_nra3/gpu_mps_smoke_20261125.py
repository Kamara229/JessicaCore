"""Short hardware smoke test only: synthetic Calibri-shaped rank-4 head, no model checkpoint."""
import json, os, platform, time
import torch
import torch.nn.functional as F

assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0", "No silent CPU fallback"
assert torch.backends.mps.is_built() and torch.backends.mps.is_available(), "MPS unavailable"
print("GPU_AVAILABLE", json.dumps({"machine":platform.machine(),"torch":torch.__version__,"MPS":True}), flush=True)
torch.manual_seed(119)
torch.set_num_threads(2)
x=torch.randn(32,320)
base=torch.randn(32,49152)*.1
gold=torch.arange(32,dtype=torch.long)+100
state=[torch.randn(320,4)*.02,torch.randn(4,49152)*.02,torch.zeros(49152)]
def train(device):
    xx,bb,yy=[v.to(device) for v in (x,base,gold)]
    params=[torch.nn.Parameter(v.to(device).clone()) for v in state]
    opt=torch.optim.Adam(params,lr=.003,foreach=False)
    def sync():
        if device=="mps":torch.mps.synchronize()
    def step():
        opt.zero_grad(set_to_none=True)
        logits=bb+(xx@params[0])@params[1]+params[2]
        assert logits.device.type==device
        ce=F.cross_entropy(logits,yy)
        eos=F.softplus(logits[:,2]-logits.gather(1,yy[:,None]).squeeze(1)+1.).mean()
        loss=ce+.25*eos
        loss.backward()
        assert all(p.grad is not None and p.grad.device.type==device for p in params)
        opt.step()
        return loss
    initial=float(step().detach().cpu())
    sync(); t=time.perf_counter()
    for _ in range(8): final=step()
    sync(); elapsed=time.perf_counter()-t
    changed=[float((p.detach().cpu()-v).norm()) for p,v in zip(params,state)]
    final=float(final.detach().cpu())
    assert all(v>0 for v in changed) and final<initial
    return {"device":device,"initial_loss":initial,"final_loss":final,"updates":9,"timed_updates":8,"seconds":elapsed,"weight_changes":changed},[p.detach().cpu() for p in params]
cpu,a=train("cpu")
gpu,b=train("mps")
delta=max(float((i-j).abs().max()) for i,j in zip(a,b))
assert abs(cpu["final_loss"]-gpu["final_loss"])<.005 and delta<.005, "CPU MPS parity"
print("GPU_SMOKE_RESULT",json.dumps({"cpu":cpu,"gpu":gpu,"max_weight_delta":delta,"speed_ratio":cpu["seconds"]/gpu["seconds"],"fallback":False,"synthetic_only":True,"gate":0}),flush=True)
