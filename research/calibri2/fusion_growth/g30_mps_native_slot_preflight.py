"""G30 GPU preflight: actual native L6 hidden states, MPS trainable future slot.
CPU prefix capture is explicit; no claim of full-core GPU training.
Run on G28 only to prepare G29 warm-start continuation, no promotion.
"""
import argparse,copy,json,os,time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from safetensors.torch import load,save
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import (
    Residual,install,parameters,PIN,restore,digest,forward,select,G05,G28_SHA)
from research.calibri2.fusion_growth.g28_l6_continuation import ByteValidity
from v176.operators import clear

def run(g05,f08,parent,out):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.manual_seed(20270117);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    model,sources,_,full,_,child,_=st
    packet=(parent/"adapter.safetensors").read_bytes();assert digest(packet)==G28_SHA
    l6=Residual();l6.load_state_dict(load(packet),strict=True)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train,_,_=select(tok);utf=ByteValidity(tok)
    frozen=list({id(p):p for p in list(parameters(restored))+list(l6.parameters())}.values())
    versions={id(p):p._version for p in frozen}
    for p in frozen:p.grad=None
    # Actual compact native head reconstruction. Dense training-only view is
    # never exported; inference still uses the original compact native head.
    w=model.lm_head.source().detach().clone();clear(sources)
    assert w.shape==(49152,320)
    head=copy.deepcopy(child.language_head).eval().requires_grad_(False)
    assert all(p.device.type=="cpu" for p in head.parameters())
    def readout(h,weight,alignment):
        return (F.linear(h,weight)+.25*alignment(h))[:,0,:]
    hs=[];targets=[];states=[];errors=[]
    # Balanced fixed sample across existing TRAIN families; no DEV tuning.
    groups={}
    for r in train:groups.setdefault(r["family"],[]).append(r)
    rows=[r for k in sorted(groups) for r in groups[k][:4]]
    assert all(r["split"]=="TRAIN" for r in rows)
    with install(st,6,l6):
        for r in rows:
            ss=utf.states(r)
            for k in range(min(4,len(r["g"])-1)):
                capture=[]
                h=model.lm_head.register_forward_pre_hook(lambda _m,a:capture.append(a[0][:,-1:,:].detach().clone()))
                try:z=forward(st,r["p"]+r["g"][:k],r["features"]).detach()
                finally:h.remove()
                assert len(capture)==1
                v=capture[0]
                err=float((readout(v,w,head)-z).abs().max())
                assert err<1e-4,err
                errors.append(err);hs.append(v);targets.append(r["g"][k+1]);states.append(ss[k+1])
    x=torch.cat(hs,dim=0);y=torch.tensor(targets);valid=torch.stack([utf.legal(s) for s in states])
    assert len(x)>=32 and valid[torch.arange(len(y)),y].all()
    # Both CPU and MPS start from the identical checkpoint and use same
    # fixed mini-batches, true gradients, and same readout. No proxy model.
    initial=Residual().state_dict()
    def optimize(device):
        slot=Residual().to(device);slot.load_state_dict(initial)
        wd=w.to(device);hd=copy.deepcopy(head).to(device);xd=x.to(device);yd=y.to(device);vd=valid.to(device)
        opt=torch.optim.AdamW(slot.parameters(),lr=.001,foreach=False)
        def objective(ix):
            z=readout(slot(xd[ix]),wd,hd)
            assert z.device.type==device
            ce=F.cross_entropy(z,yd[ix])
            # finite masking constant only in training loss, never inference
            lv=(z.logsumexp(-1)-z.masked_fill(~vd[ix],-1e9).logsumexp(-1)).mean()
            return ce+.3*lv
        def sync():
            if device=="mps":torch.mps.synchronize()
        allidx=torch.arange(len(x),device=device)
        before=float(objective(allidx).detach());sync()
        start=time.perf_counter();norms=[];updates=0
        for epoch in range(3):
            for j in range(0,len(x),8):
                ix=allidx[j:j+8];opt.zero_grad(set_to_none=True);loss=objective(ix);loss.backward()
                assert all(p.grad is not None and p.grad.device.type==device and torch.isfinite(p.grad).all() for p in slot.parameters())
                norms.append([float(p.grad.norm()) for p in slot.parameters()])
                opt.step();updates+=1
        sync();seconds=time.perf_counter()-start
        after=float(objective(allidx).detach())
        state={k:v.detach().cpu().clone() for k,v in slot.state_dict().items()}
        assert all(any(n[i]>0 for n in norms) for i in range(2))
        return {"device":device,"updates":updates,"before":before,"after":after,"seconds":seconds,"gradient_device":device},state
    cpu,cs=optimize("cpu");gpu,gs=optimize("mps")
    delta=max(float((cs[k]-gs[k]).abs().max()) for k in cs)
    raw=save({k:v.half().contiguous() for k,v in gs.items()})
    reloaded=Residual();reloaded.load_state_dict(load(raw),strict=True)
    reload_loss=float(F.cross_entropy(readout(reloaded(x),w,head),y).detach())
    assert all(p.grad is None and p._version==versions[id(p)] for p in frozen)
    size=sum(map(len,blobs))+sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())+len(packet)+len(raw)
    bpp=8*size/G05;assert bpp<=.7
    out.mkdir(parents=True,exist_ok=False)
    report={"parent_sha256":G28_SHA,"prefixes":len(x),"families":sorted(groups),"max_native_logit_error":max(errors),"cpu":cpu,"gpu":gpu,"cpu_over_mps_update_seconds":cpu["seconds"]/gpu["seconds"],"max_weight_difference":delta,"reloaded_train_slot2_ce":reload_loss,"known_payload_bpp":bpp,"candidate_accepted":False,"limitations":["Actual native L6/GT hidden states captured on CPU; only future-slot optimization and native readout run on GPU.","G28 structural preflight only; full G30b must warm-start selected G29, no quality or CPU inference certification.","No dense weight exports; no external teacher, no DEV updates."]}
    (out/"G30_MPS_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G30_MPS_REPORT",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out)
