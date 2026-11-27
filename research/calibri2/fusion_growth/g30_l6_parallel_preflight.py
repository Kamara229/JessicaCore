"""G30a: two future native token positions from ONE L6/GT core pass.
Mechanical TRAIN-only preflight, not a model quality or speed claim.
The final G30 training must warm-start from the selected G29 checkpoint.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g28_l6_continuation import (
    Residual,install,parameters,PIN,restore,digest,forward,select,G05)
from research.calibri2.fusion_growth.f06_integrated_recovery import scoped_reduced_native_head_gain
from safetensors.torch import load,save
from v176.operators import clear
G28_SHA="a53dc69d1553a5e7a30dd8f1e97bdd83c651652a5312b4e537f98484ec5a5feb"

def block(st,p,features,slot):
    """Predict y[t+1], y[t+2] from one current prefix.
    Slot 2 uses the same final GT-conditioned hidden, never token 1 or gold.
    The same native readout is called twice, but the twelve-layer core only once.
    """
    model,sources,_,full,_,child,_=st
    captures=[];core_calls=[]
    h1=model.lm_head.register_forward_pre_hook(lambda _m,a: captures.append(a[0][:,-1:,:]))
    h2=model.register_forward_pre_hook(lambda *_:core_calls.append(1))
    try:
        z1=forward(st,p,features)
    finally:
        h1.remove();h2.remove()
    assert len(captures)==len(core_calls)==1
    assert full.last_forward_coverage==tuple(range(12))
    h=captures[0]
    assert h.shape==(1,1,320)
    with torch.enable_grad(),scoped_reduced_native_head_gain(child.language_head):
        try:
            s=slot(h)
            z2=(model.lm_head(s).float()+child.language_head(s).float())[:,0,:]
        finally:clear(sources)
    assert z1.shape==z2.shape==(1,49152)
    assert torch.isfinite(z1).all() and torch.isfinite(z2).all()
    return torch.stack((z1,z2),dim=1),len(core_calls)

def generate(st,row,slot,cap=8):
    seq=[];calls=0;blocks=0
    while len(seq)<min(cap,159-len(row["p"])):
        z,n=block(st,row["p"]+seq,row["features"],slot)
        calls+=n;blocks+=1
        # Both predictions already exist before appending either token.
        for t in z.detach().argmax(-1)[0].tolist():
            if len(seq)>=min(cap,159-len(row["p"])):break
            seq.append(t)
            if t==2:break
        if seq[-1]==2:break
    assert calls==blocks
    return seq,calls

def run(g05,f08,parent,out):
    torch.manual_seed(20270116);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    packet=(parent/"adapter.safetensors").read_bytes()
    assert digest(packet)==G28_SHA
    l6=Residual();l6.load_state_dict(load(packet),strict=True)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train,_,_=select(tok)
    rows=[r for r in train if len(r["g"])>=3 and len(r["p"])+len(r["g"])<160][:8]
    assert len(rows)==8 and all(r["split"]=="TRAIN" for r in rows)
    frozen=list({id(p):p for p in list(parameters(restored))+list(l6.parameters())}.values())
    for p in frozen:p.grad=None
    versions={id(p):p._version for p in frozen}
    slot=Residual();params=list(slot.parameters())
    opt=torch.optim.AdamW(params,lr=.001)
    out.mkdir(parents=True,exist_ok=False)
    observations=[]
    with install(st,6,l6):
        r=rows[0]
        native=forward(st,r["p"],r["features"]).detach()
        logits,calls=block(st,r["p"],r["features"],slot)
        err=float((logits[:,0].detach()-native).abs().max())
        identity_slot=float((logits[:,1].detach()-native).abs().max())
        assert err<1e-4 and identity_slot<1e-4,(err,identity_slot)
        for r in rows[:4]:
            for k in range(2):
                p=r["p"]+r["g"][:k]
                logits,n=block(st,p,r["features"],slot)
                targets=torch.tensor(r["g"][k:k+2])
                assert targets.numel()==2
                loss=F.cross_entropy(logits[0,1:2],targets[1:2])
                opt.zero_grad(set_to_none=True)
                grads=torch.autograd.grad(loss,params)
                norms=[float(g.norm()) for g in grads]
                assert all(torch.isfinite(g).all() for g in grads)
                for par,g in zip(params,grads):par.grad=g
                opt.step()
                observations.append({"core_calls":n,"targets":targets.tolist(),"slot2_loss":float(loss.detach()),"gradient_norms":norms})
        assert all(any(x["gradient_norms"][j]>0 for x in observations) for j in range(len(params)))
        # Reload actual serialized FP16 future-slot weights.
        slot_packet=save({k:v.detach().cpu().half().contiguous() for k,v in slot.state_dict().items()})
        new=Residual();new.load_state_dict(load(slot_packet),strict=True)
        probes=[]
        for r in rows[4:]:
            native=forward(st,r["p"],r["features"]).detach()
            logits,n=block(st,r["p"],r["features"],new)
            assert (native-logits[:,0].detach()).abs().max()<1e-4
            ids,n=generate(st,r,new)
            probes.append({"prompt":r["q"],"ids":ids,"text":tok.decode(ids,skip_special_tokens=True),"core_calls":n,"tokens":len(ids)})
        assert all(x["core_calls"]==(x["tokens"]+1)//2 for x in probes)
        assert all(p.grad is None and p._version==versions[id(p)] for p in frozen)
    size=sum(map(len,blobs))+sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())+len(packet)+len(slot_packet)
    bpp=8*size/G05;assert bpp<=.7
    report={"parent_sha256":G28_SHA,"slot_parameters":sum(p.numel() for p in params),"first_slot_identity":err,"initial_second_slot_identity":identity_slot,"observations":observations,"probes":probes,"known_payload_bpp":bpp,"candidate_accepted":False,"limitations":["TRAIN-only mechanical preflight; eight ephemeral updates, no retained candidate weights.","Two future slots share one core pass; second native readout costs extra compute, CPU acceleration not established.","G29 still training when designed; G28 is used only for mechanics. Full training must preserve selected G29 weights.","No historical616, independent quality or production speed certification."]}
    (out/"G30A_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G30A_REPORT",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out)
