"""G17: train a small full-vocabulary contextual residual readout.
G15 found common-vector dilution through Core12 and native readout collapse.
Freeze G05 and existing language head. Cache actual causal TRAIN hidden
states and logits, compare matched raw vs centered normalized features.
Then quantize each learned residual and verify real free generation through
all twelve layers. No answer mask, routing, external teacher or G14 fitting.
"""
import argparse
import json
import math
import struct
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save,load
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import digest
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g02_attention_discrete import forward
from research.calibri2.fusion_growth.g16_russian_connected_language import data,audit,FAMILIES
from research.calibri2.fusion_growth.g14_russian_ladder import suite
from research.calibri2.fusion_growth.f09_head_rate_codec import quantize,reconstruct

SEED=20270109
RANK=8
EPOCHS=24
BATCH=32
ARMS=("RAW","CENTERED")
GAIN=.25

class Residual(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.register_buffer("mean",mean.clone().float())
        self.register_buffer("scale",scale.clone().float())
        self.down=nn.Linear(320,RANK,bias=False)
        self.up=nn.Linear(RANK,49152,bias=False)
        nn.init.normal_(self.down.weight,std=.02)
        nn.init.zeros_(self.up.weight)
    def forward(self,h):
        x=(F.layer_norm(h.float(),(320,))-self.mean)*self.scale
        return self.up(F.gelu(self.down(x)))

def cache(st,train):
    hs=[];zs=[];targets=[];weights=[];head=st[0].lm_head
    n=0
    for r in train:
        assert r["split"]=="TRAIN"
        for j,t in enumerate(r["g"]):
            cap=[]
            h=head.register_forward_pre_hook(lambda _m,args:cap.append(args[0][0,-1].detach().float().clone()))
            try:z=forward(st,r["p"]+r["g"][:j],r["features"]).detach()[0].clone()
            finally:h.remove()
            assert len(cap)==1 and cap[0].shape==(320,) and z.shape==(49152,)
            hs.append(cap[0]);zs.append(z);targets.append(t)
            weights.append(.7 if t==2 else 1./max(1,len(r["g"])-1));n+=1
        if len(hs)%128<len(r["g"]):print("G17_CACHE",n,flush=True)
    x=torch.stack(hs);z=torch.stack(zs);y=torch.tensor(targets);w=torch.tensor(weights)
    assert torch.isfinite(x).all() and torch.isfinite(z).all()
    return x,z,y,w

@contextmanager
def install(st,aux):
    # Registered before scoped_reduced_native_head_gain: the existing
    # .25 gain applies to base correction AND the additional residual.
    head=st[5].language_head
    handle=head.register_forward_hook(lambda _m,args,out:out+aux(args[0]))
    try:yield
    finally:handle.remove()

def packet(aux):
    # Quantize only the new 8 x full-vocabulary up projection to Q8.
    desc,raw=quantize(aux.up.weight.detach().half().contiguous(),8)
    other=save({"down":aux.down.weight.detach().half().contiguous(),
                "mean":aux.mean.detach().half().contiguous(),
                "scale":aux.scale.detach().half().contiguous()})
    meta=json.dumps(desc,sort_keys=True,separators=(",",":")).encode()
    return b"G17R"+struct.pack("<II",len(meta),len(other))+meta+other+raw

def unpack(blob):
    assert blob[:4]==b"G17R"
    n,m=struct.unpack_from("<II",blob,4);i=12
    desc=json.loads(blob[i:i+n]);i+=n
    other=load(blob[i:i+m]);i+=m
    aux=Residual(other["mean"],other["scale"])
    with torch.no_grad():
        aux.down.weight.copy_(other["down"].float())
        aux.up.weight.copy_(reconstruct(desc,blob[i:]).float())
    return aux.eval()

def run(g05,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tr=data(tok,"TRAIN");dv=data(tok,"DEV")
    train=[r for rows in tr.values() for r in rows];dev=[r for rows in dv.values() for r in rows]
    assert len(train)==144 and len(dev)==48
    assert not {r["q"] for r in train}&({r["q"] for r in dev}|{r["q"] for r in suite()})
    out.mkdir(parents=True,exist_ok=False)
    st,*rest=restore(blobs[:6],blobs[6],blobs[7])
    x,z,y,w=cache(st,train)
    # Fixed TRAIN-only statistics, rounded to the exact stored precision.
    h=F.layer_norm(x,(320,));mean=h.mean(0).half().float()
    rms=(h-mean).square().mean().sqrt()
    scale=(1/rms.clamp_min(.01)).half().float()
    assert torch.isfinite(scale) and scale>0
    tb=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    baseline=audit(st,tok,dev,"G17_BASE")
    rng=np.random.default_rng(SEED)
    orders=[rng.permutation(len(y)) for _ in range(EPOCHS)]
    results={}
    for arm in ARMS:
        torch.manual_seed(SEED)
        aux=Residual(mean if arm=="CENTERED" else torch.zeros_like(mean),
                     scale if arm=="CENTERED" else torch.ones_like(scale))
        assert sum(p.numel() for p in aux.parameters())==RANK*(320+49152)
        opt=torch.optim.AdamW(aux.parameters(),lr=.008,weight_decay=.0001)
        steps=0;curve=[];grad=set()
        for epoch,order in enumerate(orders,1):
            losses=[]
            for j in range(0,len(order),BATCH):
                ix=torch.from_numpy(order[j:j+BATCH].copy()).long()
                logits=z[ix]+GAIN*aux(x[ix])
                ce=F.cross_entropy(logits,y[ix],reduction="none")
                loss=(ce*w[ix]).sum()/w[ix].sum()
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward()
                grad.update(k for k,p in aux.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0)
                torch.nn.utils.clip_grad_norm_(aux.parameters(),5.,error_if_nonfinite=True);opt.step()
                steps+=1;losses.append(float(loss.detach()))
            curve.append(float(np.mean(losses)))
            if epoch%4==0:print("G17_PROGRESS",arm,epoch,steps,curve[-1],flush=True)
        assert grad=={"down.weight","up.weight"}
        blob=packet(aux);(out/(arm+".G17R")).write_bytes(blob)
        q=unpack(blob)
        size=sum(map(len,blobs))+len(blob)+tb
        bpp=8*size/G05
        assert bpp<=.7
        # Measure held-out generation using the actual restored Q8 residual;
        # no cached gold DEV prefix used for free generation.
        with install(st,q):
            # Witness exact placement and gain, not a standalone classifier.
            r=train[0];zz=forward(st,r["p"],r["features"]).detach()[0]
            expected=z[0]+GAIN*q(x[0]).detach()
            err=float((zz-expected).abs().max())
            assert err<1e-4,err
            after=audit(st,tok,dev,"G17_"+arm)
        result={"steps":steps,"epochs":EPOCHS,"cached_TRAIN_tokens":len(y),
                "first_epoch_CE":curve[0],"last_epoch_CE":curve[-1],
                "residual_bytes":len(blob),"weights_tokenizer_plus_residual_bpp":bpp,
                "residual_sha256":digest(blob),"runtime_witness_max_diff":err,
                "after_physical_reload":after}
        results[arm]=result
        print("G17_ARM",arm,json.dumps({**result,"after_physical_reload":after["summary"]},ensure_ascii=False,sort_keys=True),flush=True)
    report={"stage":"G17_CENTERED_NATIVE_RESIDUAL","geometry":G05,"TRAIN_unique":144,"DEV_unique":48,
            "base_sha256":PIN,"base_frozen":True,"train_centered_rms":float(rms),
            "train_centered_gain":float(scale),"baseline":baseline,"arms":results,
            "candidate_accepted":False,"teacher_calls":0,
            "limitations":["G05 frozen, parallel diagnostic to unfinished G16, same TRAIN/DEV corpus, not independent evaluation.",
              "Two matched arms add a rank8 full-vocabulary neural residual, no hardcoded output or routing. Centering stats derived from TRAIN only, stored FP16.",
              "Caching teacher-forced TRAIN states does not itself establish generalization; actual free greedy DEV generation after quantized reload is decisive.",
              "Both use original base head plus residual. All known stored payload bytes charged; extra runtime implementation and RAM excluded.",
              "No historical 616 replay, deployment or CPU certificate; no G14 fitting."]}
    (out/"G17_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G17_ASSESS",json.dumps({"candidate_accepted":False,"stage":report["stage"]}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
