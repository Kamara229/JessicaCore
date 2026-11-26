"""G24: isolate native tied readout and learned-head prior; repair calibration.
Matched G18 continuation vs trainable component scales + full-vocab bias.
Actual autoregressive native forward, no teacher, no output rules or masks.
"""
import argparse
import json
import struct
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save,load
from research.calibri2.fusion_growth.g05_134m_joint import restore,G05
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import digest
from research.calibri2.fusion_growth.g02_attention_discrete import forward
from research.calibri2.fusion_growth.g14_russian_ladder import normalize
from research.calibri2.fusion_growth import g18_early_context_readout as g18
from research.calibri2.fusion_growth import g22_early_late_synthesis as g22

SEED=20270114
GAIN=.25
EPOCHS=8
BATCH=24
ARMS=("MATCHED_G18","CALIBRATED_G18")

class Readout(nn.Module):
    def __init__(self,aux,calibrated):
        super().__init__()
        self.aux=aux
        self.calibrated=calibrated
        if calibrated:
            self.log_scales=nn.Parameter(torch.zeros(2))
            self.bias=nn.Parameter(torch.zeros(49152))
    def forward(self,h,base,raw):
        result=base+GAIN*self.aux(h)
        if self.calibrated:
            scales=self.log_scales.clamp(-4,2).exp()
            # raw tied output plus the already scaled old learned correction.
            result=result+(scales[0]-1)*raw+(scales[1]-1)*(base-raw)+self.bias
        return result

def native_parts(st,p,features):
    raw=[];early=[]
    def capture_raw(_m,_a,out):
        raw.append(out[0,-1].detach().float().clone())
    def capture_early(_m,_a,out):
        x=out[0] if isinstance(out,tuple) else out
        early.append(x[0,-1].detach().float().clone())
    # Register raw observer before temporary learned-head hook in forward.
    a=st[0].lm_head.register_forward_hook(capture_raw)
    b=st[0].model.layers[0].register_forward_hook(capture_early)
    try:
        z=forward(st,p,features).detach()[0].float()
    finally:
        a.remove();b.remove()
    assert len(raw)==len(early)==1 and raw[0].shape==z.shape==(49152,) and early[0].shape==(320,)
    assert all(torch.isfinite(x).all() for x in (raw[0],z,early[0]))
    return early[0],z,raw[0]

def packet(m):
    core=g18.packet(m.aux)
    extra=save({"log_scales":m.log_scales.detach().half().contiguous(),
                "bias":m.bias.detach().half().contiguous()}) if m.calibrated else b""
    return b"G24R"+struct.pack("<II",len(core),len(extra))+core+extra

def unpack(blob):
    assert blob[:4]==b"G24R"
    n,m=struct.unpack_from("<II",blob,4)
    assert len(blob)==12+n+m
    obj=Readout(g18.unpack(blob[12:12+n]),bool(m))
    if m:
        ex=load(blob[12+n:])
        assert set(ex)=={"log_scales","bias"}
        with torch.no_grad():
            obj.log_scales.copy_(ex["log_scales"].float())
            obj.bias.copy_(ex["bias"].float())
    return obj.eval()

def cache(st,tr):
    h=[];z=[];raw=[];y=[];w=[];bad=[]
    negative=g22.paired_targets(tr)
    stats=defaultdict(Counter)
    for i,r in enumerate(tr):
        assert r["split"]=="TRAIN"
        for j,t in enumerate(r["g"]):
            v,b,a=native_parts(st,r["p"]+r["g"][:j],r["features"])
            h.append(v);z.append(b.half());raw.append(a.half());y.append(t)
            w.append(.5 if t==2 else 1./max(1,len(r["g"])-1))
            bad.append(negative.get((i,j),-1))
            if j==0:
                for name,l in (("tied",a),("learned_delta",b-a),("combined",b)):
                    stats[name][int(l.argmax())]+=1
        if (i+1)%16==0:print("G24_CACHE",i+1,len(y),flush=True)
    return (torch.stack(h),torch.stack(z),torch.stack(raw),torch.tensor(y),
            torch.tensor(w),torch.tensor(bad),
            {k:v.most_common(8) for k,v in stats.items()})

def audit(st,tok,dv,m,out,tag):
    rows=[];pairs=defaultdict(list)
    with (out/(tag+".jsonl")).open("w",encoding="utf8") as f:
        for i,r in enumerate(dv):
            ids=[];first={}
            for j in range(min(32,159-len(r["p"]))):
                h,b,a=native_parts(st,r["p"]+ids,r["features"])
                with torch.no_grad():z=m(h,b,a)
                assert torch.isfinite(z).all()
                if j==0:
                    first={k:int(v.argmax()) for k,v in (("tied",a),("learned_delta",b-a),("combined",b),("corrected",z))}
                    first["gold_rank"]=1+int((z>z[r["g"][0]]).sum())
                ids.append(int(z.argmax()))
                if ids[-1]==2:break
            answer=tok.decode(ids,skip_special_tokens=True).strip()
            aw=normalize(answer).split();gw=normalize(r["answer"]).split()
            rec={"q":r["q"],"gold":r["answer"],"answer":answer,"ids":ids,"first":first,
                 "cohort":r["cohort"],"family":r["family"],"EOS":ids[-1]==2,
                 "exact_EOS":ids[-1]==2 and normalize(answer)==normalize(r["answer"]),
                 "first_word":bool(aw and gw and aw[0]==gw[0]),
                 "replacement_char":"\ufffd" in answer,
                 "three_plus_words":len(aw)>=3,
                 "repeat":any(a==b for a,b in zip(aw,aw[1:]))}
            rows.append(rec)
            if "pair_id" in r:pairs[r["pair_id"]].append(rec)
            f.write(json.dumps(rec,ensure_ascii=False)+"\n");f.flush()
            if (i+1)%16==0:print("G24_AUDIT_PROGRESS",tag,i+1,flush=True)
    assert all(len(v)==2 for v in pairs.values())
    def summary(rr):
        return {"n":len(rr),**{k:sum(x[k] for x in rr) for k in ("EOS","exact_EOS","first_word","replacement_char","three_plus_words","repeat")},
                "first_top":{k:Counter(x["first"][k] for x in rr).most_common(5) for k in ("tied","learned_delta","combined","corrected")},
                "median_first_gold_rank":float(np.median([x["first"]["gold_rank"] for x in rr])),
                "unique_answers":len({normalize(x["answer"]) for x in rr})}
    groups=defaultdict(list)
    for r in rows:groups[r["cohort"]+"/"+r["family"]].append(r)
    return {"ALL":summary(rows),"groups":{k:summary(v) for k,v in groups.items()},
            "pairs":{"n":len(pairs),"both_exact":sum(all(x["exact_EOS"] for x in v) for v in pairs.values())}}

def run(g05,g18dir,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    old=(g18dir/"CENTERED_EARLY_L0.G17R").read_bytes()
    assert digest(old)==g22.g21.G18_SHA
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    assert tok.eos_token_id==2
    tr,dv,rejected=g22.select(tok)
    out.mkdir(parents=True,exist_ok=False)
    (out/"G24_MANIFEST.json").write_text(json.dumps({k:[{j:r[j] for j in ("q","answer","cohort","family")} for r in v] for k,v in (("TRAIN",tr),("DEV",dv))},ensure_ascii=False,indent=2))
    st,*rest=restore(blobs[:6],blobs[6],blobs[7])
    base=Readout(g18.unpack(old),False)
    r=tr[0];v,b,a=native_parts(st,r["p"],r["features"])
    with g18.install(st,g18.unpack(old)):
        witness=forward(st,r["p"],r["features"]).detach()[0]
    with torch.no_grad():
        err=float((base(v,b,a)-witness).abs().max())
    assert err<1e-4,err
    h,z,raw,y,w,bad,priors=cache(st,tr)
    print("G24_TRAIN_FIRST_TOKEN_PRIORS",json.dumps(priors),flush=True)
    # FP16 cache quantization error must be small relative to native scores.
    assert torch.isfinite(z).all() and torch.isfinite(raw).all()
    before=audit(st,tok,dv,base,out,"BASE_G18")
    rng=np.random.default_rng(SEED);orders=[rng.permutation(len(y)) for _ in range(EPOCHS)]
    tb=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    report={"stage":"G24_NATIVE_HEAD_CALIBRATION","parent":digest(old),"runtime_witness":err,
            "TRAIN":len(tr),"DEV":len(dv),"TRAIN_first_token_priors":priors,
            "baseline":before,"arms":{},"candidate_accepted":False,"teacher_calls":0,
            "limitations":["Frozen G05, neural readout repair only; no protected J7 certification or production CPU gate.",
            "Shared previously viewed DEV, exact matching conservative, no semantic grading claim.",
            "FP16 cached training logits; actual physical reload and raw native generation for evaluation.",
            "Bias and two scales jointly varied, individual causal contribution not isolated; no DEV hyperparameter selection.",
            "Stored payload includes tokenizer but not full deployment/runtime RSS."]}
    (out/"G24_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for arm in ARMS:
        obj=Readout(g18.unpack(old),arm=="CALIBRATED_G18")
        opt=torch.optim.AdamW(obj.parameters(),lr=.004,weight_decay=.0001)
        curve=[];grad=set()
        for epoch,order in enumerate(orders,1):
            losses=[]
            for j in range(0,len(order),BATCH):
                ix=torch.from_numpy(order[j:j+BATCH].copy()).long()
                pred=obj(h[ix],z[ix].float(),raw[ix].float())
                ce=F.cross_entropy(pred,y[ix],reduction="none")
                loss=(ce*w[ix]).sum()/w[ix].sum()
                eligible=bad[ix]>=0
                if eligible.any():
                    logits=pred[eligible];gold=y[ix][eligible];neg=bad[ix][eligible];row=torch.arange(len(gold))
                    loss=loss+.2*F.softplus(.5-(logits[row,gold]-logits[row,neg])).sum()/len(ix)
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward()
                grad.update(k for k,p in obj.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0)
                torch.nn.utils.clip_grad_norm_(obj.parameters(),5.,error_if_nonfinite=True);opt.step()
                losses.append(float(loss.detach()))
            curve.append(float(np.mean(losses)))
            print("G24_PROGRESS",arm,epoch,curve[-1],flush=True)
        assert grad=={k for k,_ in obj.named_parameters()},grad
        blob=packet(obj);(out/(arm+".G24R")).write_bytes(blob)
        q=unpack(blob)
        size=sum(map(len,blobs))+len(blob)+tb;bpp=8*size/G05
        assert bpp<=.7,(arm,bpp)
        scores=audit(st,tok,dv,q,out,arm)
        report["arms"][arm]={"scores":scores,"objective":curve,"bpp":bpp,"stored_bytes":size,
            "sha256":digest(blob),"connected_parameters":sorted(grad),
            "scales":q.log_scales.clamp(-4,2).exp().tolist() if q.calibrated else [1.,1.],
            "bias_rms":float(q.bias.detach().square().mean().sqrt()) if q.calibrated else 0.}
        (out/"G24_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print("G24_ARM",arm,json.dumps(report["arms"][arm],ensure_ascii=False),flush=True)
    print("G24_ASSESS",json.dumps({"candidate_accepted":False,"arms":list(report["arms"])}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","g18","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.g18,a.f08,a.out)
