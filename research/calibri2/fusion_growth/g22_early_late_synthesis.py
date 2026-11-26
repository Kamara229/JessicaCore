"""G22: combine experimentally useful G-stage mechanisms in one bounded study.
G18 warm start, early causal attention, centered late-state residual in
the SAME rank8 full-vocabulary head, balanced complete answers, paired
counterfactual preference, physical reload, real generation. Matched
continued-G18 control. No teacher, rule outputs, MTP or failed own-error
objective. G05 frozen for this diagnostic.
"""
import argparse
import json
import math
import struct
from collections import Counter,defaultdict
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save,load
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import digest
from research.calibri2.fusion_growth.g05_134m_joint import restore,G05
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g02_attention_discrete import forward
from research.calibri2.fusion_growth.g14_russian_ladder import normalize
from research.calibri2.fusion_growth import g18_early_context_readout as g18
from research.calibri2.fusion_growth import g21_g18_context_attention as g21

SEED=20270112
ARMS=("CONTINUED_G18","HYBRID_EARLY_LATE")
EPOCHS=12
BATCH=24
GAIN=.25
RANK=8

class Hybrid(nn.Module):
    def __init__(self,base,hybrid,mean,scale):
        super().__init__();self.base=base;self.hybrid=hybrid
        if hybrid:
            self.query=nn.Linear(320,RANK,bias=False)
            self.key=nn.Linear(320,RANK,bias=False)
            self.mix=nn.Parameter(torch.zeros(()))
            self.late=nn.Linear(320,RANK,bias=False)
            nn.init.normal_(self.query.weight,std=.02)
            nn.init.normal_(self.key.weight,std=.02)
            nn.init.zeros_(self.late.weight)
            self.register_buffer("late_mean",mean.clone().float())
            self.register_buffer("late_scale",scale.clone().float())
    def forward(self,x,mask,late):
        assert x.ndim==3 and mask.shape==x.shape[:2] and late.shape==(len(x),320)
        h=F.layer_norm(x.float(),(320,))
        last=h[torch.arange(len(h),device=h.device),mask.sum(1)-1]
        if self.hybrid:
            score=(self.key(h)*self.query(last)[:,None,:]).sum(-1)/math.sqrt(RANK)
            score=score.masked_fill(~mask,-torch.inf)
            pooled=(score.softmax(-1)[:,:,None]*h).sum(1)
            last=last+torch.tanh(self.mix)*(pooled-last)
        feature=self.base.down((last-self.base.mean)*self.base.scale)
        if self.hybrid:
            feature=feature+self.late((F.layer_norm(late.float(),(320,))-self.late_mean)*self.late_scale)
        return self.base.up(F.gelu(feature))

def packet(model):
    raw=g18.packet(model.base)
    extra=save({k:v.detach().cpu().half().contiguous() for k,v in model.state_dict().items() if not k.startswith("base.")}) if model.hybrid else b""
    return b"G22R"+struct.pack("<II",len(raw),len(extra))+raw+extra

def unpack(blob):
    assert blob[:4]==b"G22R";n,m=struct.unpack_from("<II",blob,4);assert len(blob)==12+n+m
    base=g18.unpack(blob[12:12+n]);extra=load(blob[12+n:]) if m else {}
    model=Hybrid(base,bool(m),extra.get("late_mean",torch.zeros(320)),extra.get("late_scale",torch.ones(())))
    if m:
        assert set(extra)=={"query.weight","key.weight","mix","late.weight","late_mean","late_scale"}
        with torch.no_grad():
            for k,p in model.named_parameters():
                if k in extra:p.copy_(extra[k].float())
    return model.eval()

@contextmanager
def install(st,model):
    pending=[]
    def cap(_m,_a,out):
        assert not pending;pending.append(out[0] if isinstance(out,tuple) else out)
    def inject(_m,args,out):
        assert len(pending)==1
        early=pending.pop();late=args[0][:,-1,:]
        mask=torch.ones(early.shape[:2],dtype=torch.bool,device=early.device)
        delta=model(early,mask,late)[:,None,:]
        assert delta.shape==(out.shape[0],1,49152)
        return out+torch.cat((torch.zeros_like(out[:,:-1,:]),delta),dim=1)
    a=st[0].model.layers[0].register_forward_hook(cap)
    b=st[5].language_head.register_forward_hook(inject)
    try:yield
    finally:a.remove();b.remove();pending.clear()

def rows(tok):
    def read(split):
        items=[]
        for line in (g21.ROOT/(split+".jsonl")).read_text(encoding="utf8").splitlines():
            r=json.loads(line);assert r["split"]==split
            p,g=g21.native(tok,r["q"],r["answer"])
            assert tok.decode(g[:-1],skip_special_tokens=True).strip()==r["answer"]
            assert len(p)+len(g)<160
            assert len(p)<159
            items.append({**r,"p":p,"g":g,"features":g21.prompt_features(tok,r["q"]),"cohort":"G19"})
        return items
    tr19,dv19=read("TRAIN"),read("DEV")
    tr16=[{**r,"cohort":"G16"} for v in g21.data(tok,"TRAIN").values() for r in v]
    dv16=[{**r,"cohort":"G16"} for v in g21.data(tok,"DEV").values() for r in v]
    tr=tr19+tr16;dv=dv19+dv16
    assert len(tr)==336 and len(dv)==112
    assert len({r["q"] for r in tr})==336
    assert len({r["q"] for r in dv})==112
    assert not {r["q"] for r in tr}&({r["q"] for r in dv}|{r["q"] for r in g21.suite()})
    return tr,dv

def select(tok):
    tr,dv=rows(tok)
    # Keep complete contrast pairs; replay existing Russian word/context/
    # sentence data. Fixed selection, no outcome-dependent sampling.
    groups=defaultdict(list)
    for r in tr:
        if r["cohort"]=="G19":groups[r["family"]].append(r)
    train=[]
    for k in sorted(groups):
        v=groups[k];assert len(v)==24
        train.extend(v[:8])
    for k in ("copy","context","sentence"):
        v=[r for r in tr if r["cohort"]=="G16" and r["family"]==k]
        # Spread across each full family rather than only first templates.
        train.extend(v[::2])
    dev=[r for r in dv if r["cohort"]=="G19"]
    for k in ("copy","context","sentence"):
        dev.extend([r for r in dv if r["cohort"]=="G16" and r["family"]==k][::2])
    assert len(train)==136 and len(dev)==88
    assert not {r["q"] for r in train}&{r["q"] for r in dev}
    return train,dev

def paired_targets(train):
    groups=defaultdict(list);negative={}
    # Use only objectively contradictory target pairs with identical
    # answer prefix up to the divergent native token. No open answer
    # paraphrases labelled wrong, no malformed model-output penalty.
    families={"location","ownership","classification"}
    for i,r in enumerate(train):
        if r["cohort"]=="G19" and r["family"] in families:groups[r["pair_id"]].append((i,r))
    for v in groups.values():
        assert len(v)==2
        (i,a),(j,b)=v
        pos=next((k for k,(x,y) in enumerate(zip(a["g"],b["g"])) if x!=y),None)
        assert pos is not None and a["g"][:pos]==b["g"][:pos]
        # Do not oppose EOS: G08 showed the risk of termination collapse.
        if a["g"][pos]!=2 and b["g"][pos]!=2:
            negative[i,pos]=b["g"][pos];negative[j,pos]=a["g"][pos]
    assert len(negative)>0
    return negative

def cache(st,train):
    x=[];late=[];z=[];y=[];w=[];bad=[];negative=paired_targets(train)
    for i,r in enumerate(train):
        assert r["split"]=="TRAIN"
        for j,t in enumerate(r["g"]):
            early=[];final=[]
            a=st[0].model.layers[0].register_forward_hook(lambda _m,_a,out:early.append((out[0] if isinstance(out,tuple) else out)[0].detach().clone()))
            b=st[0].lm_head.register_forward_pre_hook(lambda _m,args:final.append(args[0][0,-1].detach().clone()))
            try:logit=forward(st,r["p"]+r["g"][:j],r["features"]).detach()[0].clone()
            finally:a.remove();b.remove()
            assert len(early)==len(final)==1 and early[0].shape==(len(r["p"])+j,320) and final[0].shape==(320,)
            assert torch.isfinite(early[0]).all() and torch.isfinite(final[0]).all() and torch.isfinite(logit).all()
            x.append(early[0].half().contiguous());late.append(final[0].half());z.append(logit);y.append(t)
            # Full answer, not only first token. Both arms share objective.
            w.append(.5 if t==2 else 1./max(1,len(r["g"])-1))
            bad.append(negative.get((i,j),-1))
        if (i+1)%16==0:print("G22_CACHE",i+1,len(y),flush=True)
    return x,torch.stack(late).float(),torch.stack(z),torch.tensor(y),torch.tensor(w),torch.tensor(bad)

def audit(st,tok,dev,model,tag,out):
    groups=defaultdict(list);pairs=defaultdict(list)
    with install(st,model),(out/(tag+".jsonl")).open("w",encoding="utf8") as f:
        for r in dev:
            ids=[]
            for _ in range(min(32,159-len(r["p"]))):
                z=forward(st,r["p"]+ids,r["features"]).detach()
                ids.append(int(z.argmax()))
                if ids[-1]==2:break
            a=tok.decode(ids,skip_special_tokens=True).strip()
            aw=normalize(a).split();gw=normalize(r["answer"]).split()
            item={"cohort":r["cohort"],"family":r["family"],"q":r["q"],"gold":r["answer"],"answer":a,"ids":ids,
                  "exact":normalize(a)==normalize(r["answer"]),"EOS":ids[-1]==2,
                  "first_word_exact":bool(aw and gw and aw[0]==gw[0]),
                  "replacement_char":"\ufffd" in a,
                  "adjacent_word_repeat":any(x==y for x,y in zip(aw,aw[1:])),
                  "three_plus_words":len(aw)>=3}
            groups[r["cohort"]+"/"+r["family"]].append(item)
            if "pair_id" in r:pairs[r["pair_id"]].append(item)
            f.write(json.dumps(item,ensure_ascii=False)+"\n");f.flush()
    def stat(rr):
        return {"n":len(rr),"exact_EOS":sum(x["exact"] and x["EOS"] for x in rr),
                "EOS":sum(x["EOS"] for x in rr),"first_word_exact":sum(x["first_word_exact"] for x in rr),
                "replacement_char":sum(x["replacement_char"] for x in rr),
                "adjacent_word_repeat":sum(x["adjacent_word_repeat"] for x in rr),
                "three_plus_words":sum(x["three_plus_words"] for x in rr),
                "dominant_first_token":Counter(x["ids"][0] for x in rr).most_common(1)[0],
                "unique_answers":len({normalize(x["answer"]) for x in rr})}
    result={k:stat(v) for k,v in groups.items()}
    result["ALL"]=stat([x for v in groups.values() for x in v])
    assert len(pairs)==32 and all(len(v)==2 for v in pairs.values())
    result["G19_pairs"]={"n":32,"both_exact_EOS":sum(all(x["exact"] and x["EOS"] for x in v) for v in pairs.values()),
                         "different_outputs":sum(normalize(v[0]["answer"])!=normalize(v[1]["answer"]) for v in pairs.values())}
    print("G22_AUDIT",tag,json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
    return result

def run(g05,g18dir,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    old=(g18dir/"CENTERED_EARLY_L0.G17R").read_bytes()
    prior=json.loads((g18dir/"G18_REPORT.json").read_text())
    assert digest(old)==g21.G18_SHA==prior["arms"]["CENTERED_EARLY_L0"]["residual_sha256"] and prior["base_sha256"]==PIN
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tr,dv=select(tok)
    out.mkdir(parents=True,exist_ok=False)
    (out/"G22_MANIFEST.json").write_text(json.dumps({k:[{j:r[j] for j in ("q","answer","cohort","family")} for r in v] for k,v in (("TRAIN",tr),("DEV",dv))},ensure_ascii=False,indent=2))
    st,*rest=restore(blobs[:6],blobs[6],blobs[7])
    base=Hybrid(g18.unpack(old),False,torch.zeros(320),torch.ones(()))
    r=tr[0]
    with g18.install(st,g18.unpack(old)):
        a=forward(st,r["p"],r["features"]).detach()
    with install(st,base):
        b=forward(st,r["p"],r["features"]).detach()
    assert float((a-b).abs().max())<1e-4
    before=audit(st,tok,dv,base,"G18_BASE",out)
    h,late,z,y,w,bad=cache(st,tr)
    normalized=F.layer_norm(late,(320,));mean=normalized.mean(0).half().float()
    scale=(1/(normalized-mean).square().mean().sqrt().clamp_min(.01)).half().float()
    assert torch.isfinite(scale) and scale>0
    rng=np.random.default_rng(SEED);orders=[rng.permutation(len(y)) for _ in range(EPOCHS)]
    tb=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    results={}
    for arm in ARMS:
        torch.manual_seed(SEED)
        obj=Hybrid(g18.unpack(old),arm.startswith("HYBRID"),mean,scale)
        x,mask=g21.collate(h,[0,1,2,3])
        with torch.no_grad():
            assert torch.equal(obj(x,mask,late[:4]),base(x,mask,late[:4]))
        opt=torch.optim.AdamW(obj.parameters(),lr=.004,weight_decay=.0001)
        curve=[];steps=0;grad=set()
        for epoch,order in enumerate(orders,1):
            losses=[]
            for j in range(0,len(order),BATCH):
                ix=torch.from_numpy(order[j:j+BATCH].copy()).long();x,mask=g21.collate(h,ix)
                pred=z[ix]+GAIN*obj(x,mask,late[ix])
                ce=F.cross_entropy(pred,y[ix],reduction="none")
                loss=(ce*w[ix]).sum()/w[ix].sum()
                eligible=bad[ix]>=0
                if eligible.any():
                    logits=pred[eligible];gold=y[ix][eligible];neg=bad[ix][eligible];row=torch.arange(len(gold))
                    # Margin on first divergent target, not common first byte.
                    loss=loss+.2*F.softplus(.5-(logits[row,gold]-logits[row,neg])).sum()/len(ix)
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward()
                grad.update(k for k,p in obj.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0)
                torch.nn.utils.clip_grad_norm_(obj.parameters(),5.,error_if_nonfinite=True);opt.step()
                steps+=1;losses.append(float(loss.detach()))
            curve.append(float(np.mean(losses)))
            print("G22_PROGRESS",arm,epoch,steps,curve[-1],flush=True)
        assert grad=={k for k,_ in obj.named_parameters()},grad
        raw=packet(obj);(out/(arm+".G22R")).write_bytes(raw)
        q=unpack(raw)
        size=sum(map(len,blobs))+len(raw)+tb;assert 8*size/G05<=.7
        after=audit(st,tok,dv,q,arm,out)
        results[arm]={"steps":steps,"TRAIN_tokens":len(y),"pair_contrast_tokens":int((bad>=0).sum()),
             "first_epoch_objective":curve[0],"last_epoch_objective":curve[-1],
             "stored_bytes":size,"known_payload_bpp":8*size/G05,"base_reference_bpp":8*size/36380480,
             "residual_bytes":len(raw),"residual_sha256":digest(raw),"connected_params":sorted(grad),
             "additional_neural_parameters":sum(p.numel() for p in q.parameters())-RANK*(320+49152),
             "mix":float(q.mix.detach()) if q.hybrid else 0.,"scores":after}
        print("G22_ARM",arm,json.dumps(results[arm],ensure_ascii=False,sort_keys=True),flush=True)
        (out/"G22_REPORT.json").write_text(json.dumps({"stage":"G22_EARLY_LATE_SYNTHESIS","baseline":before,
            "arms":results,"TRAIN_unique":len(tr),"DEV_unique":len(dv),"parent_sha256":g21.G18_SHA,
            "candidate_accepted":False,"teacher_calls":0,
            "limitations":["Two matched arms, same training contrast and schedules, architecture combined so early vs late effects not individually isolated.",
             "G05 core frozen, not joint core certification. G19 DEV template-based; G16 DEV already viewed.",
             "Actual physical Q8 residual and FP16 statistics reload. Known payload includes tokenizer, not executable/runtime RSS.",
             "Whole word and pair exact scores are conservative; no semantic judgement inferred from them.",
             "No MTP, no unsafe own-negative objective, no routing or teacher. Historical616 and CPU gates pending."]},ensure_ascii=False,indent=2))
    print("G22_ASSESS",json.dumps({"arms":list(results),"candidate_accepted":False}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","g18","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.g18,a.f08,a.out)
