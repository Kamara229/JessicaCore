"""G21: continue G18 from its actual quantized residual.
Matched warm-start control vs learned causal attention over all early
token states. Full vocabulary CE including EOS. No answer copying rules,
runtime token masks, public teacher, or test training.
Prepared while G16 runs; execute only after current comparison artifacts.
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
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import digest,native
from research.calibri2.fusion_growth.g05_134m_joint import restore,G05
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g02_attention_discrete import forward
from research.calibri2.fusion_growth.f21_native_cpu_integration import prompt_features
from research.calibri2.fusion_growth.g16_russian_connected_language import data
from research.calibri2.fusion_growth.g14_russian_ladder import suite,normalize
from research.calibri2.fusion_growth import g18_early_context_readout as g18

SEED=20270111
G18_SHA="fb42a068d4c90042bb08d1637b0c1058ebe14a51b9875ffed465827a40e02d4e"
ARMS=("G18_CONTINUED","G18_CONTEXT_ATTENTION")
EPOCHS=12
BATCH=16
RANK=8
GAIN=.25
ROOT=Path(__file__).parent/"corpora"/"g19"

class ContextReadout(nn.Module):
    def __init__(self,base,context):
        super().__init__()
        self.base=base
        self.context=context
        if context:
            self.query=nn.Linear(320,RANK,bias=False)
            self.key=nn.Linear(320,RANK,bias=False)
            self.mix=nn.Parameter(torch.zeros(()))
            nn.init.normal_(self.query.weight,std=.02)
            nn.init.normal_(self.key.weight,std=.02)
    def forward(self,x,mask):
        # x contains ONLY current causal prefix, no current/future target.
        assert x.ndim==3 and mask.shape==x.shape[:2]
        h=F.layer_norm(x.float(),(320,))
        last=h[torch.arange(h.shape[0],device=h.device),mask.sum(1)-1]
        if self.context:
            scores=(self.key(h)*self.query(last)[:,None,:]).sum(-1)/math.sqrt(RANK)
            scores=scores.masked_fill(~mask,-torch.inf)
            pooled=(scores.softmax(-1)[:,:,None]*h).sum(1)
            last=last+torch.tanh(self.mix)*(pooled-last)
        return self.base.up(F.gelu(self.base.down((last-self.base.mean)*self.base.scale)))

def pack(model):
    base=g18.packet(model.base)
    extra=save({k:v.detach().cpu().half().contiguous() for k,v in model.state_dict().items() if not k.startswith("base.")}) if model.context else b""
    return b"G21R"+struct.pack("<II",len(base),len(extra))+base+extra

def unpack(raw):
    assert raw[:4]==b"G21R"
    n,m=struct.unpack_from("<II",raw,4)
    assert len(raw)==12+n+m
    obj=ContextReadout(g18.unpack(raw[12:12+n]),bool(m))
    if m:
        extra=load(raw[12+n:]);assert set(extra)=={"query.weight","key.weight","mix"}
        with torch.no_grad():
            for k,p in obj.named_parameters():
                if k in extra:p.copy_(extra[k].float())
    return obj.eval()

@contextmanager
def install(st,model):
    layer=st[0].model.layers[0]
    head=st[5].language_head
    pending=[]
    def capture(_m,_args,out):
        assert not pending
        pending.append(out[0] if isinstance(out,tuple) else out)
    def readout(_m,args,out):
        assert len(pending)==1
        x=pending.pop()
        mask=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
        delta=model(x,mask)[:,None,:]
        # Only last logits are consumed by the autoregressive caller.
        assert out.shape[0]==delta.shape[0] and out.shape[-1]==49152
        return out+torch.cat((torch.zeros_like(out[:,:-1,:]),delta),dim=1)
    a=layer.register_forward_hook(capture)
    b=head.register_forward_hook(readout)
    try:yield
    finally:a.remove();b.remove();pending.clear()

def rows(tok):
    def read(split):
        items=[]
        for line in (ROOT/(split+".jsonl")).read_text(encoding="utf8").splitlines():
            r=json.loads(line);assert r["split"]==split
            p,g=native(tok,r["q"],r["answer"])
            assert tok.decode(g[:-1],skip_special_tokens=True).strip()==r["answer"]
            assert len(p)+max(64,len(g))<160
            items.append({**r,"p":p,"g":g,"features":prompt_features(tok,r["q"]),"cohort":"G19"})
        return items
    tr19,dv19=read("TRAIN"),read("DEV")
    tr16=[{**r,"cohort":"G16"} for v in data(tok,"TRAIN").values() for r in v]
    dv16=[{**r,"cohort":"G16"} for v in data(tok,"DEV").values() for r in v]
    tr=tr19+tr16;dv=dv19+dv16
    assert len(tr)==336 and len(dv)==112
    assert len({r["q"] for r in tr})==336
    assert len({r["q"] for r in dv})==112
    assert not {r["q"] for r in tr}&({r["q"] for r in dv}|{r["q"] for r in suite()})
    return tr,dv

def cache(st,train):
    hidden=[];logits=[];targets=[];weights=[]
    layer=st[0].model.layers[0]
    for r in train:
        assert r["split"]=="TRAIN"
        for j,t in enumerate(r["g"]):
            cap=[]
            handle=layer.register_forward_hook(lambda _m,_a,out:cap.append((out[0] if isinstance(out,tuple) else out)[0].detach().clone()))
            try:z=forward(st,r["p"]+r["g"][:j],r["features"]).detach()[0].clone()
            finally:handle.remove()
            assert len(cap)==1 and cap[0].shape==(len(r["p"])+j,320)
            assert torch.isfinite(cap[0]).all() and torch.isfinite(z).all()
            # FP16 cache for training only; runtime takes actual FP32 states.
            hidden.append(cap[0].half().contiguous());logits.append(z)
            targets.append(t);weights.append(.7 if t==2 else 1./max(1,len(r["g"])-1))
        if len(hidden)%256<len(r["g"]):print("G21_CACHE",len(hidden),flush=True)
    return hidden,torch.stack(logits),torch.tensor(targets),torch.tensor(weights)

def collate(hidden,ix):
    vv=[hidden[int(i)] for i in ix]
    x=nn.utils.rnn.pad_sequence(vv,batch_first=True).float()
    mask=torch.arange(x.shape[1])[None,:]<torch.tensor([len(v) for v in vv])[:,None]
    return x,mask

def audit(st,tok,dev,model,tag,out):
    groups=defaultdict(list);pairs=defaultdict(list)
    with install(st,model),(out/(tag+".jsonl")).open("w",encoding="utf8") as f:
        for r in dev:
            ids=[]
            for _ in range(64):
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
    print("G21_AUDIT",tag,json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
    return result

def run(g05,g18dir,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    report=json.loads((g18dir/"G18_REPORT.json").read_text())
    old=(g18dir/"CENTERED_EARLY_L0.G17R").read_bytes()
    assert report["base_sha256"]==PIN and digest(old)==G18_SHA==report["arms"]["CENTERED_EARLY_L0"]["residual_sha256"]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train,dev=rows(tok)
    out.mkdir(parents=True,exist_ok=False)
    st,*rest=restore(blobs[:6],blobs[6],blobs[7])
    baseline=ContextReadout(g18.unpack(old),False)
    r=train[0]
    with g18.install(st,g18.unpack(old)):
        expected=forward(st,r["p"],r["features"]).detach()
    with install(st,baseline):
        actual=forward(st,r["p"],r["features"]).detach()
    assert float((expected-actual).abs().max())<1e-4
    before=audit(st,tok,dev,baseline,"G18_BASE",out)
    h,z,y,w=cache(st,train)
    rng=np.random.default_rng(SEED)
    orders=[rng.permutation(len(y)) for _ in range(EPOCHS)]
    tb=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    results={}
    for arm in ARMS:
        torch.manual_seed(SEED)
        obj=ContextReadout(g18.unpack(old),arm.endswith("ATTENTION"))
        # Initialization with zero mixing gate is the original G18 readout.
        x,mask=collate(h,[0,1,2,3])
        with torch.no_grad():
            assert torch.equal(obj(x,mask),baseline(x,mask))
        opt=torch.optim.AdamW(obj.parameters(),lr=.004,weight_decay=.0001)
        curve=[];steps=0
        for epoch,order in enumerate(orders,1):
            losses=[]
            for j in range(0,len(order),BATCH):
                ix=torch.from_numpy(order[j:j+BATCH].copy()).long()
                x,mask=collate(h,ix)
                prediction=z[ix]+GAIN*obj(x,mask)
                ce=F.cross_entropy(prediction,y[ix],reduction="none")
                loss=(ce*w[ix]).sum()/w[ix].sum()
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(obj.parameters(),5.,error_if_nonfinite=True);opt.step()
                losses.append(float(loss.detach()));steps+=1
            curve.append(float(np.mean(losses)))
            print("G21_PROGRESS",arm,epoch,steps,curve[-1],flush=True)
        raw=pack(obj);(out/(arm+".G21R")).write_bytes(raw)
        q=unpack(raw)
        size=sum(map(len,blobs))+len(raw)+tb
        assert 8*size/G05<=.7
        after=audit(st,tok,dev,q,arm,out)
        results[arm]={"steps":steps,"TRAIN_tokens":len(y),"first_epoch_CE":curve[0],"last_epoch_CE":curve[-1],
             "stored_bytes":size,"known_payload_bpp":8*size/G05,"residual_sha256":digest(raw),
             "mix":float(q.mix.detach()) if q.context else 0.,"scores":after}
    (out/"G21_REPORT.json").write_text(json.dumps({"stage":"G21_G18_CONTEXT_ATTENTION",
        "baseline":before,"arms":results,"TRAIN_unique":336,"DEV_unique":112,"parent_sha256":G18_SHA,
        "candidate_accepted":False,"teacher_calls":0,
        "limitations":["G05 frozen; G18 continuation only. New attention is causal and learned, not a token-output mask.",
        "Same warm-start, schedule and datasets; context arm adds 5121 trainable parameters; no claim of identical FLOPs.",
        "FP16 TRAIN activation cache; physical Q8+FP16 residual reload and full native free generation on DEV.",
        "G16 DEV previously viewed, G19 DEV template-based. Whole first word is a stricter diagnostic than shared first byte, not semantic correctness.",
        "No historical 616 replay, CPU or complete runtime certification; no deployment."]},ensure_ascii=False,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","g18","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.g18,a.f08,a.out)
