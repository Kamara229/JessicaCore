"""G38: test whether early causal representations restore letter identity.
Frozen actual G36 Joint core/GT/L6/head/slot, new compact attention readout.
GPU learns on cached real native features; CPU free decode recomputes features.
No gold tokens, answer lookup, source token IDs or masks supplied to the readout.
"""
import argparse,json,os,random,time,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load,save
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import Residual,install,parameters,PIN,restore,digest,G05,block
from research.calibri2.fusion_growth.g36_alphabet_margin import suite,encode
from research.calibri2.fusion_growth.g32_word_foundation_mps import LETTERS
from v176.operators import clear
PARENT="fa74bd1f69c3c9817e43a841b166fb1172d8daab270e00436bba84e0442db4e8"
SEED=20270124
EPOCHS=48
BATCH=32

class Readout(nn.Module):
    def __init__(self,contextual):
        super().__init__();self.contextual=contextual
        self.q=nn.Linear(320,24,bias=False);self.k=nn.Linear(320,24,bias=False)
        self.v=nn.Linear(320,32,bias=False);self.f=nn.Linear(320,24,bias=False)
        self.out=nn.Linear(56,320,bias=False)
        self.slot=nn.Parameter(torch.randn(2,24)*.02)
        self.pos=nn.Parameter(torch.zeros(2))
        self.logscale=nn.Parameter(torch.tensor(math.log(12.)))
        nn.init.zeros_(self.out.weight)
    def forward(self,early,last,valid,weight):
        # early B,T,320; last B,2,320 (slot2 residual from same core pass).
        early=F.layer_norm(early,(320,));last=F.layer_norm(last,(320,))
        q=self.q(last)+self.slot[None,:,:];k=self.k(early);v=self.v(early)
        scores=torch.einsum("bsd,btd->bst",q,k)/math.sqrt(24)
        length=valid.sum(1).clamp_min(1)
        rel=(torch.arange(early.shape[1],device=early.device)[None,:]-length[:,None]+1).float()/length[:,None]
        scores=scores+self.pos[None,:,None]*rel[:,None,:]
        scores=scores.masked_fill(~valid[:,None,:],-1.e4)
        context=torch.einsum("bst,btd->bsd",scores.softmax(-1),v)
        if not self.contextual:context=context*0
        delta=self.out(F.gelu(torch.cat((context,self.f(last)),dim=-1)))
        # Reuse the frozen native tied output geometry, no new 49k matrix.
        return F.linear(delta,weight)*self.logscale.exp().clamp(max=80.)/math.sqrt(320)

def capture(st,p,features,slot):
    model=st[0];early=[];last=[]
    def layer0(_m,args):early.append(args[0].detach().clone())
    def head(_m,args):last.append(args[0][:,-1:,:].detach().clone())
    # Native layer0 input carries actual source embeddings + causal prefix.
    # First readout call captures final hidden; the second captures slot2.
    h0=model.model.layers[0].register_forward_pre_hook(layer0)
    h1=model.lm_head.register_forward_pre_hook(head)
    try:z,_=block(st,p,features,slot)
    finally:h0.remove();h1.remove()
    assert len(early)==1 and len(last)==2 and early[0].shape==(1,len(p),320)
    return early[0][0],torch.cat(last,dim=1)[0],z.detach()[0]

def pack(rows):
    n=len(rows);m=max(r[0].shape[0] for r in rows)
    e=torch.zeros(n,m,320);h=torch.stack([r[1] for r in rows]);z=torch.stack([r[2] for r in rows])
    mask=torch.zeros(n,m,dtype=torch.bool);y=torch.full((n,2),-100,dtype=torch.long)
    for i,(a,_,_,g) in enumerate(rows):
        e[i,:len(a)]=a;mask[i,:len(a)]=True;y[i,:len(g)]=torch.tensor(g)
    return e,h,mask,z,y

def evaluate(st,tok,rows,slot,net,weight):
    result={};records=[]
    for mode in ("sequential","parallel"):
        d={"n":len(rows),"exact_EOS":0,"EOS":0,"empty":0,"invalid":0,"first_correct":0,"overrun":0}
        for r in rows:
            ids=[]
            while len(ids)<8:
                e,h,z=capture(st,r["p"]+ids,r["features"],slot)
                with torch.no_grad():
                    correction=net(e[None],h[None],torch.ones((1,len(e)),dtype=torch.bool),weight)[0]
                    choices=(z+correction).argmax(-1).tolist()
                for t in choices[:1 if mode=="sequential" else 2]:
                    if len(ids)>=8:break
                    ids.append(t)
                    if t==2:break
                if ids[-1]==2:break
            a=tok.decode(ids,skip_special_tokens=True).strip()
            for k,v in {"exact_EOS":a==r["answer"] and ids[-1]==2,"EOS":ids[-1]==2,"empty":not a,"invalid":"\ufffd" in a,"first_correct":ids[0]==r["g"][0],"overrun":len(ids)==8 and ids[-1]!=2}.items():d[k]+=int(v)
            records.append({"mode":mode,"q":r["q"],"expected":r["answer"],"answer":a,"ids":ids})
        result[mode]=d
    return result,records

def run(g05,f08,parent,out,arm):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.manual_seed(SEED);torch.set_num_threads(2);rng=random.Random(SEED)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train,dev=suite(tok)
    assert len(train)==132 and len(dev)==66
    # These prompt forms are not in G34/G36 training. They are diagnostics,
    # not independently blinded semantic evaluation.
    fresh=[encode(tok,"Воспроизведи написанный символ «"+a+"».",a) for a in LETTERS]
    assert not {r["q"] for r in train}&{r["q"] for r in fresh+dev}
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0];model,sources,_,full,_,child,_=st
    packet=(parent/"letter_joint.safetensors").read_bytes();assert digest(packet)==PARENT
    raw=load(packet);l6=Residual();l6.load_state_dict({k[3:]:v for k,v in raw.items() if k.startswith("l6.")},strict=True)
    slot=Residual();slot.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("slot.")},strict=True)
    child.language_head.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("head.")},strict=True)
    with torch.no_grad():
        for k,v in raw.items():
            if k.startswith("core."):sources[k[5:]].centroids.copy_(v.float())
            if k=="GT.gates":full.gates.copy_(v.float())
        weight=model.lm_head.source().detach().clone()
        clear(sources)
        weight=F.normalize(weight.float(),dim=-1)
    assert weight.shape==(49152,320)
    frozen=list({id(p):p for p in list(parameters(restored))+list(l6.parameters())+list(slot.parameters())}.values())
    versions={id(p):p._version for p in frozen}
    out.mkdir(parents=True,exist_ok=False)
    cached=[]
    with install(st,6,l6):
        for i,r in enumerate(train):
            for k in range(len(r["g"])):
                e,h,z=capture(st,r["p"]+r["g"][:k],r["features"],slot)
                cached.append((e,h,z,r["g"][k:k+2]))
            if (i+1)%33==0:print("G38_CAPTURE",arm,i+1,len(cached),flush=True)
    assert all(p._version==versions[id(p)] for p in frozen)
    # Real native source features and native logits are constant through this
    # run; all new learned attention and readout gradients run on MPS.
    e,h,valid,z,y=[x.to("mps") for x in pack(cached)];del cached
    net=Readout(arm=="context").to("mps");w=weight.to("mps")
    opt=torch.optim.AdamW(net.parameters(),lr=.003,foreach=False,weight_decay=.0001)
    curve=[];seen=set();start=time.perf_counter()
    for ep in range(1,EPOCHS+1):
        order=list(range(len(e)));rng.shuffle(order);losses=[];correct=0;count=0
        for i in range(0,len(order),BATCH):
            ix=torch.tensor(order[i:i+BATCH],device="mps")
            base=z[ix];target=y[ix];pred=base+net(e[ix],h[ix],valid[ix],w)
            p=pred.reshape(-1,49152);t=target.reshape(-1);ok=t!=-100
            p=p[ok];t=t[ok];gold=p.gather(1,t[:,None]).squeeze(1)
            wrong=p.clone();wrong.scatter_(1,t[:,None],-torch.inf)
            rank=F.softplus(1.5+wrong.max(1).values-gold)
            ce=F.cross_entropy(p,t,reduction="none")
            # Equal weight for EOS and symbol at their actual causal prefix;
            # learn both rather than introduce inference-time output masks.
            loss=(ce+rank).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            for name,par in net.named_parameters():
                if par.grad is not None:
                    assert par.grad.device.type=="mps" and torch.isfinite(par.grad).all()
                    if float(par.grad.abs().sum())>0:seen.add(name)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            losses.append(float(loss.detach()));correct+=int((p.detach().argmax(1)==t).sum());count+=len(t)
        if ep==1 or ep%8==0:
            item={"epoch":ep,"loss":sum(losses)/len(losses),"teacher_forced_train_top1":correct,"train_targets":count,"seconds":time.perf_counter()-start}
            curve.append(item);print("G38_EPOCH",arm,json.dumps(item),flush=True)
            checkpoint=save({k:v.detach().cpu().half().contiguous() for k,v in net.state_dict().items()})
            (out/f"epoch_{ep}.safetensors").write_bytes(checkpoint)
    assert {"out.weight","f.weight"}<=seen
    if arm=="context":assert {"q.weight","k.weight","v.weight"}<=seen
    final=(out/f"epoch_{EPOCHS}.safetensors").read_bytes()
    net=Readout(arm=="context");net.load_state_dict(load(final),strict=True);net.eval()
    size=sum(map(len,blobs))+sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())+len(packet)+len(final)
    bpp=8*size/G05;assert bpp<=.7,bpp
    with install(st,6,l6):
        score,records=evaluate(st,tok,dev,slot,net,weight)
        new,more=evaluate(st,tok,fresh,slot,net,weight)
    assert all(p._version==versions[id(p)] for p in frozen)
    (out/"readout.safetensors").write_bytes(final)
    (out/"DEV.json").write_text(json.dumps(records+more,ensure_ascii=False))
    report={"arm":arm,"parent_sha":PARENT,"readout_sha":digest(final),"params":sum(p.numel() for p in net.parameters()),"known_payload_bpp":bpp,"train_curve":curve,"dev":score,"new_prompt_diagnostic":new,"gpu_nonzero_gradients":sorted(seen),"candidate_accepted":False,"limitations":["Cached native CPU features; all new readout learning on MPS, no core update this diagnostic.","Both arms preserve actual G36 core/GT/L6/head/slot. Final-only control zeros context, otherwise same architecture. Full native vocabulary and free decode; no source ID copy or inference mask.","33 shared letter identities with alternate prompt forms; historical J7 and fixed-work CPU certification not tested."]}
    (out/"G38_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2));print("G38_REPORT",json.dumps(report),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--arm",choices=("context","local"),required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out,a.arm)
