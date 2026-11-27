"""G39: frozen native layer-wise symbol identity diagnostics.
Use actual G36 Joint weights; compare all 12 decoder layer outputs plus input
and final normalized hidden. No training of model, no answer injection.
Train read-only probes on TRAIN only on MPS. Test distinct prompt templates.
"""
import argparse,json,os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import Residual,install,parameters,PIN,restore,digest,block
from research.calibri2.fusion_growth.g36_alphabet_margin import suite,encode
from research.calibri2.fusion_growth.g32_word_foundation_mps import LETTERS
from research.calibri2.fusion_growth.g38_context_readout import PARENT
SEED=20270125
DEPTH=14
MODES=("last","mean","attention")

def capture(st,r,slot):
    model=st[0];values={};handles=[]
    def store(i,x):
        assert i not in values and x.ndim==3 and x.shape[0]==1 and x.shape[2]==320
        values[i]=x[0].detach().clone()
    handles.append(model.model.layers[0].register_forward_pre_hook(lambda m,a:store(0,a[0])))
    for i,layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(lambda m,a,o,i=i:store(i+1,o[0] if isinstance(o,tuple) else o)))
    handles.append(model.model.norm.register_forward_hook(lambda m,a,o:store(13,o)))
    try:z,_=block(st,r["p"],r["features"],slot)
    finally:
        for h in handles:h.remove()
    assert set(values)==set(range(DEPTH))
    assert all(v.shape==(len(r["p"]),320) for v in values.values())
    return torch.stack([values[i] for i in range(DEPTH)]),int(z[0,0].argmax()),int(r["g"][0])

def dataset(rows,features):
    m=max(x.shape[1] for x in features);x=torch.zeros(len(rows),DEPTH,m,320);valid=torch.zeros(len(rows),m,dtype=torch.bool)
    for i,a in enumerate(features):x[i,:,:a.shape[1]]=a;valid[i,:a.shape[1]]=True
    y=torch.tensor([LETTERS.index(r["answer"]) for r in rows])
    return x,valid,y

class Probe(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.q=nn.Parameter(torch.randn(DEPTH,320)*.02)
        self.pos=nn.Parameter(torch.zeros(DEPTH))
        self.w=nn.Parameter(torch.randn(DEPTH,320,33)*.02)
        self.b=nn.Parameter(torch.zeros(DEPTH,33))
    def forward(self,x,m):
        x=F.layer_norm(x,(320,))
        if self.mode=="last":
            pos=m.sum(1)-1
            h=x.gather(2,pos[:,None,None,None].expand(-1,DEPTH,1,320)).squeeze(2)
        elif self.mode=="mean":
            h=(x*m[:,None,:,None]).sum(2)/m.sum(1)[:,None,None]
        else:
            scores=torch.einsum("bltd,ld->blt",x,self.q)/320**.5
            position=(torch.arange(x.shape[2],device=x.device)[None,:]-m.sum(1)[:,None]+1)/m.sum(1)[:,None]
            scores=scores+self.pos[None,:,None]*position[:,None,:]
            a=scores.masked_fill(~m[:,None,:],-1.e4).softmax(-1)
            h=torch.einsum("blt,bltd->bld",a,x)
        return torch.einsum("bld,ldc->blc",h,self.w)+self.b

def metrics(pred,y):
    pred=pred.detach().cpu();y=y.cpu()
    return (pred.argmax(-1)==y[:,None]).sum(0).tolist()

def run(g05,f08,parent,out):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.manual_seed(SEED);torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train,dev=suite(tok)
    fresh=[encode(tok,"Воспроизведи написанный символ «"+a+"».",a) for a in LETTERS]
    # New diagnostics with two distractor letters; intended target disambiguated by natural-language instruction.
    contrast=[encode(tok,"Из букв "+LETTERS[(i+7)%33]+" и "+a+" напиши вторую.",a) for i,a in enumerate(LETTERS)]
    sets={"train":train,"dev":dev,"fresh":fresh,"contrast":contrast}
    assert not {r["q"] for r in train}&{r["q"] for key in ("dev","fresh","contrast") for r in sets[key]}
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN);assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0];model,sources,_,full,_,child,_=st
    packet=(parent/"letter_joint.safetensors").read_bytes();assert digest(packet)==PARENT
    raw=load(packet);l6=Residual();l6.load_state_dict({k[3:]:v for k,v in raw.items() if k.startswith("l6.")},strict=True)
    slot=Residual();slot.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("slot.")},strict=True)
    child.language_head.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("head.")},strict=True)
    with torch.no_grad():
        for k,v in raw.items():
            if k.startswith("core."):sources[k[5:]].centroids.copy_(v.float())
            if k=="GT.gates":full.gates.copy_(v.float())
    frozen=list({id(p):p for p in list(parameters(restored))+list(l6.parameters())+list(slot.parameters())}.values())
    versions={id(p):p._version for p in frozen}
    out.mkdir(parents=True,exist_ok=False);data={};native={};separation={}
    with install(st,6,l6):
        for name,rows in sets.items():
            fs=[];correct=0
            for r in rows:
                x,p,g=capture(st,r,slot);fs.append(x);correct+=int(p==g)
            native[name]=correct
            data[name]=dataset(rows,fs)
            # Last-position normalized pairwise distances, same template blocks
            # across different letters; scale-free difference measure.
            x=F.normalize(torch.stack([a[:,-1] for a in fs]),dim=-1)
            distances=1.-torch.einsum("nld,mld->lnm",x,x)
            yy=data[name][2];different=yy[:,None]!=yy[None,:]
            separation[name]=[float(v.median()) for v in distances[:,different]]
            print("G39_CAPTURE",name,len(rows),json.dumps({"first_correct":correct,"median_cosine_distance":separation[name]}),flush=True)
    assert all(p._version==versions[id(p)] for p in frozen)
    device={k:tuple(t.to("mps") for t in v) for k,v in data.items()}
    results={};shuffled={}
    for mode in MODES:
        for control in (False,True):
            torch.manual_seed(SEED);net=Probe(mode).to("mps")
            opt=torch.optim.AdamW(net.parameters(),lr=.005,foreach=False,weight_decay=.01)
            x,m,y=device["train"]
            if control:
                # Permutation of labels, fixed balanced marginal, independent of target.
                y=y[torch.randperm(len(y),generator=torch.Generator().manual_seed(SEED+1)).to("mps")]
            for ep in range(160):
                p=net(x,m);loss=F.cross_entropy(p.reshape(-1,33),y[:,None].expand(-1,DEPTH).reshape(-1))
                opt.zero_grad(set_to_none=True);loss.backward()
                assert all(v.grad is None or v.grad.device.type=="mps" for v in net.parameters())
                opt.step()
            with torch.no_grad():
                r={k:metrics(net(xx,mm),yy) for k,(xx,mm,yy) in device.items()}
                if control:r["train_permuted"]=metrics(net(x,m),y)
            (shuffled if control else results)[mode]=r
            print("G39_PROBE",mode,"shuffled" if control else "real",json.dumps(r),flush=True)
    report={"parent_sha":PARENT,"layers":["embedding"]+[f"L{i}" for i in range(1,13)]+["final_norm"],"n":{k:len(v) for k,v in sets.items()},"native_first_correct":native,"median_cosine_distance":separation,"probe_top1":results,"shuffled_label_control":shuffled,"candidate_accepted":False,"limitations":["Read-only probes, never used in generation. Actual native model frozen. MPS trains probes only.","Shared 33 identities across splits; held-out templates, not unseen letters. Distractor instructions outside training distribution. Probe separability does not prove correct reasoning or causality.","No historical J7 or CPU speed claim. All source checkpoints unchanged."]}
    (out/"G39_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2));print("G39_REPORT",json.dumps(report),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out)
