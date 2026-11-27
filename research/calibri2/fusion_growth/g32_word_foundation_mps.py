"""G32a: letters -> syllables -> words on actual G29 L6, MPS native readout.
Full 5000-word TRAIN inventory; bounded first pilot, not full-corpus training.
Preserve both G29 checkpoints; frozen actual CPU core features, MPS head/slot.
"""
import argparse,copy,hashlib,importlib.metadata,json,os,random,re,time
from collections import Counter,defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from safetensors.torch import load,save
from wordfreq import top_n_list
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import (
    Residual,install,parameters,PIN,restore,digest,forward,G05,block,generate)
from research.calibri2.fusion_growth.g28_l6_continuation import ByteValidity
from research.calibri2.fusion_growth.g21_g18_context_attention import prompt_features
from research.calibri2.fusion_growth.g27_internal_residual_preflight import strict_chat_source_ids
from v176.operators import clear
PARENTS={"control":"ae6a2dddc6fc70a6102f3ffc9a4d31a4eb9b49644864361244f56b1ae19c71b6","repair":"3f011a8e19b5f8bc6c06655695c7445230fe626484b09a9d0457ff9093a09ba9"}
SEED=20270118
LETTERS="абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
SYLLABLES="ма мо му ми ме мя на но ну ни не ня ра ро ру ри ре ря са со су си се ся ла ло лу ли ле ля".split()

def row(tok,a,kind,split,variant=0):
    label={"letter":"букву","syllable":"слог","word":"слово"}[kind]
    q=f"Повтори {label} «{a}». Напиши только {label}." if not variant else f"Запиши без изменений: {a}."
    p=strict_chat_source_ids(tok,q)
    chat=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    ids=tok.encode(chat+a,add_special_tokens=False)
    g=ids[len(p):]+[2]
    if ids[:len(p)]!=p or len(g)<2 or 2 in g[:-1] or len(p)+len(g)>=160 or tok.decode(g[:-1],skip_special_tokens=True).strip()!=a:return None
    return {"q":q,"answer":a,"family":kind,"split":split,"p":p,"g":g}

def corpus(tok,out):
    words=[];seen=set()
    for w in top_n_list("ru",30000):
        if re.fullmatch("[а-яё]{2,14}",w) and w not in seen:
            r=row(tok,w,"word","POOL")
            if r is not None:words.append(r);seen.add(w)
        if len(words)==5500:break
    assert len(words)==5500
    # Deterministic partition by word identity; related inflections may overlap.
    rng=random.Random(SEED);rng.shuffle(words)
    train=words[:5000];dev=words[5000:]
    for r in train:r["split"]="TRAIN"
    for r in dev:r["split"]="DEV"
    assert not {r["answer"] for r in train}&{r["answer"] for r in dev}
    def balanced(rows,n):
        groups=defaultdict(list)
        for r in rows:groups[min(10,len(r["answer"]))].append(r)
        out=[]
        while len(out)<n:
            for k in sorted(groups):
                if groups[k] and len(out)<n:out.append(groups[k].pop(0))
        return out
    letters=[row(tok,a,"letter","TRAIN") for a in LETTERS]
    syll=[row(tok,a,"syllable","TRAIN") for a in SYLLABLES]
    assert all(r is not None for r in letters+syll)
    test=[row(tok,a,"letter","DEV",1) for a in LETTERS]
    test+= [row(tok,a,"syllable","DEV",1) for a in SYLLABLES[:12]]
    test+=balanced(dev,48)
    assert all(r is not None for r in test)
    pilot=letters+syll+balanced(train,96)
    dist=importlib.metadata.distribution("wordfreq")
    notice_files=dist.files or []
    # Wheels need not contain repository NOTICE.md. Preserve every bundled
    # attribution/license and complete package metadata, rather than skip it.
    texts=[]
    for p in notice_files:
        name=Path(str(p)).name.lower()
        if name.startswith(("notice","license","licence","copying","authors","copyright")) or name=="metadata":
            f=dist.locate_file(p)
            if f.is_file():
                texts.append("### "+str(p)+"\n"+f.read_text(encoding="utf8",errors="replace"))
    assert texts,"No bundled attribution or package metadata"
    source="\n\n".join(texts)
    (out/"WORDFREQ_NOTICE.md").write_text("Filtered Russian word forms from wordfreq 3.1.1 (Robyn Speer), modified by regex, tokenizer eligibility, shuffle and split. Derived word-list data CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/ ; upstream https://github.com/rspeer/wordfreq .\n\n"+source,encoding="utf8")
    for name,rows in (("TRAIN_WORDS",train),("DEV_WORDS",dev),("PILOT_TRAIN",pilot),("PILOT_DEV",test)):
        (out/(name+".jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
    stat=lambda rr:{"rows":len(rr),"target_tokens":sum(len(r["g"]) for r in rr),"unique_target_tokens":len({t for r in rr for t in r["g"]}),"families":dict(Counter(r["family"] for r in rr))}
    report={"train_inventory":stat(train),"dev_inventory":stat(dev),"pilot_train":stat(pilot),"pilot_dev":stat(test),"source":"wordfreq 3.1.1; filtered surface word forms, not manually validated dictionary lemmas"}
    print("G32_CORPUS",json.dumps(report),flush=True)
    return pilot,test,report

def audit(st,tok,rows,slot=None):
    scores=defaultdict(Counter);items=[]
    for r in rows:
        f=r["features"];cap=min(40,159-len(r["p"]));ids=[];calls=0
        if slot is None:
            for _ in range(cap):
                ids.append(int(forward(st,r["p"]+ids,f).detach().argmax()));calls+=1
                if ids[-1]==2:break
        else:ids,calls=generate(st,r,slot,cap=cap)
        ans=tok.decode(ids,skip_special_tokens=True).strip();eos=ids[-1]==2
        d=scores[r["family"]]
        for k,v in {"n":1,"exact_EOS":ans==r["answer"] and eos,"EOS":eos,"empty":not ans,"invalid":"\ufffd" in ans,"valid_nonempty":bool(ans) and "\ufffd" not in ans,"core_calls":calls,"tokens":len(ids)}.items():d[k]+=int(v)
        items.append({"family":r["family"],"q":r["q"],"gold":r["answer"],"answer":ans,"ids":ids,"calls":calls})
    return {k:dict(v) for k,v in scores.items()},items

def run(g05,f08,parent,out,arm):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.manual_seed(SEED);torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True);utf=ByteValidity(tok)
    train,dev,corpus_report=corpus(tok,out)
    for r in train+dev:r["features"]=prompt_features(tok,r["q"])
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    model,sources,_,full,_,child,_=st
    packet=(parent/"adapter.safetensors").read_bytes();assert digest(packet)==PARENTS[arm]
    l6=Residual();l6.load_state_dict(load(packet),strict=True)
    frozen=list({id(p):p for p in list(parameters(restored))+list(l6.parameters())}.values())
    versions={id(p):p._version for p in frozen}
    for p in frozen:p.grad=None
    w=model.lm_head.source().detach().clone();clear(sources)
    original=copy.deepcopy(child.language_head).eval()
    def readout(h,wd,head):return (F.linear(h,wd)+.25*head(h))[:,0,:]
    hx=[];y1=[];y2=[];s1=[];s2=[];groups=[];errors=[]
    with install(st,6,l6):
        before,raw=audit(st,tok,dev)
        (out/"PARENT_DEV.json").write_text(json.dumps(raw,ensure_ascii=False))
        print("G32_PARENT",arm,json.dumps(before),flush=True)
        for i,r in enumerate(train):
            states=utf.states(r)
            for k,t in enumerate(r["g"]):
                captures=[]
                handle=model.lm_head.register_forward_pre_hook(lambda _m,a:captures.append(a[0][:,-1:,:].detach().clone()))
                try:z=forward(st,r["p"]+r["g"][:k],r["features"]).detach()
                finally:handle.remove()
                assert len(captures)==1 and full.last_forward_coverage==tuple(range(12))
                h=captures[0];err=float((readout(h,w,original)-z).abs().max());assert err<1e-4,err
                errors.append(err);hx.append(h);y1.append(t);s1.append(states[k]);groups.append(r["family"])
                y2.append(r["g"][k+1] if k+1<len(r["g"]) else -100);s2.append(states[min(k+1,len(states)-1)])
            if (i+1)%24==0:print("G32_CAPTURE",arm,i+1,len(hx),flush=True)
    x=torch.cat(hx).to("mps");t1=torch.tensor(y1,device="mps");t2=torch.tensor(y2,device="mps")
    # Byte masks used only in loss, never at generation. Deduplicate by state.
    state_ids={s:i for i,s in enumerate(dict.fromkeys(s1+s2))}
    bank=torch.stack([utf.legal(s) for s in state_ids]).to("mps")
    si1=torch.tensor([state_ids[s] for s in s1],device="mps");si2=torch.tensor([state_ids[s] for s in s2],device="mps")
    wd=w.to("mps");head=copy.deepcopy(original).to("mps");head.requires_grad_(True)
    slot=Residual().to("mps")
    params=list(head.parameters())+list(slot.parameters())
    opt=torch.optim.AdamW(params,lr=.001,foreach=False)
    def loss(ix):
        h=x[ix];z=readout(h,wd,head);z2=readout(slot(h),wd,head)
        a=t1[ix];b=t2[ix];v=bank[si1[ix]];v2=bank[si2[ix]]
        ce=F.cross_entropy(z,a)
        invalid=(z.logsumexp(-1)-z.masked_fill(~v,-1e9).logsumexp(-1)).mean()
        # Explicitly oppose premature EOS on every nonterminal gold position.
        ongoing=(a!=2).float()
        eos=(F.softplus(1.+z[:,2]-z.gather(1,a[:,None]).squeeze(1))*ongoing).sum()/ongoing.sum().clamp_min(1)
        active=(b!=-100).float()
        ce2=(F.cross_entropy(z2,b,reduction="none",ignore_index=-100)*active).sum()/active.sum().clamp_min(1)
        invalid2=((z2.logsumexp(-1)-z2.masked_fill(~v2,-1e9).logsumexp(-1))*active).sum()/active.sum().clamp_min(1)
        return ce+.3*invalid+.5*eos+.5*ce2+.15*invalid2
    rng=random.Random(SEED);curve=[];updates=0;start=time.perf_counter()
    # Cumulative curriculum: no dropping letters/syllables when words enter.
    for stage,allowed,epochs in (("letters",{"letter"},4),("syllables",{"letter","syllable"},4),("words",{"letter","syllable","word"},8)):
        inds=[i for i,k in enumerate(groups) if k in allowed]
        for epoch in range(1,epochs+1):
            rng.shuffle(inds);total=0.
            for j in range(0,len(inds),64):
                ix=torch.tensor(inds[j:j+64],device="mps");opt.zero_grad(set_to_none=True);l=loss(ix)
                assert l.device.type=="mps" and torch.isfinite(l);l.backward()
                assert all(p.grad is not None and p.grad.device.type=="mps" and torch.isfinite(p.grad).all() for p in params)
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();updates+=1;total+=float(l.detach())*len(ix)
            item={"stage":stage,"epoch":epoch,"loss":total/len(inds),"updates":updates};curve.append(item);print("G32_EPOCH",arm,json.dumps(item),flush=True)
            raw=save({"head."+k:v.detach().cpu().half().contiguous() for k,v in head.state_dict().items()}|{"slot."+k:v.detach().cpu().half().contiguous() for k,v in slot.state_dict().items()})
            (out/(stage+"_"+str(epoch)+".safetensors")).write_bytes(raw)
    torch.mps.synchronize();seconds=time.perf_counter()-start
    assert all(p.grad is None and p._version==versions[id(p)] for p in frozen)
    raw=(out/"words_8.safetensors").read_bytes();(out/"word_head_slot.safetensors").write_bytes(raw)
    fields=load(raw);newslot=Residual();newslot.load_state_dict({k[5:]:v for k,v in fields.items() if k.startswith("slot.")},strict=True)
    child.language_head.load_state_dict({k[5:]:v for k,v in fields.items() if k.startswith("head.")},strict=True)
    with install(st,6,l6):
        sequential,seq=audit(st,tok,dev)
        parallel,par=audit(st,tok,dev,newslot)
    (out/"SEQUENTIAL_DEV.json").write_text(json.dumps(seq,ensure_ascii=False))
    (out/"PARALLEL_DEV.json").write_text(json.dumps(par,ensure_ascii=False))
    # Conservative: charge new head in full even though it replaces existing head.
    size=sum(map(len,blobs))+sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())+len(packet)+len(raw)
    bpp=8*size/G05;assert bpp<=.7
    report={"arm":arm,"parent_sha256":PARENTS[arm],"corpus":corpus_report,"parent":before,"sequential":sequential,"parallel":parallel,"max_native_logit_error":max(errors),"gpu_updates":updates,"gpu_seconds":seconds,"device":"mps","known_payload_bpp_conservative":bpp,"curve":curve,"candidate_accepted":False,"limitations":["5000-word inventory, pilot trained only 96 words + letters + syllables; not 5000-word training yet.","CPU captures actual frozen L6 features; GPU trains existing native head and future slot only; no full core GPU yet.","Letter/syllable DEV is prompt paraphrase, word DEV unseen surface forms; no general semantic test or historical616 certification.","No inference masks; FP16 reload, conservative duplicate head storage charged; no fixed-work CPU certification."]}
    (out/"G32_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G32_REPORT",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--arm",choices=tuple(PARENTS),required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out,a.arm)
