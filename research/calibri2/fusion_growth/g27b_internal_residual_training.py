"""G27b: matched native residual training, research only."""
import argparse,json,random,time
from collections import defaultdict,Counter
from contextlib import nullcontext
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import save,load
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g27_internal_residual_preflight import Residual,install,parameters,PIN,restore,digest,forward,strict_chat_source_ids
from research.calibri2.fusion_growth.g05_134m_joint import G05
from research.calibri2.fusion_growth.g14_russian_ladder import normalize
from research.calibri2.fusion_growth import g21_g18_context_attention as g21
SEED=20270113
EPOCHS=4
BATCH=8

def rows(tok):
    # Preflight actual native tokenizer before choosing any training pair.
    # Preserve the original G19 prompts/answers; do not truncate or rewrite.
    # Ineligible pairs are excluded as a unit, before model loading.
    rejected=[]
    def read(split):
        groups=defaultdict(list)
        for line in (g21.ROOT/(split+".jsonl")).read_text(encoding="utf8").splitlines():
            r=json.loads(line);assert r["split"]==split
            groups[r["pair_id"]].append(r)
        items=[]
        for pair,rr in groups.items():
            assert len(rr)==2
            valid=[];errors=[]
            for r in rr:
                p=strict_chat_source_ids(tok,r["q"])
                chat=tok.apply_chat_template([{"role":"user","content":r["q"]}],tokenize=False,add_generation_prompt=True)
                ids=tok.encode(chat+r["answer"],add_special_tokens=False)
                g=ids[len(p):]+[2]
                if ids[:len(p)]!=p or len(ids)<=len(p) or 2 in g[:-1] or len(p)+len(g)>=160 or tok.decode(g[:-1],skip_special_tokens=True).strip()!=r["answer"]:
                    errors.append({"id":r["id"],"prompt_tokens":len(p),"target_tokens":len(g),"reason":"native_prefix_length_or_roundtrip"})
                    continue
                valid.append({**r,"p":p,"g":g,"features":g21.prompt_features(tok,r["q"]),"cohort":"G19"})
            if errors:
                rejected.append({"split":split,"pair_id":pair,"ids":[r["id"] for r in rr],"errors":errors})
            else:
                assert len(valid)==2
                items.extend(valid)
        return items
    tr19,dv19=read("TRAIN"),read("DEV")
    tr16=[{**r,"cohort":"G16"} for v in g21.data(tok,"TRAIN").values() for r in v]
    dv16=[{**r,"cohort":"G16"} for v in g21.data(tok,"DEV").values() for r in v]
    tr=tr19+tr16;dv=dv19+dv16
    assert len({r["q"] for r in tr})==len(tr)
    assert len({r["q"] for r in dv})==len(dv)
    assert not {r["q"] for r in tr}&({r["q"] for r in dv}|{r["q"] for r in g21.suite()})
    print("G22_PREFLIGHT",json.dumps({"G19_TRAIN_eligible":len(tr19),"G19_DEV_eligible":len(dv19),"rejected":rejected},ensure_ascii=False),flush=True)
    return tr,dv,rejected

def select(tok):
    tr,dv,rejected=rows(tok)
    groups=defaultdict(list)
    for r in tr:
        if r["cohort"]=="G19":groups[r["family"]].append(r)
    train=[]
    for k in sorted(groups):
        v=groups[k];assert len(v)%2==0
        train.extend(v[:8])
    for k in ("copy","context","sentence"):
        v=[r for r in tr if r["cohort"]=="G16" and r["family"]==k]
        train.extend(v[::2])
    dev=[r for r in dv if r["cohort"]=="G19"]
    for k in ("copy","context","sentence"):
        dev.extend([r for r in dv if r["cohort"]=="G16" and r["family"]==k][::2])
    assert len(train)>=72 and len(dev)>=24
    assert not {r["q"] for r in train}&{r["q"] for r in dev}
    return train,dev,rejected

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
    print("G22_CONTRAST_ELIGIBLE",len(negative),flush=True)
    return negative


def audit(st,tok,dev,ctx,tag,out):
    rows=[];pairs=defaultdict(list);start=time.monotonic()
    with ctx,(out/(tag+".jsonl")).open("w",encoding="utf8") as f:
        for r in dev:
            ids=[]
            for _ in range(min(32,159-len(r["p"]))):
                z=forward(st,r["p"]+ids,r["features"]).detach()
                assert torch.isfinite(z).all()
                ids.append(int(z.argmax()))
                if ids[-1]==2:break
            ans=tok.decode(ids,skip_special_tokens=True).strip()
            aw=normalize(ans).split();gw=normalize(r["answer"]).split()
            eos=ids[-1]==2
            item={"q":r["q"],"gold":r["answer"],"answer":ans,"ids":ids,"family":r["family"],"cohort":r["cohort"],
                "exact_EOS":eos and normalize(ans)==normalize(r["answer"]),"EOS":eos,
                "first_word_exact":bool(aw and gw and aw[0]==gw[0] and (len(aw)>1 or eos)),
                "replacement_char":"\ufffd" in ans,"three_plus_words":len(aw)>=3,
                "repeated_token_block":any(ids[i:i+k]==ids[i+k:i+2*k]==ids[i+2*k:i+3*k] for k in range(1,5) for i in range(len(ids)-3*k+1))}
            rows.append(item)
            if "pair_id" in r:pairs[r["pair_id"]].append(item)
            f.write(json.dumps(item,ensure_ascii=False)+"\n");f.flush()
    assert all(len(v)==2 for v in pairs.values())
    result={k:sum(x[k] for x in rows) for k in ("exact_EOS","EOS","first_word_exact","replacement_char","three_plus_words","repeated_token_block")}
    result.update(n=len(rows),pairs=len(pairs),both_exact_EOS=sum(all(x["exact_EOS"] for x in v) for v in pairs.values()),different_outputs=sum(normalize(v[0]["answer"])!=normalize(v[1]["answer"]) for v in pairs.values()),unique_answers=len({normalize(x["answer"]) for x in rows}),dominant_first_token=Counter(x["ids"][0] for x in rows).most_common(1)[0],tokens_per_second=sum(len(x["ids"]) for x in rows)/max(.001,time.monotonic()-start))
    print("G27B_AUDIT",tag,json.dumps(result),flush=True)
    return result

def run(g05,f08,out,layer):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train,dev,rejected=select(tok)
    assert all(r["split"]=="TRAIN" and len(r["p"])+len(r["g"])<160 for r in train)
    out.mkdir(parents=True,exist_ok=False)
    (out/"manifest.json").write_text(json.dumps({"TRAIN":train,"DEV":dev,"rejected":rejected},ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x)))
    neg=paired_targets(train)
    frozen=list({id(p):p for p in parameters(restored)}.values())
    for p in frozen:p.grad=None
    versions={id(p):p._version for p in frozen}
    def unchanged():
        assert all(p.grad is None and p._version==versions[id(p)] for p in frozen)
    baseline=audit(st,tok,dev,nullcontext(),"G05_BASE",out)
    unchanged()
    torch.manual_seed(SEED);a=Residual()
    params=(a.up.weight,a.down.weight)
    assert sum(p.numel() for p in params)==5120
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
    ref=forward(st,train[0]["p"],train[0]["features"]).detach().clone()
    rng=random.Random(SEED);orders=[]
    for _ in range(EPOCHS):
        order=list(range(len(train)));rng.shuffle(order);orders.append(order)
    curve=[];steps=0;connected=set()
    with install(st,layer,a):
        assert (forward(st,train[0]["p"],train[0]["features"]).detach()-ref).abs().max()<1e-4
        for epoch,order in enumerate(orders,1):
            total=0.
            for j in range(0,len(order),BATCH):
                batch=order[j:j+BATCH];opt.zero_grad(set_to_none=True)
                for i in batch:
                    r=train[i]
                    for k,t in enumerate(r["g"]):
                        z=forward(st,r["p"]+r["g"][:k],r["features"])
                        loss=F.cross_entropy(z,torch.tensor([t]))/len(r["g"])
                        if (i,k) in neg:
                            loss=loss+.2*F.softplus(.5-(z[0,t]-z[0,neg[i,k]]))
                        assert torch.isfinite(loss)
                        total+=float(loss.detach())
                        grads=torch.autograd.grad(loss/len(batch),params)
                        for n,(p,g) in enumerate(zip(params,grads)):
                            assert torch.isfinite(g).all()
                            if g.abs().sum()>0:connected.add(n)
                            if p.grad is None:p.grad=g.detach().clone()
                            else:p.grad.add_(g.detach())
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();steps+=1
                unchanged()
                if steps%8==0:print("G27B_PROGRESS",layer,epoch,steps,flush=True)
            curve.append(total/len(train))
            print("G27B_EPOCH",layer,epoch,curve[-1],flush=True)
            (out/"progress.json").write_text(json.dumps({"layer":layer,"curve":curve,"steps":steps,"candidate_accepted":False}))
    assert connected=={0,1}
    packet=save({k:v.detach().cpu().half().contiguous() for k,v in a.state_dict().items()},metadata={"layer":str(layer)})
    (out/"adapter.safetensors").write_bytes(packet)
    reloaded=Residual();reloaded.load_state_dict(load(packet),strict=True);reloaded.eval()
    tb=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    size=sum(map(len,blobs))+tb+len(packet);bpp=8*size/G05;assert bpp<=.7
    score=audit(st,tok,dev,install(st,layer,reloaded),"L"+str(layer),out)
    unchanged()
    report={"layer":layer,"baseline":baseline,"score":score,"TRAIN_rows":len(train),"TRAIN_tokens":sum(len(r["g"]) for r in train),"DEV_rows":len(dev),"contrast_tokens":len(neg),"epochs":EPOCHS,"steps":steps,"objective":curve,"known_payload_bpp":bpp,"payload_bytes":size,"adapter_sha256":digest(packet),"frozen_tensors":len(frozen),"candidate_accepted":False,"limitations":["Reused exploratory DEV, no independent holdout or historical616 certification.","Measured generation speed depends on output length; not CPU certification.","Known payload includes tokenizer but excludes executable.","Core frozen; only rank8 adapter trained."]}
    (out/"G27B_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G27B_REPORT",json.dumps(report),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--layer",type=int,choices=(6,12),required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out,a.layer)
