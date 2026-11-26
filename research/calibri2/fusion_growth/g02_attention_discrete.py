"""G02: real 74.719M logical geometry: third FFN bank and second attention
bank at all twelve layers. Start from actual G01 recovered physical weights.
Matched 48 joint Adam steps at 54M vs 74M; then separate TRAIN-only discrete
index phase on 84 newly added matrices. No routing, no protected fitting.
Native full-prefix research path only: extra attention cache not persisted.
"""
import argparse
import copy
import json
import lzma
import struct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v176.operators import TrainWeight,clear
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import (
    BASE,GROWN,EXTRA,RATE,extra_pack,extra_open,install_growth,payload,audit,
    native,supervised,digest,open_bundle,direct,k_for,unpack_indices,prompt_features)

SEED=20261223
CORE_SHA="5f4c6683287922be5b5b34d55e9f30e4ab5ea688b7b02ccbae2388d021dab8e1"
EXTRA_SHA="2e3c10e400d7fc3fe3345a702ce872e64986636919854540e8771d26bd823839"
ATTN=12*(2*320*320+2*64*320)
G02=GROWN+EXTRA+ATTN
STEPS=48
DISCRETE=84

def rows(tok,split):
    tr=split=="TRAIN";n=12 if tr else 6;result={"math":[],"general":[]}
    for i in range(n):
        a=(29 if tr else 713)+17*i;b=2+i%7;c=3+i%8
        x,y=("Ира","Павел") if tr else ("Роман","Лидия")
        ask=bool(i%2);owner=x if ask else y;other=y if ask else x
        cases=[
            ("math",f"Число {a} увеличили на {b*c}. Чему равен результат?",a+b*c,a-b*c),
            ("math",f"Вычисли {a} + {b} × {c}. Ответ числом.",a+b*c,(a+b)*c),
            ("general",f"Случай {a}: {x} передал подарок человеку {y}. Кто {'передал' if ask else 'получил'} подарок?",owner,other),
            ("general",f"Запись {a}: у {x} {'есть' if ask else 'нет'} книги. Есть ли у {x} книга? Ответь да или нет.","да" if ask else "нет","нет" if ask else "да")]
        for kind,q,y,w in cases:
            y,w=str(y),str(w)
            p,g=native(tok,q,y);pp,bad=native(tok,q,w)
            assert p==pp and g!=bad
            div=next(j for j,(u,v) in enumerate(zip(g,bad)) if u!=v)
            result[kind].append({"q":q,"p":p,"g":g,"bad":bad,"div":div,"answer":y,"wrong":w,
                                 "features":prompt_features(tok,q),"split":split,"kind":kind})
    return result

class TripleMLP(nn.Module):
    def __init__(self,old,gate,up,down):
        super().__init__()
        # old contains original FFN bank and G01 learned second bank.
        self.first=old
        self.third_gate=gate;self.third_up=up;self.third_down=down
        self.intermediate_size=4608
    def forward(self,x):
        old=self.first
        a=old.first(x)
        b=F.linear(old.first.act_fn(F.linear(x,old.second_gate()))*F.linear(x,old.second_up()),old.second_down())
        c=F.linear(old.first.act_fn(F.linear(x,self.third_gate()))*F.linear(x,self.third_up()),self.third_down())
        for s in (old.second_gate,old.second_up,old.second_down,self.third_gate,self.third_up,self.third_down):s.clear()
        return .5*a+.25*(b+c)

class DualAttention(nn.Module):
    def __init__(self,original,secondary):
        super().__init__()
        self.original=original;self.secondary=secondary
    def forward(self,*args,**kwargs):
        # Pinned Llama 4.46 decoder passes cache-related arguments by name.
        # Each call in native research starts from the entire current prefix.
        assert len(args)<=1
        pos=kwargs.get("cache_position")
        if pos is not None:assert int(pos.flatten()[0])==0,"G02 does not implement incremental KV cache"
        a=self.original(*args,**kwargs)
        kw=dict(kwargs);kw["past_key_value"]=None
        kw["use_cache"]=False
        b=self.secondary(*args,**kw)
        for k in ("q_proj","k_proj","v_proj","o_proj"):getattr(self.secondary,k).source.clear()
        assert isinstance(a,tuple) and isinstance(b,tuple) and a[0].shape==b[0].shape
        return (.5*(a[0]+b[0]),*a[1:])

def clone_bank(src):
    src.clear()
    s=copy.deepcopy(src);s.clear()
    assert s.codes.data_ptr()!=src.codes.data_ptr() and s.centroids.data_ptr()!=src.centroids.data_ptr()
    return s

def added_from(state,second):
    sources=state[1]
    result={k:clone_bank(v) for k,v in second.items()}
    result.update({k:clone_bank(v) for k,v in sources.items() if ".self_attn." in k})
    assert len(result)==84
    assert sum(s.shape[0]*s.shape[1] for s in result.values())==EXTRA+ATTN
    return result

def install_added(state,added):
    model,sources=state[:2];clear(sources)
    for i,layer in enumerate(model.model.layers):
        p=f"model.layers.{i}.mlp."
        layer.mlp=TripleMLP(layer.mlp,*[added[p+k+".weight"] for k in ("gate_proj","up_proj","down_proj")])
        original=layer.self_attn
        secondary=copy.deepcopy(original)
        p=f"model.layers.{i}.self_attn."
        for k in ("q_proj","k_proj","v_proj","o_proj"):
            getattr(secondary,k).source=added[p+k+".weight"]
        layer.self_attn=DualAttention(original,secondary)
    assert len(model.model.layers)==12

def open_added(blob,state,second):
    out=added_from(state,second)
    assert blob[:4]==b"G01X"
    raw=lzma.decompress(blob[4:]);n,=struct.unpack_from("<I",raw)
    meta=json.loads(raw[4:4+n]);p=4+n
    assert {m["name"] for m in meta}==set(out)
    for m in meta:
        s=out[m["name"]];assert list(s.shape)==m["shape"] and s.codes.numel()==m["count"]
        items=[]
        for length in m["sizes"]:
            items.append(raw[p:p+length]);p+=length
        ids=unpack_indices(items[0],m["count"],3)
        c=np.frombuffer(items[1],dtype="<f2").astype(np.float32).reshape(8,8)
        sc=np.frombuffer(items[2],dtype="<f2").astype(np.float32)
        with torch.no_grad():
            s.codes.copy_(torch.from_numpy(ids.astype(np.int64)))
            s.centroids.zero_();s.centroids[:8].copy_(torch.from_numpy(c))
            s.base_scales.copy_(torch.from_numpy(sc))
    assert p==len(raw) and extra_pack(out)==blob
    return out

from research.calibri2.fusion_growth.f06_integrated_recovery import forward

def joint_train(state,tok,data,stage,extra=None):
    _,sources,_,full,_,child,region=state
    core=[s.centroids for s in sources.values()]
    ext=[s.centroids for s in extra.values()] if extra else []
    groups=[{"params":core+ext,"lr":.00015},{"params":region,"lr":.0003},{"params":[full.gates],"lr":.0001}]
    opt=torch.optim.Adam(groups);seen=set();xseen=set();gt=set();rg=set();losses=[]
    for i,r in enumerate(data):
        assert r["split"]=="TRAIN"
        opt.zero_grad(set_to_none=True)
        l=supervised(state,tok,r["q"],r["answer"],r["bad"])
        for k,s in sources.items():
            if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(k)
        for k,s in (extra or {}).items():
            if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:xseen.add(k)
        if full.gates.grad is not None:gt.update(j for j,v in enumerate(full.gates.grad) if float(v.abs())>0)
        for k,p in child.named_parameters():
            if p.grad is not None and float(p.grad.abs().sum())>0:rg.add(k)
        torch.nn.utils.clip_grad_norm_(core+ext+region+[full.gates],1.,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            for k,s in sources.items():s.centroids[k_for(RATE,k):].zero_()
            for s in (extra or {}).values():s.centroids[8:].zero_()
        clear(sources)
        losses.append(l)
        if (i+1)%16==0:print("G02_PROGRESS",stage,i+1,l,len(seen),len(xseen),len(gt),flush=True)
    assert len(seen)==85 and len(gt)==12 and len(rg)>0
    if extra:assert len(xseen)==len(extra)
    return {"steps":len(data),"connected_base_codebooks":len(seen),"connected_added_codebooks":len(xseen),
            "connected_GT":len(gt),"connected_region_tensors":len(rg),"first16_objective":float(np.mean(losses[:16])),
            "last16_objective":float(np.mean(losses[-16:]))}


def indexed_banks(second,added):
    return {"second/"+k:s for k,s in second.items()}|{"new/"+k:s for k,s in added.items()}

def first_loss(state,row):
    z=forward(state,row["p"],row["features"])
    target=row["g"][0]
    l=F.cross_entropy(z,torch.tensor([target]))+.15*F.softplus(.5-(z[0,target]-z[0,2]))
    if row["div"]==0:l=l+.5*F.softplus(.5-(z[0,target]-z[0,row["bad"][0]]))
    return l

def discrete_train(state,added,rows):
    # Each added matrix visited exactly once. Independent compact index
    # reassignment, not another update to the same tiny codebook.
    proposed=accepted=accepted_matrices=0;records=[]
    keys=sorted(added);assert len(keys)==DISCRETE
    for i,k in enumerate(keys):
        r=rows[i%len(rows)];assert r["split"]=="TRAIN"
        s=added[k];cap=[]
        def capture(_m,_args,z):
            z.retain_grad();cap.append(z)
        handle=s.register_forward_hook(capture)
        state[0].zero_grad(set_to_none=True)
        try:
            l=first_loss(state,r);l.backward()
        finally:handle.remove()
        assert len(cap)==1 and cap[0].grad is not None
        grad=cap[0].grad.detach().reshape(-1,8).numpy()
        assert np.isfinite(grad).all()
        centers,sc=s.fields();c=centers[:8].detach().numpy()
        ids=s.codes.detach().numpy().copy()
        row_sc=np.repeat(sc.detach().numpy(),s.shape[1]//8)
        # Linearized loss change for each group, penalize large weight jumps
        # relative to the current OWN reconstruction, not external weights.
        old=c[ids];delta=c[None,:,:]-old[:,None,:]
        g=grad*row_sc[:,None]
        norm=np.linalg.norm(g,axis=1)
        cost=np.einsum("nki,ni->nk",delta,g)/np.maximum(norm[:,None],1e-20)
        cost+=.05*np.sum(delta**2,axis=-1)
        win=cost.argmin(axis=1);gain=-cost[np.arange(len(ids)),win]
        order=np.argsort(-gain,kind="stable")
        selected=order[(gain[order]>1e-6)&(norm[order]>1e-15)][:64]
        n=len(selected);proposed+=n;before=float(l.detach());after=before;keep=False
        del cap,l,grad
        if n:
            ix=torch.from_numpy(selected.copy()).long()
            with torch.no_grad():s.codes[ix]=torch.from_numpy(win[selected].astype(np.int64))
            s.clear();clear(state[1])
            after=float(first_loss(state,r).detach())
            keep=np.isfinite(after) and after+1e-7<before
            if keep:accepted+=n;accepted_matrices+=1
            else:
                with torch.no_grad():s.codes[ix]=torch.from_numpy(ids[selected].astype(np.int64))
                s.clear();clear(state[1])
        records.append({"matrix":k,"proposed":n,"accepted":n if keep else 0,"before":before,"after_trial":after})
        if (i+1)%12==0:print("G02_DISCRETE",i+1,proposed,accepted,accepted_matrices,flush=True)
    return {"visited_matrices":len(keys),"proposed":proposed,"accepted":accepted,"accepted_matrices":accepted_matrices,"records":records}

def rate(packet,eb,ab,geometry):
    n=len(packet)+len(eb)+len(ab)
    return {"bytes":n,"geometry":geometry,"geometry_bpp":8*n/geometry,"base_reference_bpp":8*n/BASE,
            "within_geometry_cap":8*n/geometry<=.7}

def brief(d):return {k:v for k,v in d.items() if k!="records"}

def run(g01,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    assert G02==74719040
    raw=(g01/"G01_FUSED_RECOVERED_Q8.bundle").read_bytes()
    bank=(g01/"G01_FUSED_RECOVERED_EXTRA.bin").read_bytes()
    assert digest(raw)==CORE_SHA and digest(bank)==EXTRA_SHA
    archive,side,legacy=open_bundle(raw)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tr,dv=rows(tok,"TRAIN"),rows(tok,"DEV")
    assert not {r["q"] for v in tr.values() for r in v}&{r["q"] for v in dv.values() for r in v}
    rng=np.random.default_rng(SEED)
    train_rows=[r for pair in zip([tr["math"][int(i)] for i in rng.permutation(24)],[tr["general"][int(i)] for i in rng.permutation(24)]) for r in pair]
    assert len(train_rows)==STEPS
    out.mkdir(parents=True,exist_ok=False)
    def restore(packet,eb,ab=b""):
        a,s,l=open_bundle(packet);assert l==legacy
        st=direct(a,s);second=extra_open(eb,a,st[1]);install_growth(st,second)
        added={}
        if ab:added=open_added(ab,st,second);install_added(st,added)
        return st,second,added
    st,second,_=restore(raw,bank)
    baseline=audit(st,dv,tok)
    probes=dv["math"][:4]+dv["general"][:2]
    ref=[forward(st,r["p"],r["features"]).detach().clone() for r in probes]
    added=added_from(st,second);install_added(st,added)
    err=max(float((forward(st,r["p"],r["features"]).detach()-z).abs().max()) for r,z in zip(probes,ref))
    assert err<1e-4,err
    print("G02_BASE",json.dumps({"baseline":baseline,"duplicate_growth_max_logit_diff":err,"geometry":G02},sort_keys=True),flush=True)
    del st,second,added,ref
    arms={}
    for arm in ("CONTROL_54M","GROWTH_74M"):
        st,second,added=restore(raw,bank)
        if arm=="GROWTH_74M":
            added=added_from(st,second);install_added(st,added)
        trained=joint_train(st,tok,train_rows,arm,indexed_banks(second,added))
        _,packet,a,s=payload(archive,side,st,legacy)
        eb=extra_pack(second);ab=extra_pack(added) if added else b""
        (out/(arm+"_Q8.bundle")).write_bytes(packet);(out/(arm+"_SECOND.bin")).write_bytes(eb)
        if ab:(out/(arm+"_ADDED.bin")).write_bytes(ab)
        geom=G02 if added else GROWN
        assert rate(packet,eb,ab,geom)["within_geometry_cap"]
        del st,second,added
        st,second,added=restore(packet,eb,ab);post=audit(st,dv,tok)
        info={"training":trained,"after_physical_reload":post,"storage":rate(packet,eb,ab,geom),
              "sha256":{"core":digest(packet),"second":digest(eb),"added":digest(ab) if ab else None}}
        print("G02_ARM",arm,json.dumps(info,sort_keys=True),flush=True)
        if arm=="GROWTH_74M":
            discrete=discrete_train(st,added,train_rows)
            new=extra_pack(added);(out/"GROWTH_74M_DISCRETE_ADDED.bin").write_bytes(new)
            assert rate(packet,eb,new,geom)["within_geometry_cap"]
            del st,second,added
            st,second,added=restore(packet,eb,new)
            post2=audit(st,dv,tok)
            info["discrete_phase"]={"updates":discrete,"after_physical_reload":post2,
                                    "storage":rate(packet,eb,new,geom),"added_sha256":digest(new)}
            print("G02_DISCRETE_RESULT",json.dumps({**info["discrete_phase"],"updates":brief(discrete)},sort_keys=True),flush=True)
        arms[arm]=info
        del st,second,added
    report={"stage":"G02_ACTUAL_ATTENTION_AND_FFN_GROWTH","parent_core_sha256":CORE_SHA,"parent_second_sha256":EXTRA_SHA,
            "depth":12,"old_geometry":GROWN,"new_geometry":G02,"hidden_width":320,"new_FFN_width":4608,
            "query_heads_per_layer_total":10,"KV_heads_per_layer_total":2,
            "added_matrices":84,"duplicate_growth_max_logit_difference":err,"baseline":baseline,
            "arms":arms,"candidate_accepted":False,"FP32_teacher_calls":0,"output_gate":0,
            "limitations":["Actual new independent attention and FFN compact banks active each layer, fixed mixture not router.",
                           "Hidden width and neural region/GT geometry still 320; no proportional all-region expansion yet.",
                           "Both arms same 48 Adam steps; discrete phase is additional compute on growth only, not matched comparison.",
                           "Discrete proposals improve same TRAIN first-token objective only; not proof of generalization.",
                           "Secondary attention uses full-prefix native research path, no secondary KV cache persistence, no CPU certification.",
                           "Expanded logical entries not independent trainable dense parameters; physical payload known components only, tokenizer and runtime separate, no protected 616 replay."]}
    (out/"G02_REPORT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("G02_ASSESS",json.dumps({"stage":report["stage"],"geometry":G02,"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g01","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g01,a.f08,a.out)
