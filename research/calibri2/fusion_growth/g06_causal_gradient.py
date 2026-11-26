"""G06: real joint learning on 133.701M G05 checkpoint.
The historical early causal hook detached h0..h5 before the semantic child.
Restore this gradient path without changing the forward function or adding
weights. Matched 192 TRAIN exposures against the original detached path.
Physical reload, held-out counterfactual pairs and full answer+EOS metrics.
No protected fitting, no teacher, no deployment.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import (
    payload,audit,native,digest,open_bundle,prompt_features,supervised,extra_pack)
from research.calibri2.fusion_growth.g02_attention_discrete import (
    indexed_banks,joint_train,forward)
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore,region_split,rate
from research.calibri_nra3 import s119q1_real43m_onepass_causal_early as q1
from v176.operators import clear

SEED=20261229
PIN={
 "Q8.bundle":"a5a4c1ecb7beeb7c7813a47a54805feb7d7fac2947e371375c04c3f8456c5c07",
 "SECOND.bin":"a3e885e58e09ab77c83cef03f88a3b2e8749df958d9b1543d955c6615bb28ed5",
 "ADDED.bin":"ad71e047d6b47be3c57f00999d707b7aec7481d100f509de35518e1cdcc05592",
 "FOURTH.bin":"4d6a1300aa6c49f304e04803715a5f667dfe60e25f129ed31b159688d0b0e963",
 "REGIONS.safetensors":"cb09b3c44df9d002338a19aa9cdadbf8e53c7585ce0df75a086cb146786028ed",
 "FIFTH.bin":"178f581364770e61334b41ce65c269cf571e48541f5e585558146db2e1c7ea58",
 "SIXTH.bin":"221d8109e01a3cc863d0cc09821ba577a3f2062028662dab26195ca4de0f11e2",
 "REGIONS2.safetensors":"bc7d4f5c1001835baa86577f5dcd35387bae0cc963cf0499bd8496488d11f032"
}

@contextmanager
def connected_hook(model,adapter,*,contextual=True,view_mapping=q1.PLACEHOLDER_MAPPING):
    assert len(model.model.layers)==12 and adapter.output_gate==0
    assert not adapter.candidate_accepted and not adapter.activation_allowed
    assert view_mapping==q1.PLACEHOLDER_MAPPING
    layers=model.model.layers
    baseline=[set(l._forward_hooks) for l in layers[:6]]
    seen={};handles=[]
    witness={"candidate_forward_calls":0,"early_layer_calls":{},"adapter_delta_H320":None,
             "placeholder_mapping":list(view_mapping)}
    def record(i):
        def hook(_m,_a,out):
            h=out[0] if isinstance(out,tuple) else out
            assert i not in seen and h.ndim==3 and h.shape[0]==1 and h.shape[-1]==320
            assert torch.isfinite(h).all()
            seen[i]=h
            witness["early_layer_calls"][i]=1
        return hook
    def mid(_m,_a,out):
        assert set(seen)==set(range(5))
        h=out[0] if isinstance(out,tuple) else out
        assert h.ndim==3 and h.shape[0]==1 and h.shape[-1]==320
        seen[5]=h;witness["early_layer_calls"][5]=1
        x=torch.stack([seen[i] for i in view_mapping],dim=1)
        assert x.shape==(1,12,h.shape[1],320) and torch.equal(x[:,11],h)
        d=adapter(x,contextual=contextual)
        assert d.shape==(1,320) and d.requires_grad and torch.isfinite(d).all()
        witness["adapter_delta_H320"]=float(d.detach().norm())
        witness["candidate_forward_calls"]+=1
        z=h+torch.cat((torch.zeros_like(h[:,:-1]),d.unsqueeze(1)),dim=1)
        return (z,*out[1:]) if isinstance(out,tuple) else z
    try:
        for i in range(5):handles.append(layers[i].register_forward_hook(record(i)))
        handles.append(layers[5].register_forward_hook(mid))
        yield witness
        assert witness["candidate_forward_calls"]==1 and set(seen)==set(range(6))
    finally:
        for handle in handles:handle.remove()
        assert all(set(layers[i]._forward_hooks)==baseline[i] for i in range(6))

@contextmanager
def gradient_mode(connected):
    original=q1.single_pass_early_intervention
    if connected:q1.single_pass_early_intervention=connected_hook
    try:yield
    finally:q1.single_pass_early_intervention=original

def dataset(tok,split):
    tr=split=="TRAIN";n=16 if tr else 6;out={"math":[],"general":[]}
    names=[("Миша","Вика"),("Дима","Катя"),("Илья","Соня"),("Саша","Лиза")] if tr else [
        ("Глеб","Алиса"),("Павел","Рита"),("Максим","Оля")]
    objects=["ключ","мяч","письмо","зонт"] if tr else ["карандаш","билет","альбом"]
    for i in range(n):
        a=5+i%4;b=1+i//4 if tr else 2+i//4
        x,y=names[i%len(names)];obj=objects[(i//len(names))%len(objects)]
        # Same operands or context, opposite answer-changing instruction.
        tag=(2300 if tr else 8300)+i
        cases=[
         ("math","arith",f"Вычисли {a} + {b}." if tr else f"Найди сумму {a} и {b}.",str(a+b),str(a-b)),
         ("math","arith",f"Вычисли {a} - {b}." if tr else f"Из {a} вычти {b}.",str(a-b),str(a+b)),
         ("math","compare",f"Что больше: {a} или {b}?" if tr else f"Назови большее число: {b}, {a}.",str(a),str(b)),
         ("math","compare",f"Что меньше: {a} или {b}?" if tr else f"Назови меньшее число: {b}, {a}.",str(b),str(a)),
         ("general","negation",f"Запись {tag}: у {x} есть {obj}. Есть ли {obj}?" if tr else f"Случай {tag}: {x} имеет предмет {obj}. Есть ли предмет?","да","нет"),
         ("general","negation",f"Запись {tag}: у {x} нет предмета {obj}. Есть ли {obj}?" if tr else f"Случай {tag}: {x} не имеет предмета {obj}. Есть ли предмет?","нет","да"),
         ("general","owner",f"Запись {tag}: {x} держит {obj}, {y} держит книгу. Кто держит {obj}?" if tr else f"Случай {tag}: предмет {obj} у {x}, книга у {y}. У кого предмет {obj}?",x,y),
         ("general","owner",f"Запись {tag}: {y} держит {obj}, {x} держит книгу. Кто держит {obj}?" if tr else f"Случай {tag}: предмет {obj} у {y}, книга у {x}. У кого предмет {obj}?",y,x)]
        for kind,family,q,ans,wrong in cases:
            p,g=native(tok,q,ans);pp,bad=native(tok,q,wrong)
            assert p==pp and g!=bad
            div=next(j for j,(u,v) in enumerate(zip(g,bad)) if u!=v)
            out[kind].append({"q":q,"p":p,"g":g,"bad":bad,"div":div,"answer":ans,
                              "wrong":wrong,"features":prompt_features(tok,q),"split":split,
                              "kind":kind,"family":family,"pair":f"{family}/{i}"})
    assert all(len(v)==4*n and len({r["q"] for r in v})==4*n for v in out.values())
    return out

def pair_audit(st,ds,tok):
    result=audit(st,ds,tok)
    records=[]
    for kind,rows in ds.items():
        for r in rows:
            own=[]
            for _ in range(8):
                z=forward(st,r["p"]+own,r["features"]).detach()
                own.append(int(z.argmax()))
                if own[-1]==2:break
            txt=tok.decode(own,skip_special_tokens=True).strip()
            records.append({"q_sha256":digest(r["q"].encode()),"kind":kind,"family":r["family"],
                            "pair":r["pair"],"gold":r["answer"],"answer":txt,"ids":own,
                            "exact":own[-1]==2 and txt==r["answer"]})
    pairs={}
    for row in records:pairs.setdefault(row["pair"],[]).append(row)
    assert all(len(v)==2 for v in pairs.values())
    stats={}
    for fam in ("arith","compare","negation","owner"):
        rr=[v for v in pairs.values() if v[0]["family"]==fam]
        stats[fam]={"pairs":len(rr),"both_correct":sum(all(x["exact"] for x in v) for v in rr),
                    "different_outputs":sum(v[0]["ids"]!=v[1]["ids"] for v in rr),
                    "exact":sum(x["exact"] for v in rr for x in v)}
    return {"ordinary":result,"counterfactual":stats,"records":records}

def compact(a):return {k:v for k,v in a.items() if k!="records"}

def witness(st,r,connected):
    cap=[]
    def pre(_m,args): 
        x=args[0]
        if x.requires_grad:x.retain_grad()
        cap.append(x)
    h=st[5].register_forward_pre_hook(pre)
    try:
        with gradient_mode(connected):
            z=forward(st,r["p"],r["features"])
            l=F.cross_entropy(z,torch.tensor([r["g"][0]]))
            l.backward()
    finally:h.remove()
    assert len(cap)==1
    x=cap[0];norm=float(x.grad.norm()) if x.grad is not None else 0.
    assert x.requires_grad==connected
    if connected:assert norm>0 and np.isfinite(norm)
    early={}
    for k,s in st[1].items():
        if ".layers.0." in k or ".layers.5." in k:
            early[k]=float(s.centroids.grad.norm()) if s.centroids.grad is not None else 0.
    for p in st[0].parameters():p.grad=None
    for p in st[-1]+[st[3].gates]:p.grad=None
    clear(st[1])
    return {"adapter_input_requires_grad":x.requires_grad,"adapter_input_gradient_L2":norm,
            "early_core_gradient_norms":early,"first_token_CE":float(l.detach())}

def run(g05,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    src=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(src,PIN.values()))
    ar,sd,legacy=open_bundle(src[0])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tokenizer_bytes=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    tr,dv=dataset(tok,"TRAIN"),dataset(tok,"DEV")
    assert not {r["q"] for v in tr.values() for r in v}&{r["q"] for v in dv.values() for r in v}
    rng=np.random.default_rng(SEED)
    training=[]
    # 128 unique TRAIN prompts then 64 balanced replay exposures.
    for n in (64,32):
        mm=[tr["math"][int(i)] for i in rng.permutation(64)[:n]]
        gg=[tr["general"][int(i)] for i in rng.permutation(64)[:n]]
        training.extend(r for pair in zip(mm,gg) for r in pair)
    assert len(training)==192 and len({r["q"] for r in training})==128
    out.mkdir(parents=True,exist_ok=False)
    def reload(blobs):return restore(blobs[:6],blobs[6],blobs[7])
    st,*rest=reload(src)
    pre=pair_audit(st,dv,tok)
    probes=dv["math"][:3]+dv["general"][:3]
    a=[forward(st,r["p"],r["features"]).detach().clone() for r in probes]
    with gradient_mode(True):
        b=[forward(st,r["p"],r["features"]).detach().clone() for r in probes]
    err=max(float((u-v).abs().max()) for u,v in zip(a,b))
    assert err<1e-6
    wd=witness(st,training[0],False);wc=witness(st,training[0],True)
    print("G06_BASE",json.dumps({"baseline":compact(pre),"forward_max_diff":err,
                                 "detached_witness":wd,"connected_witness":wc},sort_keys=True),flush=True)
    del st,rest,a,b
    arms={}
    for arm in ("DETACHED_CONTROL","CONNECTED_CAUSAL"):
        st,second,added,fourth,reg,old,fifth,sixth,reg2=reload(src)
        ext=indexed_banks(second,added)|{"fourth/"+k:v for k,v in fourth.items()}|{"fifth/"+k:v for k,v in fifth.items()}|{"sixth/"+k:v for k,v in sixth.items()}
        assert len(ext)==324
        # Same training function, optimizer, examples and objective.
        with gradient_mode(arm=="CONNECTED_CAUSAL"):
            trained=joint_train(st,tok,training,arm,ext)
        _,packet,_,_=payload(ar,sd,st,legacy)
        rb,rb2=region_split(st,reg,old,reg2)
        vals=(packet,extra_pack(second),extra_pack(added),extra_pack(fourth),rb,extra_pack(fifth),extra_pack(sixth),rb2)
        storage=rate(vals[:6],vals[6],vals[7],G05,tokenizer_bytes)
        assert storage["within_known_weight_cap"]
        for k,v in zip(PIN,vals):(out/(arm+"_"+k)).write_bytes(v)
        del st,second,added,fourth,reg,old,fifth,sixth,reg2,ext
        st,*rest=reload(vals)
        post=pair_audit(st,dv,tok)
        train_audit=pair_audit(st,{"math":tr["math"][:16],"general":tr["general"][:16]},tok)
        del st,rest
        arms[arm]={"training":trained,"after_physical_reload":post,"train_subset":train_audit,
                   "storage":storage,"sha256":dict(zip(PIN,[digest(v) for v in vals]))}
        print("G06_ARM",arm,json.dumps({**arms[arm],"after_physical_reload":compact(post),
                                        "train_subset":compact(train_audit)},sort_keys=True),flush=True)
    report={"stage":"G06_CONNECTED_CAUSAL_JOINT_TRAIN","geometry":G05,"depth":12,
            "forward_max_diff":err,"detached_witness":wd,"connected_witness":wc,"before":pre,
            "TRAIN_unique":128,"TRAIN_exposures_per_arm":192,"DEV_unique":48,
            "arms":arms,"candidate_accepted":False,"FP32_teacher_calls":0,"output_gate":0,
            "limitations":["Matched causal gradient path ablation; no change in forward architecture or new parameters.",
                           "No protected replay or historical CPU verification; same 133.701M logical core, not conventional dense independent parameters.",
                           "Authored short-answer counterfactual pairs, no claim of universal generation. DEV separate prompts, arithmetic operand/value overlap possible.",
                           "Fixed IDs; all physical banks and GT/regions retained. Full runtime excluded from rate. No automatic acceptance."]}
    (out/"G06_REPORT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("G06_ASSESS",json.dumps({"stage":report["stage"],"geometry":G05,"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
