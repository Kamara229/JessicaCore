"""G04: 113M actual logical core width from physically recovered G03.
Split fourth FFN bank and secondary attention branch preserving initial
function. Both branches execute on every request. New matched recovery
plus discrete diversification. No hidden-width or protected replay claim.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import (
    BASE,EXTRA,extra_pack,extra_open,install_growth,payload,audit,
    native,digest,open_bundle,direct,prompt_features)
from research.calibri2.fusion_growth.g02_attention_discrete import (
    ATTN,clone_bank,install_added,open_added,indexed_banks,joint_train,discrete_train,forward)
from research.calibri2.fusion_growth.g03_region_tunnel_width import (
    G03,install_fourth,open_fourth,widen_regions,region_bytes)
from v176.operators import clear
from safetensors.torch import load

G04=G03+EXTRA+ATTN
SEED=20261227
STEPS=96
PIN={
 "Q8.bundle":"fb2d8f554aaed8b95fcb4d7c66c7f732c5fd063bccbfc7ba8eeeb9a162b82e50",
 "SECOND.bin":"a090823aa8ff745f2dce7ae62d5765dc57dc93fd4532af8dbeccd1cb6c075dcf",
 "ADDED.bin":"b883c7c1b2799518161036b331ab67e189853d6e7475decbcc8543627e5875e5",
 "FOURTH.bin":"f4c07f540b800588a48b1e1e3c45a035cb6ed7f51ed02fbd05f82e228f1866db",
 "REGIONS.safetensors":"169227dc42013db90ea7341935d2e3f21e15ddeef2a1d589b3dcbbb04b5083a2",
}

def dataset(tok,split):
    tr=split=="TRAIN";n=24 if tr else 8
    out={"math":[],"general":[]}
    for i in range(n):
        a=2+i%8;b=2+i//8+(0 if tr else 4)
        x,z=("Нина","Саша") if tr else ("Ирина","Пётр")
        yes=bool(i%2);order=bool((i//2)%2)
        tag=(300 if tr else 740)+i
        cases=[
          ("math",f"Произведение чисел {a} и {b} равно чему?" if tr else f"Умножь {a} на {b}.",str(a*b),str(a*b+1)),
          ("math",f"Найди частное {a*b} и {b}." if tr else f"Раздели {a*b} на {b}.",str(a),str(a+1)),
          ("general",f"Случай {tag}: {'ключ есть' if yes else 'ключа нет'} у {x}. Есть ли ключ?","да" if yes else "нет","нет" if yes else "да"),
          ("general",f"Запись {tag}: {'сначала '+x+', затем '+z if order else 'сначала '+z+', затем '+x} вошли в зал. Кто вошёл первым?",x if order else z,z if order else x)]
        for kind,q,y,w in cases:
            p,g=native(tok,q,y);pp,bad=native(tok,q,w)
            assert p==pp and g!=bad
            div=next(j for j,(u,v) in enumerate(zip(g,bad)) if u!=v)
            out[kind].append({"q":q,"p":p,"g":g,"bad":bad,"div":div,"answer":y,"wrong":w,
                              "features":prompt_features(tok,q),"split":split,"kind":kind})
    assert all(len(v)==2*n and len({r["q"] for r in v})==2*n for v in out.values())
    return out

class QuintMLP(nn.Module):
    def __init__(self,old,gate,up,down):
        super().__init__()
        self.first=old
        self.fifth_gate=gate;self.fifth_up=up;self.fifth_down=down
        self.intermediate_size=7680
    def forward(self,x):
        quad=self.first;triple=quad.first;wide=triple.first;orig=wide.first
        def apply(g,u,d):
            z=F.linear(orig.act_fn(F.linear(x,g()))*F.linear(x,u()),d())
            for s in (g,u,d):s.clear()
            return z
        a=orig(x)
        b=apply(wide.second_gate,wide.second_up,wide.second_down)
        c=apply(triple.third_gate,triple.third_up,triple.third_down)
        d=apply(quad.fourth_gate,quad.fourth_up,quad.fourth_down)
        e=apply(self.fifth_gate,self.fifth_up,self.fifth_down)
        return .5*a+.25*b+.125*c+.0625*(d+e)

class TripleAttention(nn.Module):
    def __init__(self,old,third):
        super().__init__()
        self.first=old;self.third=third
    def forward(self,*args,**kwargs):
        pos=kwargs.get("cache_position")
        if pos is not None:assert int(pos.flatten()[0])==0
        a=self.first.original(*args,**kwargs)
        kw=dict(kwargs);kw["past_key_value"]=None;kw["use_cache"]=False
        b=self.first.secondary(*args,**kw);c=self.third(*args,**kw)
        for mod in (self.first.secondary,self.third):
            for k in ("q_proj","k_proj","v_proj","o_proj"):getattr(mod,k).source.clear()
        assert a[0].shape==b[0].shape==c[0].shape
        return (.5*a[0]+.25*(b[0]+c[0]),*a[1:])

def fifth_from(fourth,added):
    bank={k:clone_bank(s) for k,s in fourth.items()}
    bank.update({k:clone_bank(s) for k,s in added.items() if ".self_attn." in k})
    assert len(bank)==84 and sum(s.shape[0]*s.shape[1] for s in bank.values())==EXTRA+ATTN
    return bank

def install_fifth(st,bank):
    clear(st[1])
    for i,layer in enumerate(st[0].model.layers):
        p=f"model.layers.{i}.mlp."
        layer.mlp=QuintMLP(layer.mlp,*[bank[p+k+".weight"] for k in ("gate_proj","up_proj","down_proj")])
        old=layer.self_attn
        for k in ("q_proj","k_proj","v_proj","o_proj"):getattr(old.secondary,k).source.clear()
        third=copy.deepcopy(old.secondary)
        p=f"model.layers.{i}.self_attn."
        for k in ("q_proj","k_proj","v_proj","o_proj"):getattr(third,k).source=bank[p+k+".weight"]
        layer.self_attn=TripleAttention(old,third)

def restore(blobs,fifth=b""):
    packet,eb,ab,fb,rb=blobs
    ar,sd,legacy=open_bundle(packet)
    st=direct(ar,sd)
    second=extra_open(eb,ar,st[1]);install_growth(st,second)
    added=open_added(ab,st,second);install_added(st,added)
    fourth=open_fourth(fb,added);install_fourth(st,fourth)
    st,reg,old=widen_regions(st,rb)
    extra={}
    if fifth:
        extra=open_added(fifth,st,second);install_fifth(st,extra)
    return st,second,added,fourth,reg,old,extra

def rate(blobs,fifth,geom):
    n=sum(map(len,blobs))+len(fifth)
    return {"bytes":n,"geometry":geom,"geometry_bpp":8*n/geom,
            "base_reference_bpp":8*n/BASE,"within_cap":8*n/geom<=.7,
            "tokenizer_excluded":True}

def run(g03,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    assert G04==113057600
    blobs=tuple((g03/("GROWTH_92M_REGIONS_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    ar,sd,legacy=open_bundle(blobs[0])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tr,dv=dataset(tok,"TRAIN"),dataset(tok,"DEV")
    assert not {r["q"] for v in tr.values() for r in v}&{r["q"] for v in dv.values() for r in v}
    rng=np.random.default_rng(SEED)
    training=[r for pair in zip([tr["math"][int(i)] for i in rng.permutation(48)],[tr["general"][int(i)] for i in rng.permutation(48)]) for r in pair]
    assert len(training)==STEPS
    out.mkdir(parents=True,exist_ok=False)
    st,second,added,fourth,reg,old,_=restore(blobs)
    before=audit(st,dv,tok)
    probes=dv["math"][:3]+dv["general"][:3]
    ref=[forward(st,r["p"],r["features"]).detach().clone() for r in probes]
    new=fifth_from(fourth,added);install_fifth(st,new)
    err=max(float((forward(st,r["p"],r["features"]).detach()-z).abs().max()) for r,z in zip(probes,ref))
    assert err<1e-4,err
    init=extra_pack(new);del st,second,added,fourth,reg,old,new,ref
    print("G04_BASE",json.dumps({"before":before,"geometry":G04,"duplicate_init_max_logit_difference":err},sort_keys=True),flush=True)
    arms={}
    for arm in ("CONTROL_92M","GROWTH_113M"):
        grow=arm.startswith("GROWTH")
        st,second,added,fourth,reg,old,new=restore(blobs,init if grow else b"")
        ext=indexed_banks(second,added)|{"fourth/"+k:s for k,s in fourth.items()}|{"fifth/"+k:s for k,s in new.items()}
        trained=joint_train(st,tok,training,arm,ext)
        _,packet,_,_=payload(ar,sd,st,legacy)
        vals=(packet,extra_pack(second),extra_pack(added),extra_pack(fourth),region_bytes(st,reg,old))
        nb=extra_pack(new) if grow else b""
        geom=G04 if grow else G03
        storage=rate(vals,nb,geom);assert storage["within_cap"]
        for k,v in zip(PIN,vals):(out/(arm+"_"+k)).write_bytes(v)
        if grow:(out/(arm+"_FIFTH.bin")).write_bytes(nb)
        del st,second,added,fourth,reg,old,new,ext
        st,second,added,fourth,reg,old,new=restore(vals,nb)
        post=audit(st,dv,tok)
        info={"training":trained,"after":post,"storage":storage,
              "sha256":dict(zip(PIN,[digest(v) for v in vals])),"fifth_sha256":digest(nb) if grow else None}
        print("G04_ARM",arm,json.dumps(info,sort_keys=True),flush=True)
        if grow:
            d=discrete_train(st,new,training)
            nb2=extra_pack(new)
            (out/"GROWTH_113M_DISCRETE_FIFTH.bin").write_bytes(nb2)
            assert rate(vals,nb2,geom)["within_cap"]
            del st,second,added,fourth,reg,old,new
            st,second,added,fourth,reg,old,new=restore(vals,nb2)
            info["discrete"]={"updates":d,"after":audit(st,dv,tok),"storage":rate(vals,nb2,geom),"fifth_sha256":digest(nb2)}
            print("G04_DISCRETE",json.dumps({**info["discrete"],"updates":{k:v for k,v in d.items() if k!="records"}},sort_keys=True),flush=True)
        arms[arm]=info
        del st,second,added,fourth,reg,old,new
    report={"stage":"G04_ACTUAL_113M_GROWTH","geometry":G04,"depth":12,"hidden_width":320,
            "FFN_width":7680,"query_heads_total_per_layer":15,"KV_heads_total_per_layer":3,
            "GT_rank":16,"additional_region_params":sum(v.numel() for v in load(blobs[-1]).values()),
            "duplicate_init_max_logit_difference":err,"before":before,"arms":arms,
            "matched_TRAIN_steps":STEPS,"candidate_accepted":False,"FP32_teacher_calls":0,"output_gate":0,
            "limitations":["FFN and attention bank horizontal expansion; hidden width unchanged, no new GT or region width in this round.",
                           "Logical matrix geometry, not independent dense optimizer degrees. New banks initialized from donors; discrete changes only on new banks.",
                           "Matched Adam arms; discrete phase extra compute and same-row first-token acceptance, not independent quality proof.",
                           "Small authored held-out set, not universal quality or protected 616 replay. Full-prefix only; no CPU certification or all-in tokenization/runtime budget."]}
    (out/"G04_REPORT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("G04_ASSESS",json.dumps({"stage":report["stage"],"geometry":G04,"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g03","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g03,a.f08,a.out)
