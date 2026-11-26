"""G05: near-target 133.701M actual logical core, plus GT/region growth.
Use G04 pre-discrete checkpoint because the extra discrete pass regressed
both held-out NLLs. Split the most influential original FFN/attention bank
instead of another already weakly weighted clone. Matched specialist and
mixed-domain recovery phases. No router, protected fitting, or deployment.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load,save
from v176.operators import clear
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import (
    BASE,EXTRA,extra_pack,payload,audit,native,digest,open_bundle,prompt_features)
from research.calibri2.fusion_growth.g02_attention_discrete import (
    ATTN,clone_bank,open_added,indexed_banks,joint_train,forward)
from research.calibri2.fusion_growth.g03_region_tunnel_width import TARGETS,ExtraProjection
from research.calibri2.fusion_growth.g04_113m_diversification import G04,restore as restore_g04

G05=G04+EXTRA+ATTN
SEED=20261228
PIN={
 "Q8.bundle":"c335830d051a8403f53e035d131f5879633ed04ab0b68ccbe9897d55d5e8269c",
 "SECOND.bin":"18120fcb3b36bfdeec8f4ac0c6e4d280f1734f404a7abb7240d0a9027fa7f8f9",
 "ADDED.bin":"83f6d78ceefddd61533a63ba14efc9e568202b677a3613a6401364b8ed2eceab",
 "FOURTH.bin":"1edbf4aa11be16567c4e4b6f0a5b185167f617d7291da4dd80252e634f439b88",
 "REGIONS.safetensors":"b40851f508d7a3662268bad6434caf75474fdfb0e9ca36eb7d587d788bc1f785",
 "FIFTH.bin":"b877fc43208e50c3f38cbd2f492e8f036f487649aee2c2cc92af644c11127995"}

def dataset(tok,split):
    tr=split=="TRAIN";n=32 if tr else 8
    out={"math":[],"general":[]}
    for i in range(n):
        a=2+i%8;b=1+i//8+(0 if tr else 5)
        tag=(4300 if tr else 8700)+i
        x,y=("Вера","Олег") if tr else ("Лена","Антон")
        even=bool(i%2);rev=bool((i//2)%2)
        cases=[
            ("math",f"Прибавь {b} к {a} и запиши число." if tr else f"Назови сумму {b} и {a}.",str(a+b),str(a+b+1)),
            ("math",f"Реши: x + {b} = {a+b}." if tr else f"Какое x даст {a+b} при прибавлении {b}?",str(a),str(a+1)),
            ("general",f"Карточка {tag}: {x} {'сохранила' if even else 'не сохранила'} письмо. Сохранено ли письмо?","да" if even else "нет","нет" if even else "да"),
            ("general",f"Запись {tag}: {'книга у '+x+', мяч у '+y if rev else 'мяч у '+x+', книга у '+y}. У кого книга?",x if rev else y,y if rev else x)]
        for kind,q,ans,wrong in cases:
            p,g=native(tok,q,ans);pp,bad=native(tok,q,wrong)
            assert p==pp and g!=bad
            div=next(j for j,(u,v) in enumerate(zip(g,bad)) if u!=v)
            out[kind].append({"q":q,"p":p,"g":g,"bad":bad,"div":div,"answer":ans,"wrong":wrong,
                              "features":prompt_features(tok,q),"split":split,"kind":kind})
    assert all(len(v)==2*n and len({r["q"] for r in v})==2*n for v in out.values())
    return out

class SixMLP(nn.Module):
    def __init__(self,old,gate,up,down):
        super().__init__();self.first=old
        self.sixth_gate=gate;self.sixth_up=up;self.sixth_down=down
        self.intermediate_size=9216
    def forward(self,x):
        quint=self.first;quad=quint.first;triple=quad.first;wide=triple.first;orig=wide.first
        def apply(g,u,d):
            z=F.linear(orig.act_fn(F.linear(x,g()))*F.linear(x,u()),d())
            for s in (g,u,d):s.clear()
            return z
        a=orig(x)
        b=apply(wide.second_gate,wide.second_up,wide.second_down)
        c=apply(triple.third_gate,triple.third_up,triple.third_down)
        d=apply(quad.fourth_gate,quad.fourth_up,quad.fourth_down)
        e=apply(quint.fifth_gate,quint.fifth_up,quint.fifth_down)
        f=apply(self.sixth_gate,self.sixth_up,self.sixth_down)
        return .25*(a+f)+.25*b+.125*c+.0625*(d+e)

class FourAttention(nn.Module):
    def __init__(self,old,fourth):
        super().__init__();self.first=old;self.fourth=fourth
    def forward(self,*args,**kwargs):
        pos=kwargs.get("cache_position")
        if pos is not None:assert int(pos.flatten()[0])==0
        triple=self.first;dual=triple.first
        a=dual.original(*args,**kwargs)
        kw=dict(kwargs);kw["past_key_value"]=None;kw["use_cache"]=False
        b=dual.secondary(*args,**kw);c=triple.third(*args,**kw);d=self.fourth(*args,**kw)
        for mod in (dual.secondary,triple.third,self.fourth):
            for k in ("q_proj","k_proj","v_proj","o_proj"):getattr(mod,k).source.clear()
        assert a[0].shape==b[0].shape==c[0].shape==d[0].shape
        return (.25*((a[0]+d[0])+(b[0]+c[0])),*a[1:])

def sixth_from(st):
    bank={k:clone_bank(s) for k,s in st[1].items() if ".mlp." in k or ".self_attn." in k}
    assert len(bank)==84 and sum(s.shape[0]*s.shape[1] for s in bank.values())==EXTRA+ATTN
    return bank

def install_sixth(st,bank):
    clear(st[1])
    for i,layer in enumerate(st[0].model.layers):
        p=f"model.layers.{i}.mlp."
        layer.mlp=SixMLP(layer.mlp,*[bank[p+k+".weight"] for k in ("gate_proj","up_proj","down_proj")])
        old=layer.self_attn;original=old.first.original
        for k in ("q_proj","k_proj","v_proj","o_proj"):getattr(original,k).source.clear()
        fourth=copy.deepcopy(original)
        p=f"model.layers.{i}.self_attn."
        for k in ("q_proj","k_proj","v_proj","o_proj"):getattr(fourth,k).source=bank[p+k+".weight"]
        layer.self_attn=FourAttention(old,fourth)

def widen_again(st,raw=None):
    model,_,_,full,_,child,region=st
    assert full.encoder.shape==(16,320) and full.decoder.shape==(320,16)
    torch.manual_seed(SEED)
    bank=nn.ModuleDict()
    for name,path in TARGETS.items():
        target=child.get_submodule(path)
        assert isinstance(target,nn.Linear)
        bank[name]=ExtraProjection(target.in_features,target.out_features)
    enc=torch.randn(16,320)*.02;dec=torch.zeros(320,16)
    if raw is not None:
        data=load(raw);enc=data.pop("GT.encoder").float();dec=data.pop("GT.decoder").float()
        bank.load_state_dict({k:v.float() for k,v in data.items()},strict=True)
    assert enc.shape==(16,320) and dec.shape==(320,16)
    old_enc,old_dec=full.encoder,full.decoder
    full.encoder=nn.Parameter(torch.cat((old_enc.detach(),enc),0))
    full.decoder=nn.Parameter(torch.cat((old_dec.detach(),dec),1));full.rank=32
    full.encoder.register_hook(lambda g:torch.cat((torch.zeros_like(g[:4]),g[4:]),0))
    full.decoder.register_hook(lambda g:torch.cat((torch.zeros_like(g[:,:4]),g[:,4:]),1))
    model.model.add_module("g05_expanded_regions",bank)
    for name,path in TARGETS.items():
        proj=bank[name]
        child.get_submodule(path).register_forward_hook(lambda _m,args,z,proj=proj:z+proj(args[0]))
    kept=[p for p in region if p is not old_enc and p is not old_dec]
    assert len(kept)==len(region)-2
    return (*st[:-1],kept+[full.encoder,full.decoder,*bank.parameters()]),bank

def region_split(st,reg,old,reg2=None):
    full=st[3]
    assert torch.equal(full.encoder[:4].detach(),old[0]) and torch.equal(full.decoder[:,:4].detach(),old[1])
    def packed(bank,lo,hi):
        vals={"GT.encoder":full.encoder[lo:hi].detach().half().contiguous(),
              "GT.decoder":full.decoder[:,lo:hi].detach().half().contiguous()}
        vals.update({k:v.detach().half().contiguous() for k,v in bank.state_dict().items()})
        assert all(torch.isfinite(v).all() for v in vals.values())
        return save(vals)
    return packed(reg,4,16),packed(reg2,16,32) if reg2 is not None else b""

def restore(blobs,sixth=b"",region2=None):
    st,second,added,fourth,reg,old,fifth=restore_g04(blobs[:5],blobs[5])
    bank={};reg2=None
    if sixth:
        bank=open_added(sixth,st,second);install_sixth(st,bank)
        st,reg2=widen_again(st,region2)
    return st,second,added,fourth,reg,old,fifth,bank,reg2

def rate(blobs,sixth,rb2,geom,tokenizer_bytes):
    n=sum(map(len,blobs))+len(sixth)+len(rb2)
    return {"bytes":n,"geometry":geom,"geometry_bpp":8*n/geom,
            "base_reference_bpp":8*n/BASE,"within_known_weight_cap":8*n/geom<=.7,
            "tokenizer_bytes":tokenizer_bytes,"weights_plus_tokenizer_bpp":8*(n+tokenizer_bytes)/geom}

def run(g04,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    assert G05==133701440
    blobs=tuple((g04/("GROWTH_113M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    ar,sd,legacy=open_bundle(blobs[0])
    tokenizer_bytes=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tr,dv=dataset(tok,"TRAIN"),dataset(tok,"DEV")
    assert not {r["q"] for v in tr.values() for r in v}&{r["q"] for v in dv.values() for r in v}
    rng=np.random.default_rng(SEED)
    specialist=[tr["math"][int(i)] for i in rng.permutation(64)]
    mixed=[r for pair in zip([tr["math"][int(i)] for i in rng.permutation(64)],
                              [tr["general"][int(i)] for i in rng.permutation(64)]) for r in pair]
    assert len(specialist)==64 and len(mixed)==128
    out.mkdir(parents=True,exist_ok=False)
    st,second,added,fourth,reg,old,fifth,_,_=restore(blobs)
    before=audit(st,dv,tok)
    probes=dv["math"][:3]+dv["general"][:3]
    ref=[forward(st,r["p"],r["features"]).detach().clone() for r in probes]
    bank=sixth_from(st);install_sixth(st,bank);st,reg2=widen_again(st)
    err=max(float((forward(st,r["p"],r["features"]).detach()-z).abs().max()) for r,z in zip(probes,ref))
    assert err<1e-4,err
    init=extra_pack(bank);_,init_rb=region_split(st,reg,old,reg2)
    del st,second,added,fourth,reg,old,fifth,bank,reg2,ref
    print("G05_BASE",json.dumps({"before":before,"geometry":G05,"duplicate_init_max_logit_difference":err},sort_keys=True),flush=True)
    arms={}
    for arm in ("CONTROL_113M","GROWTH_134M"):
        grow=arm.startswith("GROWTH")
        st,second,added,fourth,reg,old,fifth,bank,reg2=restore(blobs,init if grow else b"",init_rb if grow else None)
        ext=indexed_banks(second,added)|{"fourth/"+k:s for k,s in fourth.items()}|{"fifth/"+k:s for k,s in fifth.items()}|{"sixth/"+k:s for k,s in bank.items()}
        spec=joint_train(st,tok,specialist,arm+"_MATH",ext)
        # No DEV-driven early stopping, optimizer reset at fixed phase boundary.
        middle=audit(st,dv,tok)
        print("G05_SPECIALIST",arm,json.dumps({"training":spec,"DEV_diagnostic":middle},sort_keys=True),flush=True)
        recovered=joint_train(st,tok,mixed,arm+"_MIXED",ext)
        _,packet,_,_=payload(ar,sd,st,legacy)
        rb,rb2=region_split(st,reg,old,reg2)
        vals=(packet,extra_pack(second),extra_pack(added),extra_pack(fourth),rb,extra_pack(fifth))
        nb=extra_pack(bank) if grow else b""
        geom=G05 if grow else G04
        storage=rate(vals,nb,rb2,geom,tokenizer_bytes);assert storage["within_known_weight_cap"]
        changed=[]
        if grow:
            orig,new=load(init_rb),load(rb2)
            changed=[k for k,v in new.items() if not torch.equal(v,orig[k])]
            assert "GT.decoder" in changed and any(k.startswith("semantic.") for k in changed) and any(k.startswith("planning.") for k in changed)
        for k,v in zip(PIN,vals):(out/(arm+"_"+k)).write_bytes(v)
        if grow:
            (out/(arm+"_SIXTH.bin")).write_bytes(nb)
            (out/(arm+"_REGIONS2.safetensors")).write_bytes(rb2)
        del st,second,added,fourth,reg,old,fifth,bank,reg2,ext
        st,*rest=restore(vals,nb,rb2 if grow else None)
        post=audit(st,dv,tok);del st,rest
        info={"specialist_training":spec,"specialist_DEV_diagnostic":middle,"recovery_training":recovered,
              "after_physical_reload":post,"storage":storage,
              "new_region_changed":changed,"sha256":dict(zip(PIN,[digest(v) for v in vals])),
              "sixth_sha256":digest(nb) if grow else None,"regions2_sha256":digest(rb2) if grow else None}
        arms[arm]=info
        print("G05_ARM",arm,json.dumps(info,sort_keys=True),flush=True)
    report={"stage":"G05_NEAR_135M_SPECIALIST_AND_JOINT","geometry":G05,"target_geometry":135000000,
            "fraction_of_target":G05/135000000,"depth":12,"hidden_width":320,"FFN_width":9216,
            "query_heads_total_per_layer":20,"KV_heads_total_per_layer":4,"GT_rank":32,
            "region_extra_projection_ranks":[16,16],"new_core_matrices":84,
            "duplicate_init_max_logit_difference":err,"before":before,"arms":arms,
            "matched_specialist_steps":64,"matched_recovery_steps":128,
            "candidate_accepted":False,"FP32_teacher_calls":0,"output_gate":0,
            "limitations":["133.701M is 99.04% of 135M target, not exact 135M; logical compact matrix entries, not independent dense trainable parameters.",
                           "Hidden width320 retained; all FFN and attention banks execute with fixed gains, not a router or conventional wider residual stream.",
                           "Parent chosen before G04 discrete regression; no use of new DEV for updates, fixed phase lengths. G05 fixed IDs; further discrete learning needed.",
                           "GT rank16 to32 and added rank16 parallel projections, old 4 GT channels protected; all added regions charged.",
                           "Small authored DEV; protected 616, speed and complete runtime bytes not certified. Tokenizer now separately reported and included in a secondary rate, executable/runtime omitted."]}
    (out/"G05_REPORT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("G05_ASSESS",json.dumps({"stage":report["stage"],"geometry":G05,"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g04","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g04,a.f08,a.out)
