"""G03: continue actual horizontal growth from G02; 92.414M logical
core entries plus real GT recurrent rank growth and parallel learned
projections inside semantic/planning/causal regions. Compare matched
74M recovery. No new router, no protected training, no LIVE.
"""
import argparse
import json
import lzma
import struct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load,save
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import (
    BASE,EXTRA,extra_pack,extra_open,install_growth,payload,audit,
    native,digest,open_bundle,direct,unpack_indices,prompt_features)
from research.calibri2.fusion_growth.g02_attention_discrete import (
    G02,clone_bank,install_added,open_added,indexed_banks,joint_train,forward)

SEED=20261224
G03=G02+EXTRA
CORE_SHA="5f696ee4316232ab5bff93f7cfd986ed86094c8af837352879184b53545857cc"
SECOND_SHA="c72e7bbb35aff9f3ef814436e750855f5aedb3fecebab7011096466978ba2593"
ADDED_SHA="b3d714fdf07bb88728aa0a1682f171bc8b26f3054166c05a12a4f979458d1fa3"
RANK=12
REGION_RANK=16
STEPS=96
TARGETS={
 "semantic":"child.child.shadow.unified.structured_interface.semantic",
 "binding":"child.child.shadow.unified.structured_interface.fuse",
 "planning":"child.child.shadow.unified.planning_action_area.input_projection",
 "causal":"child.up",
}

def dataset(tok,split):
    tr=split=="TRAIN";n=24 if tr else 8
    rng=np.random.default_rng(SEED+(0 if tr else 1))
    out={"math":[],"general":[]}
    for i in range(n):
        a=1+i%4;b=1+(i//4)%4
        # Balanced short answers; additional numeric identifiers only in
        # nonmath examples. Both positive and wrong answers verified.
        if i%2:q=f"Сложи {a} и {b}." if tr else f"Сколько будет {b} плюс {a}?";y=a+b;w=(y%8)+1
        else:
            hi=a+b;lo=b;q=f"Вычисли {hi} - {lo}." if tr else f"Из {hi} вычти {lo}.";y=a;w=(a%4)+1
        ident=(1500 if tr else 9500)+11*i
        x,z=("Анна","Борис") if tr else ("Ольга","Марк")
        yes=bool((i//2)%2)
        cases=[
            ("math",q,str(y),str(w)),
            ("math",f"Какое число {'больше' if yes else 'меньше'}: {a} или {a+b}?" if tr else f"Назови {'большее' if yes else 'меньшее'} из {a+b} и {a}.",str(a+b if yes else a),str(a if yes else a+b)),
            ("general",f"Запись {ident}: {x} {'нашла' if yes else 'не нашла'} ключ. Нашла ли {x} ключ?","да" if yes else "нет","нет" if yes else "да"),
            ("general",f"Запись {ident}: {x} передала подарок человеку {z}. Кто {'передал' if yes else 'получил'} подарок?",x if yes else z,z if yes else x)]
        for kind,q,y,w in cases:
            p,g=native(tok,q,y);pp,bad=native(tok,q,w)
            assert p==pp and g!=bad
            div=next(j for j,(u,v) in enumerate(zip(g,bad)) if u!=v)
            out[kind].append({"q":q,"p":p,"g":g,"bad":bad,"div":div,"answer":y,"wrong":w,
                              "features":prompt_features(tok,q),"split":split,"kind":kind})
    # Avoid repeated identical math prompts masquerading as independent rows.
    for k,rs in out.items():
        unique={r["q"]:r for r in rs}
        assert all(r["answer"]==unique[r["q"]]["answer"] for r in rs)
        out[k]=list(unique.values())
    return out

class QuadMLP(nn.Module):
    def __init__(self,old,gate,up,down):
        super().__init__()
        self.first=old
        self.fourth_gate=gate;self.fourth_up=up;self.fourth_down=down
        self.intermediate_size=6144
    def forward(self,x):
        triple=self.first;wide=triple.first;orig=wide.first
        def apply(g,u,d):
            z=F.linear(orig.act_fn(F.linear(x,g()))*F.linear(x,u()),d())
            for s in (g,u,d):s.clear()
            return z
        a=orig(x)
        b=apply(wide.second_gate,wide.second_up,wide.second_down)
        c=apply(triple.third_gate,triple.third_up,triple.third_down)
        d=apply(self.fourth_gate,self.fourth_up,self.fourth_down)
        return .5*a+.25*b+.125*(c+d)

def fourth_from(added):
    out={k:clone_bank(s) for k,s in added.items() if ".mlp." in k}
    assert len(out)==36 and sum(s.shape[0]*s.shape[1] for s in out.values())==EXTRA
    return out

def install_fourth(state,bank):
    for i,layer in enumerate(state[0].model.layers):
        p=f"model.layers.{i}.mlp."
        layer.mlp=QuadMLP(layer.mlp,*[bank[p+k+".weight"] for k in ("gate_proj","up_proj","down_proj")])

def open_fourth(blob,added):
    out=fourth_from(added)
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

class ExtraProjection(nn.Module):
    def __init__(self,ni,no):
        super().__init__()
        self.down=nn.Linear(ni,REGION_RANK,bias=False)
        self.up=nn.Linear(REGION_RANK,no,bias=False)
        nn.init.normal_(self.down.weight,std=.02);nn.init.zeros_(self.up.weight)
    def forward(self,x):return self.up(F.gelu(self.down(x)))

def widen_regions(state,raw=None):
    model,_,_,full,_,child,region=state
    old_enc=full.encoder.detach().clone();old_dec=full.decoder.detach().clone()
    assert old_enc.shape==(4,320) and old_dec.shape==(320,4)
    torch.manual_seed(SEED)
    bank=nn.ModuleDict()
    for name,path in TARGETS.items():
        target=child.get_submodule(path)
        assert isinstance(target,nn.Linear),path
        bank[name]=ExtraProjection(target.in_features,target.out_features)
    # Additional recurrent channels inside the existing shared GT, not a
    # separate route or a second call to the original 12-layer model.
    enc=torch.randn(RANK,320)*.02;dec=torch.zeros(320,RANK)
    if raw is not None:
        data=load(raw);enc=data.pop("GT.encoder").float();dec=data.pop("GT.decoder").float()
        bank.load_state_dict({k:v.float() for k,v in data.items()},strict=True)
    assert enc.shape==(RANK,320) and dec.shape==(320,RANK)
    full.encoder=nn.Parameter(torch.cat((old_enc,enc),dim=0))
    full.decoder=nn.Parameter(torch.cat((old_dec,dec),dim=1));full.rank=4+RANK
    full.encoder.register_hook(lambda g:torch.cat((torch.zeros_like(g[:4]),g[4:]),dim=0))
    full.decoder.register_hook(lambda g:torch.cat((torch.zeros_like(g[:,:4]),g[:,4:]),dim=1))
    # Preserve original child serialization ABI. These are actual parallel
    # learned paths inside its existing semantic/plan/causal projections.
    model.model.add_module("g03_expanded_regions",bank)
    for name,path in TARGETS.items():
        target=child.get_submodule(path);proj=bank[name]
        target.register_forward_hook(lambda _m,args,output,proj=proj:output+proj(args[0]))
    new=[full.encoder,full.decoder,*bank.parameters()]
    return (*state[:-1],region+new),bank,(old_enc,old_dec)

def region_bytes(state,bank,old):
    full=state[3]
    assert torch.equal(full.encoder[:4].detach(),old[0]) and torch.equal(full.decoder[:,:4].detach(),old[1])
    data={"GT.encoder":full.encoder[4:].detach().half().contiguous(),
          "GT.decoder":full.decoder[:,4:].detach().half().contiguous()}
    data.update({k:v.detach().half().contiguous() for k,v in bank.state_dict().items()})
    assert all(torch.isfinite(x).all() for x in data.values())
    return save(data)

def rate(packet,eb,ab,fb,rb,geom):
    n=sum(map(len,(packet,eb,ab,fb,rb)))
    return {"bytes":n,"core_geometry":geom,"bpp_core_denominator":8*n/geom,
            "base_reference_bpp":8*n/BASE,"within_cap":8*n/geom<=.7,
            "extra_region_bytes":len(rb),"extra_region_params":sum(v.numel() for v in load(rb).values()) if rb else 0}

def run(g02,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    assert G03==92413760
    raw=(g02/"GROWTH_74M_Q8.bundle").read_bytes()
    sb=(g02/"GROWTH_74M_SECOND.bin").read_bytes()
    ab=(g02/"GROWTH_74M_DISCRETE_ADDED.bin").read_bytes()
    assert (digest(raw),digest(sb),digest(ab))==(CORE_SHA,SECOND_SHA,ADDED_SHA)
    a,s,legacy=open_bundle(raw)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tr,dv=dataset(tok,"TRAIN"),dataset(tok,"DEV")
    assert not {r["q"] for v in tr.values() for r in v}&{r["q"] for v in dv.values() for r in v}
    rng=np.random.default_rng(SEED)
    m,g=tr["math"],tr["general"]
    # Fixed balanced exposures, with repeated TRAIN rows explicitly counted.
    def sequence(rows,n):
        seq=[]
        while len(seq)<n:seq.extend(rows[int(i)] for i in rng.permutation(len(rows)))
        return seq[:n]
    training=[r for pair in zip(sequence(m,48),sequence(g,48)) for r in pair]
    assert len(training)==STEPS
    out.mkdir(parents=True,exist_ok=False)
    def restore(packet,eb,xb,fb=b"",rb=None):
        ar,sd,l=open_bundle(packet);assert l==legacy
        st=direct(ar,sd);second=extra_open(eb,ar,st[1]);install_growth(st,second)
        added=open_added(xb,st,second);install_added(st,added)
        fourth={};reg=None;old=None
        if fb:
            fourth=open_fourth(fb,added);install_fourth(st,fourth)
            st,reg,old=widen_regions(st,rb)
        return st,second,added,fourth,reg,old
    st,second,added,_,_,_=restore(raw,sb,ab)
    before=audit(st,dv,tok)
    probes=dv["math"][:3]+dv["general"][:3]
    ref=[forward(st,r["p"],r["features"]).detach().clone() for r in probes]
    fourth=fourth_from(added);install_fourth(st,fourth);st,reg,old=widen_regions(st)
    err=max(float((forward(st,r["p"],r["features"]).detach()-z).abs().max()) for r,z in zip(probes,ref))
    assert err<1e-4,err
    init_fourth=extra_pack(fourth);init_region=region_bytes(st,reg,old)
    print("G03_BASE",json.dumps({"before":before,"duplicate_init_max_logit_diff":err,
                                "TRAIN_unique":{k:len(v) for k,v in tr.items()},"geometry":G03,
                                "extra_region_params":sum(v.numel() for v in load(init_region).values())},sort_keys=True),flush=True)
    del st,second,added,fourth,reg,old,ref
    arms={}
    for name in ("CONTROL_74M","GROWTH_92M_REGIONS"):
        growth=name.startswith("GROWTH")
        st,second,added,fourth,reg,old=restore(raw,sb,ab,init_fourth if growth else b"",init_region if growth else None)
        extras=indexed_banks(second,added)|{"fourth/"+k:v for k,v in fourth.items()}
        trained=joint_train(st,tok,training,name,extras)
        rb=region_bytes(st,reg,old) if growth else b""
        if growth:
            pre,post=load(init_region),load(rb)
            changed=[k for k,v in post.items() if not torch.equal(v,pre[k])]
            assert "GT.decoder" in changed and any(k.startswith("semantic.") for k in changed) and any(k.startswith("planning.") for k in changed),changed
        else:changed=[]
        _,packet,ar,sd=payload(a,s,st,legacy)
        eb=extra_pack(second);xb=extra_pack(added);fb=extra_pack(fourth) if growth else b""
        geom=G03 if growth else G02
        storage=rate(packet,eb,xb,fb,rb,geom);assert storage["within_cap"]
        for suffix,blob in (("Q8.bundle",packet),("SECOND.bin",eb),("ADDED.bin",xb),("FOURTH.bin",fb),("REGIONS.safetensors",rb)):
            if blob:(out/(name+"_"+suffix)).write_bytes(blob)
        del st,second,added,fourth,reg,old,extras
        st,*rest=restore(packet,eb,xb,fb,rb if growth else None)
        after=audit(st,dv,tok);del st,rest
        arms[name]={"training":trained,"after":after,"storage":storage,"changed_new_region_tensors":changed,
                    "sha256":{"core":digest(packet),"second":digest(eb),"added":digest(xb),"fourth":digest(fb) if fb else None,"regions":digest(rb) if rb else None}}
        print("G03_ARM",name,json.dumps(arms[name],sort_keys=True),flush=True)
    report={"stage":"G03_REAL_FFN_GT_REGION_GROWTH","before":before,"arms":arms,"depth":12,
            "new_core_geometry":G03,"new_FFN_width":6144,"GT_rank":16,"region_parallel_rank":REGION_RANK,
            "old_core_geometry":G02,"duplicate_init_max_logit_difference":err,
            "training_unique":{k:len(v) for k,v in tr.items()},"matched_training_exposures":STEPS,
            "candidate_accepted":False,"FP32_teacher_calls":0,"output_gate":0,
            "limitations":["One extra FFN bank (36 independent compact matrices), true GT rank4 to16, learned parallel nonlinear rank16 projections in existing semantic/binding/planning/causal modules.",
                           "Hidden width320 and existing attention geometry unchanged. Not final proportional architecture; original frozen NRA3 tensors retained.",
                           "Fixed compact indices in joint Adam stage; G02 discrete indices inherited. Exact counts are logical core entries, not independent dense trainable degrees.",
                           "Small authored diagnostics; repeated TRAIN exposure counted separately; no protected 616/CPU/all-in certification. Known bytes include all extra region tensors but exclude tokenizer/runtime."]}
    (out/"G03_REPORT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("G03_ASSESS",json.dumps({"stage":report["stage"],"geometry":G03,"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g02","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g02,a.f08,a.out)
