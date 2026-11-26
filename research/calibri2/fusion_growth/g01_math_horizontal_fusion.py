"""G01: first actual horizontal growth pilot, math specialist -> fused FFN.
User authorized starting growth before quality plateau. Train a physical
specialist on the existing integrated F10 model, combine both sets of FFN
neurons in each of the 12 layers, average compatible shared non-FFN and
region weights, then joint mixed-domain recovery. No model router.
This is first width step, not proportional hidden/attention/region growth.
"""
import argparse
import json
import lzma
import struct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load,save
from v176.operators import TrainWeight,clear
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,digest,metrics
from research.calibri2.fusion_growth.f03_rate_train_core import decode,encode,k_for,pack_indices,unpack_indices
from research.calibri2.fusion_growth.f06_integrated_recovery import forward,evaluate,OLD_PREFIX
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native,supervised
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,open_bundle,HDR
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f20_state_diagnostic import F10_SHA
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features

RATE="K16_embed_K8_body"
SEED=20261222
BASE=36380480
EXTRA=12*3*320*1536
GROWN=BASE+EXTRA
SPECIAL_STEPS=64
RECOVERY_STEPS=48

def rows(tok,split):
    tr=split=="TRAIN";n=16 if tr else 6
    math=[];general=[]
    for i in range(n):
        a=(37 if tr else 403)+13*i;b=2+i%9;c=4+i%7
        defs=[(f"Вычисли ({a} + {b}) - {c}.",a+b-c,a+b+c),
              (f"Найди значение {a} + {b} × {c}.",a+b*c,(a+b)*c),
              (f"Реши уравнение x - {b} = {a}. Найди x.",a+b,a-b),
              (f"Найди {b} из {c+b} равных частей числа {(c+b)*a}.",b*a,c*a)]
        for q,y,w in defs:math.append((q+" Запиши только число.",str(y),str(w)))
        x,y=("Иван","Мира") if tr else ("Лев","Лана")
        general.extend([(f"Запись {a}: {x} передал {y} ключ. Кто получил ключ?",y,x),
                        (f"Карточка {a}: у {x} есть зонт. Есть ли у {x} зонт?","да","нет")])
    def pack(data,kind):
        out=[]
        for q,y,w in data:
            p,g=native(tok,q,y);p2,bad=native(tok,q,w)
            assert p==p2 and g!=bad
            div=next(j for j,(u,v) in enumerate(zip(g,bad)) if u!=v)
            out.append({"q":q,"p":p,"g":g,"bad":bad,"div":div,"answer":y,
                        "wrong":w,"features":prompt_features(tok,q),"split":split,"kind":kind})
        return out
    return {"math":pack(math,"math"),"general":pack(general,"general")}

def audit(state,ds,tok):
    return {k:evaluate(state,v,tok) for k,v in ds.items()}

def payload(archive,side,state,legacy):
    _,sources,_,full,_,child,_=state
    ns=dict(side);original=load(side[NAMES[3]])
    child_state={OLD_PREFIX+k:v.detach().cpu().half().contiguous() for k,v in child.state_dict().items()}
    assert set(child_state)==set(original)
    assert all(torch.equal(v,original[k]) for k,v in child_state.items() if ".nra3_neural_weights." in k)
    ns[NAMES[3]]=save(child_state)
    key=next(iter(load(side[NAMES[4]])))
    ns[NAMES[4]]=save({key:full.gates.detach().cpu().half().contiguous()})
    fp=encode(archive.blob,sources,RATE,ns)
    a,s=decode(fp);assert s==ns
    compact,head,_=split_head(ns,8)
    cp=encode(archive.blob,sources,RATE,compact)
    packet=HDR.pack(b"FG09",len(cp),len(head),len(legacy))+cp+head+legacy
    qa,qs,ql=open_bundle(packet);assert qa.blob==a.blob and ql==legacy
    return fp,packet,qa,qs

def mlp_names(sources):
    names=sorted(k for k in sources if ".mlp." in k)
    assert len(names)==36
    return names

class WideMLP(nn.Module):
    """Algebraically one 3072-neuron SwiGLU, two independent compact banks.
    Split down-projection sum avoids concatenated dense weight allocation.
    """
    def __init__(self,old,gate,up,down):
        super().__init__()
        self.first=old
        self.second_gate=gate;self.second_up=up;self.second_down=down
        self.intermediate_size=3072
    def forward(self,x):
        second=(self.first.act_fn(torch.nn.functional.linear(x,self.second_gate()))*
                torch.nn.functional.linear(x,self.second_up()))
        second=torch.nn.functional.linear(second,self.second_down())
        first=self.first(x)
        self.second_gate.clear();self.second_up.clear();self.second_down.clear()
        return .5*(first+second)

def install_growth(state,extra):
    model,sources=state[:2]
    assert set(extra)==set(mlp_names(sources))
    for i,layer in enumerate(model.model.layers):
        prefix=f"model.layers.{i}.mlp."
        layer.mlp=WideMLP(layer.mlp,*[extra[prefix+k+".weight"] for k in ("gate_proj","up_proj","down_proj")])
    assert len(model.model.layers)==12
    assert all(layer.mlp.intermediate_size==3072 for layer in model.model.layers)
    assert sum(s.shape[0]*s.shape[1] for s in extra.values())==EXTRA
    assert all(extra[k].codes.data_ptr()!=sources[k].codes.data_ptr() for k in extra)

def extra_from(archive,sources):
    out={}
    for k in mlp_names(sources):
        s=TrainWeight(archive.handles[k]);s.log_scales.requires_grad_(False)
        assert tuple(s.shape)==tuple(sources[k].shape)
        out[k]=s
    return out

def extra_pack(extra):
    meta=[];parts=[]
    for k,s in sorted(extra.items()):
        c,scale=s.fields()
        c=c.detach().cpu().numpy().astype("<f2");scale=scale.detach().cpu().numpy().astype("<f2")
        ids=s.codes.detach().cpu().numpy()
        assert np.all(ids<8) and not np.any(c[8:]) and np.isfinite(c).all() and np.isfinite(scale).all()
        items=(pack_indices(ids,3),c[:8].tobytes(),scale.tobytes())
        meta.append({"name":k,"shape":list(s.shape),"count":len(ids),"sizes":[len(x) for x in items]})
        parts.extend(items)
    h=json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
    return b"G01X"+lzma.compress(struct.pack("<I",len(h))+h+b"".join(parts),preset=6)

def extra_open(blob,archive,sources):
    assert blob[:4]==b"G01X"
    raw=lzma.decompress(blob[4:]);n,=struct.unpack_from("<I",raw)
    meta=json.loads(raw[4:4+n]);p=4+n;out=extra_from(archive,sources)
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

def mix(state,special):
    _,sources,_,full,_,child,_=state
    _,ss,_,sf,_,sc,_=special
    with torch.no_grad():
        for k,s in sources.items():
            assert torch.equal(s.codes,ss[k].codes) and torch.equal(s.base_scales,ss[k].base_scales)
            if ".mlp." not in k:s.centroids.lerp_(ss[k].centroids,.5)
        full.gates.lerp_(sf.gates,.5)
        for (k,p),(j,q) in zip(child.named_parameters(),sc.named_parameters()):
            assert k==j and p.shape==q.shape
            if p.requires_grad:p.lerp_(q,.5)
            else:assert torch.equal(p,q)

def train(state,tok,data,stage,extra=None):
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
        if (i+1)%16==0:print("G01_PROGRESS",stage,i+1,l,len(seen),len(xseen),len(gt),flush=True)
    assert len(seen)==85 and len(gt)==12 and len(rg)>0
    if extra:assert len(xseen)==36
    return {"steps":len(data),"connected_base_codebooks":len(seen),"connected_added_codebooks":len(xseen),
            "connected_GT":len(gt),"connected_region_tensors":len(rg),"first16_objective":float(np.mean(losses[:16])),
            "last16_objective":float(np.mean(losses[-16:]))}

def physical_rate(packet,extra=b""):
    n=len(packet)+len(extra)
    return {"bytes":n,"base_reference_bpp":8*n/BASE,
            "expanded_geometry_bpp":8*n/GROWN if extra else None,
            "known_payload_cap":int(.7*(GROWN if extra else BASE)/8),
            "within_geometry_budget":8*n/(GROWN if extra else BASE)<=.7}

def run(f10,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    raw=(f10/"F10_FP16.bundle").read_bytes();assert digest(raw)==F10_SHA
    archive,side=decode(raw);assert archive.meta["original_unique_parameter_count"]==BASE
    legacy=(f08/"legacy_extras.bundle").read_bytes();assert len(legacy)==137712
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tr,dv=rows(tok,"TRAIN"),rows(tok,"DEV")
    assert len(tr["math"])==SPECIAL_STEPS and len(tr["general"])==32 and len(dv["math"])==24 and len(dv["general"])==12
    assert not {r["q"] for v in tr.values() for r in v}&{r["q"] for v in dv.values() for r in v}
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(SEED)
    state=direct(archive,side);base_audit=audit(state,dv,tok)
    print("G01_BASE",json.dumps(base_audit,sort_keys=True),flush=True)
    # Geometry-only check: duplicate the same physical weights before mixing.
    probe=dv["math"][:4]+dv["general"][:2]
    ref=[forward(state,r["p"],r["features"]).detach().clone() for r in probe]
    duplication=extra_from(archive,state[1]);install_growth(state,duplication)
    parity=[float((forward(state,r["p"],r["features"]).detach()-z).abs().max()) for r,z in zip(probe,ref)]
    assert max(parity)<1e-4
    del state,duplication,ref
    print("G01_WIDTH_PARITY",json.dumps({"max_logit_difference":max(parity),"probes":len(parity),"expanded_geometry":GROWN}),flush=True)
    # Phase 1: math-specialized actual full model, no FP32 teacher.
    spec=direct(archive,side)
    spec_training=train(spec,tok,[tr["math"][int(i)] for i in rng.permutation(SPECIAL_STEPS)],"MATH_SPECIALIST")
    fp,q,sa,sd=payload(archive,side,spec,legacy)
    (out/"CALIBRI_2_0_1_1_MATH_FP16.bundle").write_bytes(fp)
    (out/"CALIBRI_2_0_1_1_MATH_Q8.bundle").write_bytes(q)
    spec_eval=direct(sa,sd);spec_audit=audit(spec_eval,dv,tok);del spec_eval,spec
    print("G01_SPECIALIST",json.dumps({"audit":spec_audit,"training":spec_training,"storage":physical_rate(q),"sha256":digest(q)},sort_keys=True),flush=True)
    # Phase 2: combine actual physically reloaded specialist with original
    # base. Keep original FFN bank and add specialist FFN neurons in EVERY layer.
    sa,ss=decode(fp)
    spec=direct(sa,ss);state=direct(archive,side)
    mix(state,spec);extra=extra_from(sa,state[1]);install_growth(state,extra);del spec
    _,pre,a,s=payload(archive,side,state,legacy);eb=extra_pack(extra)
    (out/"G01_FUSED_BEFORE_Q8.bundle").write_bytes(pre)
    (out/"G01_FUSED_BEFORE_EXTRA.bin").write_bytes(eb)
    del state,extra
    state=direct(a,s);extra=extra_open(eb,a,state[1]);install_growth(state,extra)
    pre_audit=audit(state,dv,tok)
    print("G01_FUSED_BEFORE",json.dumps({"audit":pre_audit,"storage":physical_rate(pre,eb)},sort_keys=True),flush=True)
    # Phase 3: joint recovery, balance 24 math and 24 general TRAIN rows.
    mm=[tr["math"][int(i)] for i in rng.permutation(64)[:24]]
    gg=[tr["general"][int(i)] for i in rng.permutation(32)[:24]]
    recovery=[r for pair in zip(mm,gg) for r in pair];assert len(recovery)==RECOVERY_STEPS
    # Start recovery from full precision side tensors rather than quantizing twice.
    del state,extra
    state=direct(archive,side);spec=direct(sa,ss);mix(state,spec);extra=extra_from(sa,state[1]);install_growth(state,extra);del spec
    trained=train(state,tok,recovery,"FUSED_JOINT_RECOVERY",extra)
    f,q,a,s=payload(archive,side,state,legacy);eb=extra_pack(extra)
    (out/"G01_FUSED_RECOVERED_Q8.bundle").write_bytes(q)
    (out/"G01_FUSED_RECOVERED_EXTRA.bin").write_bytes(eb)
    assert physical_rate(q,eb)["within_geometry_budget"]
    del state,extra
    state=direct(a,s);extra=extra_open(eb,a,state[1]);install_growth(state,extra)
    post=audit(state,dv,tok)
    report={"stage":"G01_MATH_SPECIALIST_AND_HORIZONTAL_FUSION","base_sha256":F10_SHA,
            "base_geometry":BASE,"expanded_geometry":GROWN,"added_logical_matrix_entries":EXTRA,
            "depth":12,"hidden_width":320,"old_ffn_width":1536,"new_ffn_width":3072,
            "independent_added_compact_matrices":36,"duplicate_init_max_logit_difference":max(parity),
            "base_DEV":base_audit,"specialist_DEV":spec_audit,"specialist_training":spec_training,
            "fusion_before_DEV":pre_audit,"fusion_after_DEV":post,"recovery_training":trained,
            "fused_storage":physical_rate(q,eb),"fused_core_sha256":digest(q),"fused_extra_sha256":digest(eb),
            "candidate_accepted":False,"output_gate":0,"FP32_teacher_calls":0,
            "limitations":["Actual FFN horizontal fusion only; hidden width, attention, GT and region geometry unchanged; not final proportional 135M architecture.",
                           "Expanded logical matrix entries, not that many independent optimizer degrees of freedom; each bank has independent physical codes/scales and trainable codebooks, discrete IDs fixed in this round.",
                           "Both FFN banks contribute to every layer and request; no task router. Shared non-FFN and trainable region tensors averaged, old frozen weights unchanged.",
                           "Separate extra physical packet charged in full; known payload only, tokenizer/runtime excluded. No CPU or protected 616 certification.",
                           "Research F10 base not accepted and F37 legacy retention is a different path. Specialist name provisional; independent quality not assumed."]}
    (out/"G01_REPORT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("G01_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f10,a.f08,a.out)
