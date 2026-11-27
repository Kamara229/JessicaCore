"""G40: causal, reversible L6 component ablation on actual G36 Joint.
No parameter updates, no deployment. Capture raw decoder output before any
layer hook, then each existing hook, learned L6 residual, and L7 input
AFTER temporary NRA3/NPA1 intervention. Distinguish hook-order artifacts.
"""
import argparse,json
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
import torch
from torch.nn import functional as F
from safetensors.torch import load
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import Residual,install,parameters,PIN,restore,digest,block
from research.calibri2.fusion_growth.g35_full_mps_preflight import native
from research.calibri2.fusion_growth.g36_alphabet_margin import suite,encode
from research.calibri2.fusion_growth.g32_word_foundation_mps import LETTERS
from research.calibri2.fusion_growth.g38_context_readout import PARENT
from v176.operators import clear

def hidden(o):
    return o[0] if isinstance(o,tuple) else o
def replace(o,h):
    return (h,*o[1:]) if isinstance(o,tuple) else h
def med(x):
    return float(torch.tensor(x).median())
def separation(xs):
    h=F.normalize(torch.stack(xs).float(),dim=-1)
    d=1-h@h.T
    mask=torch.triu(torch.ones_like(d,dtype=torch.bool),diagonal=1)
    return float(d[mask].median())
def rel_delta(before,after):
    return float((after-before).norm()/before.norm().clamp_min(1.e-9))

@contextmanager
def instrument(st,l6,arm,static):
    model,_,_,_,_,child,_=st
    layer=model.model.layers[5];values={};relative={};handles=[]
    def record(k,h):
        assert k not in values,(k,arm)
        assert h.ndim==3 and h.shape[0]==1 and h.shape[-1]==320
        values[k]=h[0,-1].detach().clone()
    def cap_pre(k):
        return lambda m,a:record(k,a[0])
    def cap_post(k):
        return lambda m,a,o:record(k,hidden(o))
    def blend(o,old,g):
        if g==1.:return o
        return replace(o,old+(hidden(o)-old)*g)
    handles.append(model.model.layers[4].register_forward_hook(cap_post("L5_after_static")))
    handles.append(layer.register_forward_pre_hook(cap_pre("L6_input")))
    handles.append(layer.input_layernorm.register_forward_pre_hook(cap_pre("attention_residual_input")))
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(cap_pre("after_attention_residual")))
    handles.append(layer.register_forward_hook(cap_post("L6_raw_before_hooks"),prepend=True))
    handles.append(model.model.layers[6].register_forward_pre_hook(cap_pre("L7_input_after_NRA3")))
    handles.append(model.model.norm.register_forward_hook(cap_post("final_norm")))
    for sub,name in [(layer.self_attn,"attention"),(layer.mlp,"mlp")]:
        def subhook(m,a,o,name=name):
            record(name+"_delta",hidden(o))
            if arm==name+"_zero":return replace(o,hidden(o)*0.)
        handles.append(sub.register_forward_hook(subhook))
    # Wrap exact installed static hooks without moving them, and preserve
    # their side effects (e.g. recurrent GT state). Scale only emitted delta.
    for key,hook,label in static:
        def wrapped(m,a,o,hook=hook,label=label):
            old=hidden(o)
            z=hook(m,a,o)
            z=o if z is None else z
            relative[label]=rel_delta(old.detach(),hidden(z).detach())
            gain=0. if arm in ("static_all_zero","zero_"+label) else 1.
            z=blend(z,old,gain)
            record(label,hidden(z))
            return z
        layer._forward_hooks[key]=wrapped
    def residual(m,a,o):
        old=hidden(o);new=l6(old)
        relative["L6_adapter"]=rel_delta(old.detach(),new.detach())
        new=old if arm=="l6_zero" else new
        record("after_L6_adapter",new)
        return replace(o,new)
    handles.append(layer.register_forward_hook(residual))
    def childhook(m,a,o):
        assert o.shape==(1,320)
        relative["NRA3_delta_norm"]=float(o.detach().norm())
        relative["NRA3_relative_to_pre_injection"]=float(o.detach().norm()/values["after_L6_adapter"].norm().clamp_min(1.e-9))
        gain={"nra_zero":0.,"nra_quarter":.25}.get(arm,1.)
        return o*gain if gain!=1. else o
    handles.append(child.register_forward_hook(childhook))
    try:yield values,relative
    finally:
        for h in handles:h.remove()
        for key,hook,label in static:layer._forward_hooks[key]=hook

def capture(st,r,slot,l6,arm,static):
    with instrument(st,l6,arm,static) as (values,relative):
        z,z2=native(st,r["p"],r["features"],slot)
        logits=z.detach()[0];second=z2.detach()[0]
        t=r["g"][0]
        ranks=int((logits>logits[t]).sum())+1
        alt=logits.clone();alt[t]=-torch.inf
        return values,relative,{"first":int(logits.argmax()),"second":int(second.argmax()),"gold":t,"rank":ranks,"margin":float(logits[t]-alt.max()),"first_correct":int(logits.argmax())==t}

def run(g05,f08,parent,out):
    torch.manual_seed(20270126);torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    _,dev=suite(tok)
    fresh=[encode(tok,"Воспроизведи написанный символ «"+a+"».",a) for a in LETTERS]
    # G36 DEV has two prompt forms; use fixed groups to avoid conflating
    # between-template differences with between-letter representation distance.
    sets={"fresh":fresh,"dev_a":dev[::2],"dev_b":dev[1::2]}
    assert all(len(v)==33 and len({r["answer"] for r in v})==33 for v in sets.values())
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
    frozen=list({id(p):p for p in list(parameters(restored))+list(l6.parameters())+list(slot.parameters())}.values())
    versions={id(p):p._version for p in frozen}
    layer=model.model.layers[5]
    static=[(k,h,"hook_"+str(i)+"_"+getattr(h,"__qualname__",type(h).__name__)) for i,(k,h) in enumerate(layer._forward_hooks.items())]
    assert not any(k in layer._forward_hooks_with_kwargs for k,_,_ in static),"unexpected static hook kwargs"
    # Check exact control identity against the original runtime, not just
    # the same instrumented implementation.
    r=fresh[0]
    with install(st,6,l6):ref=block(st,r["p"],r["features"],slot)[0].detach()
    with instrument(st,l6,"base",static):
        a,b=native(st,r["p"],r["features"],slot)
        err=float((torch.stack((a,b),dim=1).detach()-ref).abs().max())
    assert err<1.e-4,err
    out.mkdir(parents=True,exist_ok=False)
    arms=["base","nra_zero","nra_quarter","l6_zero","static_all_zero","attention_zero","mlp_zero"]+["zero_"+label for _,_,label in static]
    results={}
    print("G40_HOOK_ORDER",json.dumps({"static":[label for _,_,label in static],"control_logit_error":err}),flush=True)
    for arm in arms:
        result={}
        for name,rows in sets.items():
            vectors=defaultdict(list);ratios=defaultdict(list);records=[]
            for r in rows:
                vals,rel,item=capture(st,r,slot,l6,arm,static)
                for k,v in vals.items():vectors[k].append(v)
                for k,v in rel.items():ratios[k].append(v)
                records.append({"q":r["q"],**item})
            result[name]={"n":len(rows),"first_correct":sum(x["first_correct"] for x in records),"unique_first":len({x["first"] for x in records}),"median_gold_rank":med([x["rank"] for x in records]),"median_gold_margin":med([x["margin"] for x in records]),"cosine_distance":{k:separation(v) for k,v in vectors.items()},"median_relative_delta":{k:med(v) for k,v in ratios.items()}}
            (out/(str(arms.index(arm))+"_"+name+".json")).write_text(json.dumps(records,ensure_ascii=False))
        results[arm]=result
        assert all(p._version==versions[id(p)] for p in frozen)
        print("G40_ARM",arm,json.dumps(result),flush=True)
        (out/"partial.json").write_text(json.dumps(results,indent=2))
    with install(st,6,l6):post=block(st,r["p"],r["features"],slot)[0].detach()
    with install(st,6,l6):again=block(st,r["p"],r["features"],slot)[0].detach()
    assert torch.equal(post,again)
    report={"parent_sha":PARENT,"control_logit_error":err,"static_hook_order":[label for _,_,label in static],"results":results,"model_weights_changed":False,"candidate_accepted":False,"limitations":["CPU causal diagnostic, no training; temporary ablation only. Static hooks retain recurrent side effects, so output attenuation is not equivalent to deleting a tunnel.","G39 layer hooks were installed before temporary NRA3 injection; G40 captures L7 input after injection. Raw attention/MLP zero arms are destructive counterfactuals, not proposed production changes.","No high precision original same-trained L6 available; do not fabricate K16 from K8. No historical J7, speed, or independent alphabet certification."]}
    (out/"G40_REPORT.json").write_text(json.dumps(report,indent=2));print("G40_REPORT",json.dumps(report),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out)
