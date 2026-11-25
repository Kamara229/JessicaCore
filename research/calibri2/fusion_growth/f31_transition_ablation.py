"""F31: identify signal changes before/after layer-5 child and layer-6 block.
Actual original K32 and mixed F10, frozen weights, reversible ablations.
No optimization, no protected prompts, no new weights.
"""
import argparse
import json
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from research.calibri2.fusion_growth.f30_gradient_signal import (
    PAIRS,F10_SHA,NAMES,PIN,digest,unpack,raw_core,CB59,decode,direct,
    forward,native,prompt_features,clear,norm,cosine)

MODES=("FULL","ZERO_CHILD_DELTA","ZERO_GT_GATES","ZERO_EXPANSION",
       "ZERO_ALL_THREE")

@contextmanager
def ablate(state,mode):
    _,_,_,full,expansion,child,_=state
    g=full.gates.detach().clone();u=expansion.up.detach().clone()
    hook=None
    try:
        with torch.no_grad():
            if mode in ("ZERO_GT_GATES","ZERO_ALL_THREE"):full.gates.zero_()
            if mode in ("ZERO_EXPANSION","ZERO_ALL_THREE"):expansion.up.zero_()
        if mode in ("ZERO_CHILD_DELTA","ZERO_ALL_THREE"):
            # Preserve differentiability, execution and original child hook
            # contracts; only suppress its residual contribution.
            hook=child.register_forward_hook(lambda _m,_a,out:out*0.)
        yield
    finally:
        if hook is not None:hook.remove()
        with torch.no_grad():full.gates.copy_(g);expansion.up.copy_(u)
        assert torch.equal(full.gates,g) and torch.equal(expansion.up,u)

def observe(state,tok,q,y,wrong,mode):
    model=state[0];child=state[5]
    p,g=native(tok,q,y);p2,w=native(tok,q,wrong)
    assert p==p2 and len(g)==len(w)==2
    cap={};hooks=[]
    def save(name,h):
        assert name not in cap and h.shape[-1]==320
        cap[name]=h[0,-1].detach().cpu().double().numpy().copy()
    def pre(name):
        def cb(_m,args,kwargs):
            h=args[0] if args else kwargs["hidden_states"]
            save(name,h)
        return cb
    def post(name):
        def cb(_m,args,out):
            h=out[0] if isinstance(out,tuple) else out
            save(name,h)
        return cb
    def delta(_m,args,out):
        assert out.shape==(1,320)
        cap["child_delta"]=out[0].detach().cpu().double().numpy().copy()
    with ablate(state,mode):
        try:
            # prepend observes raw block output before existing GT/expansion
            # residual hooks; appended observer precedes dynamically installed
            # child injection. Next block prehook sees actual injected state.
            for j in (5,6):
                m=model.model.layers[j]
                hooks.append(m.register_forward_pre_hook(pre(f"L{j}_input"),with_kwargs=True))
                hooks.append(m.register_forward_hook(post(f"L{j}_raw"),prepend=True))
                hooks.append(m.register_forward_hook(post(f"L{j}_after_fixed_hooks")))
            hooks.append(child.register_forward_hook(delta))
            hooks.append(model.lm_head.register_forward_pre_hook(pre("final_head_input"),with_kwargs=True))
            z=forward(state,p,prompt_features(tok,q)).detach()[0].cpu().double().numpy().copy()
        finally:
            for h in hooks:h.remove()
    assert len(cap)==8 and z.shape==(49152,) and all(np.isfinite(a).all() for a in (*cap.values(),z))
    if mode in ("ZERO_CHILD_DELTA","ZERO_ALL_THREE"):assert norm(cap["child_delta"])==0
    assert np.allclose(cap["L6_input"],cap["L5_after_fixed_hooks"]+cap["child_delta"],rtol=1e-5,atol=2e-6)
    clear(state[1])
    return cap,z,g[0],w[0]

def diff(a,b):
    return {"relative_difference":norm(a-b)/max(1e-30,norm(a)),
            "absolute_difference":norm(a-b),"cosine":cosine(a,b),
            "norm_a":norm(a),"norm_b":norm(b)}

def run(f01,f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(20261216)
    old=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(old[k])==v
    ar=CB59(raw_core(old[NAMES[0]]));side={k:old[k] for k in NAMES[1:]}
    p=(f10/"F10_FP16.bundle").read_bytes();assert digest(p)==F10_SHA
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    out.mkdir(parents=True,exist_ok=False);results={}
    for name,a,s in (("ORIGINAL_K32",ar,side),("F10_MIXED",*decode(p))):
        state=direct(a,s);arms={};ref=None
        for mode in (*MODES,"FULL_RESTORE"):
            rows=[];finger=[]
            for q,y,q2,y2 in PAIRS:
                h,z,t,w=observe(state,tok,q,y,y2,"FULL" if mode=="FULL_RESTORE" else mode)
                i,v,t2,w2=observe(state,tok,q2,y2,y,"FULL" if mode=="FULL_RESTORE" else mode)
                assert t==w2 and w==t2 and set(h)==set(i)
                finger.extend([digest(z.tobytes()),digest(v.tobytes())])
                rows.append({"states":{k:diff(h[k],i[k]) for k in h},
                             "same_argmax":int(z.argmax())==int(v.argmax()),
                             "first_correct":int(z.argmax()==t)+int(v.argmax()==t2),
                             "paired_margin":float((z[t]-z[w]+v[t2]-v[w2])/2),
                             "child_delta_norm":(norm(h["child_delta"])+norm(i["child_delta"]))/2})
            if mode=="FULL":ref=finger
            if mode=="FULL_RESTORE":assert finger==ref;continue
            summary={"mean_relative_difference":{k:float(np.mean([r["states"][k]["relative_difference"] for r in rows])) for k in rows[0]["states"]},
                     "mean_absolute_difference":{k:float(np.mean([r["states"][k]["absolute_difference"] for r in rows])) for k in rows[0]["states"]},
                     "same_argmax_pairs":sum(r["same_argmax"] for r in rows),
                     "first_correct":sum(r["first_correct"] for r in rows),
                     "paired_margin":float(np.mean([r["paired_margin"] for r in rows])),
                     "child_delta_norm":float(np.mean([r["child_delta_norm"] for r in rows]))}
            arms[mode]={"summary":summary,"records":rows}
            print("F31_ARM",name,mode,json.dumps(summary,sort_keys=True),flush=True)
        results[name]=arms;del state
    report={"stage":"F31_TRANSITION_ABLATION","arms":results,"restored_full_logits_exact":True,
            "optimizer_steps":0,"new_parameters":0,"FP32_teacher_calls":0,
            "candidate_accepted":False,"output_gate":0,
            "limitations":["Four counterfactual pairs reused from F30 for diagnostic attribution, not independent benchmark.",
                           "GT ablation suppresses full-layer gate feedback only, not every original tunnel/bridge.",
                           "Expansion and child ablation are diagnostic interventions, not candidate removals.",
                           "No historical replay, speed, or new training claim."]}
    (out/"F31_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F31_ASSESS",json.dumps({"stage":report["stage"],"restored_full_logits_exact":True},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f10,a.f08,a.out)
