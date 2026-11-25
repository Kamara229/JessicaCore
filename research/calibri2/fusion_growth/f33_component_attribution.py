"""F33: measure individual residual component magnitude, selectivity and effect.
Frozen real F10 and K32, diagnostic only. Eight masks of existing three
tanh gates, with prompt pairs from F30. No optimized gate selection on DEV.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from research.calibri2.fusion_growth.f30_gradient_signal import (
    PAIRS,F10_SHA,NAMES,PIN,digest,unpack,raw_core,CB59,decode,direct,
    forward,native,prompt_features,clear,norm,cosine)

LABELS=("CORE","STRUCTURE_NRA3","PLAN_NPA1")
MASKS=tuple(itertools.product((0,1),repeat=3))

def capture(state,tok,q,y,w,mask):
    model,_,_,_,_,child,_=state
    inner=child.child.child
    p,g=native(tok,q,y);p2,b=native(tok,q,w)
    assert p==p2 and len(g)==len(b)==2
    original=inner.midlayer_gates.detach().clone()
    cap={}
    # Wrap actual component method once, preserving its values and gradients.
    original_method=inner.delta_components
    def components(*args,**kwargs):
        c=original_method(*args,**kwargs)
        assert c.shape==(1,3,320) and "components" not in cap
        cap["components"]=c.detach().cpu().double().numpy()[0].copy()
        return c
    def pre(_m,args,kwargs):
        h=args[0] if args else kwargs["hidden_states"]
        cap["next_input"]=h[0,-1].detach().cpu().double().numpy().copy()
    def final(_m,args):
        cap["head"]=args[0][0,-1].detach().cpu().double().numpy().copy()
    def after(_m,args,out):
        h=out[0] if isinstance(out,tuple) else out
        cap["before_delta"]=h[0,-1].detach().cpu().double().numpy().copy()
    handles=[]
    try:
        with torch.no_grad():inner.midlayer_gates.mul_(torch.tensor(mask,dtype=original.dtype))
        effective=inner.midlayer_gates.tanh().detach().cpu().double().numpy()
        inner.delta_components=components
        handles.append(model.model.layers[5].register_forward_hook(after))
        handles.append(model.model.layers[6].register_forward_pre_hook(pre,with_kwargs=True))
        handles.append(model.lm_head.register_forward_pre_hook(final))
        z=forward(state,p,prompt_features(tok,q)).detach()[0].cpu().double().numpy().copy()
        assert set(cap)=={"components","before_delta","next_input","head"}
        cap["weighted"]=cap["components"]*effective[:,None]
        assert np.allclose(cap["next_input"],cap["before_delta"]+cap["weighted"].sum(0),rtol=2e-5,atol=3e-6)
        assert np.isfinite(z).all()
    finally:
        for h in handles:h.remove()
        # Remove temporary instance override to recover original class method.
        del inner.delta_components
        with torch.no_grad():inner.midlayer_gates.copy_(original)
        clear(state[1])
        assert torch.equal(inner.midlayer_gates,original)
    return cap,z,g[0],b[0]

def compare(a,b):
    return {"norm_mean":(norm(a)+norm(b))/2,
            "absolute_difference":norm(a-b),
            "relative_difference":norm(a-b)/max(1e-30,(norm(a)+norm(b))/2),
            "cosine":cosine(a,b)}

def run(f01,f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(20261217)
    old=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(old[k])==v
    ar=CB59(raw_core(old[NAMES[0]]));side={k:old[k] for k in NAMES[1:]}
    p=(f10/"F10_FP16.bundle").read_bytes();assert digest(p)==F10_SHA
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    out.mkdir(parents=True,exist_ok=False);results={}
    for name,a,s in (("ORIGINAL_K32",ar,side),("F10_MIXED",*decode(p))):
        state=direct(a,s);arms={};reference=None
        for mask in ((1,1,1),*(m for m in MASKS if m!=(1,1,1)),(1,1,1)):
            mode="".join(str(x) for x in mask);rows=[];finger=[]
            for q,y,q2,y2 in PAIRS:
                h,z,t,w=capture(state,tok,q,y,y2,mask)
                i,v,t2,w2=capture(state,tok,q2,y2,y,mask)
                assert t==w2 and w==t2
                finger.extend([digest(z.tobytes()),digest(v.tobytes())])
                rows.append({"raw_components":{k:compare(h["components"][j],i["components"][j]) for j,k in enumerate(LABELS)},
                             "weighted_components":{k:compare(h["weighted"][j],i["weighted"][j]) for j,k in enumerate(LABELS)},
                             "combined_delta":compare(h["weighted"].sum(0),i["weighted"].sum(0)),
                             "before_delta":compare(h["before_delta"],i["before_delta"]),
                             "next_input":compare(h["next_input"],i["next_input"]),
                             "head":compare(h["head"],i["head"]),
                             "same_argmax":int(z.argmax())==int(v.argmax()),
                             "first_correct":int(z.argmax()==t)+int(v.argmax()==t2),
                             "pair_margin":float((z[t]-z[w]+v[t2]-v[w2])/2)})
            if mode=="111":
                if reference is not None:assert finger==reference;continue
                reference=finger
            def avg(items):return {k:float(np.mean([i[k] for i in items])) for k in items[0]}
            summary={"raw_components":{k:avg([r["raw_components"][k] for r in rows]) for k in LABELS},
                     "weighted_components":{k:avg([r["weighted_components"][k] for r in rows]) for k in LABELS},
                     **{k:avg([r[k] for r in rows]) for k in ("combined_delta","before_delta","next_input","head")},
                     "first_correct":sum(r["first_correct"] for r in rows),
                     "same_argmax_pairs":sum(r["same_argmax"] for r in rows),
                     "pair_margin":float(np.mean([r["pair_margin"] for r in rows]))}
            arms[mode]={"summary":summary,"records":rows}
            print("F33_ARM",name,mode,json.dumps(summary,sort_keys=True),flush=True)
        results[name]=arms;del state
    report={"stage":"F33_RESIDUAL_COMPONENT_ATTRIBUTION","arms":results,"mask_order":LABELS,
            "restored_full_logits_exact":True,"optimizer_steps":0,"new_parameters":0,
            "candidate_accepted":False,"output_gate":0,"FP32_teacher_calls":0,
            "limitations":["Same four diagnostic pairs as F30/F31; no independent quality claim.",
                           "Mask interventions change effective gate contributions, not component weights.",
                           "No selection or promotion of a mask based on this diagnostic; no protected J7 or CPU certification."]}
    (out/"F33_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F33_ASSESS",json.dumps({"stage":report["stage"],"restored_full_logits_exact":True},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f10,a.f08,a.out)
