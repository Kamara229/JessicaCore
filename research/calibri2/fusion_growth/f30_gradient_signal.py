"""F30: audit answer-dependent gradients vs common lexical learning.
Frozen K32 and F10 native path. Same-context counterfactual question pairs.
Measure actual full-vocab CE gradients and paired logit-difference gradients.
No optimization, no protected prompts, no new weights.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,unpack
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f03_rate_train_core import decode
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f06_integrated_recovery import forward
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native
from research.calibri2.fusion_growth.f20_state_diagnostic import F10_SHA
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from direct_codebook_v59 import CB59
from v176.operators import clear

PAIRS=(
    ("Коды: Сава — 2, Нора — 4. Назови код Савы одной цифрой.","2","Коды: Сава — 2, Нора — 4. Назови код Норы одной цифрой.","4"),
    ("Ряд: мыло, ваза, перо, стекло. Назови номер вазы, считая от 1.","2","Ряд: мыло, ваза, перо, стекло. Назови номер стекла, считая от 1.","4"),
    ("У Таи код 1, у Дэна код 3. Какой код у Таи? Ответь цифрой.","1","У Таи код 1, у Дэна код 3. Какой код у Дэна? Ответь цифрой.","3"),
    ("По порядку: банка, лента, папка, кисть. Назови позицию банки от 1.","1","По порядку: банка, лента, папка, кисть. Назови позицию папки от 1.","3"),
)

def groups(state):
    _,sources,_,full,_,child,params=state
    result={}
    for name,s in sorted(sources.items()):
        g="EMBED" if name=="model.embed_tokens.weight" else "L"+name.split(".")[2].zfill(2)
        result.setdefault(g,[]).append(s.centroids)
    result["GT"]=[full.gates]
    allowed={id(p) for p in params}
    result["NATIVE_HEAD"]=[p for n,p in child.named_parameters() if id(p) in allowed and n.startswith("language_head.")]
    result["NEURAL_REGIONS"]=[p for n,p in child.named_parameters() if id(p) in allowed and not n.startswith("language_head.")]
    assert len(result)==16 and all(result.values())
    flat=[p for ps in result.values() for p in ps]
    assert len(flat)==len({id(p) for p in flat}) and all(p.requires_grad for p in flat)
    return result,flat

def measure(state,tok,q,answer,wrong,group,flat):
    p,g=native(tok,q,answer);p2,w=native(tok,q,wrong)
    assert p==p2 and len(g)==len(w)==2 and g[-1]==w[-1]==2 and g[0]!=w[0]
    cap={};hooks=[]
    def hook(i):
        def cb(_m,_a,out):
            h=out[0] if isinstance(out,tuple) else out
            cap[i]=h[0,-1].detach().float().cpu().numpy().copy()
        return cb
    try:
        for i,m in enumerate(state[0].model.layers):hooks.append(m.register_forward_hook(hook(i)))
        z=forward(state,p,prompt_features(tok,q))
    finally:
        for h in hooks:h.remove()
    assert set(cap)==set(range(12)) and z.shape==(1,49152)
    ce=F.cross_entropy(z,torch.tensor([g[0]]))
    contrast=z[0,g[0]]-z[0,w[0]]
    def vector(loss,retain):
        grad=torch.autograd.grad(loss,flat,retain_graph=retain,allow_unused=True)
        values={id(p):(v.detach().float().reshape(-1).cpu().numpy().copy() if v is not None else np.zeros(p.numel(),np.float32)) for p,v in zip(flat,grad)}
        return {k:np.concatenate([values[id(p)] for p in ps]).astype(np.float64) for k,ps in group.items()}
    a=vector(ce,True);b=vector(contrast,False)
    assert all(np.isfinite(x).all() for x in (*a.values(),*b.values()))
    result={"target":g[0],"wrong":w[0],"rank":1+int((z[0]>z[0,g[0]]).sum()),"argmax":int(z.argmax()),"gap":float(contrast.detach()),"CE":float(ce.detach())}
    clear(state[1])
    return result,a,b,np.stack([cap[i] for i in range(12)]).astype(np.float64),z.detach().cpu().numpy()[0].astype(np.float64)

def norm(x):return float(np.linalg.norm(x))
def cosine(a,b):return float(np.dot(a,b)/max(1e-30,norm(a)*norm(b)))
def pair_metrics(a,b):
    return {"norm_a":norm(a),"norm_b":norm(b),"cosine":cosine(a,b),
            "pair_mean_norm":norm((a+b)/2),"mean_individual_norm":(norm(a)+norm(b))/2,
            "mean_to_individual_ratio":norm(a+b)/max(1e-30,norm(a)+norm(b))}

def run(f01,f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(20261215)
    old=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(old[k])==v
    ar=CB59(raw_core(old[NAMES[0]]));side={k:old[k] for k in NAMES[1:]}
    p=(f10/"F10_FP16.bundle").read_bytes();assert digest(p)==F10_SHA
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    out.mkdir(parents=True,exist_ok=False)
    results={}
    for name,a,s in (("ORIGINAL_K32",ar,side),("F10_MIXED",*decode(p))):
        state=direct(a,s);group,flat=groups(state);records=[]
        for q,y,q2,y2 in PAIRS:
            x,c,d,h,z=measure(state,tok,q,y,y2,group,flat)
            v,e,f,i,w=measure(state,tok,q2,y2,y,group,flat)
            assert x["target"]==v["wrong"] and v["target"]==x["wrong"]
            record={"first":x,"second":v,
                    "same_argmax":x["argmax"]==v["argmax"],
                    "output_centered_relative_difference":norm((z-z.mean())-(w-w.mean()))/max(1e-30,norm(z-z.mean())),
                    "layers":{str(j):{"cosine":cosine(h[j],i[j]),"relative_difference":norm(h[j]-i[j])/max(1e-30,norm(h[j]))} for j in range(12)},
                    "CE_gradients":{k:pair_metrics(c[k],e[k]) for k in group},
                    "correct_minus_wrong_gradients":{k:pair_metrics(d[k],f[k]) for k in group}}
            records.append(record)
        del state,group,flat
        summary={"pairs":len(records),"same_argmax":sum(r["same_argmax"] for r in records),
                 "mean_layer_relative_difference":{str(j):float(np.mean([r["layers"][str(j)]["relative_difference"] for r in records])) for j in range(12)},
                 "mean_CE_pair_gradient_ratio":{k:float(np.mean([r["CE_gradients"][k]["mean_to_individual_ratio"] for r in records])) for k in records[0]["CE_gradients"]},
                 "mean_contrast_pair_gradient_ratio":{k:float(np.mean([r["correct_minus_wrong_gradients"][k]["mean_to_individual_ratio"] for r in records])) for k in records[0]["CE_gradients"]}}
        results[name]={"summary":summary,"records":records}
        print("F30_ARM",name,json.dumps(summary,sort_keys=True),flush=True)
    report={"stage":"F30_FROZEN_GRADIENT_SIGNAL","arms":results,"pairs":PAIRS,"optimizer_steps":0,
            "new_parameters":0,"FP32_teacher_calls":0,"candidate_accepted":False,"output_gate":0,
            "limitations":["Four authored same-context counterfactual pairs, not independent performance benchmark.",
                           "First-token CE and correct-minus-wrong gradient only; no optimizer or causal claim about training failure.",
                           "A low pair gradient ratio can indicate cancellation of common or discriminative gradients, interpreted alongside actual norms and states.",
                           "No J7 or deployment. Original K32 means original compressed model, not FP32."]}
    (out/"F30_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("F30_ASSESS",json.dumps({"stage":report["stage"],"arms":{k:v["summary"] for k,v in results.items()}},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f10,a.f08,a.out)
