"""F20: locate native output collapse without modifying model weights.

Frozen 12-layer states from actual K32/Z73, F10 and F12 checkpoints. Fit
small diagnostic ridge probes on authored TRAIN only; test disjoint DEV
wordings. Probe is NOT a decoder, runtime component or candidate model.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,unpack
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f03_rate_train_core import decode
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f06_integrated_recovery import forward
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from direct_codebook_v59 import CB59

F10_SHA="694d031e340825c4d9667f99a674f95b69c4ade29c962e178d01a465bad49018"
F12_SHA="86dc040a05caa1c2f6564f91a95e8c52577b2e84692cd052b8447945647993eb"
FAMILIES=("difference","largest_position","owner_binding","negation")
LABELS=(1,2,3,4)
NAMES_T=("Анна","Борис","Вера","Глеб")
NAMES_D=("Лида","Марк","Нина","Олег")

def corpus(split):
    tr=split=="TRAIN";n=4 if tr else 2
    names=NAMES_T if tr else NAMES_D
    out=[]
    for i in range(n):
        for y in LABELS:
            a=(37 if tr else 173)+19*i+7*y
            order=[(i+j)%4+1 for j in range(4)]
            # Different source positions and explicit numeric responses.
            high=a+13
            vals=[a+j for j in range(4)];vals[y-1]=high
            ordered=sorted(names,key=lambda x:(names.index(x)+i+y)%4)
            pairs=list(zip(ordered,order))
            owner=next(x for x,v in pairs if v==y)
            maptext="; ".join(f"{x}: {v}" for x,v in pairs)
            nums=", ".join(map(str,vals))
            other=", ".join(str(j) for j in LABELS if j!=y)
            questions=[
                ("difference",f"Вычисли {a} минус {a-y}." if tr else f"Найди разность чисел {a} и {a-y}."),
                ("largest_position",f"Числа по порядку: {nums}. Назови номер наибольшего (считая с 1)." if tr else f"В списке [{nums}] укажи позицию максимума от 1 до 4."),
                ("owner_binding",f"Коды: {maptext}. Какой код у человека {owner}?" if tr else f"Запись кодов ({maptext}). Какое число соответствует имени {owner}?"),
                ("negation",f"Коробки {other} пусты. Остальная коробка не пуста. Какой у неё номер? Метка {a}." if tr else f"Из коробок 1–4 только {other} не содержат предмета. Назови номер непустой коробки. Метка {a}.")]
            for kind,q in questions:
                out.append({"split":split,"kind":kind,"q":q+" Ответь одной цифрой.","answer":str(y),"label":y-1})
    assert len(out)==16*n and len({x["q"] for x in out})==len(out)
    assert Counter(x["label"] for x in out)=={j:4*n for j in range(4)}
    return out

def collect(state,tok,rows):
    model=state[0];res=[];native_correct=0;ranks=[];first=Counter()
    for r in rows:
        p,g=native(tok,r["q"],r["answer"])
        # Distinct target digits must have distinct native FIRST token IDs.
        assert g[-1]==tok.eos_token_id and len(g)>=2
        states={};hooks=[]
        def capture(i):
            def cb(_m,_args,out):
                h=out[0] if isinstance(out,tuple) else out
                assert h.shape[0]==1 and h.shape[-1]==320
                states[i]=h[0,-1].detach().float().cpu().numpy().copy()
            return cb
        try:
            for i,m in enumerate(model.model.layers):
                hooks.append(m.register_forward_hook(capture(i)))
            def final(_m,args):
                states[12]=args[0][0,-1].detach().float().cpu().numpy().copy()
            hooks.append(model.lm_head.register_forward_pre_hook(final))
            z=forward(state,p,prompt_features(tok,r["q"])).detach()
        finally:
            for h in hooks:h.remove()
        assert set(states)==set(range(13)) and z.shape==(1,49152)
        t=g[0];a=int(z.argmax());first[a]+=1;native_correct+=int(a==t)
        ranks.append(1+int((z[0]>z[0,t]).sum()))
        res.append((np.stack([states[i] for i in range(13)]),r["label"],t))
    for label in range(4):
        assert len({t for _,y,t in res if y==label})==1
    assert len({t for _,_,t in res})==4
    h=np.stack([x[0] for x in res]).astype(np.float64)
    assert np.isfinite(h).all()
    return h,np.array([x[1] for x in res]),{"first_correct":native_correct,"n":len(rows),
            "median_target_rank":float(np.median(ranks)),"first_token_histogram":dict(first)}

def probe(x,y,v,w,train,dev):
    # Fit normalization and ridge only on TRAIN; no DEV hyperparameter selection.
    mean=x.mean(0);scale=x.std(0).clip(1e-6)
    a=(x-mean)/scale;b=(v-mean)/scale
    a=np.column_stack([a,np.ones(len(a))]);b=np.column_stack([b,np.ones(len(b))])
    targets=np.eye(4)[y]
    # Dual ridge: small, fixed positive regularizer. No full-vocab answer masks.
    weights=a.T@np.linalg.solve(a@a.T+10.*np.eye(len(a)),targets)
    p=(a@weights).argmax(1);q=(b@weights).argmax(1)
    return {"train_correct":int((p==y).sum()),"dev_correct":int((q==w).sum()),
            "train_n":len(y),"dev_n":len(w),
            "dev_by_family":{k:int(sum(q[i]==w[i] for i,r in enumerate(dev) if r["kind"]==k)) for k in FAMILIES},
            "centered_effective_rank":int(np.linalg.matrix_rank(x-mean,tol=1e-5)),
            "train_feature_std_mean":float(x.std(0).mean())}

def run(f01,f10,f12,f08,out):
    torch.set_num_threads(2);torch.manual_seed(20261208)
    old=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(old[k])==v
    original=CB59(raw_core(old[NAMES[0]]))
    old_side={k:old[k] for k in NAMES[1:]}
    p10=(f10/"F10_FP16.bundle").read_bytes();assert digest(p10)==F10_SHA
    p12=(f12/"CENTROIDS_AND_CODES_FP16.bundle").read_bytes();assert digest(p12)==F12_SHA
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train,dev=corpus("TRAIN"),corpus("DEV")
    assert not set(r["q"] for r in train)&set(r["q"] for r in dev)
    out.mkdir(parents=True,exist_ok=False)
    arms={}
    for name,ar,side in [("ORIGINAL_K32_Z73_NATIVE",original,old_side),
                         ("F10",*decode(p10)),("F12",*decode(p12))]:
        state=direct(ar,side)
        a,y,n1=collect(state,tok,train);b,w,n2=collect(state,tok,dev);del state
        layers={str(i):probe(a[:,i],y,b[:,i],w,train,dev) for i in range(13)}
        arms[name]={"native_train":n1,"native_dev":n2,"layers":layers}
        print("F20_ARM",name,json.dumps(arms[name],sort_keys=True),flush=True)
    report={"stage":"F20_FROZEN_STATE_DIAGNOSTIC","arms":arms,"train_n":64,"dev_n":32,
            "balanced_chance_dev":8,"model_optimizer_steps":0,"candidate_accepted":False,
            "output_gate":0,"FP32_teacher_calls":0,
            "limitations":["Fixed four-class diagnostic on frozen last-position states, not a generative decoder; no runtime weight export.",
                           "Original K32/Z73 uses the historical native child and GT configuration, not legacy J7 decoder; not an FP32 teacher.",
                           "Different TRAIN and DEV wording/operands, authored small dataset, not independent general intelligence benchmark.",
                           "Probe results diagnose whether information is linearly accessible, not proof of reasoning or a certified causal mechanism.",
                           "No protected J7 data or model updates. No model promotion."]}
    (out/"F20_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F20_ASSESS",json.dumps({"stage":report["stage"],"native_dev":{k:v["native_dev"] for k,v in arms.items()},"last_state_probe":{k:v["layers"]["12"] for k,v in arms.items()}},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f10","f12","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f10,a.f12,a.f08,a.out)
