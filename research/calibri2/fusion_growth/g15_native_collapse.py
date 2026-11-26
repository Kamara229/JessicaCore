"""G15: localize elementary Russian generation failure observed in G14.
No learning or new weights. Compare actual same checkpoint with temporary
readout ablations, plain vs appended instruction. Measure layer separation,
target ranks, raw greedy sequences. A probe is not a replacement decoder.
"""
import argparse
import json
import statistics
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import digest,native
from research.calibri2.fusion_growth.g05_134m_joint import restore,G05
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g02_attention_discrete import forward
from research.calibri2.fusion_growth.f21_native_cpu_integration import prompt_features
from research.calibri2.fusion_growth.g13_open_135m_benchmark import norm

MODES=("FULL","NO_HEAD_CORRECTION","NO_HEAD_BIAS")
SUFFIX=" Ответь только кратким ответом."
# Fresh explicit copying tasks, same template and only target changed.
ITEMS={"digit":("2","4","5","6","7","9"),"letter":("б","в","к","о","р","с"),
       "word":("мост","сад","рыба","чай","снег","рука")}
TEMPLATES={"digit":"Напиши цифру {}.","letter":"Напиши букву {}.","word":"Повтори слово {}."}

@contextmanager
def ablate(st,mode):
    head=st[5].language_head
    if mode=="FULL":
        yield;return
    if mode=="NO_HEAD_CORRECTION":
        params=list(head.parameters())
    else:
        assert mode=="NO_HEAD_BIAS"
        params=[head.output_bias]
    old=[p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            for p in params:p.zero_()
        yield
    finally:
        with torch.no_grad():
            for p,v in zip(params,old):p.copy_(v)
        assert all(torch.equal(p,v) for p,v in zip(params,old))

def capture(st,p,f):
    model=st[0];h={};handles=[];raw=[]
    def cb(i):
        def hook(_m,_a,out):
            x=out[0] if isinstance(out,tuple) else out
            h[i]=x[0,-1].detach().float().numpy().copy()
        return hook
    try:
        for i,l in enumerate(model.model.layers):handles.append(l.register_forward_hook(cb(i)))
        handles.append(model.lm_head.register_forward_pre_hook(lambda _m,args:h.__setitem__(12,args[0][0,-1].detach().float().numpy().copy())))
        handles.append(model.lm_head.register_forward_hook(lambda _m,_a,out:raw.append(out[0,-1].detach().float().clone())))
        z=forward(st,p,f).detach()[0]
    finally:
        for handle in handles:handle.remove()
    assert set(h)==set(range(13)) and len(raw)==1 and z.shape==raw[0].shape==(49152,)
    return z,raw[0],np.stack([h[i] for i in range(13)])

def rank(z,t):return 1+int((z>z[t]).sum())
def geometry(xs):
    x=np.stack(xs).astype(np.float64)
    center=x.mean(0)
    diff=x-center
    sv=np.linalg.svd(diff,compute_uv=False)**2
    prob=sv/max(1e-30,sv.sum());p=prob[prob>0]
    return {"relative_centered_rms":float(np.linalg.norm(diff)/max(np.linalg.norm(x),1e-30)),
            "effective_rank":float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,
            "mean_vector_norm":float(np.linalg.norm(center))}

def run(g05,f08,out):
    torch.manual_seed(20270107);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    st,*rest=restore(blobs[:6],blobs[6],blobs[7])
    out.mkdir(parents=True,exist_ok=False)
    rows=[{"family":k,"gold":a,"q":TEMPLATES[k].format(a)+s,"format":fmt}
          for k,vals in ITEMS.items() for a in vals
          for fmt,s in (("PLAIN",""),("SUFFIX",SUFFIX))]
    assert len(rows)==36 and len({r["q"] for r in rows})==36
    records=[];vectors={}
    for mode in MODES:
        with ablate(st,mode):
            for r in rows:
                p,g=native(tok,r["q"],r["gold"]);f=prompt_features(tok,r["q"])
                z,raw,h=capture(st,p,f);ids=[int(z.argmax())]
                for _ in range(7):
                    if ids[-1]==2:break
                    v=forward(st,p+ids,f).detach()
                    ids.append(int(v.argmax()))
                text=tok.decode(ids,skip_special_tokens=True).strip()
                vectors[(mode,r["q"])]=h
                t=g[0];delta=z-raw
                records.append({**r,"mode":mode,"answer":text,"ids":ids,"EOS":ids[-1]==2,
                      "exact":norm(text)==norm(r["gold"]),"first_correct":ids[0]==t,
                      "first_rank":rank(z,t),"raw_tied_rank":rank(raw,t),"raw_tied_first":int(raw.argmax()),
                      "readout_delta_L2":float(delta.norm()),"raw_logits_L2":float(raw.norm()),
                      "first_target_margin":float(z[t]-z.max())})
    # Readout-only interventions cannot change initial hidden states.
    parity=max(float(np.max(np.abs(vectors[("FULL",r["q"])]-vectors[(m,r["q"])])))
               for m in MODES[1:] for r in rows)
    assert parity<1e-6
    result={}
    for mode in MODES:
        result[mode]={}
        for fmt in ("PLAIN","SUFFIX"):
            rr=[r for r in records if r["mode"]==mode and r["format"]==fmt]
            families={}
            for k in ITEMS:
                v=[r for r in rr if r["family"]==k]
                families[k]={"n":len(v),"exact":sum(r["exact"] for r in v),
                             "first_correct":sum(r["first_correct"] for r in v),
                             "median_first_rank":statistics.median(r["first_rank"] for r in v),
                             "median_raw_tied_rank":statistics.median(r["raw_tied_rank"] for r in v),
                             "layer_geometry":{str(i):geometry([vectors[(mode,r["q"])][i] for r in v]) for i in range(13)}}
            result[mode][fmt]={"n":len(rr),"exact":sum(r["exact"] for r in rr),
                 "exact_EOS":sum(r["exact"] and r["EOS"] for r in rr),"EOS":sum(r["EOS"] for r in rr),
                 "first_correct":sum(r["first_correct"] for r in rr),
                 "median_first_rank":statistics.median(r["first_rank"] for r in rr),
                 "median_raw_tied_rank":statistics.median(r["raw_tied_rank"] for r in rr),
                 "first_token_histogram":Counter(r["ids"][0] for r in rr),
                 "answer_histogram":Counter(r["answer"] for r in rr),
                 "families":families}
        print("G15_ARM",mode,json.dumps(result[mode],ensure_ascii=False,sort_keys=True),flush=True)
    # Confirm no residual effect from temporary interventions.
    r=rows[0];p,g=native(tok,r["q"],r["gold"])
    z,_,_=capture(st,p,prompt_features(tok,r["q"]))
    assert int(z.argmax())==records[0]["ids"][0]
    report={"stage":"G15_NATIVE_COLLAPSE_LOCALIZATION","geometry":G05,"n_prompts":36,"arms":result,"records":records,
            "hidden_parity_max_diff":parity,"training_updates":0,"candidate_accepted":False,
            "limitations":["Temporary head ablations only, not a proposed deployment. No optimization or new model weights.",
                           "18 fresh copy targets x two prompt formats; closely related to G14 but different exact prompts. No claims of general comprehension.",
                           "Layer geometry measures sensitivity of last-position activations, not accessible semantic information. Head ablation may worsen useful behavior.",
                           "G05 only; no protected historical replay. No public model output used as teacher."]}
    (out/"G15_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G15_ASSESS",json.dumps({"stage":report["stage"],"hidden_parity_max_diff":parity,"candidate_accepted":False}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
