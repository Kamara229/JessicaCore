"""F29: distinguish failed TRAIN fitting from generalization and EOS failure.
Frozen actual F06 and F11 physical weights. Read-only stratified audit of
preexisting broad training/DEV manifest; no protected J7, no new optimization.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from research.calibri2.fusion_growth.f01_physical_budget import digest
from research.calibri2.fusion_growth.f03_rate_train_core import decode
from research.calibri2.fusion_growth.f09_head_rate_codec import open_bundle
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f11_broad_recovery import FAMILIES,corpus,pack
from research.calibri2.fusion_growth.f21_native_cpu_integration import last_forward,keep_keyword

PARENT="7dcc391ce6001f88f6e7c12498c4f371b83431b049e4db2cb65d7f0b89ab722a"
F11_FP16="3bdcfff2c038e0922a4279e27ac441a74f51d63fb00a9dbb2d32c4908e627d13"
F11_Q8="0b37e0d161473ecaf9b9a756c28043a22021390c20e4551c6db44e2561dbfac9"
MANIFEST="d42fb449836c1459d2196f8c8a2ae290da4415ac41d4f74811b8128d22079ea5"
SEED=20261214

def subset(rows,split):
    # Fixed positions across each family, independent of model outcomes.
    positions=(0,8,16,24,32,40) if split=="TRAIN" else (0,3,6,9)
    return [group[i] for k in FAMILIES for group in [[r for r in rows if r["kind"]==k]] for i in positions]

def rank(z,t):
    return 1+int((z>z[t]).sum())

def metrics(records):
    n=len(records);assert n
    hist=Counter(r["own"][0] for r in records)
    return {"n":n,"first_correct":sum(r["first_correct"] for r in records),
            "teacher_forced_content_all_top1":sum(r["content_all_top1"] for r in records),
            "teacher_forced_EOS_top1":sum(r["gold_EOS_rank"]==1 for r in records),
            "first_rank_median":float(np.median([r["first_rank"] for r in records])),
            "gold_EOS_rank_median":float(np.median([r["gold_EOS_rank"] for r in records])),
            "first_entropy_mean":float(np.mean([r["first_entropy"] for r in records])),
            "first_token_distinct":len(hist),"first_token_histogram":dict(hist),
            "NLL":float(np.mean([r["NLL"] for r in records])),
            "greedy_exact_native_EOS":sum(r["own"]==r["g"] for r in records),
            "greedy_EOS":sum(r["own"][-1]==2 for r in records),
            "greedy_empty":sum(not r["decoded"].strip() for r in records),
            "greedy_adjacent_repeat_cases":sum(any(a==b for a,b in zip(r["own"],r["own"][1:])) for r in records),
            "teacher_forced_mean_EOS_probability":float(np.mean([r["gold_EOS_probability"] for r in records])),
            "first_EOS_probability":float(np.mean([r["initial_EOS_probability"] for r in records])),
            "first_pair_correct":sum(r["pair_gap"]>0 for r in records)}

def audit(state,tok,rows):
    key=keep_keyword(state[0]);records=[]
    for row in rows:
        p,g,f=row["p"],row["g"],row["features"]
        costs=[];ranks=[];initial=None;gap=None;last=None
        for i,t in enumerate(g):
            z=last_forward(state,p+g[:i],f,key)[0]
            assert z.shape==(49152,) and torch.isfinite(z).all()
            costs.append(float(F.cross_entropy(z[None],torch.tensor([t]))))
            ranks.append(rank(z,t))
            if i==0:initial=z.clone()
            if i==row["div"]:gap=float(z[t]-z[row["bad"][i]])
            if i==len(g)-1:last=z
        assert g[-1]==2 and initial is not None and last is not None and gap is not None
        prob=initial.softmax(0);entropy=float(-(prob*initial.log_softmax(0)).sum())
        own=[int(initial.argmax())]
        for _ in range(7):
            if own[-1]==2:break
            own.append(int(last_forward(state,p+own,f,key).argmax()))
        records.append({"id":row["id"],"kind":row["kind"],"g":g,"own":own,
                        "decoded":tok.decode(own,skip_special_tokens=True),
                        "first_correct":own[0]==g[0],"first_rank":ranks[0],
                        "content_all_top1":all(k==1 for k in ranks[:-1]),
                        "gold_EOS_rank":ranks[-1],"first_entropy":entropy,
                        "gold_EOS_probability":float(last.softmax(0)[2]),
                        "initial_EOS_probability":float(prob[2]),
                        "NLL":sum(costs)/len(costs),"pair_gap":gap})
    return {"all":metrics(records),"by_family":{k:metrics([r for r in records if r["kind"]==k]) for k in FAMILIES},"records":records}

def run(f06,f11,f08,out):
    torch.set_num_threads(2);torch.manual_seed(SEED)
    p=(f06/"F06_JOINT.bundle").read_bytes();assert digest(p)==PARENT
    a=(f11/"F11_FP16.bundle").read_bytes();assert digest(a)==F11_FP16
    b=(f11/"F11_Q8.bundle").read_bytes();assert digest(b)==F11_Q8
    manifest=(f11/"F11_DATA_MANIFEST.json").read_bytes();assert digest(manifest)==MANIFEST
    data=json.loads(manifest);assert data["TRAIN"]==corpus("TRAIN") and data["DEV"]==corpus("DEV")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tr=pack(tok,subset(data["TRAIN"],"TRAIN"));dv=pack(tok,subset(data["DEV"],"DEV"))
    assert len(tr)==48 and len(dv)==32 and not set(r["q"] for r in tr)&set(r["q"] for r in dv)
    qcore,qside,extras=open_bundle(b)
    assert extras==(f08/"legacy_extras.bundle").read_bytes()
    fp,side=decode(a);assert fp.blob==qcore.blob
    out.mkdir(parents=True,exist_ok=False)
    results={}
    for name,ar,sd in (("F06_PARENT",*decode(p)),("F11_FP16",fp,side),("F11_Q8",qcore,qside)):
        state=direct(ar,sd)
        results[name]={"TRAIN":audit(state,tok,tr),"DEV":audit(state,tok,dv)};del state
        print("F29_ARM",name,json.dumps({k:v["all"] for k,v in results[name].items()},sort_keys=True),flush=True)
    report={"stage":"F29_BROAD_TRAIN_FIT_AUDIT","manifest_sha256":MANIFEST,"train_n":48,"dev_n":32,
            "arms":results,"optimizer_steps":0,"FP32_teacher_calls":0,"new_parameters":0,
            "candidate_accepted":False,"output_gate":0,
            "limitations":["Fixed subset of original F11 authored TRAIN and DEV, not an independent benchmark.",
                           "Teacher-forced metrics diagnostic only; actual free decoding scored separately on full vocabulary plus EOS.",
                           "F21 last-logit path used without packed substitution, equivalence previously tested on F10, not separately certified on F11.",
                           "No protected J7, no model updates, no deployment or speed certification."]}
    (out/"F29_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("F29_ASSESS",json.dumps({"stage":report["stage"],"arms":{a:{s:v["all"] for s,v in ar.items()} for a,ar in results.items()}},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f06","f11","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f06,a.f11,a.f08,a.out)
