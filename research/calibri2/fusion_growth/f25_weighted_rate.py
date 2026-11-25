"""F25: source-only, budget-constrained heterogeneous rate allocation.

Both candidate layouts are frozen before any historical inference. Optimize
physical weight distortion, not protected answers. K32/16/8 per tensor,
row-scale-weighted codebook merging, Q4 native head charged. Original
legacy runtime remains only an audit; no new router or neural parameters.
"""
import argparse
import json
import struct
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from research.calibri2.fusion_growth import f03_rate_train_core as codec
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,open_bundle,HDR
from research.calibri2.fusion_growth.f13_profile_rebase import (
    BANK_NAME,BANK_SHA,KEYS,bank_open,bank_pack,extra_open,extra_pack,projection)
from research.calibri2.fusion_growth.f15_legacy_j7_replay import (
    configure_original_selector,model_state,replay,output_digest,s30,s87,
    source_questions_456,materialize_train_only,new_train_paraphrase_probes)
from research.calibri2.fusion_growth.f18_retention_attribution import H584,H32
from direct_codebook_v59 import CB59,unpack5

CAP=3183292
LEVELS=(8,16,32)
ARMS=("ABSOLUTE_WEIGHT_ERROR","RELATIVE_OPERATOR_ERROR")

def options(h):
    m,n=h.shape;g=n//8
    old=unpack5(h.packed.tobytes(),m*g)
    c=h.centroids.astype(np.float64)
    weight=np.bincount(old,weights=np.repeat(h.row_scales.astype(np.float64)**2,g),minlength=32)
    energy=float((weight[:,None]*c*c).sum())
    result={}
    for k in LEVELS:
        if k==32:
            result[k]={"centers":h.centroids.copy(),"ids":old.copy(),"error":0.}
            continue
        chosen=[int(weight.argmax())]
        for _ in range(k-1):
            d=((c[:,None]-c[chosen][None])**2).sum(-1).min(1)
            score=d*weight;score[chosen]=-1
            chosen.append(int(score.argmax()))
        centers=c[chosen].copy()
        for _ in range(30):
            assignment=((c[:,None]-centers[None])**2).sum(-1).argmin(1)
            for j in range(k):
                w=weight*(assignment==j)
                if w.sum()>0:centers[j]=(w[:,None]*c).sum(0)/w.sum()
        centers=centers.astype(np.float16).astype(np.float32)
        assignment=((c[:,None]-centers[None])**2).sum(-1).argmin(1)
        padded=np.zeros((32,8),np.float32);padded[:k]=centers
        error=float((weight[:,None]*(c-centers[assignment])**2).sum())
        result[k]={"centers":padded,"ids":assignment[old].astype(np.uint8),"error":error}
    return result,max(energy,1e-30)

def ranked_plan(opts,energy,objective):
    plan={n:8 for n in opts};steps=[]
    # All possible upgrades form a deterministic ordered chain. A later
    # upgrade of an operator is considered only after its prior upgrade.
    while True:
        choices=[]
        for n,k in plan.items():
            if k==32:continue
            nxt=k*2
            cost=(len(opts[n][k]["ids"])+7)//8+16*(nxt-k)
            gain=opts[n][k]["error"]-opts[n][nxt]["error"]
            if objective==ARMS[1]:gain/=energy[n]
            choices.append((gain/cost,n,k,nxt,cost))
        if not choices:break
        _,n,k,nxt,cost=max(choices,key=lambda x:(x[0],x[1]))
        steps.append((n,k,nxt,cost));plan[n]=nxt
    assert len(steps)==170
    return steps

def adapted_bank(old,archive,plan,inventory,extras,source_sha):
    if all(plan[n]==32 for n in KEYS):
        return extra_pack(inventory,extras),extras
    h,chunks=bank_open(extras[BANK_NAME])
    for name in KEYS:
        k=plan[name]
        if k==32:continue
        a,b=old.handles[name],archive.handles[name];m,n=a.shape
        assert a.shape==b.shape and np.array_equal(a.row_scales,b.row_scales)
        x=unpack5(a.packed.tobytes(),m*n//8).astype(np.int64)
        y=unpack5(b.packed.tobytes(),m*n//8).astype(np.int64)
        w=np.repeat(a.row_scales.astype(np.float64)**2,n//8)
        weights=np.bincount(x*k+y,weights=w,minlength=32*k).reshape(32,k)
        occupied=weights.sum(0)>0
        for profile in ("S87","S91"):
            alt=np.frombuffer(chunks[profile,name],dtype="<f2").reshape(32,8).astype(np.float64)
            padded=np.zeros((32,8),dtype="<f2")
            # Unoccupied new slots never contribute to a matrix product.
            padded[np.flatnonzero(occupied)]=projection(weights[:,occupied],alt)
            chunks[profile,name]=padded.tobytes()
    bank=bank_pack(h,chunks);n,=struct.unpack_from("<I",bank,8)
    header=json.loads(bank[12:12+n]);header["fusion_parent_sha256"]=source_sha
    header["fusion_operator_rate_sha256"]=digest(json.dumps(plan,sort_keys=True).encode())
    raw=json.dumps(header,sort_keys=True,separators=(",",":")).encode()
    bank=bank[:8]+struct.pack("<I",len(raw))+raw+bank[12+n:]
    assert bank_open(bank)[1]==chunks
    blobs=dict(extras);blobs[BANK_NAME]=bank
    return extra_pack(inventory,blobs),blobs

def build_candidates(old,side,inventory,extras,out):
    opts={};energy={}
    for n,h in sorted(old.handles.items()):
        if n not in old.meta["tensor_descriptors"]:continue
        opts[n],energy[n]=options(h)
    assert len(opts)==85
    compact,head,_=split_head(side,4)
    model,sources=codec.make(old)
    assert set(sources)==set(opts)
    def serialize(plan,label):
        with torch.no_grad():
            for n,s in sources.items():
                v=opts[n][plan[n]]
                s.centroids.copy_(torch.from_numpy(v["centers"]))
                s.codes.copy_(torch.from_numpy(v["ids"].astype(np.int64)))
        with patch.object(codec,"k_for",lambda arm,name:plan[name]):
            cp=codec.encode(old.blob,sources,label,compact)
        a,sd=codec.decode(cp)
        extra,_=adapted_bank(old,a,plan,inventory,extras,digest(cp))
        packet=HDR.pack(b"FG09",len(cp),len(head),len(extra))+cp+head+extra
        ar,rs,ex=open_bundle(packet)
        assert ar.blob==a.blob and ex==extra
        assert all(rs[n]==side[n] for n in NAMES[1:] if n!=NAMES[3])
        for n,k in plan.items():
            h=ar.handles[n];v=opts[n][k]
            assert np.array_equal(unpack5(h.packed.tobytes(),len(v["ids"])),v["ids"])
            assert np.array_equal(h.centroids,v["centers"])
            assert np.array_equal(h.row_scales,old.handles[n].row_scales)
        return packet
    low={n:8 for n in opts}
    start=serialize(low,"F25_LOW")
    assert len(start)<=CAP,("base exceeds physical cap",len(start))
    candidates={}
    for arm in ARMS:
        steps=ranked_plan(opts,energy,arm)
        allowance=CAP-len(start)
        total=0;ix=0
        for _,_,_,cost in steps:
            if total+cost>allowance:break
            total+=cost;ix+=1
        def plan_at(j):
            p=dict(low)
            for n,k,nxt,_ in steps[:j]:
                assert p[n]==k;p[n]=nxt
            return p
        # Exact serialized byte count, never protected outputs, determines
        # cutoff. If proxy overshoots, retreat; then try bounded forward
        # additions. All candidates locked before reading historical prompts.
        j=ix;plan=plan_at(j);packet=serialize(plan,arm)
        while len(packet)>CAP and j>0:
            j-=1;plan=plan_at(j);packet=serialize(plan,arm)
        assert len(packet)<=CAP
        for _ in range(12):
            if j==len(steps):break
            trial=plan_at(j+1);p=serialize(trial,arm)
            if len(p)>CAP:break
            j+=1;plan=trial;packet=p
        assert len(packet)<=CAP
        info={"plan":plan,"levels":dict(Counter(plan.values())),"upgrades":j,
              "absolute_weight_error":sum(opts[n][k]["error"] for n,k in plan.items()),
              "mean_relative_operator_error":sum(opts[n][k]["error"]/energy[n] for n,k in plan.items())/len(plan),
              "Q4_known_payload":metrics(len(packet)),"sha256":digest(packet)}
        (out/(arm+".bundle")).write_bytes(packet)
        candidates[arm]=info
        print("F25_LAYOUT",arm,json.dumps(info,sort_keys=True),flush=True)
    del model,sources
    return candidates

def run(f01,f08,f15,out):
    torch.set_num_threads(2);torch.manual_seed(20261212)
    files=unpack((f01/"F01.xz.bundle").read_bytes())
    for n,v in PIN.items():assert digest(files[n])==v
    raw=raw_core(files[NAMES[0]]);assert digest(raw)=="73b05e4cd542e523b4c5cb828d31b358004513c8e64e5e8f76fde2814bdb0aba"
    old=CB59(raw);side={n:files[n] for n in NAMES[1:]}
    inventory,extras=extra_open((f08/"legacy_extras.bundle").read_bytes())
    assert digest(extras[BANK_NAME])==BANK_SHA
    out.mkdir(parents=True,exist_ok=False)
    candidates=build_candidates(old,side,inventory,extras,out)
    manifest=json.dumps(candidates,sort_keys=True)
    (out/"F25_FROZEN_LAYOUTS.json").write_text(manifest)
    # No protected question or prediction is loaded until both physical
    # layouts have been serialized and their hashes frozen.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    s30.TOK=tok
    oq=source_questions_456(s87.families(tok));nq,_,_=materialize_train_only(tok)
    probes=new_train_paraphrase_probes()
    rows={**{"OLD/"+k:{"question":q} for k,q in oq.items()},
          **{"NEW/"+r["id"]:{"question":r["question"]} for r in nq},
          **{f"PROBE/{i:02d}":{"question":q} for i,(q,_,_) in enumerate(probes)}}
    del oq,nq,probes
    forced={k for k in rows if k.startswith("PROBE/")}
    keys={k:digest(v["question"].encode()) for k,v in rows.items()}
    assert len(rows)==len(set(keys.values()))==616
    probe_ids={keys[k] for k in forced}
    ref=json.loads((f15/"F15_BASELINE_IDS.json").read_text())
    assert set(ref)==set(keys.values())
    assert output_digest({k:v for k,v in ref.items() if k not in probe_ids})==H584
    assert output_digest({k:v for k,v in ref.items() if k in probe_ids})==H32
    configure_original_selector()
    state=model_state(old,side,extras,False);pred,_,_=replay(state,tok,rows,forced);del state
    assert pred==ref
    results={}
    for arm,info in candidates.items():
        packet=(out/(arm+".bundle")).read_bytes();assert digest(packet)==info["sha256"]
        a,s,e=open_bundle(packet);_,b=extra_open(e)
        state=model_state(a,s,b,False);pred,_,counts=replay(state,tok,rows,forced);del state
        same={k for k in ref if pred[k]==ref[k]}
        results[arm]={**info,"exact_616":len(same),"drift":616-len(same),
                      "exact_groups":{g:sum(keys[k] in same for k in rows if k.startswith(g+"/")) for g in ("OLD","NEW","PROBE")},
                      "route_counts":counts}
        (out/(arm+"_IDS.json")).write_text(json.dumps(pred,sort_keys=True))
        print("F25_ARM",arm,json.dumps({k:v for k,v in results[arm].items() if k!="plan"},sort_keys=True),flush=True)
    report={"stage":"F25_SOURCE_WEIGHTED_RATE_ALLOCATION","baseline_exact_616":616,
            "layout_manifest_sha256":digest(manifest.encode()),"arms":results,"depth":12,
            "optimizer_steps":0,"FP32_teacher_calls":0,"new_parameters":0,
            "candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Weight-domain distortion allocation and byte cap only, no protected fitting or adaptive selection by J7 results.",
                           "Both layouts charge Q4 native child; historical decoder does not activate that head. No native generation quality claim.",
                           "Original legacy GT gates, no new router. Full tokenizer and runtime costs remain separate.",
                           "Retention compares historical output sequences, not independent correctness. No CPU speed certification."]}
    (out/"F25_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F25_ASSESS",json.dumps({**report,"arms":{k:{i:j for i,j in v.items() if i!="plan"} for k,v in results.items()}},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f08","f15","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f08,a.f15,a.out)
