"""F26: functional sensitivity, not weight MSE alone, for heterogeneous K.
Single-group perturbations of original compressed K32 native model, 6 new
calibration prompts, 13 groups. Dynamic programming allocates rate budget.
No training, no protected prompts until both physical layouts frozen.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from research.calibri2.fusion_growth.f25_weighted_rate import (
    CAP,LEVELS,options,adapted_bank,codec,NAMES,PIN,digest,metrics,unpack,
    raw_core,split_head,open_bundle,HDR,extra_open,BANK_NAME,BANK_SHA,
    configure_original_selector,model_state,replay,output_digest,s30,s87,
    source_questions_456,materialize_train_only,new_train_paraphrase_probes,
    H584,H32,CB59,unpack5)
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f06_integrated_recovery import forward
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import strict_chat_source_ids
from v176.operators import clear

CAL=("В коробке лежит синий шар. Какого цвета шар?",
     "Назови число, которое следует за 347.",
     "В записи Юрий: 19; Павел: 27. Какой код у Павла?",
     "Сравни 78 и 91. Какое число меньше?",
     "Repeat exactly: the small boat.",
     "Вера открыла дверь после звонка. Что произошло раньше?")
ARMS=("LOGIT_SENSITIVITY","HIDDEN_SENSITIVITY")
UNIT=1024

def group_name(n):
    return "EMBED" if n=="model.embed_tokens.weight" else "L"+n.split(".")[2].zfill(2)

def load_group(sources,opts,ns,k):
    with torch.no_grad():
        for n in ns:
            v=opts[n][k];s=sources[n]
            s.centroids.copy_(torch.from_numpy(v["centers"]))
            s.codes.copy_(torch.from_numpy(v["ids"].astype(np.int64)))
    clear(sources)

def witness(state,probes):
    zz=[];hh=[]
    for p,f in probes:
        cap=[]
        def cb(_m,args):cap.append(args[0][0,-1].detach().float().clone())
        handle=state[0].lm_head.register_forward_pre_hook(cb)
        try:z=forward(state,p,f).detach().float().clone()
        finally:handle.remove()
        assert len(cap)==1 and cap[0].shape==(320,) and z.shape==(1,49152)
        assert torch.isfinite(z).all() and torch.isfinite(cap[0]).all()
        zz.append(z[0].numpy());hh.append(cap[0].numpy())
    return np.stack(zz).astype(np.float64),np.stack(hh).astype(np.float64)

def relative(a,b):
    return float(np.mean(np.sum((a-b)**2,axis=-1)/np.maximum(np.sum(a*a,axis=-1),1e-20)))

def sensitivity(old,side,opts,groups,tok):
    state=direct(old,side);sources=state[1]
    probes=[(strict_chat_source_ids(tok,q),prompt_features(tok,q)) for q in CAL]
    z,h=witness(state,probes);z=z-z.mean(-1,keepdims=True)
    costs={}
    for g,ns in groups.items():
        costs[g]={32:{"logit":0.,"hidden":0.,"top1_changes":0}}
        for k in (8,16):
            load_group(sources,opts,ns,k)
            a,b=witness(state,probes);a=a-a.mean(-1,keepdims=True)
            costs[g][k]={"logit":relative(z,a),"hidden":relative(h,b),
                         "top1_changes":int(np.sum(z.argmax(-1)!=a.argmax(-1)))}
        load_group(sources,opts,ns,32)
        print("F26_SENSITIVITY",g,json.dumps(costs[g]),flush=True)
    a,b=witness(state,probes)
    assert np.array_equal(z,a-a.mean(-1,keepdims=True)) and np.array_equal(h,b)
    del state,sources
    return costs

def dp(groups,costs,proxy,field,budget):
    # Exact dynamic programming on rounded-up raw incremental byte costs.
    # At most 13*3*(budget/1KiB) states, no protected inference in search.
    dp={0:(0.,{})}
    for g in groups:
        nxt={}
        for used,(err,plan) in dp.items():
            for k in LEVELS:
                t=used+math.ceil(proxy[g][k]/UNIT)
                if t>budget//UNIT:continue
                q=(err+costs[g][k][field],{**plan,g:k})
                if t not in nxt or q[0]<nxt[t][0]:nxt[t]=q
        dp=nxt
    assert dp
    return min(dp.values(),key=lambda x:x[0])[1]

def build_candidates(old,side,inventory,extras,out,f08):
    opts={};groups={}
    for n,h in sorted(old.handles.items()):
        if n not in old.meta["tensor_descriptors"]:continue
        opts[n],_=options(h);groups.setdefault(group_name(n),[]).append(n)
    assert len(opts)==85 and len(groups)==13
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    costs=sensitivity(old,side,opts,groups,tok)
    (out/"F26_CALIBRATION.json").write_text(json.dumps({"prompts":CAL,"costs":costs},ensure_ascii=False,indent=2))
    proxy={g:{k:sum((len(opts[n][8]["ids"])*(k.bit_length()-4)+7)//8+16*(k-8) for n in ns) for k in LEVELS} for g,ns in groups.items()}
    compact,head,_=split_head(side,4);model,sources=codec.make(old)
    assert set(sources)==set(opts)
    def serialize(layout,label):
        plan={n:layout[g] for g,ns in groups.items() for n in ns}
        for g,ns in groups.items():load_group(sources,opts,ns,layout[g])
        with patch.object(codec,"k_for",lambda arm,name:plan[name]):
            cp=codec.encode(old.blob,sources,label,compact)
        a,_=codec.decode(cp)
        extra,_=adapted_bank(old,a,plan,inventory,extras,digest(cp))
        packet=HDR.pack(b"FG09",len(cp),len(head),len(extra))+cp+head+extra
        ar,rs,ex=open_bundle(packet)
        assert ar.blob==a.blob and ex==extra
        assert all(rs[n]==side[n] for n in NAMES[1:] if n!=NAMES[3])
        for n,k in plan.items():
            h=ar.handles[n];v=opts[n][k]
            assert np.array_equal(unpack5(h.packed.tobytes(),len(v["ids"])),v["ids"])
            assert np.array_equal(h.centroids,v["centers"])
        return packet,plan
    low={g:8 for g in groups};packet,_=serialize(low,"F26_LOW")
    allowance=CAP-len(packet);assert allowance>0
    result={}
    for arm,field in zip(ARMS,("logit","hidden")):
        layout=dp(groups,costs,proxy,field,allowance)
        packet,plan=serialize(layout,arm)
        # Physical cap enforced, conservative proxy adjusted only by bytes.
        while len(packet)>CAP:
            choices=[]
            for g,k in layout.items():
                if k==8:continue
                gain=costs[g][k//2][field]-costs[g][k][field]
                saving=proxy[g][k]-proxy[g][k//2]
                choices.append((gain/max(1,saving),g))
            assert choices
            _,g=min(choices);layout[g]//=2;packet,plan=serialize(layout,arm)
        # Try every feasible alternative in ranked order, skipping large
        # upgrades rather than stopping at the first one that cannot fit.
        for _ in range(6):
            choices=[]
            for g,k in layout.items():
                if k==32:continue
                gain=costs[g][k][field]-costs[g][k*2][field]
                saving=proxy[g][k*2]-proxy[g][k]
                if gain>0:choices.append((gain/max(1,saving),g))
            improved=False
            for _,g in sorted(choices,reverse=True):
                trial=dict(layout);trial[g]*=2
                pp,p=serialize(trial,arm)
                if len(pp)<=CAP:
                    layout,packet,plan=trial,pp,p;improved=True;break
            if not improved:break
        assert len(packet)<=CAP
        info={"plan":plan,"group_layout":layout,"levels":dict(Counter(plan.values())),
              "calibration_proxy_logit_error":sum(costs[g][k]["logit"] for g,k in layout.items()),
              "calibration_proxy_hidden_error":sum(costs[g][k]["hidden"] for g,k in layout.items()),
              "Q4_known_payload":metrics(len(packet)),"sha256":digest(packet)}
        (out/(arm+".bundle")).write_bytes(packet);result[arm]=info
        print("F26_LAYOUT",arm,json.dumps({k:v for k,v in info.items() if k!="plan"},sort_keys=True),flush=True)
    del model,sources
    return result

def run(f01,f08,f15,out):
    torch.set_num_threads(2);torch.manual_seed(20261212)
    files=unpack((f01/"F01.xz.bundle").read_bytes())
    for n,v in PIN.items():assert digest(files[n])==v
    raw=raw_core(files[NAMES[0]]);assert digest(raw)=="73b05e4cd542e523b4c5cb828d31b358004513c8e64e5e8f76fde2814bdb0aba"
    old=CB59(raw);side={n:files[n] for n in NAMES[1:]}
    inventory,extras=extra_open((f08/"legacy_extras.bundle").read_bytes())
    assert digest(extras[BANK_NAME])==BANK_SHA
    out.mkdir(parents=True,exist_ok=False)
    candidates=build_candidates(old,side,inventory,extras,out,f08)
    manifest=json.dumps(candidates,sort_keys=True)
    (out/"F26_FROZEN_LAYOUTS.json").write_text(manifest)
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
        print("F26_ARM",arm,json.dumps({k:v for k,v in results[arm].items() if k!="plan"},sort_keys=True),flush=True)
    report={"stage":"F26_FUNCTIONAL_RATE_ALLOCATION","baseline_exact_616":616,
            "layout_manifest_sha256":digest(manifest.encode()),"arms":results,"depth":12,
            "optimizer_steps":0,"FP32_teacher_calls":0,"new_parameters":0,
            "candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Layer sensitivity measured on fresh authored prompts against original compressed K32 native reference (not FP32 teacher). No protected fitting or adaptive selection by J7 results.",
                           "Q4 child charged. Allocation uses original native-path logit or hidden drift, while historical decoder does not activate that child. Transfer across paths is unproven.",
                           "Original legacy GT gates, no new router. Full tokenizer and runtime costs remain separate.",
                           "Retention compares historical output sequences, not independent correctness. No CPU speed certification."]}
    (out/"F26_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F26_ASSESS",json.dumps({**report,"arms":{k:{i:j for i,j in v.items() if i!="plan"} for k,v in results.items()}},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f08","f15","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f08,a.f15,a.out)
