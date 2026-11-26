"""G02D: measure scaled CPU speed/working-memory tradeoff on actual 74M.
Packed tied head in every arm. Immutable body weight cache, 32MiB and full.
No additional learned weights, no deployment, no claim of compact RAM.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import MethodType
import torch
from research.calibri2.fusion_growth.g02c_packed_native import (
    restore,PROMPTS,build_v2,install_packed,keep_keyword,last_forward,generate,
    prompt_features,strict_chat_source_ids,digest)
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native
from research.calibri2.fusion_growth.g02_attention_discrete import G02
from v176.operators import clear

MODES=("PACKED_UNCACHED","PACKED_32MIB","PACKED_ALL_BODY")

def all_body(st,second,added):
    result={"base/"+k:v for k,v in st[1].items() if k!="model.embed_tokens.weight"}
    result.update({"second/"+k:v for k,v in second.items()})
    result.update({"added/"+k:v for k,v in added.items()})
    assert len(result)==204 and len({id(v) for v in result.values()})==204
    return result

def install_cache(st,second,added,budget):
    sources=all_body(st,second,added)
    # Static precommitted small-matrix-first cache, not selected by DEV.
    # Small matrices pay high fixed reconstruction cost per resident byte.
    chosen=[];used=0;start=time.perf_counter()
    for k,s in sorted(sources.items(),key=lambda kv:(kv[1].shape[0]*kv[1].shape[1],kv[0])):
        size=4*s.shape[0]*s.shape[1]
        if used+size>budget:continue
        with torch.no_grad():
            s.clear();w=s().detach().contiguous();s.clear()
        assert w.numel()*w.element_size()==size
        s.forward=MethodType(lambda self,w=w:w,s)
        chosen.append(k);used+=size
    return {"cache_bytes":used,"cached_matrices":len(chosen),
            "body_matrices":len(sources),"initialization_seconds":time.perf_counter()-start,
            "cached_names":chosen}

def run(g02,f08,out):
    torch.set_num_threads(2);torch.manual_seed(20261226)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    out.mkdir(parents=True,exist_ok=False)
    records=[];inventory={}
    with tempfile.TemporaryDirectory() as td:
        lib,kernel=build_v2(td);states={};keys={};counters={}
        for m,budget in zip(MODES,(0,32*1024**2,2**63-1)):
            st,ar,second,added=restore(g02,"74M")
            inventory[m]=install_cache(st,second,added,budget)
            counters[m]=install_packed(st,ar,kernel);states[m]=st;keys[m]=keep_keyword(st[0])
        assert inventory[MODES[0]]["cached_matrices"]==0
        assert inventory[MODES[-1]]["cached_matrices"]==204
        q="Назови число 7.";p=strict_chat_source_ids(tok,q);f=prompt_features(tok,q)
        for m in MODES:generate(states[m],p,f,"PACKED_LAST",keys[m])
        for i,(q,a) in enumerate(PROMPTS):
            p,g=native(tok,q,a);f=prompt_features(tok,q)
            order=MODES[i%3:]+MODES[:i%3]
            ts={m:[] for m in MODES};diff={m:0. for m in MODES};same={m:0 for m in MODES}
            for j in range(len(g)):
                zz={}
                for m in order:
                    start=time.perf_counter()
                    zz[m]=last_forward(states[m],p+g[:j],f,keys[m])
                    ts[m].append(time.perf_counter()-start)
                ref=zz[MODES[0]]
                for m in MODES:
                    diff[m]=max(diff[m],float((zz[m]-ref).abs().max()))
                    same[m]+=int(zz[m].argmax()==ref.argmax())
            gr={}
            for m in order:
                seq,_,dt=generate(states[m],p,f,"PACKED_LAST",keys[m]);gr[m]={"ids":seq,"seconds":dt}
            records.append({"prompt_sha256":digest(q.encode()),"n":len(g),
                            "max_logit_difference":diff,"same_top1":same,
                            "forced_seconds":{m:sum(ts[m]) for m in MODES},"greedy":gr})
        result={}
        for m in MODES:
            matched=[r for r in records if r["greedy"][m]["ids"]==r["greedy"][MODES[0]]["ids"]]
            result[m]={**inventory[m],"max_logit_difference":max(r["max_logit_difference"][m] for r in records),
                       "same_top1":sum(r["same_top1"][m] for r in records),"forced_steps":sum(r["n"] for r in records),
                       "greedy_parity":len(matched),"prompts":len(records),
                       "fixed_prefix_median_speedup":statistics.median(r["forced_seconds"][MODES[0]]/r["forced_seconds"][m] for r in records),
                       "same_greedy_median_speedup":statistics.median(r["greedy"][MODES[0]]["seconds"]/r["greedy"][m]["seconds"] for r in matched) if matched else None,
                       "packed_calls":counters[m]}
            print("G02D_ARM",m,json.dumps({k:v for k,v in result[m].items() if k!="cached_names"},sort_keys=True),flush=True)
        report={"stage":"G02D_IMMUTABLE_BODY_CACHE_BUDGET","geometry":G02,"arms":result,"records":records,
                "candidate_accepted":False,"CPU_speed_certified":False,"new_learned_weights":0,
                "limitations":["All arms use actual same 74M discrete checkpoint and same packed tied head. Cache only for immutable inference; not trainable.",
                               "Cache bytes are additional dense FP32 reconstructed weights, not whole process RSS; cannot claim compact runtime memory.",
                               "No secondary incremental KV cache or historical CPU reference. Eight authored prompts, parity not accuracy. No changes to G03 or deployment."]}
        (out/"G02D_REPORT.json").write_text(json.dumps(report,indent=2))
        print("G02D_ASSESS",json.dumps({"stage":report["stage"],"geometry":G02,"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g02","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g02,a.f08,a.out)
