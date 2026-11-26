"""G02C: independent CPU and independence audit of actual scaled G02.
Reuse F19 packed native tied head and gathered embedding on both 54M and
74M physical checkpoints. Compare full-prefix and packed-last at identical
forced prefixes and independently greedy sequences. No new weights.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import (
    GROWN,extra_open,install_growth,open_bundle,direct,digest)
from research.calibri2.fusion_growth.g02_attention_discrete import G02,open_added,install_added
from research.calibri2.fusion_growth.f21_native_cpu_integration import (
    keep_keyword,last_forward,install_packed,generate,build_v2,forward,
    strict_chat_source_ids,prompt_features)
from v176.operators import clear

SOURCES={
 "54M":("CONTROL_54M_Q8.bundle","CONTROL_54M_SECOND.bin",None,
 "58c4f908bd7f4030e2756e0e1cf8bb7283a1dadd2a30744cfad339b70153a8b6",
 "7aad5d4ced8a28d0cba27aa20c8a2fc344780df0af954818ac90a24cb7800c18",None),
 "74M":("GROWTH_74M_Q8.bundle","GROWTH_74M_SECOND.bin","GROWTH_74M_DISCRETE_ADDED.bin",
 "5f696ee4316232ab5bff93f7cfd986ed86094c8af837352879184b53545857cc",
 "c72e7bbb35aff9f3ef814436e750855f5aedb3fecebab7011096466978ba2593",
 "b3d714fdf07bb88728aa0a1682f171bc8b26f3054166c05a12a4f979458d1fa3")}
PROMPTS=(
 ("Сколько будет 9 плюс 7?","16"),("Повтори слово облако.","облако"),
 ("У Глеба нет ключа. Есть ли у Глеба ключ?","нет"),
 ("В коробке лежит мел, в сумке ручка. Где ручка?","сумка"),
 ("Назови меньшее число: 39 или 81.","39"),
 ("Repeat exactly: green glass.","green glass"),
 ("What is 72 minus 8?","64"),
 ("Сначала был снег, потом дождь. Что было раньше?","снег"))
MODES=("FULL_PREFIX","PACKED_LAST")

def restore(root,name):
    c,s,a,ch,sh,ah=SOURCES[name]
    packet=(root/c).read_bytes();eb=(root/s).read_bytes()
    assert digest(packet)==ch and digest(eb)==sh
    ar,sd,_=open_bundle(packet);st=direct(ar,sd)
    second=extra_open(eb,ar,st[1]);install_growth(st,second)
    added={}
    if a:
        ab=(root/a).read_bytes();assert digest(ab)==ah
        added=open_added(ab,st,second);install_added(st,added)
    return st,ar,second,added

def divergence(st,second,added):
    records=[]
    for k,s in sorted(added.items()):
        ref=second[k] if ".mlp." in k else st[1][k]
        assert tuple(s.shape)==tuple(ref.shape)
        ids=s.codes.detach();old=ref.codes.detach()
        with torch.no_grad():
            w=s();v=ref()
            rel=float(torch.linalg.vector_norm(w-v)/torch.linalg.vector_norm(v).clamp_min(1e-20))
            s.clear();ref.clear()
        records.append({"name":k,"group":"FFN" if ".mlp." in k else "ATTENTION",
                        "different_indices":int((ids!=old).sum()),"index_count":ids.numel(),
                        "relative_weight_L2":rel})
    clear(st[1])
    return {group:{"matrices":len(rr),"changed_index_count":sum(x["different_indices"] for x in rr),
                   "total_indices":sum(x["index_count"] for x in rr),
                   "median_relative_weight_L2":statistics.median(x["relative_weight_L2"] for x in rr)}
            for group in ("FFN","ATTENTION") if (rr:=[x for x in records if x["group"]==group])},records

def run(g02,f08,out):
    torch.set_num_threads(2);torch.manual_seed(20261225)
    from transformers import AutoTokenizer
    from research.calibri2.fusion_growth.f10_joint_own_error_learning import native
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    out.mkdir(parents=True,exist_ok=False)
    results={};details={}
    with tempfile.TemporaryDirectory() as td:
        lib,kernel=build_v2(td)
        for name in SOURCES:
            dense,ar,second,added=restore(g02,name)
            independence,ind_records=divergence(dense,second,added)
            packed,_,_,_=restore(g02,name)
            counts=install_packed(packed,ar,kernel);key=keep_keyword(packed[0])
            states={"FULL_PREFIX":dense,"PACKED_LAST":packed}
            warm=strict_chat_source_ids(tok,"Назови число 5.");feat=prompt_features(tok,"Назови число 5.")
            for mode in MODES:generate(states[mode],warm,feat,mode,key)
            records=[]
            for i,(q,a) in enumerate(PROMPTS):
                p,g=native(tok,q,a);f=prompt_features(tok,q)
                order=MODES if i%2==0 else MODES[::-1]
                timings={m:[] for m in MODES};err=0.;same=0;n=0
                # Same exact prefix lengths and token IDs even if models
                # generate different outputs across geometries.
                for j in range(len(g)):
                    results_at={}
                    for m in order:
                        start=time.perf_counter()
                        z=forward(dense,p+g[:j],f).detach().clone() if m=="FULL_PREFIX" else last_forward(packed,p+g[:j],f,key)
                        timings[m].append(time.perf_counter()-start);results_at[m]=z
                    u,v=[results_at[m] for m in MODES]
                    err=max(err,float((u-v).abs().max()));same+=int(u.argmax()==v.argmax());n+=1
                gr={}
                for m in order:
                    seq,_,dt=generate(states[m],p,f,m,key);gr[m]={"ids":seq,"seconds":dt}
                records.append({"prompt_sha256":digest(q.encode()),"matched_prefix_steps":n,"same_top1":same,
                                "max_logit_error":err,"forced_seconds":{m:sum(timings[m]) for m in MODES},
                                "greedy":gr,"greedy_exact_parity":gr[MODES[0]]["ids"]==gr[MODES[1]]["ids"]})
            assert counts["packed_head"]==counts["gather_embedding"] and counts["packed_head"]>0
            exact=[x for x in records if x["greedy_exact_parity"]]
            info={"geometry":GROWN if name=="54M" else G02,
                  "forced_steps":sum(x["matched_prefix_steps"] for x in records),
                  "same_top1":sum(x["same_top1"] for x in records),
                  "max_logit_error":max(x["max_logit_error"] for x in records),
                  "greedy_parity":len(exact),"prompts":len(records),
                  "forced_median_speedup":statistics.median(x["forced_seconds"]["FULL_PREFIX"]/x["forced_seconds"]["PACKED_LAST"] for x in records),
                  "same_greedy_median_speedup":statistics.median(x["greedy"]["FULL_PREFIX"]["seconds"]/x["greedy"]["PACKED_LAST"]["seconds"] for x in exact) if exact else None,
                  "packed_calls":counts,"bank_independence":independence}
            results[name]=info;details[name]={"rows":records,"matrix_divergence":ind_records}
            print("G02C_ARM",name,json.dumps(info,sort_keys=True),flush=True)
            del dense,packed,states,ar,second,added
    a,b=[details[k]["rows"] for k in ("54M","74M")]
    assert all(x["prompt_sha256"]==y["prompt_sha256"] and x["matched_prefix_steps"]==y["matched_prefix_steps"] for x,y in zip(a,b))
    cross={m:statistics.median(x["forced_seconds"][m]/y["forced_seconds"][m] for x,y in zip(a,b)) for m in MODES}
    report={"stage":"G02C_SCALED_PACKED_CPU_AUDIT","arms":results,
            "cross_geometry_74M_throughput_relative_54M":cross,
            "candidate_accepted":False,"CPU_speed_certified":False,"optimizer_steps":0,
            "limitations":["Fixed-prefix timings on same host; independent greedy parity within geometry only, not correct answers.",
                           "Packed tied head and gathered embedding, body still reconstructs; no secondary incremental KV cache.",
                           "54M control and 74M discrete checkpoints have different learned weights. No historical J7 CPU reference or total resident memory claim.",
                           "Relative matrix differences measure distinct physical representations, not independent knowledge. No training or changes to G03."]}
    (out/"G02C_REPORT.json").write_text(json.dumps({**report,"details":details},indent=2))
    print("G02C_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g02","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g02,a.f08,a.out)
