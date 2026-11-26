"""G12: same-host throughput reference against real original-size F10.
Compare original native full-prefix, original packed last, scaled packed
uncached, scaled FP16 FFN, and scaled FP16 FFN+attention. Fixed prefixes
identical across geometries; greedy parity only within G05. No historical
J7 CPU certificate, no extra learned weights or changes to deployment.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import MethodType
import torch
from torch.nn import functional as F
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import BASE,open_bundle,digest,native
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g11_half_cache_cpu import install_ffn,PROMPTS
from research.calibri2.fusion_growth.f21_native_cpu_integration import (
    build_v2,install_packed,last_forward,keep_keyword,generate,prompt_features,
    strict_chat_source_ids,forward)
from research.calibri2.fusion_growth.f03_rate_train_core import decode
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f17_packed_cpu import PARENT
from v176.operators import clear

MODES=("F10_FULL","F10_PACKED","G05_PACKED","G05_FFN16","G05_ALL16")

def cache_attention(st):
    count=0;nbytes=0;start=time.perf_counter()
    for layer in st[0].model.layers:
        four=layer.self_attn
        triple=four.first;dual=triple.first
        modules=(dual.original,dual.secondary,triple.third,four.fourth)
        for mod in modules:
            for name in ("q_proj","k_proj","v_proj","o_proj"):
                proj=getattr(mod,name);s=proj.source
                s.clear()
                with torch.no_grad():w=s().detach().half().contiguous()
                s.clear()
                assert w.shape==s.shape
                nbytes+=w.numel()*w.element_size();count+=1
                # Preserve the original attention and rotary/cache semantics;
                # only q,k,v,o matrix products use cached FP16.
                def call(self,x,w=w):
                    return F.linear(x.to(w.dtype),w).to(x.dtype)
                proj.forward=MethodType(call,proj)
    assert count==192
    clear(st[1])
    return {"attention_cache_bytes":nbytes,"attention_matrices":count,
            "attention_setup_seconds":time.perf_counter()-start}

def run(g05,f10,f08,out):
    torch.manual_seed(20270104);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    a,_,_=open_bundle(blobs[0])
    old=(f10/"F10_FP16.bundle").read_bytes();assert digest(old)==PARENT
    oa,os=decode(old)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    out.mkdir(parents=True,exist_ok=False)
    states={};keys={};inventory={};rows=[]
    with tempfile.TemporaryDirectory() as td:
        lib,kernel=build_v2(td)
        for mode in MODES:
            if mode.startswith("F10"):
                st=direct(oa,os)
                inv={"geometry":BASE,"FFN_cache_bytes":0,"attention_cache_bytes":0}
                if mode=="F10_PACKED":install_packed(st,oa,kernel)
            else:
                st,second,added,fourth,reg,oldgt,fifth,sixth,reg2=restore(blobs[:6],blobs[6],blobs[7])
                inv={"geometry":G05,"FFN_cache_bytes":0,"attention_cache_bytes":0}
                install_packed(st,a,kernel)
                if mode!="G05_PACKED":
                    v,_=install_ffn(st,(st[1],second,added,fourth,fifth,sixth),"FUSED_FP16");inv.update(v)
                if mode=="G05_ALL16":inv.update(cache_attention(st))
                del second,added,fourth,reg,oldgt,fifth,sixth,reg2
            inv["total_additional_cache_bytes"]=inv["FFN_cache_bytes"]+inv["attention_cache_bytes"]
            states[mode]=st;keys[mode]=keep_keyword(st[0]);inventory[mode]=inv
        def single(m,p,f):
            return forward(states[m],p,f).detach() if m=="F10_FULL" else last_forward(states[m],p,f,keys[m])
        def greedy(m,p,f):
            return generate(states[m],p,f,"FULL_PREFIX" if m=="F10_FULL" else "PACKED_LAST",keys[m])
        q="Запиши число 2.";p=strict_chat_source_ids(tok,q);f=prompt_features(tok,q)
        for m in MODES:greedy(m,p,f)
        for i,(q,a) in enumerate(PROMPTS):
            p,g=native(tok,q,a);f=prompt_features(tok,q)
            ts={m:[] for m in MODES};diff={m:0. for m in MODES};agree={m:0 for m in MODES}
            for j in range(len(g)):
                shift=(i+j)%len(MODES);order=MODES[shift:]+MODES[:shift]
                z={}
                for m in order:
                    t=time.perf_counter();z[m]=single(m,p+g[:j],f);ts[m].append(time.perf_counter()-t)
                for m in MODES:
                    ref=z["F10_FULL" if m.startswith("F10") else "G05_PACKED"]
                    diff[m]=max(diff[m],float((z[m]-ref).abs().max()))
                    agree[m]+=int(z[m].argmax()==ref.argmax())
            res={}
            for m in MODES[i%len(MODES):]+MODES[:i%len(MODES)]:
                ids,_,t=greedy(m,p,f);res[m]={"ids":ids,"seconds":t}
            rows.append({"q_sha256":digest(q.encode()),"fixed_steps":len(g),
                         "fixed_seconds":{m:sum(ts[m]) for m in MODES},"within_geometry_max_error":diff,
                         "within_geometry_top1":agree,"greedy":res})
        results={}
        for m in MODES:
            ref="F10_FULL" if m.startswith("F10") else "G05_PACKED"
            same=[r for r in rows if r["greedy"][m]["ids"]==r["greedy"][ref]["ids"]]
            results[m]={**inventory[m],"fixed_steps":sum(r["fixed_steps"] for r in rows),
                        "fixed_prefix_tokens_per_second":sum(r["fixed_steps"] for r in rows)/sum(r["fixed_seconds"][m] for r in rows),
                        "median_throughput_relative_F10_FULL":statistics.median(r["fixed_seconds"]["F10_FULL"]/r["fixed_seconds"][m] for r in rows),
                        "median_throughput_relative_F10_PACKED":statistics.median(r["fixed_seconds"]["F10_PACKED"]/r["fixed_seconds"][m] for r in rows),
                        "median_throughput_relative_same_geometry":statistics.median(r["fixed_seconds"][ref]/r["fixed_seconds"][m] for r in rows),
                        "within_geometry_top1":sum(r["within_geometry_top1"][m] for r in rows),
                        "within_geometry_max_error":max(r["within_geometry_max_error"][m] for r in rows),
                        "within_geometry_greedy_parity":len(same),"prompts":len(rows),
                        "same_output_greedy_speedup":statistics.median(r["greedy"][ref]["seconds"]/r["greedy"][m]["seconds"] for r in same) if same else None}
            print("G12_ARM",m,json.dumps(results[m],sort_keys=True),flush=True)
    report={"stage":"G12_SCALED_CPU_REFERENCE","arms":results,"records":rows,
            "candidate_accepted":False,"CPU_speed_certified":False,"new_learned_weights":0,
            "limitations":["F10 original-size native research reference, not historical S194 J7 runtime. Same fixed prefixes across geometries but different learned weights; no cross-geometry output parity or quality claim.",
                           "FP16 cached products can change outputs. Within-geometry greedy parity checked; same-output ratios only for matching cases.",
                           "Additional cache bytes are not total RSS; no deployment or changes to concurrent training.",
                           "Eight reused authored prompts, same runner 2 threads. Full-prefix recomputation retained, no incremental KV equivalence claim."]}
    (out/"G12_REPORT.json").write_text(json.dumps(report,indent=2))
    print("G12_ASSESS",json.dumps({"stage":report["stage"],"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f10,a.f08,a.out)
