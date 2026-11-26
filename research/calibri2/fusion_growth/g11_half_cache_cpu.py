"""G11: compare immutable fused FFN cache precision at actual 133.701M.
G10 showed that dense cache, not fusion alone, drove acceleration.
Test FP32, FP16 and BF16 fused caches against uncached same checkpoint.
All bytes explicit, no training or physical weight changes.
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
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import open_bundle,digest,native
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.f21_native_cpu_integration import (
    build_v2,install_packed,last_forward,keep_keyword,generate,prompt_features,
    strict_chat_source_ids)
from v176.operators import clear

MODES=("UNCACHED","FUSED_FP32","FUSED_FP16","FUSED_BF16")
DTYPES={"FUSED_FP32":torch.float32,"FUSED_FP16":torch.float16,"FUSED_BF16":torch.bfloat16}
GAINS=(.25,.25,.125,.0625,.0625,.25)
PROMPTS=(
 ("Сколько будет 8 плюс 6?","14"),
 ("Повтори слово трамвай.","трамвай"),
 ("У Игоря нет зонта. Есть ли у него зонт?","нет"),
 ("В ящике лежит нож, на столе ложка. Где ложка?","на столе"),
 ("Что меньше: 81 или 46?","46"),
 ("Repeat exactly: red leaf.","red leaf"),
 ("What is 61 minus 9?","52"),
 ("Сначала была осень, затем зима. Что было раньше?","осень"))

def install_ffn(st,banks,mode):
    assert len(banks)==6
    nbytes=0;matrix_count=0;start=time.perf_counter()
    counters={"layer_calls":0,"linear_calls":0}
    for i,layer in enumerate(st[0].model.layers):
        prefix=f"model.layers.{i}.mlp."
        triples=[[b[prefix+k+".weight"] for k in ("gate_proj","up_proj","down_proj")] for b in banks]
        assert len({id(s) for t in triples for s in t})==18
        assert all(tuple(s.shape)==shape for t in triples for s,shape in zip(t,((1536,320),(1536,320),(320,1536))))
        matrix_count+=18
        mlp=layer.mlp
        # Keep original module registrations and hooks. Only inference
        # callable changes, no physical checkpoint or state_dict mutation.
        if mode=="UNCACHED":
            original=mlp.forward
            def call(self,x,original=original):
                counters["layer_calls"]+=1;counters["linear_calls"]+=18
                return original(x)
            mlp.forward=MethodType(call,mlp)
            continue
        weights=[]
        with torch.no_grad():
            for t in triples:
                w=[]
                for s in t:
                    s.clear();w.append(s().detach().contiguous());s.clear()
                weights.append(w)
        assert mode in DTYPES
        dtype=DTYPES[mode]
        # Block concatenation is algebraically equivalent:
        # sum_i gain_i Wdown_i [SiLU(Wgate_i x)*Wup_i x].
        # Binary gains are exact; FP32 summation order may differ.
        gu=torch.cat([t[0] for t in weights]+[t[1] for t in weights],0).to(dtype).contiguous()
        down=torch.cat([gain*t[2] for gain,t in zip(GAINS,weights)],1).to(dtype).contiguous()
        assert gu.shape==(18432,320) and down.shape==(320,9216)
        nbytes+=sum(v.numel()*v.element_size() for v in (gu,down))
        def call(self,x,gu=gu,down=down):
            counters["layer_calls"]+=1;counters["linear_calls"]+=2
            gate,up=F.linear(x.to(gu.dtype),gu).split(9216,dim=-1)
            return F.linear(F.silu(gate)*up,down).to(x.dtype)
        mlp.forward=MethodType(call,mlp)
        del weights
    assert matrix_count==216
    clear(st[1])
    expected=12*6*3*1536*320*(2 if mode in ("FUSED_FP16","FUSED_BF16") else 4)
    assert nbytes==(0 if mode=="UNCACHED" else expected)
    return {"FFN_cache_bytes":nbytes,"cache_dtype":str(DTYPES.get(mode,"none")),"logical_FFN_matrices":matrix_count,
            "setup_seconds":time.perf_counter()-start},counters

def run(g05,f08,out):
    torch.manual_seed(20270103);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    ar,_,_=open_bundle(blobs[0])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    out.mkdir(parents=True,exist_ok=False)
    states={};keys={};inventory={};ffn_counts={};head_counts={};rows=[]
    with tempfile.TemporaryDirectory() as td:
        lib,kernel=build_v2(td)
        for mode in MODES:
            st,second,added,fourth,reg,old,fifth,sixth,reg2=restore(blobs[:6],blobs[6],blobs[7])
            inv,cnt=install_ffn(st,(st[1],second,added,fourth,fifth,sixth),mode)
            inventory[mode]=inv;ffn_counts[mode]=cnt
            head_counts[mode]=install_packed(st,ar,kernel)
            keys[mode]=keep_keyword(st[0]);states[mode]=st
        q="Напиши число 3."
        p=strict_chat_source_ids(tok,q);f=prompt_features(tok,q)
        for mode in MODES:generate(states[mode],p,f,"PACKED_LAST",keys[mode])
        for i,(q,a) in enumerate(PROMPTS):
            p,g=native(tok,q,a);f=prompt_features(tok,q)
            order=MODES[i%len(MODES):]+MODES[:i%len(MODES)]
            ts={m:[] for m in MODES};err={m:0. for m in MODES}
            agree={m:0 for m in MODES}
            for j in range(len(g)):
                zz={}
                for m in order:
                    start=time.perf_counter()
                    zz[m]=last_forward(states[m],p+g[:j],f,keys[m])
                    ts[m].append(time.perf_counter()-start)
                ref=zz[MODES[0]]
                for m in MODES:
                    err[m]=max(err[m],float((zz[m]-ref).abs().max()))
                    agree[m]+=int(zz[m].argmax()==ref.argmax())
            greedy={}
            for m in order:
                ids,_,dt=generate(states[m],p,f,"PACKED_LAST",keys[m])
                greedy[m]={"ids":ids,"seconds":dt}
            rows.append({"q_sha256":digest(q.encode()),"prefix_steps":len(g),
                         "max_error":err,"top1_agreement":agree,
                         "forced_seconds":{m:sum(ts[m]) for m in MODES},
                         "greedy":greedy})
        result={}
        for m in MODES:
            same=[r for r in rows if r["greedy"][m]["ids"]==r["greedy"][MODES[0]]["ids"]]
            count=ffn_counts[m]
            assert count["layer_calls"]==12*head_counts[m]["packed_head"]
            assert count["linear_calls"]==count["layer_calls"]*(2 if m in DTYPES else 18)
            result[m]={**inventory[m],"FFN_calls":count,"head_calls":head_counts[m],
                       "max_logit_error":max(r["max_error"][m] for r in rows),
                       "same_top1":sum(r["top1_agreement"][m] for r in rows),
                       "fixed_prefix_steps":sum(r["prefix_steps"] for r in rows),
                       "greedy_parity":len(same),"prompts":len(rows),
                       "fixed_prefix_median_speedup":statistics.median(r["forced_seconds"][MODES[0]]/r["forced_seconds"][m] for r in rows),
                       "relative_to_fused_fp32":statistics.median(r["forced_seconds"]["FUSED_FP32"]/r["forced_seconds"][m] for r in rows),
                       "same_greedy_median_speedup":statistics.median(r["greedy"][MODES[0]]["seconds"]/r["greedy"][m]["seconds"] for r in same) if same else None}
            print("G11_ARM",m,json.dumps(result[m],sort_keys=True),flush=True)
    report={"stage":"G11_FUSED_FFN_CACHE_PRECISION","geometry":G05,"depth":12,
            "arms":result,"records":rows,"new_learned_weights":0,
            "candidate_accepted":False,"CPU_speed_certified":False,
            "limitations":["Pinned G05; no changes to concurrent G06/G08/G09 or deployment.",
                           "Same packed tied head and identical physical checkpoints. FP16/BF16 caches and activations rounded during inference; half cache bytes versus fused FP32. Not total RSS or compact resident memory.",
                           "Four attention branches retain full-prefix path; no historical CPU reference, protected replay or quality claim.",
                           "Reduced precision changes numeric computation; no guaranteed parity. Timings on eight reused authored prompts, same host, not quality or historical speed certification."]}
    (out/"G11_REPORT.json").write_text(json.dumps(report,indent=2))
    print("G11_ASSESS",json.dumps({"stage":report["stage"],"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
