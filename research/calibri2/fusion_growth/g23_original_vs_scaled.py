"""G23: same-native-path Russian comparison of original-size and scaled CALIBRI.
Original K32/Z73 physical bundle, F10 compact original-size descendant,
G05 scaled and G18 experimental. Same tokenizer and prompts; no training.
Do not confuse original historical 43.26M reference with 36.38M unique
core entries or historical J7 selector output with native generation.
"""
import argparse
import json
import statistics
import time
from collections import Counter,defaultdict
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from direct_codebook_v59 import CB59
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN as ORIGINAL_PIN,unpack,digest
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f03_rate_train_core import decode
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f17_packed_cpu import PARENT
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g14_russian_ladder import CAPS,score,normalize
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import native
from research.calibri2.fusion_growth.f21_native_cpu_integration import prompt_features,strict_chat_source_ids,forward
from research.calibri2.fusion_growth.g20_cross_evaluation import all_rows,summary
from research.calibri2.fusion_growth import g18_early_context_readout as g18

BASE=36380480
HISTORICAL=43261760
ARMS=("ORIGINAL_K32_Z73","F10_ORIGINAL_SIZE","G05_SCALED","G18_SCALED_EARLY")
G18_SHA="fb42a068d4c90042bb08d1637b0c1058ebe14a51b9875ffed465827a40e02d4e"

def generate(st,tok,row):
    p=strict_chat_source_ids(tok,row["q"]);f=prompt_features(tok,row["q"])
    cap=min(CAPS[row["level"]],159-len(p));assert cap>0
    ids=[];start=time.perf_counter()
    for _ in range(cap):
        z=forward(st,p+ids,f).detach()
        ids.append(int(z.argmax()))
        if ids[-1]==2:break
    seconds=time.perf_counter()-start
    text=tok.decode(ids,skip_special_tokens=True).strip()
    return {"text":text,"ids":ids,"seconds":seconds,"eos":ids[-1]==2,
            "accepted_exact":score(row,text),"first_token_id":ids[0],
            "first_word_exact":bool(normalize(text).split() and any(normalize(a).split() and normalize(text).split()[0]==normalize(a).split()[0] for a in row["accepted"])),
            "replacement_char":"\ufffd" in text}

def pair_probe(st,tok,rows):
    # Same object and same preceding answer tokens across a contrast pair
    # where possible; evaluate both directions, never train.
    groups=defaultdict(list);res=[]
    for r in rows:
        if r.get("pair_id"):groups[r["pair_id"]].append(r)
    assert len(groups)==32
    for pair,rr in groups.items():
        assert len(rr)==2
        prompts=[];targets=[];states=[];logits=[]
        for r in rr:
            p,g=native(tok,r["q"],r["accepted"][0]);prompts.append(p);targets.append(g)
        pos=next((i for i,(a,b) in enumerate(zip(*targets)) if a!=b),None)
        if pos is None or 2 in (targets[0][pos],targets[1][pos]):continue
        for i,r in enumerate(rr):
            hidden={};handles=[]
            def capture(j):
                def cb(_m,_a,out):
                    x=out[0] if isinstance(out,tuple) else out
                    hidden[j]=x[0,-1].detach().float().numpy().copy()
                return cb
            try:
                for j,layer in enumerate(st[0].model.layers):handles.append(layer.register_forward_hook(capture(j)))
                handles.append(st[0].lm_head.register_forward_pre_hook(lambda _m,a:hidden.__setitem__(12,a[0][0,-1].detach().float().numpy().copy())))
                z=forward(st,prompts[i]+targets[i][:pos],prompt_features(tok,r["q"])).detach()[0]
            finally:
                for h in handles:h.remove()
            assert set(hidden)==set(range(13))
            states.append(np.stack([hidden[j] for j in range(13)]))
            logits.append(z)
        a,b=targets[0][pos],targets[1][pos]
        gaps=[float(logits[0][a]-logits[0][b]),float(logits[1][b]-logits[1][a])]
        d=np.linalg.norm(states[0]-states[1],axis=1)/np.maximum(1e-12,(np.linalg.norm(states[0],axis=1)+np.linalg.norm(states[1],axis=1))/2)
        res.append({"pair_id":pair,"family":rr[0]["family"],"divergence":pos,
                    "target_ids":[a,b],"gaps":gaps,"both_preferred":all(v>0 for v in gaps),
                    "relative_hidden_difference":d.tolist()})
    return {"n":len(res),"both_preferred":sum(r["both_preferred"] for r in res),
            "layer_median_relative_difference":np.median([r["relative_hidden_difference"] for r in res],axis=0).tolist(),
            "records":res}

def run(f01,f10,g05,g18dir,f08,out):
    torch.manual_seed(20270113);torch.set_num_threads(2)
    raw=(f01/"F01.xz.bundle").read_bytes();files=unpack(raw)
    assert all(digest(files[k])==v for k,v in ORIGINAL_PIN.items())
    original=CB59(raw_core(files[NAMES[0]]))
    original_side={k:files[k] for k in NAMES[1:]}
    compact=(f10/"F10_FP16.bundle").read_bytes();assert digest(compact)==PARENT
    oa,os=decode(compact)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    aux=(g18dir/"CENTERED_EARLY_L0.G17R").read_bytes();assert digest(aux)==G18_SHA
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    rows=all_rows(tok);assert len(rows)==192
    tb=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    # Report actual artifact lengths, not a complete deployment inventory.
    out.mkdir(parents=True,exist_ok=False)
    (out/"G23_SUITE.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    results={}
    for arm in ARMS:
        if arm=="ORIGINAL_K32_Z73":
            st=direct(original,original_side);rest=None;weight_bytes=len(raw);geom=BASE
        elif arm=="F10_ORIGINAL_SIZE":
            st=direct(oa,os);rest=None;weight_bytes=len(compact);geom=BASE
        else:
            st,*rest=restore(blobs[:6],blobs[6],blobs[7]);weight_bytes=sum(map(len,blobs));geom=G05
        assert len(st[0].model.layers)==12
        context=nullcontext()
        if arm=="G18_SCALED_EARLY":
            context=g18.install(st,g18.unpack(aux));weight_bytes+=len(aux)
        answers=[]
        with context,(out/(arm+".jsonl")).open("w",encoding="utf8") as f:
            # Warmup, not included in measurements.
            generate(st,tok,{"q":"Напиши число 2.","level":"L0","accepted":["2"]})
            for i,row in enumerate(rows,1):
                a=generate(st,tok,row);answers.append(a)
                rec={"id":row["id"],"q":row["q"],"accepted":row["accepted"],**a}
                f.write(json.dumps(rec,ensure_ascii=False)+"\n");f.flush()
                if i%16==0:print("G23_PROGRESS",arm,i,flush=True)
            probe=pair_probe(st,tok,rows)
        closed=[a for a in answers if a["accepted_exact"] is not None]
        groups=summary(rows,answers)
        item={"logical_core_entries":geom,"historical_original_reference":HISTORICAL,
              "stored_weight_artifact_bytes":weight_bytes,"tokenizer_bytes":tb,
              "known_stored_bytes":weight_bytes+tb,"known_payload_bpp_core_denominator":8*(weight_bytes+tb)/geom,
              "original_reference_bpp":8*(weight_bytes+tb)/BASE,
              "exact_closed":sum(a["accepted_exact"] is True for a in closed),"closed_n":len(closed),
              "EOS":sum(a["eos"] for a in answers),"first_word_exact":sum(a["first_word_exact"] for a in closed),
              "replacement_char":sum(a["replacement_char"] for a in answers),
              "generated_tokens":sum(len(a["ids"]) for a in answers),
              "native_generation_seconds":sum(a["seconds"] for a in answers),
              "native_tokens_per_second":sum(len(a["ids"]) for a in answers)/sum(a["seconds"] for a in answers),
              "median_response_seconds":statistics.median(a["seconds"] for a in answers),
              "scores":groups,"paired_diagnostic":probe}
        results[arm]=item
        print("G23_ARM",arm,json.dumps({**item,"paired_diagnostic":{k:v for k,v in probe.items() if k!="records"}},ensure_ascii=False,sort_keys=True),flush=True)
        (out/"G23_REPORT.json").write_text(json.dumps({"stage":"G23_ORIGINAL_VS_SCALED","arms":results,
            "questions_per_arm":192,"training_updates":0,"candidate_accepted":False,
            "limitations":["Original historical 43.261760M reference is not the unique 36.380480M core denominator. Same native path, not historical routed J7.",
             "Original K32/Z73 vs F10 vs G05 differ in training, compression and architecture; no causal attribution to parameter count alone.",
             "Weight artifact lengths are physical files, not full deployment package/RSS. F01 xz envelope vs F10 bundle vs G05 separate banks.",
             "Two CPU threads on one runner, full-prefix native reference; sequential arms and different generated lengths, not strict fixed-token production speed certification.",
             "G14 viewed previously; G19 template-based held-out data. Exact matching misses paraphrases; open generation requires semantic review.",
             "Pair probes use shared gold prefix and evaluate both target preferences; not free generation. No teacher, no protected J7 fitting or promotion."]},ensure_ascii=False,indent=2))
        del st,rest,context
    print("G23_ASSESS",json.dumps({"arms":list(results),"candidate_accepted":False}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f10","g05","g18","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f10,a.g05,a.g18,a.f08,a.out)
