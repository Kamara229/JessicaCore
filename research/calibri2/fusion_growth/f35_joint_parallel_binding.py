"""F35: balanced parallel binding, frozen head-only vs joint core+head.
Both arms optimize identical two-token span likelihood on fresh paired
TRAIN questions; no native autoregressive target or new weights.
Separate same-pool held-out contexts and lexical-transfer contexts.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from safetensors.torch import load,save
from research.calibri2.fusion_growth.f34_parallel_prefill_learning import (
    PARENT,PARALLEL,HDR,open_bundle,extra_open,extra_pack,direct,
    forward,native,prompt_features,clear,normalized_candidate_spans,
    joint_span_nll,new_head,collect,audit,brief,score,digest,metrics,native_audit)
from research.calibri2.fusion_growth.f01_physical_budget import NAMES
from research.calibri2.fusion_growth.f03_rate_train_core import encode,k_for
from research.calibri2.fusion_growth.f06_integrated_recovery import OLD_PREFIX
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head

SEED=20261219
RATE="K16_embed_K8_body"
ARMS=("HEAD_ONLY","JOINT_CORE_HEAD")
STEPS=48
MODE="FINAL"

def contexts(tok,transfer):
    colors=("white","yellow","pink","brown") if transfer else ("red","blue","green","black")
    nouns=("chair","lamp","clock","bag","pen","shoe","plate","bell") if transfer else ("book","key","box","cup","ball","door","ring","hat")
    names=("Лея","Ганс") if transfer else ("Ада","Клим")
    labels=[f"{c} {w}" for c in colors for w in nouns
            if len(tok.encode(f"{c} {w}",add_special_tokens=False))==2]
    assert len(labels)>=8
    # Randomized complete pair pool; no first-label-fixed nested-loop prefix.
    pairs=[(a,b) for a in labels for b in labels if a.split()[0]!=b.split()[0] and a.split()[1]!=b.split()[1]]
    rng=np.random.default_rng(SEED+int(transfer))
    rows=[];seen=set()
    for ix in rng.permutation(len(pairs)):
        a,b=pairs[int(ix)]
        if tuple(sorted((a,b))) in seen:continue
        pair=[]
        for who,y in zip(names,(a,b)):
            q=f"Карточки: {names[0]} имеет код {a}; {names[1]} имеет код {b}. Выпиши полностью код, который принадлежит {who}."
            p,g=native(tok,q,y)
            if len(g)!=3 or g[-1]!=2:break
            canon=normalized_candidate_spans(tok,p)
            if not all(bool((canon==torch.tensor(tok.encode(x,add_special_tokens=False))).all(-1).any()) for x in (a,b)):break
            if not bool((canon==torch.tensor(g[:2])).all(-1).any()):break
            pair.append({"q":q,"p":p,"g":g,"pair":len(rows)//2,
                         "features":prompt_features(tok,q),"canon":canon})
        if len(pair)!=2:continue
        seen.add(tuple(sorted((a,b))));rows.extend(pair)
        if len(rows)==(24 if transfer else 72):break
    assert len(rows)==(24 if transfer else 72)
    assert len({r["q"] for r in rows})==len(rows)
    return rows

def data(tok):
    a=contexts(tok,False);b=contexts(tok,True)
    result={"TRAIN":a[:48],"DEV_CONTEXT":a[48:],"DEV_LEXICAL":b}
    seen=set()
    for split,rows in result.items():
        for r in rows:r["split"]=split
        keys={r["q"] for r in rows}
        assert not seen&keys;seen|=keys
        # Fail if a single answer dominates; all selected labels are verified
        # independently from the model, not from its generations.
        hist=Counter(tuple(r["g"][:2]) for r in rows)
        assert len(hist)>=8 and max(hist.values())<=len(rows)//4
    return result

def live_final(state,row):
    cap=[]
    h=state[0].lm_head.register_forward_pre_hook(lambda _m,args:cap.append(args[0].float()))
    try:z=forward(state,row["p"],row["features"])
    finally:h.remove()
    assert len(cap)==1 and cap[0].shape==(1,len(row["p"]),320) and cap[0].requires_grad
    assert z.shape==(1,49152)
    return {**row,"states":{MODE:cap[0]}}

def packet_with_head(archive,side,sources,full,child,items,blobs,head):
    original=load(side[NAMES[3]])
    ns=dict(side)
    changed={OLD_PREFIX+k:v.detach().cpu().half().contiguous() for k,v in child.state_dict().items()}
    assert set(changed)==set(original)
    assert all(torch.equal(v,original[k]) for k,v in changed.items() if ".nra3_neural_weights." in k or ".language_head." in k)
    ns[NAMES[3]]=save(changed)
    gate_key=next(iter(load(side[NAMES[4]])))
    ns[NAMES[4]]=save({gate_key:full.gates.detach().cpu().half().contiguous()})
    compact,packed_head,_=split_head(ns,8)
    cp=encode(archive.blob,sources,RATE,compact)
    new_blobs=dict(blobs)
    new_blobs[PARALLEL]=save({k:v.detach().cpu().half().contiguous() for k,v in head.state_dict().items()})
    assert all(new_blobs[k]==v for k,v in blobs.items() if k!=PARALLEL)
    legacy=extra_pack(items,new_blobs)
    packet=HDR.pack(b"FG09",len(cp),len(packed_head),len(legacy))+cp+packed_head+legacy
    a,s,l=open_bundle(packet)
    assert extra_open(l)[1]==new_blobs
    return packet,a,s,l

def run(f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(SEED)
    raw=(f10/"F10_Q8.bundle").read_bytes();assert digest(raw)==PARENT
    archive,side,legacy=open_bundle(raw)
    assert legacy==(f08/"legacy_extras.bundle").read_bytes()
    # Train from FP16 child to avoid a second rounding of Q8 donor tensors.
    from research.calibri2.fusion_growth.f03_rate_train_core import decode
    from research.calibri2.fusion_growth.f20_state_diagnostic import F10_SHA
    fp=(f10/"F10_FP16.bundle").read_bytes();assert digest(fp)==F10_SHA
    fp_archive,fp_side=decode(fp);assert fp_archive.blob==archive.blob
    items,blobs=extra_open(legacy)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    splits=data(tok)
    out.mkdir(parents=True,exist_ok=False)
    state=direct(fp_archive,fp_side)
    cached={k:[collect(state,r) for r in rows] for k,rows in splits.items()}
    before={k:audit(new_head(blobs[PARALLEL]),rows,MODE) for k,rows in cached.items()}
    del state
    print("F35_BEFORE",json.dumps({k:brief(v) for k,v in before.items()},sort_keys=True),flush=True)
    order=[int(i) for _ in range(2) for i in np.random.default_rng(SEED+_).permutation(24)]
    assert len(order)==STEPS
    results={}
    for arm in ARMS:
        state=direct(fp_archive,fp_side);_,sources,_,full,_,child,region=state
        head=new_head(blobs[PARALLEL])
        core=[s.centroids for s in sources.values()]
        native_ids={id(p) for p in child.language_head.parameters()}
        nonhead=[p for p in region if id(p) not in native_ids]
        params=list(head.parameters())
        groups=[{"params":params,"lr":.002}]
        if arm=="JOINT_CORE_HEAD":
            groups.extend([{"params":core,"lr":.00015},{"params":nonhead,"lr":.0003},{"params":[full.gates],"lr":.0001}])
            params=params+core+nonhead+[full.gates]
        opt=torch.optim.Adam(groups)
        seen=set();gt=set();regions=set();curve=[]
        for step,i in enumerate(order):
            opt.zero_grad(set_to_none=True)
            pair=(cached["TRAIN"] if arm=="HEAD_ONLY" else splits["TRAIN"])[2*i:2*i+2]
            for r in pair:
                assert r["split"]=="TRAIN"
                row=r if arm=="HEAD_ONLY" else live_final(state,r)
                z=score(head,row,MODE)
                loss=joint_span_nll(z,r["canon"][None],torch.tensor([r["g"][:2]]))
                assert torch.isfinite(loss);(loss/2).backward()
                curve.append(float(loss.detach()))
                del row,z,loss
            if arm=="JOINT_CORE_HEAD":
                for k,s in sources.items():
                    if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(k)
                if full.gates.grad is not None:gt.update(i for i,v in enumerate(full.gates.grad) if float(v.abs())>0)
                for k,p in child.named_parameters():
                    if p.grad is not None and float(p.grad.abs().sum())>0:regions.add(k)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            if arm=="JOINT_CORE_HEAD":
                with torch.no_grad():
                    for k,s in sources.items():s.centroids[k_for(RATE,k):].zero_()
            if (step+1)%12==0:print("F35_PROGRESS",arm,step+1,float(np.mean(curve[-24:])),len(seen),len(gt),flush=True)
        assert len(curve)==STEPS*2
        packet,a,s,l=packet_with_head(fp_archive,fp_side,sources,full,child,items,blobs,head)
        if arm=="HEAD_ONLY":assert a.blob==fp_archive.blob and s==side
        assert len(packet)<=3183292
        (out/(arm+"_Q8.bundle")).write_bytes(packet)
        del opt,state,sources,full,child,region,core,params,groups,nonhead,head
        restored=direct(a,s)
        loaded=new_head(extra_open(l)[1][PARALLEL])
        collected={k:[collect(restored,r) for r in rows] for k,rows in splits.items()}
        after={k:audit(loaded,rows,MODE) for k,rows in collected.items()}
        native_result=native_audit(restored,collected["DEV_CONTEXT"])
        del restored,collected
        results[arm]={"after":after,"native_DEV_CONTEXT":native_result,
                      "steps":STEPS,"connected_core":len(seen),"connected_GT":len(gt),
                      "connected_regions":sorted(regions),"known_payload":metrics(len(packet)),
                      "sha256":digest(packet),"first24_loss":float(np.mean(curve[:24])),
                      "last24_loss":float(np.mean(curve[-24:]))}
        print("F35_ARM",arm,json.dumps({**results[arm],"after":{k:brief(v) for k,v in after.items()},
              "native_DEV_CONTEXT":brief(native_result)},sort_keys=True),flush=True)
    report={"stage":"F35_JOINT_PARALLEL_BINDING","arms":results,"before":before,
            "manifest_sha256":digest(json.dumps({k:[(r["q"],r["g"]) for r in v] for k,v in splits.items()},ensure_ascii=False).encode()),
            "new_parameters":0,"FP32_teacher_calls":0,"candidate_accepted":False,"output_gate":0,
            "limitations":["Authored source-bound two-token span copy, not unrestricted generation; EOS follows original fixed slot protocol.",
                           "F34 nested-loop sampling biased label distribution. F35 precommits randomized distinct unordered contexts, same-pool DEV and lexical-transfer DEV.",
                           "Joint gradient through actual final states, no native-head optimizer; frozen original old NRA3 and expansion, not every parameter trainable.",
                           "No protected J7 data or runtime routing changes; profile bank unchanged and not certified for new core. No wall-clock CPU certification or all-in budget."]}
    (out/"F35_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("F35_ASSESS",json.dumps({"stage":report["stage"],"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f10,a.f08,a.out)
