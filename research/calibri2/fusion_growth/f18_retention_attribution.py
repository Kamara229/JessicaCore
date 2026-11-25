"""F18: frozen retention attribution, not training on protected J7.

F15 has real 549/616 exact historical sequences for both rebased banks.
Distinguish loss caused by rate reduction, joint core updates, and GT updates.
No selection or parameter fitting based on protected prompts/results.
"""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
import torch
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f03_rate_train_core import make,reduced,encode,decode,k_for
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,HDR,open_bundle
from research.calibri2.fusion_growth.f13_profile_rebase import (
    BANK_NAME,BANK_SHA,KEYS,bank_open,bank_pack,extra_open,extra_pack,projection)
from research.calibri2.fusion_growth.f15_legacy_j7_replay import (
    PARENT_SHA,ARM_SHA,configure_original_selector,model_state,replay,output_digest,
    s30,s87,source_questions_456,materialize_train_only,new_train_paraphrase_probes)
from direct_codebook_v59 import CB59,unpack5

H584="3b3cfe8cac8c80940750a8f15ffc86917e87288be7d8593d2e9cc5d7118fb7b0"
H32="e8b3d993c5ffab2c20528e429a9302330f82715580e3626d0e30cc03ff0732be"
RATE="K16_embed_K8_body"

def rate_only(old,side,rate):
    model,sources=make(old)
    with torch.no_grad():
        for name,s in sources.items():
            c,i=reduced(s.centroids.detach().numpy(),s.codes.numpy(),k_for(rate,name))
            s.centroids.copy_(torch.from_numpy(c));s.codes.copy_(torch.from_numpy(i))
    packet=encode(old.blob,sources,rate,side)
    del model,sources
    a,b=decode(packet);assert b==side
    return packet,a

def rebase(old,new,inventory,extras,rate,source_sha):
    h,chunks=bank_open(extras[BANK_NAME])
    for name in KEYS:
        a,b=old.handles[name],new.handles[name]
        m,n=a.shape;k=k_for(rate,name)
        assert a.shape==b.shape and np.array_equal(a.row_scales,b.row_scales)
        x=unpack5(a.packed.tobytes(),m*n//8).astype(np.int64)
        y=unpack5(b.packed.tobytes(),m*n//8).astype(np.int64)
        assert y.max()<k
        w=np.repeat(a.row_scales.astype(np.float64)**2,n//8)
        weights=np.bincount(x*k+y,weights=w,minlength=32*k).reshape(32,k)
        assert (weights.sum(0)>0).all()
        for profile in ("S87","S91"):
            alt=np.frombuffer(chunks[profile,name],dtype="<f2").reshape(32,8).astype(np.float64)
            padded=np.zeros((32,8),dtype="<f2");padded[:k]=projection(weights,alt)
            chunks[profile,name]=padded.tobytes()
    # F13 writer hard-codes F06 provenance; replace only manifest provenance
    # with the actual new physical parent, preserve encoded payload and hashes.
    bank=bank_pack(h,chunks)
    n,=struct.unpack_from("<I",bank,8)
    header=json.loads(bank[12:12+n]);header["fusion_parent_sha256"]=source_sha
    header["fusion_profile_rate"]=rate
    raw=json.dumps(header,sort_keys=True,separators=(",",":")).encode()
    bank=bank[:8]+struct.pack("<I",len(raw))+raw+bank[12+n:]
    assert bank_open(bank)[1]==chunks
    b=dict(extras);b[BANK_NAME]=bank
    packed=extra_pack(inventory,b)
    assert extra_open(packed)[1]==b
    return packed,b

def q8_size(packet,extra,rate):
    a,s=decode(packet);model,sources=make(a)
    compact,h,_=split_head(s,8);cp=encode(a.blob,sources,rate,compact)
    qp=HDR.pack(b"FG09",len(cp),len(h),len(extra))+cp+h+extra
    ar,ss,ee=open_bundle(qp);assert ar.blob==a.blob and ee==extra
    del model,sources
    return {**metrics(len(qp)),"sha256":digest(qp),"head_active_in_J7":False}

def run(f01,f06,f08,f13,f15,out):
    torch.set_num_threads(2);torch.manual_seed(20261206)
    raw=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(raw[k])==v
    original=CB59(raw_core(raw[NAMES[0]]));side={k:raw[k] for k in NAMES[1:]}
    trained=(f06/"F06_JOINT.bundle").read_bytes();assert digest(trained)==PARENT_SHA
    mixed,ts=decode(trained)
    extra=(f08/"legacy_extras.bundle").read_bytes()
    inventory,extras=extra_open(extra);assert digest(extras[BANK_NAME])==BANK_SHA
    db=(f13/"DIRECT_legacy.bundle").read_bytes();assert digest(db)==ARM_SHA["DIRECT"]
    _,rebased=extra_open(db)
    prior=json.loads((f15/"F15_REPORT.json").read_text())
    assert prior["baseline_exact_historical"] and prior["arms"]["DIRECT"]["exact_616"]==549
    assert prior["baseline_584_digest"]==H584 and prior["baseline_32_digest"]==H32
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
    out.mkdir(parents=True,exist_ok=False)
    configure_original_selector()
    # Confirm baseline on current runner before any treatment scoring.
    state=model_state(original,side,extras,False)
    fresh,_,_=replay(state,tok,rows,forced);del state
    assert fresh==ref,"historical runtime baseline drift"
    results={}
    def arm(name,archive,sd,blob,blob_map,trained_gt,packet,rate):
        state=model_state(archive,sd,blob_map,trained_gt)
        pred,_,counts=replay(state,tok,rows,forced);del state
        same={k for k in ref if pred[k]==ref[k]}
        result={"exact_616":len(same),"drift":616-len(same),
                "exact_groups":{g:sum(keys[k] in same for k in rows if k.startswith(g+"/")) for g in ("OLD","NEW","PROBE")},
                "route_counts":counts,"trained_GT":trained_gt}
        if packet is not None:
            result["FP16_plus_legacy"]=metrics(len(packet)+len(blob))
            result["Q8_plus_legacy_storage_only"]=q8_size(packet,blob,rate)
        (out/(name+"_IDS.json")).write_text(json.dumps(pred,sort_keys=True))
        results[name]=result
        print("F18_ARM",name,json.dumps(result,sort_keys=True),flush=True)
    # Fixed diagnostic arms predeclared; no adaptive selection on J7.
    arm("ORIGINAL_K32_TRAINED_GT",original,ts,extra,extras,True,None,None)
    for rate in ("K16_all",RATE):
        p,a=rate_only(original,side,rate)
        b,e=rebase(original,a,inventory,extras,rate,digest(p))
        arm("RATE_ONLY_"+rate,a,side,b,e,False,p,rate)
    # Hold the exact F06 trained core and DIRECT bank fixed while restoring
    # only the original inherited GT route gates.
    arm("F06_ORIGINAL_GT",mixed,ts,db,rebased,False,trained,RATE)
    arm("F06_TRAINED_GT",mixed,ts,db,rebased,True,trained,RATE)
    assert results["F06_TRAINED_GT"]["exact_616"]==549
    report={"stage":"F18_RETENTION_ATTRIBUTION","parent_sha256":PARENT_SHA,
            "baseline_exact_616":616,"arms":results,"optimizer_steps":0,
            "protected_labels_in_model":False,"FP32_teacher_calls":0,
            "candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Frozen diagnostic only: no adaptation from protected J7 outputs or scores.",
                           "Rate-only changes code geometry and corresponding alternate banks together; not a pure single-factor codebook-only intervention.",
                           "F06 GT ablation keeps child unused; native joint decoder is not active.",
                           "Q8 size is an actual serialized storage variant, not native generation equivalence or CPU certification. F06_ORIGINAL_GT size charges the saved trained gate file, not an exported reverted checkpoint.",
                           "Historical reference sequences include 32 outputs that are not necessarily correct answers. Exact retention is not accuracy."]}
    (out/"F18_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F18_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f06","f08","f13","f15","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f06,a.f08,a.f13,a.f15,a.out)
