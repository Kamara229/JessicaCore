"""F13: rebase the actual S94 alternate profiles to mixed K16/K8 code IDs.

Read-only weight geometry audit: two alternative codebooks were trained on
original K32 IDs. Copying their values into K8 slots is semantically wrong.
Compute weighted projections using original and F06 physical group IDs.
No protected J7 prompts, forward, training, or claims of functional retention.
"""
import argparse
import copy
import json
import lzma
import struct
import zlib
from pathlib import Path
import numpy as np
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f03_rate_train_core import decode
from direct_codebook_v59 import CB59,unpack5

PARENT_SHA="7dcc391ce6001f88f6e7c12498c4f371b83431b049e4db2cb65d7f0b89ab722a"
BANK_SHA="3ea693cc95cae5af3f09a01e72ac739ce2899251d10a7949455d60405c3ca8f5"
BANK_NAME="S94_ALTERNATE_EXACT_CORE_TUNNEL_BANK.bin"
KEYS=("model.layers.3.mlp.down_proj.weight","model.layers.9.mlp.down_proj.weight")
MAGIC=b"S94BANK1"

def extra_open(packet):
    assert packet[:4]==b"FG08"
    data=lzma.decompress(packet[4:])
    n,=struct.unpack_from("<I",data);items=json.loads(data[4:4+n]);p=4+n
    blobs={}
    for i in items:
        b=data[p:p+i["bytes"]];p+=len(b)
        assert digest(b)==i["sha256"] and i["name"] not in blobs
        blobs[i["name"]]=b
    assert p==len(data)
    return items,blobs

def extra_pack(items,blobs):
    items=copy.deepcopy(items)
    for i in items:
        b=blobs[i["name"]];i["bytes"]=len(b);i["sha256"]=digest(b)
    h=json.dumps(items,sort_keys=True,separators=(",",":")).encode()
    p=b"FG08"+lzma.compress(struct.pack("<I",len(h))+h+b"".join(blobs[i["name"]] for i in items),preset=6)
    assert extra_open(p)[1]==blobs
    return p

def bank_open(blob):
    assert blob[:8]==MAGIC
    n,=struct.unpack_from("<I",blob,8)
    h=json.loads(blob[12:12+n]);data=zlib.decompress(blob[12+n:])
    assert digest(data)==h["uncompressed_sha256"] and len(data)==h["uncompressed_bytes"]
    chunks={}
    for i in h["tensors"]:
        b=data[i["offset"]:i["offset"]+i["size"]]
        assert digest(b)==i["sha256"]
        key=(i["profile"],i["name"]);assert key not in chunks
        chunks[key]=b
    return h,chunks

def bank_pack(h,chunks):
    h=copy.deepcopy(h);data=bytearray()
    for i in h["tensors"]:
        b=chunks[(i["profile"],i["name"])]
        i["offset"]=len(data);i["size"]=len(b);i["sha256"]=digest(b);data.extend(b)
    packed=zlib.compress(bytes(data),9)
    h.update(uncompressed_sha256=digest(bytes(data)),uncompressed_bytes=len(data),deflate_bytes=len(packed),
             source_profiles_are_frozen=False,
             fusion_profile_rebase_research_only=True,
             fusion_parent_sha256=PARENT_SHA)
    raw=json.dumps(h,sort_keys=True,separators=(",",":")).encode()
    p=MAGIC+struct.pack("<I",len(raw))+raw+packed
    assert bank_open(p)[1]==chunks
    return p

def geometry(old,new):
    m,n=old.shape
    assert old.shape==new.shape and n%8==0
    assert np.array_equal(old.row_scales,new.row_scales)
    a=unpack5(old.packed.tobytes(),m*n//8).astype(np.int64)
    b=unpack5(new.packed.tobytes(),m*n//8).astype(np.int64)
    assert b.max()<8
    # Squared physical weight error is proportional to squared row scales.
    w=np.repeat(old.row_scales.astype(np.float64)**2,n//8)
    counts=np.bincount(a*8+b,minlength=256).reshape(32,8)
    weight=np.bincount(a*8+b,weights=w,minlength=256).reshape(32,8)
    assert (counts>0).sum(1).max()==1,"F06 unexpectedly changed discrete IDs"
    assert counts.sum()==len(a) and (weight.sum(0)>0).all()
    return counts,weight

def projection(w,values):
    return (w.T@values)/w.sum(0)[:,None]

def mse(w,old,new):
    delta=old[:,None,:]-new[None,:,:]
    return float((w[:,:,None]*delta**2).sum()/(8*w.sum()))

def run(f01,f06,f08,out):
    old_files=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(old_files[k])==v
    old=CB59(raw_core(old_files[NAMES[0]]))
    f06_packet=(f06/"F06_JOINT.bundle").read_bytes();assert digest(f06_packet)==PARENT_SHA
    mixed,_=decode(f06_packet)
    extras=(f08/"legacy_extras.bundle").read_bytes();assert len(extras)==137712
    inventory,blobs=extra_open(extras)
    bank=blobs[BANK_NAME];assert digest(bank)==BANK_SHA
    h,chunks=bank_open(bank)
    out.mkdir(parents=True,exist_ok=False)
    report={"stage":"F13_PHYSICAL_PROFILE_REBASE","source_sha256":PARENT_SHA,
            "original_bank_sha256":BANK_SHA,"depth":12,"models":[],"arms":{},
            "optimizer_steps":0,"FP32_teacher_calls":0,"candidate_accepted":False,
            "output_gate":0,"all_in_certified":False,
            "limitations":["Weight-domain projection only; no real J7 or CPU inference.",
                           "Direct projection minimizes physical alternate weight MSE under fixed mixed IDs; delta transfer preserves trained mixed base shift.",
                           "Router, tunnel tensors and other legacy files unchanged; functional compatibility must be tested.",
                           "Research bank replaces old bank, does not claim old historical hash or exact weights.",
                           "Tokenizer and full executable runtime not charged."]}
    outputs={"DIRECT":dict(chunks),"DELTA":dict(chunks)}
    for k in KEYS:
        a,b=old.handles[k],mixed.handles[k]
        count,w=geometry(a,b)
        base=a.centroids.astype(np.float64);trained=b.centroids[:8].astype(np.float64)
        for p in ("S87","S91"):
            alt=np.frombuffer(chunks[p,k],dtype="<f2").reshape(32,8).astype(np.float64)
            direct=projection(w,alt)
            shifted=trained+projection(w,alt-base)
            naive=alt[:8]
            stats={"profile":p,"tensor":k,"original_groups":int(count.sum()),
                   "naive_MSE":mse(w,alt,naive),
                   "direct_MSE":mse(w,alt,direct.astype("<f2").astype(np.float64)),
                   "delta_MSE":mse(w,alt,shifted.astype("<f2").astype(np.float64)),
                   "baseline_mixed_MSE":mse(w,base,trained),
                   "active_old_ids":int((count.sum(1)>0).sum()),
                   "active_new_ids":int((count.sum(0)>0).sum())}
            assert stats["direct_MSE"]<=stats["naive_MSE"]+1e-9
            report["models"].append(stats)
            for mode,c in (("DIRECT",direct),("DELTA",shifted)):
                assert np.isfinite(c).all()
                padded=np.zeros((32,8),dtype="<f2");padded[:8]=c
                outputs[mode][p,k]=padded.tobytes()
    for mode,new_chunks in outputs.items():
        changed=[k for k in chunks if chunks[k]!=new_chunks[k]]
        assert set(changed)=={(p,k) for p in ("S87","S91") for k in KEYS}
        new_bank=bank_pack(h,new_chunks)
        new_blobs=dict(blobs);new_blobs[BANK_NAME]=new_bank
        p=extra_pack(inventory,new_blobs)
        (out/(mode+"_legacy.bundle")).write_bytes(p)
        (out/(mode+"_bank.bin")).write_bytes(new_bank)
        report["arms"][mode]={"bank_bytes":len(new_bank),"bank_sha256":digest(new_bank),
                             "legacy_bytes":len(p),"legacy_sha256":digest(p),
                             "F06_plus_legacy":metrics(len(f06_packet)+len(p)),
                             "changed_tensors":len(changed)}
    (out/"F13_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F13_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("f01","f06","f08","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f06,a.f08,a.out)
