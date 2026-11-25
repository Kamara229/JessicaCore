"""F24: lossless K32 entropy coding, without shrinking codebooks or training.
Static rANS per tensor and deterministic row-scale contexts. Original CB59
bytes, including exact code IDs, row scales, centroids and checksum, restored.
Also measure optional Q8/Q4 child-head storage with exact core; no acceptance.
"""
import argparse
import hashlib
import json
import lzma
import struct
import time
from pathlib import Path
import numpy as np
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f02_entropy_core import raw_core,sf_parts,shuffle
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,merge_head
from research.calibri2.fusion_growth.f13_profile_rebase import extra_open,BANK_NAME,BANK_SHA
from direct_codebook_v59 import CB59,HEADER,pack5,unpack5

PREC=12
TOTAL=1<<PREC
LOW=1<<23
MAGIC=b"FG24"
CONTEXTS=(1,4,16)
FRAME=struct.Struct("<4sIII")

def frequencies(ids):
    counts=np.bincount(ids,minlength=32).astype(np.float64)
    expect=counts/counts.sum()*(TOTAL-32)
    freq=np.floor(expect).astype(np.int64)+1
    remainder=TOTAL-int(freq.sum())
    order=np.argsort(-(expect-np.floor(expect)),kind="stable")
    freq[order[:remainder]]+=1
    assert freq.sum()==TOTAL and np.all(freq>0)
    return freq.tolist()

def rans_encode(ids,freq):
    start=np.cumsum([0]+freq[:-1]).tolist()
    maxima=[((LOW>>PREC)<<8)*f for f in freq]
    x=LOW;out=bytearray()
    for symbol in reversed(ids.tolist()):
        f=freq[symbol]
        while x>=maxima[symbol]:
            out.append(x&255);x>>=8
        x=((x//f)<<PREC)+(x%f)+start[symbol]
    assert LOW<=x<1<<32
    return struct.pack("<I",x)+bytes(reversed(out))

def rans_decode(raw,n,freq):
    assert len(freq)==32 and min(freq)>0 and sum(freq)==TOTAL and len(raw)>=4
    start=np.cumsum([0]+freq[:-1]).tolist()
    lut=np.repeat(np.arange(32,dtype=np.uint8),freq).tolist()
    x,=struct.unpack_from("<I",raw);assert x>=LOW
    p=4;out=bytearray(n)
    for i in range(n):
        slot=x&(TOTAL-1);s=lut[slot];out[i]=s
        x=freq[s]*(x>>PREC)+slot-start[s]
        while x<LOW:
            assert p<len(raw)
            x=(x<<8)|raw[p];p+=1
    assert p==len(raw) and x==LOW
    return np.frombuffer(out,dtype=np.uint8)

def contexts(scales,n):
    # Stable sort by serialized FP16 scales. Decoder derives the exact same
    # row partition from restored scales; no per-row context labels stored.
    order=np.argsort(scales,kind="stable")
    return np.array_split(order,min(n,len(order)))

def encode_core(raw,ctx):
    a=CB59(raw);_,_,mlen,_=HEADER.unpack_from(raw);base=HEADER.size+mlen
    skeleton=bytearray(raw);manifest=[];streams=[];entropy=0.;symbols=0
    for name,d in sorted(a.meta["tensor_descriptors"].items()):
        if d["dtype"]!="CB59_K32V8":continue
        h=a.handles[name];m,n=h.shape;g=n//8
        ids=unpack5(h.packed.tobytes(),m*g).reshape(m,g)
        po,pn=d["parts"]["packed"];so,sn=d["parts"]["row_scales"]
        po+=base;so+=base
        skeleton[po:po+pn]=bytes(pn)
        parts=[]
        for rows in contexts(h.row_scales,ctx):
            v=ids[rows].reshape(-1)
            f=frequencies(v);s=rans_encode(v,f)
            counts=np.bincount(v,minlength=32);prob=counts[counts>0]/len(v)
            entropy+=float(-len(v)*np.sum(prob*np.log2(prob)));symbols+=len(v)
            parts.append([f,len(s)]);streams.append(s)
        manifest.append([name,m,g,po,pn,so,sn,parts])
    assert len(manifest)==85
    return bytes(skeleton),manifest,streams,{"symbols":symbols,"empirical_entropy_bits":entropy,
          "bits_per_index_entropy":entropy/symbols,"rans_stream_bytes":sum(map(len,streams))}

def side_parts(side):
    extras=[];entries=[];unique={}
    for name in NAMES[1:]:
        header,parts=sf_parts(side[name]);i=len(extras);extras.append(header);refs=[]
        for _,w,value in parts:
            k=(digest(value),w)
            if k not in unique:unique[k]=len(extras);extras.append(shuffle(value,w))
            refs.append([unique[k],w])
        entries.append([name,i,refs,digest(side[name])])
    return extras,entries

def pack(core,side,head,legacy,ctx):
    skeleton,manifest,streams,_=core
    extras,entries=side_parts(side)
    meta={"context_count":ctx,"skeleton_bytes":len(skeleton),"fields":manifest,
          "side_entries":entries,"extra_sizes":[len(x) for x in extras]}
    h=json.dumps(meta,separators=(",",":")).encode()
    payload=lzma.compress(struct.pack("<I",len(h))+h+skeleton+b"".join(streams)+b"".join(extras),preset=6)
    return FRAME.pack(MAGIC,len(payload),len(head),len(legacy))+payload+head+legacy

def restore(packet):
    magic,n,h,e=FRAME.unpack_from(packet)
    assert magic==MAGIC and FRAME.size+n+h+e==len(packet)
    i=FRAME.size;payload=lzma.decompress(packet[i:i+n]);head=packet[i+n:i+n+h];legacy=packet[i+n+h:]
    j,=struct.unpack_from("<I",payload);m=json.loads(payload[4:4+j]);p=4+j
    raw=bytearray(payload[p:p+m["skeleton_bytes"]]);p+=len(raw)
    for name,rows,g,po,pn,so,sn,parts in m["fields"]:
        scales=np.frombuffer(bytes(raw[so:so+sn]),dtype="<f2")
        assert len(scales)==rows
        groups=contexts(scales,m["context_count"]);assert len(groups)==len(parts)
        ids=np.empty((rows,g),dtype=np.uint8)
        for rr,(f,k) in zip(groups,parts):
            v=rans_decode(payload[p:p+k],len(rr)*g,f);p+=k
            ids[rr]=v.reshape(len(rr),g)
        v=pack5(ids.reshape(-1));assert len(v)==pn;raw[po:po+pn]=v
    extras=[]
    for size in m["extra_sizes"]:extras.append(payload[p:p+size]);p+=size
    assert p==len(payload)
    side={}
    for name,i,refs,hsh in m["side_entries"]:
        v=extras[i]+b"".join(shuffle(extras[j],w,True) for j,w in refs)
        assert digest(v)==hsh;side[name]=v
    if head:side=merge_head(side,head)
    archive=CB59(bytes(raw))
    return archive.blob,side,legacy

def run(f01,f08,out):
    files=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(files[k])==v
    raw=raw_core(files[NAMES[0]]);side={k:files[k] for k in NAMES[1:]}
    extra=(f08/"legacy_extras.bundle").read_bytes()
    _,extras=extra_open(extra);assert digest(extras[BANK_NAME])==BANK_SHA
    from safetensors.torch import load
    variants={"EXACT":(side,b"")}
    for b in (8,4):
        s,h,_=split_head(side,b);variants[f"Q{b}"]=(s,h)
    out.mkdir(parents=True,exist_ok=False)
    arms={}
    for ctx in CONTEXTS:
        t=time.perf_counter();core=encode_core(raw,ctx);enc=time.perf_counter()-t
        results={}
        for mode,(s,h) in variants.items():
            p=pack(core,s,h,extra,ctx)
            a,b,c=restore(p)
            assert a==raw and c==extra and set(b)==set(side)
            if mode=="EXACT":assert b==side
            else:
                assert all(b[k]==side[k] for k in side if k!=NAMES[3])
                old=load(side[NAMES[3]]);new=load(b[NAMES[3]])
                assert set(old)==set(new)
                assert all(old[k].equal(new[k]) for k in old if not k.endswith(("language_head.up.weight","language_head.output_bias")))
            results[mode]={**metrics(len(p)),"sha256":digest(p),"core_byte_exact":True,
                           "legacy_byte_exact":True,"all_sidecars_byte_exact":mode=="EXACT"}
            (out/f"F24_C{ctx}_{mode}.bundle").write_bytes(p)
        arms[str(ctx)]={"entropy":core[3],"encode_core_seconds":enc,"variants":results}
        print("F24_ARM",ctx,json.dumps(arms[str(ctx)],sort_keys=True),flush=True)
    report={"stage":"F24_LOSSLESS_K32_ENTROPY","raw_CB59_sha256":digest(raw),
            "arms":arms,"depth":12,"optimizer_steps":0,"FP32_teacher_calls":0,
            "candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["All K32 core bytes and legacy extras exact; Q8/Q4 change only two native-head tensors, no claim of exact full native behavior for those variants.",
                           "Entropy statistics are empirical, not guaranteed lower bounds for all possible coding schemes.",
                           "No J7 replay, CPU throughput, or full tokenizer/runtime accounting. This is a storage codec, not a new model or runtime kernel."]}
    (out/"F24_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F24_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f08,a.out)
