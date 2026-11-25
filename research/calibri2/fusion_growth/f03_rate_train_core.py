"""F03: rate-aware retraining of the actual paid CB59 core.

Three matched 5/4/mixed-3-bit codebook arms; 12 layers, tied head unchanged.
Physical compact packet includes all four inherited sidecars, which are not
attached in this core-only experiment. No FP32 reference, no J7 or promotion.
"""
import argparse
import hashlib
import json
import lzma
import struct
import zlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from research.calibri2.fusion_growth.f01_physical_budget import NAMES, PIN, unpack, metrics, digest
from research.calibri2.fusion_growth.f02_entropy_core import raw_core, sf_parts, shuffle
from direct_codebook_v59 import CB59, HEADER, pack5, unpack5
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import MODEL_SPEC
from v176.operators import clear

ARMS=("K32_control","K16_all","K16_embed_K8_body")
STEPS=32
SEED=20261125

def data():
    # Explicitly authored, deterministic bilingual core recovery curriculum.
    # Disjoint operands and lexical templates in held-out diagnostic.
    train=[]
    for i in range(16):
        a,b=12+3*i,5+i
        train.extend([(f"Calculate {a} + {b}.",str(a+b)),
                      (f"Вычисли {a} - {b}.",str(a-b))])
    nouns=[("water","liquid"),("ice","solid"),("steam","gas"),("sparrow","bird"),
           ("oak","tree"),("salmon","fish"),("rose","flower"),("copper","metal")]
    for a,b in nouns:
        train.extend([(f"Complete: {a} is a",b),(f"Copy the word {a}.",a),
                      (f"Write the word {b}.",b),(f"Repeat exactly: {a} {b}.",f"{a} {b}")])
    dev=[]
    for i in range(8):
        a,b=71+4*i,17+i
        dev.extend([(f"What is {a} plus {b}?",str(a+b)),
                    (f"Сколько будет {a} минус {b}?",str(a-b))])
    dev.extend([("Repeat the word: river.","river"),("Copy exactly: blue stone.","blue stone"),
                ("Complete: a tulip is a","flower"),("Complete: iron is a","metal"),
                ("Напиши слово лес.","лес"),("Повтори слово дом.","дом"),
                ("Repeat exactly: snow falls.","snow falls"),("Write the word cloud.","cloud")])
    assert len(train)==64 and len(dev)==24
    assert not set(p for p,a in train)&set(p for p,a in dev)
    return train,dev

def k_for(arm,name):
    if arm=="K32_control":return 32
    if arm=="K16_all" or name=="model.embed_tokens.weight":return 16
    return 8

def reduced(centers,ids,k):
    if k==32:return centers.copy(),ids.copy()
    # Weighted codebook fitting on the source's actual code occupancy.
    counts=np.bincount(ids,minlength=32).astype(np.float64)
    x=centers.astype(np.float64)
    chosen=[int(counts.argmax())]
    for _ in range(k-1):
        distance=((x[:,None]-x[chosen][None])**2).sum(-1).min(1)
        score=distance*counts
        score[chosen]=-1
        chosen.append(int(score.argmax()))
    c=x[chosen].copy()
    for _ in range(20):
        assign=((x[:,None]-c[None])**2).sum(-1).argmin(1)
        for j in range(k):
            w=counts*(assign==j)
            if w.sum():c[j]=(x*w[:,None]).sum(0)/w.sum()
    c=c.astype(np.float16).astype(np.float32)
    assign=((x[:,None]-c[None])**2).sum(-1).argmin(1)
    full=np.zeros_like(centers);full[:k]=c
    return full,assign[ids].astype(np.int64)

def pack_indices(ids,bits):
    ids=np.asarray(ids,dtype=np.uint8)
    assert np.all(ids < (1<<bits))
    stream=((ids[:,None]>>np.arange(bits,dtype=np.uint8))&1).reshape(-1)
    return np.packbits(stream,bitorder="little").tobytes()

def unpack_indices(raw,n,bits):
    x=np.unpackbits(np.frombuffer(raw,np.uint8),bitorder="little")[:n*bits].reshape(n,bits)
    return (x.astype(np.uint16)*(1<<np.arange(bits,dtype=np.uint16))).sum(1).astype(np.uint8)

def encode(raw,sources,arm,side):
    archive=CB59(raw)
    _,_,mlen,_=HEADER.unpack_from(raw);start=HEADER.size+mlen
    skeleton=bytearray(raw);skeleton[-36:]=bytes(36)
    fields=[]
    blobs=[]
    for name,src in sorted(sources.items()):
        d=archive.meta["tensor_descriptors"][name]
        k=k_for(arm,name);bits=k.bit_length()-1
        ids=src.codes.detach().cpu().numpy().astype(np.uint8)
        c=src.centroids.detach().cpu().numpy().astype("<f2")
        assert np.all(ids<k) and np.isfinite(c).all()
        if k<32:assert not np.any(c[k:])
        po,pn=d["parts"]["packed"];co,cn=d["parts"]["centroids"]
        po+=start;co+=start
        skeleton[po:po+pn]=bytes(pn);skeleton[co:co+cn]=bytes(cn)
        p=pack_indices(ids,bits);v=c[:k].tobytes()
        fields.append([po,pn,co,cn,len(ids),bits,len(p),len(v)])
        blobs.extend([p,v])
    # All inherited bytes accounted; dedup exact payload across safetensors.
    entries=[];unique={};extras=[]
    for name in NAMES[1:]:
        header,parts=sf_parts(side[name]);hidx=len(extras);extras.append(header)
        refs=[]
        for _,w,value in parts:
            key=(digest(value),w)
            if key not in unique:unique[key]=len(extras);extras.append(shuffle(value,w))
            refs.append([unique[key],w])
        entries.append([name,hidx,refs,digest(side[name])])
    meta={"arm":arm,"skeleton_len":len(skeleton),"fields":fields,
          "entries":entries,"extra_sizes":[len(x) for x in extras]}
    h=json.dumps(meta,separators=(",",":")).encode()
    return b"FG03"+lzma.compress(struct.pack("<I",len(h))+h+bytes(skeleton)+b"".join(blobs)+b"".join(extras),preset=6)

def decode(packet):
    assert packet[:4]==b"FG03"
    raw=lzma.decompress(packet[4:]);n,=struct.unpack("<I",raw[:4]);m=json.loads(raw[4:4+n])
    p=4+n;s=bytearray(raw[p:p+m["skeleton_len"]]);p+=len(s)
    for po,pn,co,cn,count,bits,plen,vlen in m["fields"]:
        ids=unpack_indices(raw[p:p+plen],count,bits);p+=plen
        code=pack5(ids);assert len(code)==pn;s[po:po+pn]=code
        v=raw[p:p+vlen];p+=vlen
        assert vlen<=cn;s[co:co+cn]=v+bytes(cn-vlen)
    s[-36:-32]=struct.pack("<I",zlib.crc32(s[:-36]));s[-32:]=hashlib.sha256(s[:-32]).digest()
    rebuilt=CB59(bytes(s))
    extras=[]
    for size in m["extra_sizes"]:extras.append(raw[p:p+size]);p+=size
    assert p==len(raw)
    side={}
    for name,i,refs,h in m["entries"]:
        value=extras[i]+b"".join(shuffle(extras[j],w,True) for j,w in refs)
        assert digest(value)==h;side[name]=value
    return rebuilt,side

def make(archive):
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    assert len(model.model.layers)==12 and len(sources)==85
    model.eval()
    return model,sources

def examples(tok,rows):
    out=[]
    for p,a in rows:
        prefix=tok.encode(p+"\nAnswer:",add_special_tokens=False)
        answer=tok.encode(" "+a,add_special_tokens=False)+[tok.eos_token_id]
        assert prefix and answer and len(prefix)+len(answer)<64
        out.append((prefix,answer,a))
    return out

def loss(model,sources,row):
    prefix,answer,_=row
    ids=torch.tensor([prefix+answer],dtype=torch.long)
    clear(sources)
    logits=model(input_ids=ids,use_cache=False).logits[0,len(prefix)-1:-1]
    return F.cross_entropy(logits.float(),torch.tensor(answer))

@torch.no_grad()
def evaluate(model,sources,rows,tok):
    losses=[];correct=0;finished=0
    for row in rows:
        losses.append(float(loss(model,sources,row)))
        prefix,answer,a=row
        seq=prefix.copy();gen=[]
        clear(sources)
        for _ in range(12):
            logits=model(input_ids=torch.tensor([seq]),use_cache=False).logits[0,-1]
            token=int(logits.argmax());gen.append(token);seq.append(token)
            if token==tok.eos_token_id:break
        finished+=int(gen[-1]==tok.eos_token_id)
        correct+=int(gen[-1]==tok.eos_token_id and tok.decode(gen,skip_special_tokens=True).strip()==a)
        clear(sources)
    return {"n":len(rows),"answer_NLL":sum(losses)/len(losses),"greedy_exact_with_EOS":correct,"EOS":finished}

def run(root,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    files=unpack((root/"F01.xz.bundle").read_bytes())
    assert set(files)==set(NAMES)
    for k,v in PIN.items():assert digest(files[k])==v
    raw=raw_core(files[NAMES[0]]);archive=CB59(raw)
    side={k:files[k] for k in NAMES[1:]}
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    tr,dv=data();train=examples(tok,tr);dev=examples(tok,dv)
    out.mkdir(parents=True,exist_ok=False)
    model,sources=make(archive)
    baseline=evaluate(model,sources,dev,tok)
    del model,sources
    print("F03_BASELINE",json.dumps(baseline),flush=True)
    order=np.random.default_rng(SEED).permutation(len(train))
    results={}
    for arm in ARMS:
        model,sources=make(archive)
        with torch.no_grad():
            for name,s in sources.items():
                k=k_for(arm,name)
                centers,ids=reduced(s.centroids.detach().numpy(),s.codes.numpy(),k)
                s.centroids.copy_(torch.from_numpy(centers));s.codes.copy_(torch.from_numpy(ids))
        clear(sources)
        pre=evaluate(model,sources,dev,tok)
        opt=torch.optim.Adam([s.centroids for s in sources.values()],lr=0.0008)
        gradients=set();losses=[]
        for i in range(STEPS):
            opt.zero_grad(set_to_none=True)
            for j in range(2):
                l=loss(model,sources,train[int(order[2*i+j])])
                assert torch.isfinite(l)
                (l/2).backward()
                losses.append(float(l.detach()))
                clear(sources)
            for name,s in sources.items():
                g=s.centroids.grad
                if g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum())>0:gradients.add(name)
                if g is not None:assert bool(torch.isfinite(g).all())
            torch.nn.utils.clip_grad_norm_([s.centroids for s in sources.values()],1.)
            opt.step()
            with torch.no_grad():
                for name,s in sources.items():s.centroids[k_for(arm,name):].zero_()
        assert len(gradients)==85
        clear(sources)
        packet=encode(raw,sources,arm,side)
        (out/(arm+".bundle")).write_bytes(packet)
        restored,restored_side=decode(packet)
        assert restored_side==side
        for name,s in sources.items():
            h=restored.handles[name]
            assert np.array_equal(unpack5(h.packed.tobytes(),s.codes.numel()),s.codes.numpy())
            assert np.array_equal(h.centroids,s.centroids.detach().numpy().astype(np.float16).astype(np.float32))
            assert np.array_equal(h.row_scales,archive.handles[name].row_scales)
        del opt,model,sources
        model,sources=make(restored)
        post=evaluate(model,sources,dev,tok)
        del model,sources
        results[arm]={"before_train":pre,"after_physical_reload":post,
                      **metrics(len(packet)),"sha256":digest(packet),
                      "optimizer_steps":STEPS,"train_rows":len(train),
                      "gradient_connected_codebooks":len(gradients),
                      "train_first8_NLL":sum(losses[:8])/8,"train_last8_NLL":sum(losses[-8:])/8,
                      "sidecars_byte_exact":True}
        print("F03_ARM",arm,json.dumps(results[arm],sort_keys=True),flush=True)
    report={"stage":"F03_RATE_AWARE_CORE_TRAIN","baseline":baseline,"arms":results,
            "depth":12,"core_only":True,"sidecars_charged_not_active":True,
            "source_train_dev_sha256":digest(json.dumps([tr,dv],ensure_ascii=False).encode()),
            "FP32_teacher_calls":0,"output_gate":0,"candidate_accepted":False,"all_in_certified":False,
            "limitations":["Bare actual paid 36.38M CB59 core; GT/NRA3/NPA1/head sidecars charged but not attached.",
                           "24 internal authored diagnostic rows, no independent language claim or protected J7 replay.",
                           "Prototype compact packet restores native K32 container; no optimized runtime/CPU claim."]}
    (out/"F03_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F03_ASSESS",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
