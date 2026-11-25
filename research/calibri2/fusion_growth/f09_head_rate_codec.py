"""F09: reserve physical budget by compact encoding of the existing native head.

No optimization, no new model parameters, no protected evaluation. Quantization
is intentionally lossy; other learned tensors and the paid core are exact.
The package includes the F08 legacy extras, not a complete deployment runtime.
"""
import argparse
import json
import lzma
import struct
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load,save
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,digest,metrics
from research.calibri2.fusion_growth.f03_rate_train_core import make,encode,decode

PARENT_SHA="9e710de6db41d6bb2a99776080f045c537d4af2f6e971529ba6527bf993828d4"
RATE="K16_embed_K8_body"
GROUP=64
HDR=struct.Struct("<4sIII")
HEAD_SUFFIXES=("language_head.up.weight","language_head.output_bias")

def quantize(t,bits):
    assert t.dtype==torch.float16 and bits in (4,8)
    x=t.detach().cpu().numpy().astype(np.float32).reshape(-1)
    qmax=(1<<(bits-1))-1
    pad=(-len(x))%GROUP
    groups=np.pad(x,(0,pad)).reshape(-1,GROUP)
    scales=np.maximum(np.abs(groups).max(1)/qmax,2.**-24).astype("<f2")
    assert np.isfinite(scales).all() and np.all(scales>0)
    q=np.rint(groups/scales.astype(np.float32)[:,None]).clip(-qmax,qmax).astype(np.int8).reshape(-1)
    if bits==8:
        codes=q.tobytes()
    else:
        u=(q.astype(np.int16)+8).astype(np.uint8)
        codes=(u[::2]|(u[1::2]<<4)).tobytes()
    desc={"shape":list(t.shape),"count":t.numel(),"bits":bits,
          "group":GROUP,"scale_bytes":scales.nbytes,"code_bytes":len(codes)}
    return desc,scales.tobytes()+codes

def reconstruct(desc,raw):
    n=desc["count"];bits=desc["bits"];g=desc["group"];nb=desc["scale_bytes"]
    assert g==GROUP and bits in (4,8) and len(raw)==nb+desc["code_bytes"]
    s=np.frombuffer(raw[:nb],dtype="<f2").astype(np.float32)
    assert np.isfinite(s).all() and np.all(s>0)
    if bits==8:q=np.frombuffer(raw[nb:],dtype=np.int8)
    else:
        u=np.frombuffer(raw[nb:],dtype=np.uint8)
        q=np.stack([u&15,u>>4],axis=1).reshape(-1).astype(np.int16)-8
    assert len(q)==len(s)*g and n<=len(q)<n+g
    assert np.all(np.abs(q.astype(np.int16))<=(1<<(bits-1))-1)
    a=(q.reshape(-1,g).astype(np.float32)*s[:,None]).reshape(-1)[:n].astype("<f2")
    assert np.isfinite(a).all()
    return torch.from_numpy(a.reshape(desc["shape"]).copy())

def split_head(side,bits):
    data=load(side[NAMES[3]])
    chosen={k:v for k,v in data.items() if k.endswith(HEAD_SUFFIXES)}
    assert len(chosen)==2 and sorted(t.numel() for t in chosen.values())==[49152,196608]
    remainder={k:v for k,v in data.items() if k not in chosen}
    assert sum(t.numel() for t in data.values())==421793
    manifest=[];parts=[]
    for k,t in sorted(chosen.items()):
        d,p=quantize(t,bits)
        manifest.append({"name":k,**d});parts.append(p)
    h=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
    packed=lzma.compress(struct.pack("<I",len(h))+h+b"".join(parts),preset=6)
    new_side=dict(side);new_side[NAMES[3]]=save(remainder)
    return new_side,packed,chosen

def merge_head(side,packed):
    plain=lzma.decompress(packed)
    n,=struct.unpack("<I",plain[:4]);manifest=json.loads(plain[4:4+n])
    off=4+n;data=load(side[NAMES[3]])
    for d in manifest:
        k=d["name"];length=d["scale_bytes"]+d["code_bytes"]
        assert k not in data and k.endswith(HEAD_SUFFIXES)
        data[k]=reconstruct(d,plain[off:off+length]);off+=length
    assert off==len(plain) and sum(t.numel() for t in data.values())==421793
    result=dict(side);result[NAMES[3]]=save(data)
    return result

def open_bundle(packet):
    magic,a,b,c=HDR.unpack_from(packet)
    assert magic==b"FG09" and HDR.size+a+b+c==len(packet)
    i=HDR.size
    core,head,legacy=packet[i:i+a],packet[i+a:i+a+b],packet[i+a+b:]
    archive,side=decode(core)
    return archive,merge_head(side,head),legacy

def stats(a,b):
    x=a.float().flatten();y=b.float().flatten();d=y-x
    return {"elements":x.numel(),"max_abs":float(d.abs().max()),
            "RMSE":float(d.square().mean().sqrt()),
            "relative_L2":float(d.norm()/x.norm().clamp_min(1e-12)),
            "unchanged_elements":int((x==y).sum())}

def run(f03,f08,out):
    torch.set_num_threads(2)
    parent=(f03/"K16_embed_K8_body.bundle").read_bytes()
    assert digest(parent)==PARENT_SHA
    archive,side=decode(parent)
    legacy=(f08/"legacy_extras.bundle").read_bytes()
    previous=json.loads((f08/"F08_REPORT.json").read_text())
    assert previous["parent_sha256"]==PARENT_SHA and len(legacy)==previous["additional_packed_bytes"]==137712
    old_total=len(parent)+len(legacy)
    model,sources=make(archive)
    original=load(side[NAMES[3]])
    out.mkdir(parents=True,exist_ok=False)
    results={}
    for bits in (8,4):
        compact,head,chosen=split_head(side,bits)
        core=encode(archive.blob,sources,RATE,compact)
        packet=HDR.pack(b"FG09",len(core),len(head),len(legacy))+core+head+legacy
        restored,rs,restored_legacy=open_bundle(packet)
        assert restored.blob==archive.blob and restored_legacy==legacy
        assert all(rs[k]==side[k] for k in NAMES[1:] if k!=NAMES[3])
        after=load(rs[NAMES[3]])
        assert set(after)==set(original)
        for k in set(original)-set(chosen):
            assert torch.equal(after[k],original[k]),k
        for k in chosen:
            assert after[k].dtype==torch.float16 and torch.isfinite(after[k]).all()
        result={**metrics(len(packet)),"saving_vs_F08":old_total-len(packet),
                "quantized_head_payload_bytes":len(head),
                "quantized_elements":sum(t.numel() for t in chosen.values()),
                "other_child_elements_exact":sum(t.numel() for k,t in original.items() if k not in chosen),
                "tensor_errors":{k:stats(original[k],after[k]) for k in chosen},
                "core_byte_exact":True,"legacy_byte_exact":True,
                "physical_reload":True,"sha256":digest(packet)}
        (out/f"F09_Q{bits}.bundle").write_bytes(packet)
        results[f"Q{bits}"]=result
    report={"stage":"F09_NATIVE_HEAD_STORAGE_RESERVE","source_sha256":PARENT_SHA,
            "baseline_F08":metrics(old_total),"arms":results,"depth":12,
            "optimizer_steps":0,"FP32_teacher_calls":0,"candidate_accepted":False,
            "all_in_certified":False,"output_gate":0,
            "tokenizer_bytes_separate":previous["tokenizer_bytes_separate"],
            "limitations":["Lossy encoding of only native head up projection and output bias. No claim of preserved model function.",
                           "Original sidecars restored except two head tensors; 12-layer compressed core and legacy extras exact.",
                           "Physical package includes known F08 extras, but not tokenizer, full deployment code or native CPU integration.",
                           "No forward or training. Quality and protected J7/CPU gates pending. Use only as a storage candidate."]}
    (out/"F09_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F09_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("f03","f08","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.f03,a.f08,a.out)
