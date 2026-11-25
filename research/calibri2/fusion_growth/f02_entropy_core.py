"""F02: actual CB59 entropy layout + exact deduplication of inherited tensors.
Only lossless changes. No teacher, no optimizer, no change in model function.
Archive prototype, not deployed decoder or all-in plateau certification.
"""
import argparse
import hashlib
import json
import lzma
import struct
import tempfile
import time
import zlib
from pathlib import Path
import numpy as np
from research.calibri2.fusion_growth.f01_physical_budget import NAMES, PIN, unpack, metrics, digest

BASELINE=3904405
MAGIC=b"FG02"
HEADER=struct.Struct("<4sBII")
CB61=struct.Struct("<8sQI32s")

def raw_core(envelope):
    magic,n,crc,h=CB61.unpack_from(envelope)
    raw=zlib.decompress(envelope[CB61.size:])
    assert magic==b"CB61ZL01" and len(raw)==n and zlib.crc32(raw)==crc and hashlib.sha256(raw).digest()==h
    return raw

def split_core(raw):
    from direct_codebook_v59 import CB59
    model=CB59(raw)
    magic,version,mlen,blen=HEADER.unpack_from(raw)
    head=HEADER.size+mlen
    segments=[]
    for name,desc in model.meta["tensor_descriptors"].items():
        for part,(off,size) in desc["parts"].items():
            mode="code" if part=="packed" else "shuffle"
            width=4 if desc["dtype"]=="f4" else 2
            segments.append((head+off,size,mode,width))
    return sorted(segments)

def transform(raw,segments,mode,inverse=False):
    # Fixed-length bit-plane layout: only aligned groups of 8 code IDs.
    # Preserve every other byte including checksums, gaps and padding.
    buf=bytearray(raw)
    for off,n,kind,width in segments:
        x=raw[off:off+n]
        if kind=="code" and mode=="planes" and n%5==0:
            g=n//5
            if not inverse:
                a=np.frombuffer(x,np.uint8).reshape(g,5).astype(np.uint64)
                w=sum(a[:,i] << np.uint64(8*i) for i in range(5))
                codes=np.stack([(w >> np.uint64(5*i)) & 31 for i in range(8)],axis=1).astype(np.uint8)
                y=b"".join(np.packbits((codes.reshape(-1)>>i)&1,bitorder="little").tobytes() for i in range(5))
            else:
                bits=np.frombuffer(x,np.uint8).reshape(5,g)
                code=sum(np.unpackbits(bits[i],bitorder="little").astype(np.uint64)<<np.uint64(i) for i in range(5))
                code=code.reshape(g,8)
                w=sum(code[:,i]<<np.uint64(5*i) for i in range(8))
                y=np.stack([(w>>np.uint64(8*i))&255 for i in range(5)],axis=1).astype(np.uint8).tobytes()
            assert len(y)==n
            buf[off:off+n]=y
        elif kind=="shuffle" and width>1 and n%width==0:
            a=np.frombuffer(x,np.uint8)
            y=(a.reshape(width,-1).T if inverse else a.reshape(-1,width).T).copy().tobytes()
            buf[off:off+n]=y
    return bytes(buf)

def sf_parts(raw):
    n,=struct.unpack("<Q",raw[:8])
    h=8+n
    meta=json.loads(raw[8:h])
    parts=[]
    for k,v in meta.items():
        if k=="__metadata__":continue
        a,b=v["data_offsets"]
        width={"F64":8,"F32":4,"I32":4,"F16":2,"BF16":2,"I16":2,"U16":2,"I64":8,"U64":8}.get(v["dtype"],1)
        parts.append((a,b,k,width))
    parts.sort()
    assert all(a==(parts[i-1][1] if i else 0) for i,(a,b,k,w) in enumerate(parts))
    assert (parts[-1][1] if parts else 0)==len(raw)-h
    return raw[:h],[(k,w,raw[h+a:h+b]) for a,b,k,w in parts]

def shuffle(raw,w,inverse=False):
    a=np.frombuffer(raw,np.uint8)
    assert len(raw)%w==0
    if not len(raw):return raw
    return (a.reshape(w,-1).T if inverse else a.reshape(-1,w).T).copy().tobytes()

def encode(files,mode):
    raw=raw_core(files[NAMES[0]])
    seg=split_core(raw)
    core=transform(raw,seg,mode)
    # Keep original safetensors headers, deduplicate actual tensor bytes
    # across the four files (including the inherited S93 tensors).
    blobs=[core]; seen={}; entries=[]
    for name in NAMES[1:]:
        header,parts=sf_parts(files[name])
        idx=len(blobs);blobs.append(header)
        refs=[]
        for key,w,value in parts:
            keyhash=(digest(value),w)
            if keyhash not in seen:
                seen[keyhash]=len(blobs)
                blobs.append(shuffle(value,w))
            refs.append([seen[keyhash],w])
        entries.append([name,idx,refs,digest(files[name])])
    # Charge all offsets, lengths, transforms, hashes and codec framing.
    meta={"mode":mode,"segments":seg,"core_sha":digest(raw),"entries":entries,
          "sizes":[len(b) for b in blobs]}
    h=json.dumps(meta,separators=(",",":")).encode()
    packet=struct.pack("<I",len(h))+h+b"".join(blobs)
    return MAGIC+lzma.compress(packet,preset=6),len(seen)

def decode(packet):
    assert packet[:4]==MAGIC
    p=lzma.decompress(packet[4:])
    n,=struct.unpack("<I",p[:4]);m=json.loads(p[4:4+n])
    cursor=4+n;blobs=[]
    for size in m["sizes"]:
        blobs.append(p[cursor:cursor+size]);cursor+=size
    assert cursor==len(p)
    raw=transform(blobs[0],m["segments"],m["mode"],True)
    assert digest(raw)==m["core_sha"]
    files={}
    for name,i,refs,sha in m["entries"]:
        f=blobs[i]+b"".join(shuffle(blobs[j],w,True) for j,w in refs)
        assert digest(f)==sha
        files[name]=f
    return raw,files

def run(root,out):
    old=(root/"F01.xz.bundle").read_bytes()
    assert len(old)==BASELINE
    prior=json.loads((root/"F01_REPORT.json").read_text())
    files=unpack(old)
    assert set(files)==set(NAMES)
    for k,v in PIN.items():assert digest(files[k])==v
    for row in prior["inventory"]:assert digest(files[row["name"]])==row["sha256"]
    raw=raw_core(files[NAMES[0]])
    out.mkdir(parents=True,exist_ok=False)
    results={}
    for mode in ("plain","planes"):
        t=time.perf_counter()
        blob,unique=encode(files,mode)
        enc=time.perf_counter()-t;t=time.perf_counter()
        recovered,side=decode(blob)
        dec=time.perf_counter()-t
        assert recovered==raw and side=={k:files[k] for k in NAMES[1:]}
        (out/f"F02.{mode}.bundle").write_bytes(blob)
        results[mode]={**metrics(len(blob)),"saving_vs_F01":BASELINE-len(blob),
                       "encode_seconds":enc,"decode_seconds":dec,"unique_side_payloads":unique,
                       "raw_CB59_and_all_four_sidecars_byte_exact":True}
    best=min(results,key=lambda m:results[m]["bytes"])
    # Verify the actual source decoder can reopen the reconstructed CB59;
    # the historical outer zlib envelope is not a model weight and can be rebuilt.
    from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import load_actual_original
    import torch
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);restored,side=decode((out/f"F02.{best}.bundle").read_bytes())
        new=CB61.pack(b"CB61ZL01",len(restored),zlib.crc32(restored),hashlib.sha256(restored).digest())+zlib.compress(restored,9)
        (d/NAMES[0]).write_bytes(new)
        for k,v in side.items():(d/k).write_bytes(v)
        model,*_=load_actual_original(*(d/n for n in NAMES[:3]))
        assert len(model.model.layers)==12
    report={"stage":"F02_ENTROPY_CORE_DEDUP","prior_F01_bytes":BASELINE,
            "raw_CB59_bytes":len(raw),"raw_CB59_sha256":digest(raw),
            "arms":results,"best":best,"depth":12,"core_loader_reopened":True,
            "all_in_certified":False,"candidate_accepted":False,"output_gate":0,
            "optimizer_steps":0,"FP32_teacher_calls":0,
            "limitations":["Known five source files only; raw CB59 and all sidecars exact, historical zlib envelope rebuilt.",
                           "Archive prototype needs native loader integration and runtime inventory; no full child attachment, J7, CPU throughput or quality measurement.",
                           "If cap not met, next step must change learned representation rather than count inactive parameters."]}
    (out/"F02_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F02_ASSESS",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
