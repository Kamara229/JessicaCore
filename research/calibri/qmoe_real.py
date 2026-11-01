"""CALIBRI v0.9: real published QMoE Switch-base-128 shard comparison.

Reads ONE expert tensor (optionally two) from OFFICIAL pretrained QMoE
checkpoint exp00.pt with torch.load(weights_only=True) and immutable
Hugging Face commit SHA. Counts per-matrix official indexes/scales and
CL7/CL8 encoded symbol/index bytes. QMoE's shared 512 KiB dictionary
is not charged per matrix (amortized over a complete checkpoint).
This experiment tests LOSSLESS REPACKING of the exact same 3-valued
trained quantized weights, NOT re-quantization or whole-model quality.
"""
from __future__ import annotations
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

from codec import encode as cl7_encode, decode as cl7_decode
from index import encode as cl8_encode, decode as cl8_decode
from benchmark import two_bit_zlib

MODEL="ISTA-DASLab/switch-base-128_qmoe"
FILENAME="exp00.pt"
ROWS_BY_NAME={"wi":3072,"wo":768}
COLS_BY_NAME={"wi":768,"wo":3072}

def dictionary():
    """Known official construction of the QMoE 65536-sequence dictionary."""
    p0=0.885
    ps=[p0,(1.-p0)/2.,(1.-p0)/2.]
    probabilities=[p*q for p in ps for q in ps]
    queue=[(-1.,())];out=[]
    while len(out)<65536:
        negp,seq=heapq.heappop(queue)
        if negp!=-1.:out.append(seq)
        if len(seq)==14:continue
        for j,p in enumerate(probabilities):
            heapq.heappush(queue,(negp*p,seq+(j,)))
    return [tuple(v for pair in seq for v in (pair%3,pair//3)) for seq in out]

def decode_qmoe(words,row_off,width,words_dict):
    words=np.asarray(words).view(np.uint16).ravel()
    offs=np.asarray(row_off,dtype=np.int64).ravel()
    if len(offs)<2 or offs[0]!=0 or offs[-1]!=len(words):
        raise ValueError("invalid QMoE offsets")
    if np.any(offs[1:]<offs[:-1]):raise ValueError("nonmonotone offsets")
    rows=len(offs)-1
    q=np.zeros((rows,width),dtype=np.int8)
    for i in range(rows):
        cursor=0
        for w in words[offs[i]:offs[i+1]]:
            for t in words_dict[int(w)]:
                if cursor==width:break
                q[i,cursor]=0 if t==0 else -1 if t==1 else 1
                cursor+=1
        if cursor!=width:raise ValueError(f"dictionary mismatch: row {i} length {cursor}, expected {width}")
    return q

def bench(path,max_matrices=1):
    try:
        d=torch.load(path,map_location="cpu",weights_only=True)
    except Exception as e:
        raise RuntimeError("Official artifact could not be safely loaded with weights_only=True") from e
    if not isinstance(d,dict):raise ValueError("expected tensor dictionary")
    candidates=sorted(k[:-7] for k in d if k.endswith(".w_comp"))
    if not candidates:raise ValueError("No compressed expert weights")
    candidates=[k for k in candidates if k.rsplit(".",1)[-1] in ROWS_BY_NAME]
    dd=dictionary();records=[]
    for name in candidates[:max_matrices]:
        part=name.rsplit(".",1)[-1]
        width=COLS_BY_NAME[part]
        wc=d[name+".w_comp"];offset=d[name+".row_off"];levels=d[name+".ter_minmax"]
        # Correct upstream QMoE requires these additional tables; no silent scale loss.
        q=decode_qmoe(wc.numpy(),offset.numpy(),width,dd)
        if levels.numel()!=2*q.shape[0]:raise ValueError("scale count mismatch")
        if q.shape[0]!=ROWS_BY_NAME[part]:raise ValueError(f"unexpected matrix shape {q.shape}")
        cb7=cl7_encode(q);cb8=cl8_encode(q)
        if not np.array_equal(q,cl7_decode(cb7)):raise AssertionError("CL7 symbols mismatch")
        if not np.array_equal(q,cl8_decode(cb8)):raise AssertionError("CL8 symbols mismatch")
        shared=levels.numel()*levels.element_size()
        qmoe_bytes=int(wc.numel()*wc.element_size()+offset.numel()*offset.element_size()+shared)
        data={"name":name,"rows":int(q.shape[0]),"cols":int(q.shape[1]),
              "weights":int(q.size),"actual_zero_fraction":float(np.mean(q==0)),
              "level_bytes":int(shared),
              "official_qmoe_bytes_excluding_shared_dictionary":qmoe_bytes,
              "cl7_bytes_including_scales":len(cb7)+shared,
              "cl8_bytes_including_scales":len(cb8)+shared,
              "two_bit_zlib_bytes_including_scales":two_bit_zlib(q)+shared,
              "cl7_bpw":8*(len(cb7)+shared)/q.size,
              "cl8_bpw":8*(len(cb8)+shared)/q.size,
              "official_bpw_excluding_shared_dictionary":8*qmoe_bytes/q.size,
              "cl8_saving_vs_official_bytes_percent":
                  100*(1-(len(cb8)+shared)/qmoe_bytes),
              "roundtrip_lossless":True}
        records.append(data)
    return records

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/calibri_v09_real_qmoe.json")
    p.add_argument("--max-matrices",type=int,default=1)
    a=p.parse_args()
    if not 1<=a.max_matrices<=2:raise ValueError("only 1-2 matrices in CI")
    revision=HfApi().model_info(MODEL).sha
    path=hf_hub_download(repo_id=MODEL,filename=FILENAME,revision=revision)
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    rows=bench(path,a.max_matrices)
    report={"model":MODEL,"revision":revision,"file":FILENAME,
            "file_sha256":h.hexdigest(),"tensor_count":len(rows),
            "comparison_scope":"official real QMoE expert ternary symbols; not whole model",
            "qmoe_shared_dictionary_bytes_excluded":65536*8,
            "caveats":["Original QMoE quantization quality preserved by identical decoded ternary symbols and row scales",
                       "No GPU latency measurement","No complete checkpoint bpp benchmark",
                       "Shared QMoE dictionary (~512KiB) amortized, not charged to one tensor"],
            "matrices":rows}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({"model":MODEL,"revision":revision,"file_sha256":h.hexdigest(),
                      "matrices":rows},indent=2),flush=True)

if __name__=="__main__":main()
