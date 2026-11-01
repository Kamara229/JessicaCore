"""Measured real-weight occupancy-mask Huffman vs CL8, QMoE-style and CL12-PFOR.

Uses the same trained Switch-base-8 paired expert Wi/Wo weights. Naive
88.5%-zero ternarization intentionally matches all other codecs; this
does NOT measure the original model's task quality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import time
import numpy as np
import torch
from huggingface_hub import HfApi,hf_hub_download
from dual_real_bench import MODEL,CKPT,PREFIX,TENSORS,row_ternarize,reconstructed
from index import encode as cl8_encode
from hybrid_v12 import encode as hybrid_encode,decode as hybrid_decode
from huffman_masks_v12 import encode as hf_encode,decode as hf_decode,read_row,direct_matvec,CODE,MAX_BITS

def measure_one(q,scales,name):
    result={"tensor":name,"rows":q.shape[0],"cols":q.shape[1],"weights":int(q.size)}
    scales_size=4*len(q)
    result['CL8_bytes_with_scales']=len(cl8_encode(q))+scales_size
    hybrid,decision=hybrid_encode(q,64)
    if not np.array_equal(q,hybrid_decode(hybrid)):
        raise AssertionError('hybrid failed real-symbol roundtrip')
    result['hybrid_auto_choice']=decision['chosen']
    result['hybrid_auto_bytes_with_scales']=len(hybrid)+scales_size
    runs=[]
    for g in (16,64):
        start=time.perf_counter()
        blob=hf_encode(q,g,index_mode='auto')
        elapsed=time.perf_counter()-start
        old_index=hf_encode(q,g,index_mode='u16')
        explicit_index=hf_encode(q,g,index_mode='u8_escape')
        if len(blob)!=min(len(old_index),len(explicit_index)):
            raise AssertionError('auto row index not optimal')
        start=time.perf_counter()
        q2=hf_decode(blob)
        decode_elapsed=time.perf_counter()-start
        if not np.array_equal(q2,q):
            raise AssertionError('mask Huffman did not roundtrip')
        rng=np.random.default_rng(2457)
        selected=rng.integers(0,len(q),size=80)
        lat=[]
        for i in selected:
            start=time.perf_counter_ns()
            _=read_row(blob,int(i))
            lat.append((time.perf_counter_ns()-start)/1e6)
        sub={'group_rows':g,'bytes_with_scales':len(blob)+scales_size,
             'physical_index':blob[:4].decode('ascii'),
             'u16_index_bytes':len(old_index)+scales_size,
             'u8_escaped_index_bytes':len(explicit_index)+scales_size,
             'bits_per_ternary_weight_with_scales':8*(len(blob)+scales_size)/q.size,
             'saving_percent_vs_CL8':100*(1-(len(blob)+scales_size)/result['CL8_bytes_with_scales']),
             'CPU_python_encode_seconds':elapsed,
             'CPU_python_full_decode_seconds':decode_elapsed,
             'CPU_python_random_row_p50_ms':float(np.percentile(lat,50)),
             'CPU_python_random_row_p95_ms':float(np.percentile(lat,95))}
        if g==16:
            x=rng.normal(size=q.shape[1]);start=time.perf_counter()
            got=direct_matvec(blob,x,scales)
            sub['direct_cpu_matvec_seconds']=time.perf_counter()-start
            expected=reconstructed(q,scales)@x
            sub['direct_cpu_matvec_relative_error']=float(np.linalg.norm(got-expected)/max(np.linalg.norm(expected),1e-20))
            sub['direct_cpu_matvec_max_absolute_error']=float(np.max(np.abs(got-expected)))
            if sub['direct_cpu_matvec_relative_error']>1e-10:
                raise AssertionError('Huffman direct matvec inaccurate')
        runs.append(sub)
    result['runs']=runs;return result

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',default='results/calibri_v12_huffman_real_moe.json')
    a=pa.parse_args()
    revision=HfApi().model_info(MODEL).sha
    path=hf_hub_download(repo_id=MODEL,filename=CKPT,revision=revision)
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    st=torch.load(path,map_location='cpu',weights_only=True)
    results=[]
    for t in TENSORS:
        q,scale=row_ternarize(st[PREFIX+t])
        results.append(measure_one(q,scale,PREFIX+t))
    cl8=sum(x['CL8_bytes_with_scales'] for x in results)
    r16=sum(next(y['bytes_with_scales'] for y in x['runs'] if y['group_rows']==16) for x in results)
    r64=sum(next(y['bytes_with_scales'] for y in x['runs'] if y['group_rows']==64) for x in results)
    weight_count=sum(x['weights'] for x in results)
    hybrid=sum(x['hybrid_auto_bytes_with_scales'] for x in results)
    report={"experiment":"CALIBRI_v12_exact_mask_huffman_on_trained_switch_expert",
            "model":MODEL,"model_revision":revision,
            "original_model_sha256":h.hexdigest(),
            "weights":weight_count,
            "shared_codebook":"fixed 94-symbol Huffman built deterministically from p_nonzero=0.115, no trained corpus sidecar",
            "max_huffman_mask_prefix_bits":MAX_BITS,
            "not_original_qmoe_quantizer":True,
            "not_full_model_quality":True,
            "no_native_gpu_kernel":True,
            "baseline_CL8_pair_bytes":cl8,
            "huffman_g16_pair_bytes":r16,
            "huffman_g64_pair_bytes":r64,
            "huffman_best_pair_bpw":8*min(r16,r64)/weight_count,
            "auto_hybrid_pair_bytes":hybrid,
            "auto_hybrid_selected":[x["hybrid_auto_choice"] for x in results],
            "auto_hybrid_improvement_vs_CL8_pct":100*(1-hybrid/cl8),
            "huffman_best_saving_vs_CL8_pct":100*(1-min(r16,r64)/cl8),
            "pair_details":results}
    out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in (
        "weights","baseline_CL8_pair_bytes","huffman_g16_pair_bytes",
        "huffman_g64_pair_bytes","huffman_best_pair_bpw",
        "auto_hybrid_pair_bytes","auto_hybrid_selected","auto_hybrid_improvement_vs_CL8_pct",
        "huffman_best_saving_vs_CL8_pct")},indent=2),flush=True)
if __name__=='__main__':main()
