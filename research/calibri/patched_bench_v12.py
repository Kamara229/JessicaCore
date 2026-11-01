"""CPU-only, pretrained expert test of patched QMoE-style word indices.

Matches EXACT SAME ternary symbols with CL8 / v11 dictionary stage.
DOES NOT claim end-to-end quantization quality, GPU speed or official
published QMoE checkpoint equality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import statistics
import time
import numpy as np
import torch
from huggingface_hub import HfApi,hf_hub_download
from dual_codec import StaticPairDictionary,encode_grouped,decode_grouped,LUT_BYTES
from patched_words_v12 import encode_patched,decode_patched,read_row,codeword_matvec
from index import encode as cl8_encode
from dual_real_bench import row_ternarize,reconstructed,MODEL,CKPT,PREFIX,TENSORS

def measured_row_reads(blob,read_fn,rows,seed=4217,n=80):
    rng=np.random.default_rng(seed)
    choices=rng.integers(0,rows,size=n)
    measured=[]
    for i in choices:
        tic=time.perf_counter_ns()
        read_fn(blob,int(i))
        measured.append((time.perf_counter_ns()-tic)/1e6)
    return {"cpu_python_random_row_p50_ms":float(np.percentile(measured,50)),
            "cpu_python_random_row_p95_ms":float(np.percentile(measured,95)),
            "reads":int(n)}

def test_one(d,q,scales,x,name):
    words,off=d.encode(q)
    if not np.array_equal(d.decode(words,off,q.shape[1]),q):
        raise AssertionError("Stage1 not lossless")
    levels_bytes=4*q.shape[0]
    cl8_bytes=len(cl8_encode(q))+levels_bytes
    qmoe_style_bytes=2*len(words)+4*len(off)+levels_bytes
    options=[]
    baselines=[]
    for grp in (4,16,64):
        blob,modes=encode_grouped(words,off,q.shape[1],group=grp,selector="adaptive")
        w2,off2,width=decode_grouped(blob)
        if not np.array_equal(w2,words) or not np.array_equal(off2,off):
            raise AssertionError('v11 roundtrip')
        baselines.append({"codec":"v11-dual","rows_per_group":grp,
                          "bytes_with_scales":len(blob)+levels_bytes,
                          "mode_counts":modes})
    for grp,block in ((1,128),(4,64),(4,128),(16,64),(16,128),(16,256),(64,128),(64,256)):
        tic=time.perf_counter()
        blob,hist=encode_patched(words,off,q.shape[1],group_rows=grp,block_words=block)
        enc_seconds=time.perf_counter()-tic
        recovered,offset,width=decode_patched(blob)
        if (not np.array_equal(recovered,words) or
            not np.array_equal(offset,off) or width!=q.shape[1]):
            raise AssertionError("Patched word code lossless roundtrip failure")
        if grp==16 and block==128:
            vals=measured_row_reads(blob,read_row,len(q))
            tic=time.perf_counter()
            y=codeword_matvec(blob,d,x,scales)
            direct_sec=time.perf_counter()-tic
            original=reconstructed(q,scales)@x
            rmse=float(np.linalg.norm(y-original)/max(np.linalg.norm(original),1e-20))
            maxerr=float(np.max(np.abs(y-original)))
            if rmse>1e-10 or maxerr>1e-10:
                raise AssertionError(f"Fused matvec mismatch: {rmse}, {maxerr}")
        else:
            vals=None;direct_sec=None;rmse=None;maxerr=None
        options.append({"rows_per_group":grp,
                        "words_per_subblock":block,
                        "bytes_with_scales":len(blob)+levels_bytes,
                        "bytes_bit_per_quantized_weight":8*(len(blob)+levels_bytes)/q.size,
                        "saving_percent_vs_original_dictionary":100*(1-(len(blob)+levels_bytes)/qmoe_style_bytes),
                        "saving_percent_vs_CL8":100*(1-(len(blob)+levels_bytes)/cl8_bytes),
                        "physical_word_width_histogram":hist,
                        "cpu_python_encode_sec":round(enc_seconds,3),
                        "random_access":vals,
                        "direct_matvec_cpu_python_sec":direct_sec,
                        "direct_matvec_rel_error":rmse,
                        "direct_matvec_max_absolute_error":maxerr})
    return {"tensor":name,"rows":q.shape[0],"cols":q.shape[1],
            "weights":int(q.size),"zero_fraction":float(np.mean(q==0)),
            "qmoe_style_first_stage_bytes":qmoe_style_bytes,
            "CL8_bytes_with_scales":cl8_bytes,
            "v11_baselines":baselines,"pfor_candidates":options,
            "best_pfor":min(options,key=lambda v:v["bytes_with_scales"])}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/calibri_v12_patched_pretrained_moe.json')
    a=p.parse_args()
    revision=HfApi().model_info(MODEL).sha
    path=hf_hub_download(repo_id=MODEL,filename=CKPT,revision=revision)
    dig=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):dig.update(buf)
    state=torch.load(path,map_location='cpu',weights_only=True)
    d=StaticPairDictionary()
    rng=np.random.default_rng(20261102)
    collected=[]
    for tensor in TENSORS:
        key=PREFIX+tensor
        q,scales=row_ternarize(state[key])
        x=rng.normal(size=q.shape[1]).astype(np.float64)
        collected.append(test_one(d,q,scales,x,key))
    baseline=sum(t['qmoe_style_first_stage_bytes'] for t in collected)
    cl8=sum(t['CL8_bytes_with_scales'] for t in collected)
    candidate=sum(t['best_pfor']['bytes_with_scales'] for t in collected)
    best_setting=[(t['best_pfor']['rows_per_group'],t['best_pfor']['words_per_subblock']) for t in collected]
    report={"experiment":"CALIBRI_v0.12_pretrained_switch_MoE_pfor_word_code",
            "model":MODEL,"model_revision":revision,
            "checkpoint_sha256":dig.hexdigest(),
            "weight_count":sum(t['weights'] for t in collected),
            "quantization":"simple per-row 88.5% forced-zero, NOT QMoE GPTQ",
            "symbols": "lossless identical ternary symbols across all physical codecs",
            "shared_static_dictionary_lut_bytes":LUT_BYTES,
            "qmoe_style_first_stage_pair_bytes_excluding_shared_lut":baseline,
            "CL8_pair_bytes":cl8,
            "best_patched_pair_bytes_excluding_shared_lut":candidate,
            "chosen_group_and_block_per_tensor":best_setting,
            "patched_vs_first_dictionary_percent":100*(1-candidate/baseline),
            "patched_vs_CL8_percent":100*(1-candidate/cl8),
            "patched_pair_bpw_excl_shared_lut":8*candidate/sum(t['weights'] for t in collected),
            "patched_pair_bpw_incl_shared_lut":8*(candidate+LUT_BYTES)/sum(t['weights'] for t in collected),
            "no_inference_quality_claim":True,
            "no_gpu_or_phone_latency_claim":True,
            "note":"PFOR is a published integer-compression approach; not a CALIBRI invention",
            "matrices":collected}
    out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({"baseline_dictionary_bytes":baseline,"CL8_bytes":cl8,
                      "best_patched_bytes":candidate,
                      "patched_vs_dictionary_saving_pct":report["patched_vs_first_dictionary_percent"],
                      "patched_vs_CL8_saving_pct":report["patched_vs_CL8_percent"],
                      "patched_pair_bpw_excl_lut":report["patched_pair_bpw_excl_shared_lut"],
                      "settings":best_setting},indent=2),flush=True)

if __name__=='__main__':main()
