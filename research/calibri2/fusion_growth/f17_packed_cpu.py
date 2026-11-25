"""F17 real F10 packed CPU operator benchmark. No full-model speed claim."""
import argparse
import ctypes as C
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from research.calibri2.fusion_growth.f01_physical_budget import digest
from research.calibri2.fusion_growth.f03_rate_train_core import decode,k_for,pack_indices
from direct_codebook_v59 import unpack5

PARENT="694d031e340825c4d9667f99a674f95b69c4ade29c962e178d01a465bad49018"
RATE="K16_embed_K8_body"
FLOAT=np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
U8=np.ctypeslib.ndpointer(dtype=np.uint8,flags="C_CONTIGUOUS")
def build_lib(tmp):
    src=Path(__file__).with_suffix(".c");so=Path(tmp)/"f17.so"
    subprocess.run(["cc","-O3","-march=native","-fopenmp","-fPIC","-shared","-ffp-contract=off",str(src),"-o",str(so)],check=True)
    lib=C.CDLL(str(so));f=lib.fg17_gemv
    f.argtypes=[U8,C.c_size_t,FLOAT,FLOAT,FLOAT,FLOAT,C.c_size_t,C.c_size_t,C.c_int]
    f.restype=C.c_int
    return lib,f
def data(h,k):
    m,n=h.shape
    ids=unpack5(h.packed.tobytes(),m*n//8)
    assert np.all(ids<k)
    p=np.frombuffer(pack_indices(ids,k.bit_length()-1),np.uint8)
    centers=np.ascontiguousarray(h.centroids[:k],dtype=np.float32)
    scales=np.ascontiguousarray(h.row_scales,dtype=np.float32)
    return p,centers,scales,ids
def operator(f,h,k,x):
    p,c,s,ids=data(h,k);m,n=h.shape
    out=np.empty(m,np.float32)
    def packed():
        rc=f(p,len(p),c,s,x,out,m,n,k.bit_length()-1)
        if rc:raise RuntimeError(f"kernel error {rc}")
        return out
    def materialize():
        return torch.from_numpy(c[ids].reshape(m,n)*s[:,None])
    xt=torch.from_numpy(x);dense=materialize()
    ref=torch.mv(dense,xt).numpy()
    actual=packed().copy()
    err=np.abs(actual-ref)
    assert np.isfinite(actual).all()
    # Relative to L2 of the entire output; near-zero individual outputs
    # do not get an artificially large relative error.
    rel=float(np.linalg.norm(err)/max(1e-12,np.linalg.norm(ref)))
    assert rel<2e-5,(h.name,rel)
    return packed,lambda:torch.mv(dense,xt),lambda:torch.mv(materialize(),xt),{
        "shape":[m,n],"bits_per_group":k.bit_length()-1,
        "max_abs_error":float(err.max()),"relative_L2_error":rel,
        "packed_codes_bytes":len(p),"physical_FP16_centers_scales_bytes":2*(c.size+s.size),
        "dense_FP32_weight_bytes":dense.numel()*4,
        "native_workspace_bytes":(n//8)*k*4+m*4,
        "native_resident_fields_bytes":len(p)+c.nbytes+s.nbytes}
def timing(funcs):
    # Same runner, thread count, repeated alternating order. Reconstruction
    # intentionally included only in the uncached baseline.
    samples={k:[] for k in funcs};keys=list(funcs)
    for f in funcs.values():
        for _ in range(3):f()
    for i in range(21):
        for k in (keys if i%2==0 else keys[::-1]):
            t=time.perf_counter_ns();funcs[k]();samples[k].append((time.perf_counter_ns()-t)/1e6)
    return {k:statistics.median(v) for k,v in samples.items()}
def run(f10,out):
    torch.set_num_threads(2);torch.manual_seed(20261205)
    p=(f10/"F10_FP16.bundle").read_bytes();assert digest(p)==PARENT
    archive,_=decode(p);rng=np.random.default_rng(20261205)
    handles={k:h for k,h in archive.handles.items() if k in archive.meta["tensor_descriptors"]}
    assert len(handles)==85
    out.mkdir(parents=True,exist_ok=False)
    results={};shapes=set()
    with tempfile.TemporaryDirectory() as tmp:
        lib,f=build_lib(tmp)
        for name,h in sorted(handles.items()):
            k=k_for(RATE,name)
            x=np.ascontiguousarray(rng.standard_normal(h.shape[1]),dtype=np.float32)
            a,b,c,report=operator(f,h,k,x)
            sig=(tuple(h.shape),k)
            if sig not in shapes:
                shapes.add(sig);t=timing({"packed":a,"cached_dense":b,"reconstructed_dense":c})
                report["median_ms"]=t
                report["ratio_vs_cached_dense"]=t["cached_dense"]/t["packed"]
                report["ratio_vs_reconstructed_dense"]=t["reconstructed_dense"]/t["packed"]
            results[name]=report
            del a,b,c
    report={"stage":"F17_PACKED_CPU_OPERATOR","source_sha256":PARENT,"operators":results,
            "checked":len(results),"timed_shape_classes":len(shapes),
            "max_relative_L2_error":max(v["relative_L2_error"] for v in results.values()),
            "packed_code_bytes":sum(v["packed_codes_bytes"] for v in results.values()),
            "native_resident_fields_bytes":sum(v["native_resident_fields_bytes"] for v in results.values()),
            "dense_FP32_weight_bytes":sum(v["dense_FP32_weight_bytes"] for v in results.values()),
            "c_source_bytes":Path(__file__).with_suffix(".c").stat().st_size,
            "candidate_accepted":False,"CPU_speed_certified":False,"output_gate":0,
            "limitations":["Actual trained F10 weight tensors; synthetic deterministic input vector per unique operator, not actual model hidden states.",
                           "Direct packed single-vector GEMV, not batched prefill or end-to-end CPU generation. Numerical tolerance not bitwise/token equivalence.",
                           "Dense baselines are PyTorch cached and reconstruct-per-call, not historical optimized S194 native runtime.",
                           "No training or J7; full model runtime integration and speed gate not certified."]}
    (out/"F17_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F17_ASSESS",json.dumps({**report,"operators":{k:v for k,v in results.items() if "median_ms" in v}},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--f10",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.f10,a.out)
