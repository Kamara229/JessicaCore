"""F19: compare packed row kernels on real native activation vectors.
No training, no protected prompts, no end-to-end speed claim.
"""
import argparse
import ctypes as C
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch
from research.calibri2.fusion_growth.f01_physical_budget import digest
from research.calibri2.fusion_growth.f03_rate_train_core import decode,k_for
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f06_integrated_recovery import forward
from research.calibri2.fusion_growth.f17_packed_cpu import PARENT,RATE,build_lib,data,timing,FLOAT,U8
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import strict_chat_source_ids
from v176.operators import TrainLinear

PROBES=("Запиши слово «парус».","Если ключ лежит в коробке, где ключ?")
def build_v2(tmp):
    so=Path(tmp)/"fg19.so"
    subprocess.run(["cc","-O3","-march=native","-fopenmp","-fPIC","-shared","-ffp-contract=off",
                    str(Path(__file__).with_suffix(".c")),"-o",str(so)],check=True)
    lib=C.CDLL(str(so));f=lib.fg19_gemv
    f.argtypes=[U8,C.c_size_t,FLOAT,FLOAT,FLOAT,FLOAT,C.c_size_t,C.c_size_t,C.c_int]
    f.restype=C.c_int
    return lib,f

def capture(archive,side,tok):
    state=direct(archive,side)
    model,sources=state[:2];reverse={id(s):k for k,s in sources.items()}
    seen={};hooks=[]
    def hook(m,args):
        k=reverse[id(m.source)]
        x=args[0].detach().reshape(-1,args[0].shape[-1])[-1].cpu().numpy().astype(np.float32).copy()
        assert np.isfinite(x).all()
        seen.setdefault(k,[]).append(x)
    for m in model.modules():
        if isinstance(m,TrainLinear):hooks.append(m.register_forward_pre_hook(hook))
    assert len(hooks)==85
    try:
        for q in PROBES:
            z=forward(state,strict_chat_source_ids(tok,q),prompt_features(tok,q)).detach()
            assert z.shape==(1,49152) and torch.isfinite(z).all()
    finally:
        for h in hooks:h.remove()
    assert set(seen)==set(sources) and all(len(v)==len(PROBES) for v in seen.values())
    return seen

def check_operator(f1,f2,h,k,inputs,benchmark):
    p,c,s,ids=data(h,k);m,n=h.shape
    dense=torch.from_numpy(c[ids].reshape(m,n)*s[:,None])
    results=[]
    for x in inputs:
        x=np.ascontiguousarray(x,dtype=np.float32)
        y1=np.empty(m,np.float32);y2=np.empty(m,np.float32);xt=torch.from_numpy(x)
        def call(f,y):
            rc=f(p,len(p),c,s,x,y,m,n,k.bit_length()-1)
            if rc:raise RuntimeError(f"native kernel error {rc}")
            return y
        def cached():return torch.mv(dense,xt)
        ref=cached().numpy()
        old=call(f1,y1).copy();new=call(f2,y2).copy()
        def rel(y):
            assert np.isfinite(y).all()
            return float(np.linalg.norm(y-ref)/max(1e-12,np.linalg.norm(ref)))
        r={"f17_relative_L2":rel(old),"f19_relative_L2":rel(new),
           "f19_max_abs":float(np.abs(new-ref).max()),
           "same_top1":int(new.argmax())==int(ref.argmax())}
        assert max(r["f17_relative_L2"],r["f19_relative_L2"])<2e-5
        if benchmark:
            t=timing({"F17":lambda:call(f1,y1),"F19":lambda:call(f2,y2),"cached_dense":cached})
            r.update(median_ms=t,F19_vs_F17=t["F17"]/t["F19"],
                     F19_vs_cached_dense=t["cached_dense"]/t["F19"])
        results.append(r)
    return {"shape":[m,n],"bits":k.bit_length()-1,"samples":results,
            "packed_resident_fields":len(p)+c.nbytes+s.nbytes,
            "dense_bytes":dense.numel()*4}

def run(f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(20261207)
    p=(f10/"F10_FP16.bundle").read_bytes();assert digest(p)==PARENT
    archive,side=decode(p)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    vectors=capture(archive,side,tok)
    out.mkdir(parents=True,exist_ok=False)
    results={};shapes=set()
    with tempfile.TemporaryDirectory() as tmp:
        lib1,f1=build_lib(tmp);lib2,f2=build_v2(tmp)
        for name,inputs in sorted(vectors.items()):
            h=archive.handles[name];k=k_for(RATE,name);sig=(h.shape,k)
            results[name]=check_operator(f1,f2,h,k,inputs,sig not in shapes);shapes.add(sig)
    samples=[s for v in results.values() for s in v["samples"]]
    report={"stage":"F19_REAL_ACTIVATION_CPU_OPERATOR","parent_sha256":PARENT,
            "operators":results,"unique_operators":len(results),"real_prompt_forwards":len(PROBES),
            "activation_vectors":len(samples),"timed_shapes":len(shapes),
            "max_relative_L2":max(s["f19_relative_L2"] for s in samples),
            "top1_agreement":sum(s["same_top1"] for s in samples),
            "candidate_accepted":False,"CPU_speed_certified":False,"output_gate":0,
            "limitations":["Real F10 native hidden vectors, two authored prompts; no protected J7.",
                           "Single-vector kernels only, no full-system inference substitution or end-to-end speed result.",
                           "F19 uses different accumulation order, numerical parity not bitwise identity.",
                           "Dense comparison uses cached FP32 PyTorch weight, not the historical CPU reference. No additional model weights."]}
    (out/"F19_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F19_ASSESS",json.dumps({**report,"operators":{k:v for k,v in results.items() if "median_ms" in v["samples"][0]}},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f10,a.f08,a.out)
