"""Real (non-QMoE) pretrained Switch MoE expert weight benchmark.

Google Switch-base-8 checkpoint is 1.24GB BF16 and different from QMoE.
We deliberately inspect exactly ONE expert tensor to keep CI memory bounded.
This DOES NOT evaluate quantization quality or real compressed inference.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from codec import encode as cl7
from index import encode as cl8,decode as cl8decode
from benchmark import two_bit_zlib

MODEL="google/switch-base-8";FILE="pytorch_model.bin"

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/calibri_v09_google_switch_expert.json")
    a=p.parse_args()
    revision=HfApi().model_info(MODEL).sha
    path=hf_hub_download(repo_id=MODEL,filename=FILE,revision=revision)
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    data=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(data,dict):raise ValueError("unexpected checkpoint")
    candidates=[(k,v) for k,v in data.items() if isinstance(v,torch.Tensor)
                and v.ndim==2 and "experts." in k and k.endswith("wi.weight")]
    if not candidates:raise ValueError(f"no expert tensor found among {len(data)} keys")
    name,w=sorted(candidates,key=lambda t:t[0])[0]
    wf=w.float().numpy()
    threshold=np.quantile(np.abs(wf),0.885,axis=1,keepdims=True)
    q=np.where(np.abs(wf)>threshold,np.sign(wf),0).astype(np.int8)
    b7=cl7(q);b8=cl8(q)
    if not np.array_equal(cl8decode(b8),q):raise AssertionError("lost ternary symbols")
    # Two fp16 row scales as in experimental ternarization; quantizer
    # itself is not official QMoE GPTQ.
    scales=4*q.shape[0]
    metrics={"source":"real pretrained Switch MoE expert, then NEW simple ternarization",
             "model":MODEL,"revision":revision,"file_sha256":h.hexdigest(),
             "tensor":name,"shape":list(w.shape),"weights":int(q.size),
             "original_weight_dtype":str(w.dtype),
             "zero_fraction_after_simple_ternarization":float(np.mean(q==0)),
             "cl7_bytes_with_scales":len(b7)+scales,
             "cl8_bytes_with_scales":len(b8)+scales,
             "zlib_2bit_bytes_with_scales":two_bit_zlib(q)+scales,
             "cl8_bits_per_ternary_weight_with_scales":8*(len(b8)+scales)/q.size,
             "cl8_saving_vs_cl7_pct":100*(1-(len(b8)+scales)/(len(b7)+scales)),
             "CL8_roundtrip":True,
             "warnings":["Weights are true pretrained model weights, not official QMoE ternary checkpoint",
                         "Aggressive simple ternarization quality untested",
                         "One matrix cannot represent full MoE checkpoint quality or compressed runtime"]}
    out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
