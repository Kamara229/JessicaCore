"""S59 bounded GPU availability/numerical preflight; CPU-portable, NOT quality proof.

No model weights or secrets are read or submitted. Equivalent CPU/GPU
FP32 matrix operations are compared with an explicit numerical tolerance.
Designed to complete fast on Kaggle's free quota, or run on CPU elsewhere.
"""
from __future__ import annotations
import json, os, platform, time
from pathlib import Path
import torch

torch.set_num_threads(2)
torch.manual_seed(59059)
cuda=bool(torch.cuda.is_available())
device=torch.device("cuda:0" if cuda else "cpu")
size=192
left=torch.randn(size,size,dtype=torch.float32)
right=torch.randn(size,size,dtype=torch.float32)
ref=left @ right
start=time.perf_counter()
if cuda:
    gpuinfo=torch.cuda.get_device_properties(0)
    cap={"name":gpuinfo.name,"memory_bytes":gpuinfo.total_memory,
         "cuda_device_count":torch.cuda.device_count()}
else:
    cap={"name":"CPU_ONLY","memory_bytes":0,"cuda_device_count":0}
a=left.to(device)
b=right.to(device)
got=a @ b
if cuda: torch.cuda.synchronize()
relative=((got.detach().cpu()-ref).abs() / ref.abs().clamp_min(.01)).max().item()
absolute=(got.detach().cpu()-ref).abs().max().item()
record={
    "stage":"CALIBRI_S59_GPU_CAPABILITY_PREFLIGHT",
    "torch_version":torch.__version__,
    "device":str(device),
    "accelerator":cap,
    "float32_192matmul_max_abs_difference":absolute,
    "float32_192matmul_max_relative_difference":relative,
    "tolerance_abs":.0005,
    "numerical_probe_pass":absolute < .0005,
    "cpu_portable":True,
    "model_quality_testing_performed":False,
    "no_train_no_model_changes":True,
    "duration_seconds":time.perf_counter()-start,
    "quota_fallback_contract":"Same PyTorch test runs on CPU; GPU optional."
}
Path("/kaggle/working" if Path("/kaggle/working").is_dir() else ".").joinpath(
   "calibri_s59_gpu_preflight.json").write_text(json.dumps(record,indent=2))
print("CALIBRI_S59_PUBLIC_PREFLIGHT="+json.dumps(record),flush=True)
if not record["numerical_probe_pass"]: raise SystemExit("numerical probe failed")
