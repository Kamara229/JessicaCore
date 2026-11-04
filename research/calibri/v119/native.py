"""True native activation VJP through frozen 5bit CB59 (no dense original W)."""
import ctypes,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from direct_runtime_v59 import NativeKernel,PackedLinear
P=ctypes.c_void_p
class BackwardNative:
    def __init__(self,path):
        f=ctypes.CDLL(str(path)).cb59_grad_input
        f.argtypes=[P,ctypes.c_size_t,P,P,P,
                    ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,P]
        f.restype=ctypes.c_int
        self.fn=f
    def input_grad(self,h,gout):
        g=np.ascontiguousarray(gout,dtype=np.float32)
        if g.ndim!=2 or g.shape[1]!=h.shape[0]:raise ValueError("grad geometry")
        dx=np.empty((len(g),h.shape[1]),dtype=np.float32)
        ptr=NativeKernel.pointer
        rc=self.fn(ptr(h.packed),len(h.packed),
                   ptr(h.row_scales),ptr(h.centroids),ptr(g),
                   len(g),h.shape[0],h.shape[1],ptr(dx))
        if rc or not np.isfinite(dx).all():raise ValueError("packed VJP failed")
        return dx
def build(out):
    folder=Path(out);folder.mkdir(exist_ok=True,parents=True)
    root=Path(__file__).resolve().parents[1]
    so=folder/"calibri_v119_native_autograd.so"
    p=subprocess.run(["gcc","-O3","-std=c11","-fPIC","-shared",
                      str(root/"direct_kernel_v59.c"),
                      str(Path(__file__).with_name("native_backward.c")),
                      "-o",str(so)],capture_output=True,text=True,timeout=120)
    if p.returncode:raise RuntimeError(p.stderr)
    return NativeKernel(so),BackwardNative(so)
class PackedVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,h,kernel,reverse):
        if x.device.type!="cpu" or x.dtype!=torch.float32:
            raise ValueError("native CB59 gradient needs CPU FP32")
        ctx.shape=tuple(x.shape);ctx.h=h;ctx.reverse=reverse
        raw=x.detach().reshape(-1,h.shape[1]).contiguous().numpy()
        result=kernel.linear(h,raw)
        return torch.from_numpy(result).reshape(tuple(x.shape[:-1])+(h.shape[0],))
    @staticmethod
    def backward(ctx,gradient):
        h=ctx.h
        data=gradient.detach().reshape(-1,h.shape[0]).contiguous().numpy()
        dx=ctx.reverse.input_grad(h,data)
        return torch.from_numpy(dx).reshape(ctx.shape),None,None,None
class FrozenDiff(nn.Module):
    def __init__(self,packed,kernel,reverse):
        super().__init__()
        self.original=packed
        self.kernel=kernel
        self.reverse=reverse
    def forward(self,x):
        y=PackedVJP.apply(x,self.original.handle,self.kernel,self.reverse)
        if self.original.bias is not None:y=y+self.original.bias
        return y
def install(model,kernel,reverse):
    originals={}
    for name,module in list(model.named_modules()):
        if isinstance(module,PackedLinear):
            originals[name]=module
            parent,sep,leaf=name.rpartition(".")
            owner=model.get_submodule(parent) if sep else model
            setattr(owner,leaf,FrozenDiff(module,kernel,reverse))
    if len(originals)<200:raise AssertionError("not full native 135M CB59")
    return originals
def restore(model,originals):
    for name,module in originals.items():
        parent,sep,leaf=name.rpartition(".")
        owner=model.get_submodule(parent) if sep else model
        setattr(owner,leaf,module)
