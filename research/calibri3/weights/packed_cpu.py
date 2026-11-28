"""Packed 3/4/5-bit CPU codebook operator, no dense weight reconstruction."""
import ctypes as C
import numpy as np
import torch

def pack_codes(codes,bits):
    values=codes.detach().cpu().numpy().reshape(-1).astype(np.uint8)
    assert np.all(values<(1<<bits))
    b=((values[:,None]>>np.arange(bits,dtype=np.uint8))&1).reshape(-1)
    # One zero sentinel byte makes the final two-byte extraction safe.
    packed=np.concatenate((np.packbits(b,bitorder="little"),np.zeros(1,dtype=np.uint8)))
    assert np.array_equal(np.sum(np.unpackbits(packed,bitorder="little")[:len(b)].reshape(-1,bits)*(1<<np.arange(bits)),axis=1),values)
    return packed

class PackedCPU:
    def __init__(self,path):
        self.lib=C.CDLL(str(path.resolve()))
        self.fn=self.lib.cb_linear
        self.fn.argtypes=[C.c_void_p]*5+[C.c_int]*5
        self.fn.restype=C.c_int
        self.plans={}
        self.stats={"calls":0,"plan_builds":0,"packed_bytes":0,"logical_weights":0}
    def __call__(self,x,source):
        assert x.device.type=="cpu" and x.dtype==torch.float32
        c,s=source.fields();o,i=source.shape
        assert i%8==0 and x.shape[-1]==i
        codes=source.codes
        stamp=(id(codes),codes.data_ptr(),codes._version,tuple(source.shape))
        key=id(source);old=self.plans.get(key)
        if old is None or old[0]!=stamp:
            hi=int(codes.max());lo=int(codes.min())
            assert 0<=lo<=hi<source.centroids.shape[0]
            bits=next(b for b in (3,4,5) if hi<1<<b)
            packed=pack_codes(codes,bits)
            self.plans[key]=(stamp,bits,packed)
            self.stats["plan_builds"]+=1
        _,bits,packed=self.plans[key]
        self.stats["packed_bytes"]=sum(v[2].nbytes for v in self.plans.values())
        a=x.detach().contiguous().reshape(-1,i)
        c=c[:1<<bits].detach().contiguous();s=s.detach().contiguous()
        out=torch.empty((a.shape[0],o),dtype=torch.float32)
        status=self.fn(a.data_ptr(),c.data_ptr(),s.data_ptr(),packed.ctypes.data,out.data_ptr(),a.shape[0],o,i//8,1<<bits,bits)
        assert status==0,status
        self.stats["calls"]+=1;self.stats["logical_weights"]+=o*i
        # Inference-only. Do not silently allow training through detached C output.
        return _InferenceOnly.apply(out.reshape(*x.shape[:-1],o),x,source.centroids)

class _InferenceOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx,y,x,c):return y
    @staticmethod
    def backward(ctx,g):raise RuntimeError("PackedCPU is inference-only; use differentiable V001 for training")
