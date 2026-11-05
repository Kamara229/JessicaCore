"""Direct 4D weight tile source, C CPU execution, no dense matrix residency."""
import ctypes
import numpy as np
from direct_runtime_v59 import NativeKernel,compile_kernel
from .codec import Tile4D

ADDRESS=ctypes.c_void_p
INT=ctypes.c_int64

class Kernel187(NativeKernel):
    def __init__(self,path):
        super().__init__(path)
        fn=self.lib.cb187_linear
        fn.argtypes=[ADDRESS,ctypes.c_size_t,ADDRESS,ADDRESS,ADDRESS,
                     INT,INT,INT,ADDRESS]
        fn.restype=ctypes.c_int
        fn=self.lib.cb187_embedding
        fn.argtypes=[ADDRESS,ctypes.c_size_t,ADDRESS,ADDRESS,ADDRESS,
                     INT,INT,INT,ADDRESS]
        fn.restype=ctypes.c_int

    def linear(self,h,x):
        h.validate()
        if x.ndim!=2 or x.shape[1]!=h.shape[1] or x.dtype!=np.float32:
            raise ValueError("4D tiled kernel requires exact FP32 activation geometry")
        x=np.ascontiguousarray(x)
        packed=np.frombuffer(h.packed,dtype=np.uint8)
        scales=np.ascontiguousarray(h.row_scales,dtype=np.float32)
        centers=np.ascontiguousarray(h.centroids,dtype=np.float32)
        out=np.empty((x.shape[0],h.shape[0]),dtype=np.float32)
        rc=self.lib.cb187_linear(
            self.pointer(packed),len(packed),self.pointer(scales),
            self.pointer(centers),self.pointer(x),x.shape[0],
            h.shape[0],h.shape[1],self.pointer(out))
        if rc or not np.isfinite(out).all():
            raise RuntimeError(f"4D native packed linear code {rc}")
        return out

    def embedding(self,h,ids):
        h.validate()
        ids=np.ascontiguousarray(ids,dtype=np.int64).reshape(-1)
        packed=np.frombuffer(h.packed,dtype=np.uint8)
        scales=np.ascontiguousarray(h.row_scales,dtype=np.float32)
        centers=np.ascontiguousarray(h.centroids,dtype=np.float32)
        out=np.empty((len(ids),h.shape[1]),dtype=np.float32)
        rc=self.lib.cb187_embedding(
            self.pointer(packed),len(packed),self.pointer(scales),
            self.pointer(centers),self.pointer(ids),len(ids),
            h.shape[0],h.shape[1],self.pointer(out))
        if rc or not np.isfinite(out).all():
            raise RuntimeError(f"4D native packed embedding code {rc}")
        return out
