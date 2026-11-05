"""Native packed CPU runner for variable-width CB177 matrices."""
import ctypes
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_runtime_v59 import compile_kernel, bind_into_llama, NativeKernel
from .codec import CB177

PTR = ctypes.c_void_p
I64 = ctypes.c_int64


class MixedNativeKernel:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        for name in ("cb177_linear", "cb177_embedding"):
            fn = getattr(self.lib, name)
            fn.argtypes = [PTR, ctypes.c_size_t, PTR, PTR, PTR,
                           I64, I64, I64, I64, PTR]
            fn.restype = ctypes.c_int

    @staticmethod
    def ptr(array):
        return PTR(int(array.ctypes.data))

    def linear(self, handle, x):
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != handle.shape[1]:
            raise ValueError("Native mixed bit linear requires 2D FP32 CPU inputs")
        x = np.ascontiguousarray(x)
        out = np.empty((x.shape[0],handle.shape[0]),dtype=np.float32)
        rc = self.lib.cb177_linear(self.ptr(handle.packed), len(handle.packed),
            self.ptr(handle.row_scales),self.ptr(handle.centroids),
            self.ptr(x),x.shape[0],handle.shape[0],handle.shape[1],handle.bits,
            self.ptr(out))
        if rc or not np.isfinite(out).all():
            raise ValueError(f"Variable-bit native C linear invalid: {rc}")
        return out

    def embedding(self, handle, ids):
        ids = np.ascontiguousarray(ids,dtype=np.int64).reshape(-1)
        out = np.empty((len(ids),handle.shape[1]),dtype=np.float32)
        rc = self.lib.cb177_embedding(self.ptr(handle.packed),len(handle.packed),
            self.ptr(handle.row_scales),self.ptr(handle.centroids),
            self.ptr(ids),len(ids),handle.shape[0],handle.shape[1],handle.bits,
            self.ptr(out))
        if rc or not np.isfinite(out).all():
            raise ValueError(f"Variable-bit native C embedding invalid: {rc}")
        return out


def build_native(blob, folder):
    archive = CB177(blob)
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    cfg = AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation = "eager"
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    kernel = MixedNativeKernel(compile_kernel(Path(__file__).with_name("kernel.c"),
                                               folder/"libcb177.so"))
    evidence = bind_into_llama(model,archive,kernel)
    if any(p.ndim>=2 for p in model.parameters()):
        raise AssertionError("Source expanded into dense model in native inference")
    if model.model.embed_tokens.handle is not model.lm_head.handle:
        raise AssertionError("Embedding and output head not tied")
    return model, archive, evidence
