"""A drop-in v141 native packed residual runner for physically paid CB140.

Only changes C execution, NEVER bytes, source dictionary, original weights,
1-bit plane or logits mathematics. Must compare against v140 numerically.
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np

from v140.runtime import OneBitNative,PTR,I64,SIZE


def compile_lut(destination):
    dst=Path(destination)
    src=Path(__file__).with_name("onebit_lut_kernel.c")
    result=subprocess.run(["gcc","-O3","-std=c11","-fPIC","-shared",
                           "-o",str(dst),str(src)],
                          text=True,capture_output=True,timeout=120)
    if result.returncode:
        raise RuntimeError("LUT CPU code failed to compile: "+result.stderr)
    return dst


class LUTNative(OneBitNative):
    def __init__(self, library, handle, physical):
        self.lib=ctypes.CDLL(str(library))
        self.handle=handle
        self.bits=np.ascontiguousarray(physical.bits,dtype=np.uint8)
        self.centers=np.ascontiguousarray(physical.centers,dtype=np.float32)
        self.base=np.ascontiguousarray(handle.packed,dtype=np.uint8)
        self.scales=np.ascontiguousarray(handle.row_scales,dtype=np.float32)
        self.rows,self.dim=handle.shape
        self.embed_fn=self.lib.calibri141_embedding_delta
        self.linear_fn=self.lib.calibri141_output_delta
        contract=[PTR,SIZE,PTR,SIZE,PTR,PTR,PTR,I64,I64,I64,PTR]
        self.embed_fn.argtypes=contract
        self.linear_fn.argtypes=contract
        self.embed_fn.restype=ctypes.c_int
        self.linear_fn.restype=ctypes.c_int
        if self.rows!=49_152 or self.dim!=576:
            raise ValueError("runtime must use actual SmolLM2 tied source shape")
        if len(self.bits)!=(self.rows*self.dim//8+7)//8:
            raise ValueError("unpaid or missing 1-bit packed plane")


def install(model,physical,folder):
    from direct_runtime_v59 import PackedLinear,PackedEmbedding
    from v140.runtime import Embedding,OutputHead
    initial_embedding=model.model.embed_tokens
    initial_head=model.lm_head
    if not isinstance(initial_embedding,PackedEmbedding) or not isinstance(initial_head,PackedLinear):
        raise ValueError("expected packed CB59 base, not dense or v140 wrappers")
    if initial_embedding.handle is not initial_head.handle:
        raise ValueError("one physical codebook must back original tied embeddings/head")
    shared=LUTNative(compile_lut(Path(folder)/"libcalibri_lut_v141.so"),
                     initial_embedding.handle,physical)
    model.model.embed_tokens=Embedding(initial_embedding,shared)
    model.lm_head=OutputHead(initial_head,shared)
    if model.model.embed_tokens.kernel is not model.lm_head.kernel:
        raise AssertionError("runtime decoded separately instead of sharing one source")
    if any(p.ndim>=2 for p in model.parameters()):
        raise AssertionError("dense trainable original weights materialized")
    return shared
