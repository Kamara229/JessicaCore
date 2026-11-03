"""Read-only exact module rescue: temporary *diagnostic* FP32 swap only.

These oracle interventions have grossly unpaid dense storage. Their
functional gain cannot be reported as an achieved compact model and their
timing cannot be compared to native packed inference speed.
"""
from __future__ import annotations
from contextlib import contextmanager
from torch import nn
from direct_runtime_v59 import PackedLinear,PackedEmbedding

@contextmanager
def temporary_oracle_modules(packed_model,original_fp32,paths):
    if not paths or len(paths)!=len(set(paths)):
        raise AssertionError("oracle probe requires unique nonempty modules")
    assigned=[]
    try:
        for name in paths:
            parent_path,sep,leaf=name.rpartition(".")
            parent=packed_model.get_submodule(parent_path) if sep else packed_model
            if not hasattr(parent,leaf):
                raise AssertionError("required native packed module not found: "+name)
            old=getattr(parent,leaf)
            exact=original_fp32.get_submodule(name)
            if not isinstance(old,(PackedLinear,PackedEmbedding)):
                raise AssertionError("replace only directly packed module: "+name)
            if isinstance(old,PackedLinear):
                if not isinstance(exact,nn.Linear) or tuple(exact.weight.shape)!=tuple(old.handle.shape):
                    raise AssertionError("mismatched original FP32 linear module")
            else:
                if not isinstance(exact,nn.Embedding) or tuple(exact.weight.shape)!=tuple(old.handle.shape):
                    raise AssertionError("mismatched original FP32 embedding module")
            setattr(parent,leaf,exact)
            assigned.append((parent,leaf,old))
        yield
    finally:
        for parent,leaf,old in reversed(assigned):
            setattr(parent,leaf,old)
