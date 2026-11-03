"""Counterfactual FP32 replacement on the ACTUAL packed student's input.

Diagnostic oracle only: these are full FP32 pretrained weights, not a
physically paid compact model. Restores all modules on success or failure.
"""
from contextlib import contextmanager
from torch import nn
from direct_runtime_v59 import PackedLinear
from v84.sources import TARGETS

SCENARIOS = {
    "sham_baseline": (),
    "head_only": ("lm_head",),
    "embedding_only": ("model.embed_tokens",),
    "down_11_28_29": TARGETS,
    "head_and_down": ("lm_head",) + TARGETS,
    "embedding_and_head": ("model.embed_tokens", "lm_head"),
    "boundary_and_down": ("model.embed_tokens", "lm_head") + TARGETS,
}

@contextmanager
def intervene(student, teacher, paths):
    if len(paths) != len(set(paths)):
        raise AssertionError("duplicate diagnostic oracle")
    originals = []
    try:
        for path in paths:
            old = student.get_submodule(path)
            original_fp32 = teacher.get_submodule(path)
            if path in TARGETS or path == "lm_head":
                if not isinstance(old, PackedLinear) or not isinstance(original_fp32, nn.Linear):
                    raise AssertionError("expected original packed and pretrained FP32")
            elif path == "model.embed_tokens":
                if not isinstance(original_fp32, nn.Embedding):
                    raise AssertionError("missing original pretrained FP32 embedding")
            else:
                raise AssertionError("unknown intervention: " + path)
            prefix, _, name = path.rpartition(".")
            parent = student.get_submodule(prefix) if prefix else student
            originals.append((parent, name, old))
            setattr(parent, name, original_fp32)
        yield
    finally:
        for parent, name, old in reversed(originals):
            setattr(parent, name, old)
