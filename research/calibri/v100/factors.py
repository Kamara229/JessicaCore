"""2x2 original FP32 hidden-state/head oracle, completely reversible.

A frozen native head + native final norm.
B original reference output at final norm, native head.
C native final norm, temporary original FP32 head.
D exact teacher final norm output AND temporary original FP32 head.

These are diagnostic causal oracles, NEVER physically compressed
or valid CPU-speed inference. Every teacher intervention is unpaid.
"""
from __future__ import annotations
from contextlib import ExitStack
import torch
from v97.oracle import temporary_oracle_modules
from v99.patch import capture_one,overwrite_one_student_stage,native_logits

@torch.inference_mode()
def factor_logits(teacher,student,ids,variant):
    if variant not in ("native","norm_only","head_only","norm_and_head"):
        raise AssertionError("invalid preregistered factorial oracle")
    patch=variant in ("norm_only","norm_and_head")
    head=variant in ("head_only","norm_and_head")
    original_head=student.get_submodule("lm_head")
    teacher_state=None
    if patch:
        with capture_one(teacher,"model.norm") as samples:
            _=teacher(input_ids=ids,use_cache=False)
        if len(samples)!=1:raise AssertionError("missing exact final norm reference")
        teacher_state=samples[0]
    try:
        with ExitStack() as stack:
            if head:
                stack.enter_context(temporary_oracle_modules(student,teacher,("lm_head",)))
            if patch:
                stack.enter_context(overwrite_one_student_stage(
                    student,"model.norm",teacher_state))
            result=native_logits(student,ids)
    finally:
        if student.get_submodule("lm_head") is not original_head:
            raise AssertionError("a temporary dense head survived the diagnostic probe")
        if student.get_submodule("model.norm")._forward_hooks:
            raise AssertionError("a temporary reference activation hook survived")
    return result
