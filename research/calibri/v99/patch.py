"""Paired same-token FP32 activation transplants, never stored or deployed.

Teacher is run on identical token input first; its hidden state at the
selected stage is injected into the *student* only within try/finally.
During generation the teacher is invoked again on the native model's
own evolving prefix: teacher assistance is UNPAID and the result is an
upper-bound diagnostic, NOT a compressed LLM or runtime benchmark.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch

def hidden_tensor(output):
    state=output[0] if isinstance(output,tuple) else output
    if not isinstance(state,torch.Tensor) or state.ndim!=3:
        raise AssertionError("expected [batch, tokens, hidden] residual tensor")
    if not bool(torch.isfinite(state).all()):
        raise AssertionError("nonfinite reference state")
    return state

@contextmanager
def capture_one(model,stage):
    saved=[]
    def hook(_module,_args,output):
        if saved:
            raise AssertionError("selected transformer stage called twice")
        saved.append(hidden_tensor(output).detach().clone())
    handle=model.get_submodule(stage).register_forward_hook(hook)
    try:yield saved
    finally:handle.remove()

@contextmanager
def overwrite_one_student_stage(student,stage,fp32_state):
    if fp32_state.requires_grad:
        raise AssertionError("unpaid reference must be detached")
    if not bool(torch.isfinite(fp32_state).all()):
        raise AssertionError("nonfinite transplanted activation")
    count=[0]
    def replace(_module,_args,output):
        count[0]+=1
        if count[0]>1:
            raise AssertionError("target stage unexpectedly invoked more than once")
        current=hidden_tensor(output)
        if current.shape!=fp32_state.shape or current.dtype!=fp32_state.dtype:
            raise AssertionError("reference/native activation geometry mismatched")
        injected=fp32_state.detach().clone()
        return (injected,)+tuple(output[1:]) if isinstance(output,tuple) else injected
    handle=student.get_submodule(stage).register_forward_hook(replace)
    try:
        yield count
        if count[0]!=1:raise AssertionError("the native stage was never patched")
    finally:
        handle.remove()

@torch.inference_mode()
def paired_one_token_logits(teacher,student,ids,stage):
    if not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.numel()<2:
        raise AssertionError("invalid common token prefix")
    with capture_one(teacher,stage) as observed:
        exact=teacher(input_ids=ids,use_cache=False).logits[0,-1].detach().float().clone()
    if len(observed)!=1:raise AssertionError("reference stage was not reached")
    with overwrite_one_student_stage(student,stage,observed[0]):
        packed=student(input_ids=ids,use_cache=False).logits[0,-1].detach().float().clone()
    if not bool(torch.isfinite(packed).all() and torch.isfinite(exact).all()):
        raise AssertionError("nonfinite next token output")
    return exact,packed

@torch.inference_mode()
def native_logits(model,ids):
    logits=model(input_ids=ids,use_cache=False).logits[0,-1].detach().float().clone()
    if not bool(torch.isfinite(logits).all()):
        raise AssertionError("nonfinite next-token logits")
    return logits
