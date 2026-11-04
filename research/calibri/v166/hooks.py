"""Read-only original hidden activations and illegal teacher-oracle swap.

Using the original pretrained FP32 state directly inside a compressed
student is NOT a Green Tunnel implementation, only a causal upper bound.
No model buffers, parameters or source codewords are modified. Hook
restoration is exception safe. The original HF decoder block may return
a tuple; preserve all other output members and replace ONLY hidden[0].
"""
from contextlib import contextmanager
import torch
from .protocol import TEACHER_ONLY_STAGES,EXPECTED_HIDDEN,WINDOW,check


def stage_module(model,name):
    if name=="input_embedding":
        return model.model.embed_tokens
    if name=="final_norm":
        return model.model.norm
    if name.startswith("block_") and name.endswith("_output"):
        idx=int(name[6:8])
        if idx>=30:raise ValueError("invalid source transformer block")
        return model.model.layers[idx]
    raise ValueError("nonpreregistered teacher hidden stage")


def extract_hidden(output):
    hidden=output[0] if isinstance(output,(tuple,list)) else output
    if not isinstance(hidden,torch.Tensor) or tuple(hidden.shape)!=(
            1,WINDOW,EXPECTED_HIDDEN):
        raise ValueError("source hidden state geometry mismatch")
    if not torch.isfinite(hidden).all():
        raise ValueError("unusable reference hidden state")
    return hidden


@contextmanager
def record_reference(model):
    """Capture five source stage tensors ONLY during original FP32 forward."""
    check()
    collected={}
    handles=[]
    try:
        for name in TEACHER_ONLY_STAGES:
            def capture(_module,_args,output,stage=name):
                if stage in collected:
                    raise ValueError("source reference stage called more than once per window")
                collected[stage]=extract_hidden(output).detach().clone()
            handles.append(stage_module(model,name).register_forward_hook(capture))
        yield collected
    finally:
        for handle in reversed(handles):
            handle.remove()


@contextmanager
def inject_illegal_teacher_oracle(model,stage,teacher_state):
    if stage not in TEACHER_ONLY_STAGES:
        raise ValueError("only five preregistered original source boundaries")
    reference=teacher_state.detach().clone()
    if tuple(reference.shape)!=(1,WINDOW,EXPECTED_HIDDEN) or not torch.isfinite(reference).all():
        raise ValueError("original teacher activation must match entire source window")

    def replace_output(_module,_args,output):
        student_hidden=extract_hidden(output)
        if student_hidden.shape!=reference.shape:
            raise ValueError("student architecture not compatible with FP32 oracle")
        ref=reference.to(device=student_hidden.device,dtype=student_hidden.dtype)
        if isinstance(output,tuple):
            return (ref,*output[1:])
        if isinstance(output,list):
            return [ref,*output[1:]]
        return ref

    handle=stage_module(model,stage).register_forward_hook(replace_output)
    try:
        yield
    finally:
        handle.remove()
