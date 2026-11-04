"""Hook the SAME original source 2D operator in FP32 and native models."""
from contextlib import contextmanager

import torch

from .protocol import OPERATOR,WINDOW


@contextmanager
def capture_input(model):
    samples=[]
    module=model.get_submodule(OPERATOR)
    def get_input(_module,inputs):
        if len(inputs)!=1 or not isinstance(inputs[0],torch.Tensor):
            raise ValueError("Llama v_proj must have one tensor input")
        x=inputs[0].detach().to(dtype=torch.float32,device="cpu").clone()
        if x.ndim!=3 or x.shape[0]!=1 or x.shape[1]!=WINDOW or x.shape[2]!=576:
            raise ValueError("unexpected layer6 original v_proj hidden geometry")
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite source input activation")
        samples.append(x[0])
    hook=module.register_forward_pre_hook(get_input)
    try:
        yield samples
    finally:
        hook.remove()


@torch.inference_mode()
def captured_fit(model,domain_tokens):
    if len(domain_tokens)!=WINDOW*4:
        raise ValueError("must only read 4 original TRAIN FIT source windows")
    with capture_input(model) as observations:
        for k in range(4):
            x=domain_tokens[k*WINDOW:(k+1)*WINDOW][None,:]
            model(input_ids=x,use_cache=False)
    if len(observations)!=4:
        raise ValueError("missing original transformer operator input")
    return torch.cat(observations,dim=0)
