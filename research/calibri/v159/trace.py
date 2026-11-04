"""Capture after each real source Llama decoder block, plus input and norm.

Read-only forward hooks are disposed on success or error. The dense FP32
teacher is a diagnostic reference ONLY, not a charged/packed candidate.
No tensor sampled here is fitted into a hidden sidecar or codebook.
"""
from contextlib import contextmanager

import torch

from .protocol import TRACE_KEYS,LAYERS,TOKENS_EACH,WINDOW


def _output_tensor(value):
    # HF eager LlamaDecoderLayer returns (hidden, ...) in 4.46.3.
    x=value[0] if isinstance(value,(tuple,list)) else value
    if not isinstance(x,torch.Tensor):
        raise ValueError("real decoder did not return original hidden tensor")
    if x.ndim!=3 or tuple(x.shape)!=(1,WINDOW,576):
        raise ValueError("original 135M hidden geometry altered")
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite upstream transformer activation")
    return x.detach().cpu().clone()[0]


@contextmanager
def hook_all_layers(model):
    layers=model.model.layers
    if len(layers)!=LAYERS:
        raise ValueError("not original 30-layer SmolLM2 source")
    modules=[model.model.embed_tokens,*layers,model.model.norm]
    if len(modules)!=len(TRACE_KEYS):
        raise ValueError("every traced original operator must exist")
    values={key:[] for key in TRACE_KEYS}
    hooks=[]
    try:
        for key,module in zip(TRACE_KEYS,modules):
            def collect(_module,_inputs,out,stage=key):
                values[stage].append(_output_tensor(out))
            hooks.append(module.register_forward_hook(collect))
        yield values
    finally:
        for hook in reversed(hooks):
            hook.remove()


@torch.inference_mode()
def run_original_fit_trace(model,tokens):
    if tokens.ndim!=1 or len(tokens)!=TOKENS_EACH:
        raise ValueError("must capture exactly 4x64 original TRAIN FIT source tokens")
    with hook_all_layers(model) as rows:
        for window in range(TOKENS_EACH//WINDOW):
            ids=tokens[window*WINDOW:(window+1)*WINDOW][None,:]
            result=model(input_ids=ids,use_cache=False)
            if result.logits.shape!=(1,WINDOW,49152):
                raise ValueError("full original LLM vocabulary changed")
    if tuple(rows)!=TRACE_KEYS or any(len(rows[k])!=4 for k in TRACE_KEYS):
        raise ValueError("some original layer did not execute exactly four times")
    merged={k:torch.cat(rows[k],dim=0).contiguous()
            for k in TRACE_KEYS}
    if any(v.shape!=(TOKENS_EACH,576) for v in merged.values()):
        raise ValueError("source trace missing hidden positions")
    return merged
