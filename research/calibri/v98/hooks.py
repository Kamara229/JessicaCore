"""Paired layer forward hooks: minimum retained teacher/native activations.

No edits to model attributes or trained weights, registered hooks removed
on exception as well as success. A layer is treated as transformer residual.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch

def stage_names(model):
    if not hasattr(model,"model") or not hasattr(model.model,"layers"):
        raise AssertionError("not a supported decoder-only pretrained model")
    n=len(model.model.layers)
    if n<20 or n>64:
        raise AssertionError("unexpected frozen original transformer depth")
    return ["model.embed_tokens"]+[f"model.layers.{i}" for i in range(n)]+["model.norm"]

@contextmanager
def capture_stages(model,stages):
    captured={}
    registrations=[]
    try:
        for name in stages:
            module=model.get_submodule(name)
            def grab(_module,_args,output,key=name):
                state=output[0] if isinstance(output,tuple) else output
                if not isinstance(state,torch.Tensor):
                    raise AssertionError("layer hook returned no transformer residual")
                captured[key]=state.detach().clone()
            registrations.append(module.register_forward_hook(grab))
        yield captured
    finally:
        for hook in registrations:hook.remove()

def compare_same_context(teacher,native,ids,tokenizer):
    stages=stage_names(teacher)
    if stages!=stage_names(native):
        raise AssertionError("teacher/native transformer layer mismatch")
    with torch.inference_mode():
        with capture_stages(teacher,stages) as t:
            teacher_last=teacher(input_ids=ids,use_cache=False).logits[0,-1,:].detach().clone()
        with capture_stages(native,stages) as q:
            native_last=native(input_ids=ids,use_cache=False).logits[0,-1,:].detach().clone()
    from .metrics import drift,logits_topk,layer_delta,rank_first_divergences
    trace=[]
    previous=None
    for name in stages:
        if name not in t or name not in q:
            raise AssertionError("missing stage captured: "+name)
        value=drift(t[name],q[name])
        trace.append({"stage":name,"metrics":value,
                      "delta_from_previous_stage":layer_delta(previous,value)})
        previous=value
    teacher_out=logits_topk(teacher_last,tokenizer)
    native_out=logits_topk(native_last,tokenizer)
    teacher_set=set(teacher_out["top10_token_ids"])
    native_set=set(native_out["top10_token_ids"])
    return {"stages":trace,"stage_drift_summary":rank_first_divergences(trace),
        "teacher_next_token":teacher_out,"native_next_token":native_out,
        "top10_vocab_overlap_count":len(teacher_set & native_set),
        "top1_prediction_matches_FP32":
            teacher_out["top10_token_ids"][0]==native_out["top10_token_ids"][0],
        "same_input_token_prefix":True,
        "no_weight_changes_or_gradients":True}
