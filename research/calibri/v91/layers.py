"""Read-only per-block error attribution on actual pretrained teacher TRAIN inputs.
Local errors do NOT prove causal effect; separate frozen-module rescue measures it.
"""
from __future__ import annotations
import math
from collections import defaultdict
import torch
from torch import nn
from direct_runtime_v59 import PackedLinear,PackedEmbedding
from v84.sources import original_teacher

MAX_TOKENS=48
PROMPTS_PER_SPLIT=1

def numerics(exact,approx):
    if exact.shape!=approx.shape:raise AssertionError("local operators change shape")
    e=exact.detach().float().reshape(-1)
    a=approx.detach().float().reshape(-1)
    if not torch.isfinite(e).all() or not torch.isfinite(a).all():
        raise AssertionError("nonfinite error attribution")
    error=a-e
    denom=torch.sqrt(torch.mean(e.square())).item()
    rms=torch.sqrt(torch.mean(error.square())).item()
    cosine=float(torch.nn.functional.cosine_similarity(
        e.double()[None,:],a.double()[None,:],dim=-1,eps=1e-12).item())
    return {"relative_RMS_error":float(rms/max(denom,1e-10)),
            "absolute_RMS_error":float(rms),"teacher_RMS":float(denom),
            "cosine":cosine,"max_abs_error":float(error.abs().max().item())}

def layerwise_probes(student,examples):
    teacher=original_teacher()
    quant_names=[name for name,mod in student.named_modules()
                 if isinstance(mod,PackedLinear)]
    if len(quant_names)<100 or "lm_head" not in quant_names:
        raise AssertionError("expected original pretrained 2D whole model, all quantized projections")
    teacher_modules={}
    for name in quant_names:
        mod=teacher.get_submodule(name)
        if not isinstance(mod,nn.Linear):
            raise AssertionError("teacher pretrained linear mismatch "+name)
        teacher_modules[name]=mod
    local=defaultdict(list)
    drifts=defaultdict(list)
    diagnostics=[]
    # Causal swaps need the real pretrained teacher modules; do not train or save them.
    try:
        for split,items in examples.items():
            item=items[0]
            tokens=item["tokens"][-MAX_TOKENS:]
            ids=torch.tensor(tokens,dtype=torch.long)[None,:]
            pre={};post={};a_layers={};b_layers={}
            hooks=[]
            try:
                for name,module in teacher_modules.items():
                    def save_input(_m,args,key=name):
                        pre[key]=args[0].detach().clone()
                    def save_output(_m,_args,out,key=name):
                        post[key]=out.detach().clone()
                    hooks.append(module.register_forward_pre_hook(save_input))
                    hooks.append(module.register_forward_hook(save_output))
                for idx in range(len(teacher.model.layers)):
                    key=f"model.layers.{idx}"
                    def save_t(_m,_args,out,key=key):
                        a_layers[key]=(out[0] if isinstance(out,tuple) else out).detach().clone()
                    hooks.append(teacher.get_submodule(key).register_forward_hook(save_t))
                with torch.inference_mode():
                    gold_logits=teacher(input_ids=ids,use_cache=False).logits[0,-1].float().clone()
            finally:
                for hook in hooks:hook.remove()
            with torch.inference_mode():
                for name in quant_names:
                    output=student.get_submodule(name)(pre[name].float())
                    local[name].append(numerics(post[name],output))
                original_emb=teacher.get_input_embeddings()(ids)
                packed_emb=student.get_input_embeddings()(ids)
                local["model.embed_tokens"].append(numerics(original_emb,packed_emb))
                student_hooks=[]
                try:
                    for idx in range(len(teacher.model.layers)):
                        key=f"model.layers.{idx}"
                        def save_b(_m,_args,out,key=key):
                            b_layers[key]=(out[0] if isinstance(out,tuple) else out).detach().clone()
                        student_hooks.append(student.get_submodule(key).register_forward_hook(save_b))
                    student_logits=student(input_ids=ids,use_cache=False).logits[0,-1].float().clone()
                finally:
                    for hook in student_hooks:hook.remove()
                for key in a_layers:
                    drifts[key].append(numerics(a_layers[key],b_layers[key]))
            diagnostics.append({"split":split,"prompt_SHA":item["prompt_SHA"],"truncated_tokens":len(tokens),
                "tokens":tokens,"teacher_logits":gold_logits,
                "baseline_logits":student_logits})
            del pre,post,a_layers,b_layers
        def avg(scans):
            keys=scans[0]
            return {k:float(sum(x[k] for x in scans)/len(scans)) for k in keys}
        metrics={k:avg(v) for k,v in local.items()}
        drift={k:avg(v) for k,v in drifts.items()}
        return teacher,metrics,drift,diagnostics
    except BaseException:
        del teacher
        raise
