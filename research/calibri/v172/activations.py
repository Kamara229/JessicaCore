"""Capture only ORIGINAL FP32 operator INPUT ENERGY on historical prompts.

The old 16 v94 diagnostic prompts are preexisting/fixed and are NOT the
new 40 v169 behavioral evaluation samples. We never look at QA labels
or generated teacher responses during fitting. All original linear
operators are sampled once with hooks, then hooks removed.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from v94.generation import NEW_PROMPTS
from v108.generation import build_chat
from .protocol import ACTIVATION_INPUTS,check


@torch.inference_mode()
def collect_original_linear_input_energy(teacher,tokenizer):
    check()
    totals={}
    counts={}
    handles=[]
    source_modules={n:m for n,m in teacher.named_modules() if isinstance(m,nn.Linear)}
    if len(source_modules)<210:
        raise ValueError("not all pretrained 30-layer original linear operators")
    for name,module in source_modules.items():
        def on_input(mod,input_args,*,name=name):
            x=input_args[0]
            if x.ndim not in (2,3) or x.shape[-1]!=mod.in_features:
                raise ValueError("wrong original transformer input shape")
            values=x.detach().reshape(-1,x.shape[-1]).float()
            # diagonal activation covariance proxy; no labels/logits.
            moments=(values*values).sum(dim=0,dtype=torch.float64).cpu().numpy()
            if name in totals:
                totals[name]+=moments
                counts[name]+=len(values)
            else:
                totals[name]=moments.astype(np.float64,copy=True)
                counts[name]=len(values)
        handles.append(module.register_forward_pre_hook(on_input))
    try:
        for category,question in NEW_PROMPTS[:ACTIVATION_INPUTS]:
            prompt=build_chat(tokenizer,question)
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            if not 4<len(ids)<160:
                raise ValueError("historical unlabeled source prompt is invalid")
            teacher(input_ids=torch.tensor([ids],dtype=torch.long),use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if set(source_modules)!=set(totals):
        raise ValueError("one original FP32 pretrained operator was never observed")
    normalized={}
    for name,m in totals.items():
        values=m/max(1,counts[name])
        if not np.isfinite(values).all() or np.mean(values)<=0:
            raise ValueError("FP32 original activation feature energy invalid")
        # cap extreme outliers to prevent a 16-prompt accidental overfit.
        values=np.clip(values/np.mean(values),0.1,10.0)
        normalized[name]=values.astype(np.float32)
    if len(normalized)<210:
        raise ValueError("source measurements incomplete")
    return normalized,{
        "historical_source_prompts":ACTIVATION_INPUTS,
        "actual_original_FP32_linear_operators":len(normalized),
        "original_teacher_input_energy_only":True,
        "no_generated_teacher_answer_or_QA_gold_observed":True,
        "original_INPUT_normalized_diagonal_energy":{
            name:{"features":len(v),"min":float(np.min(v)),
                  "max":float(np.max(v)),
                  "mean":float(np.mean(v))} for name,v in normalized.items()
        },
    }
