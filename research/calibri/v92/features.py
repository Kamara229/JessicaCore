"""Train/holdout activation energy and labels from ORIGINAL BoolQ train split.
Uses pinned original FP32 teacher, full block input pre-hook only; no TEST.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch

TRAIN_CASES=12
VALID_CASES=12
MAX_TOKENS=48

@torch.inference_mode()
def capture_energy(teacher,block,cases):
    if set(cases)!={"TRAIN","TRAIN_VALID"}:
        raise AssertionError("no external evaluation split allowed in model rewrite")
    out={}
    module=teacher.get_submodule(block)
    width=module.weight.shape[1]
    def energy(rows):
        sums=torch.zeros(width,dtype=torch.float64)
        seen=0
        digests=[]
        for row in rows:
            inputs=[]
            def hook(_mod,args):
                inputs.append(args[0].detach().float().reshape(-1,width).clone())
            h=module.register_forward_pre_hook(hook)
            try:
                ids=torch.tensor(row["tokens"][-MAX_TOKENS:],dtype=torch.long)[None,:]
                teacher(input_ids=ids,use_cache=False)
            finally:h.remove()
            if len(inputs)!=1:raise AssertionError("expected exactly one original block execution")
            x=inputs[0].double()
            sums+=x.square().sum(0)
            seen+=len(x)
            digests.append(row["prompt_SHA"])
        if not seen:raise AssertionError("empty original pretrained TRAIN input")
        return np.clip((sums/seen).numpy(),1e-9,None),{
            "observations":seen,"example_count":len(rows),
            "source_prompt_SHA256":hashlib.sha256("".join(digests).encode()).hexdigest(),
            "relative_train_tokens_only":True}
    out["TRAIN"],train_stats=energy(cases["TRAIN"][:TRAIN_CASES])
    out["TRAIN_VALID"],valid_stats=energy(cases["TRAIN_VALID"][:VALID_CASES])
    return out,{"TRAIN":train_stats,"TRAIN_VALID":valid_stats,
                "no_official_TEST_or_BoolQ_validation_used":True}
