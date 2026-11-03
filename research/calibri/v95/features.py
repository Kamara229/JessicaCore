"""Collect cheap per-input activation statistics on original public TRAIN FIT only.

Using original FP32 operator inputs only to build *native codebooks* is allowed:
we do not optimize student probability-distribution resemblance.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch

@torch.inference_mode()
def train_energies(original_model,module_path,train_boolq):
    if len(train_boolq)<24:raise AssertionError("not enough FIT examples")
    module=original_model.get_submodule(module_path)
    width=module.weight.shape[1]
    subsets={"FIT":train_boolq[:16],"FIT_INTERNAL_HOLDOUT":train_boolq[16:24]}
    results={};origins={}
    for name,rows in subsets.items():
        squares=torch.zeros(width,dtype=torch.float64)
        count=0
        for row in rows:
            captured=[]
            def pre(_mod,args):captured.append(args[0].detach().float().reshape(-1,width).clone())
            listener=module.register_forward_pre_hook(pre)
            try:
                x=torch.tensor(row["tokens"][-48:],dtype=torch.long)[None,:]
                original_model(input_ids=x,use_cache=False)
            finally:listener.remove()
            if len(captured)!=1:raise AssertionError("teacher source module not reached")
            z=captured[0].double()
            squares+=z.square().sum(dim=0)
            count+=len(z)
        if not count:raise AssertionError("empty original FIT activations")
        results[name]=np.clip((squares/count).numpy(),1e-9,None)
        origins[name]={"count":len(rows),"tokens":count,
            "prompt_SHA256":hashlib.sha256("".join(r["prompt_SHA"] for r in rows).encode()).hexdigest(),
            "actual_original_public_split":"train","is_official_TEST":False}
    return results,origins
