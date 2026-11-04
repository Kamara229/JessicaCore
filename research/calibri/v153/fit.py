"""Physical source-only re-quantize ONE original 30-layer transformer block.

Per-module K32 original 5bit codewords remain frozen; fit + ONE extra bit
per 8D group from original pretrained FP32 weights. Unlike preceding
output-logit priors, the operator itself is changed in its original layer.
All source 2D weights remain compressed; no fine-tuning on VALID or QA.
"""
import numpy as np
import torch
from direct_codebook_v59 import CB59,unpack5
from v137.protocol import groups
from v139.refinement import train as train_centers
from v140.train import assign_full_extra_bits
from .protocol import (FIT_GROUP_SAMPLE_MAX,FIT_STEPS,
                       EXPECTED_BLOCK_MATRICES)
from .codec import Patch


def fit_one(original_w,handle):
    target=np.asarray(original_w,dtype=np.float32)
    if target.shape!=handle.shape or target.ndim!=2 or target.shape[1]%8:
        raise ValueError("not an original trained 2D source")
    total=target.size//8
    code=unpack5(handle.packed.tobytes(),total)
    # Training sample is systematic, determined by physical tensor shape
    # and original source only, never influenced by model behavior or VALID.
    indices=np.linspace(0,total-1,min(FIT_GROUP_SAMPLE_MAX,total),
                        dtype=np.int64)
    _,normalized,scales=groups(target,handle.row_scales,indices)
    residual=train_centers(normalized,code[indices],handle.centroids,
                           energy=np.maximum(scales.astype(np.float64)**2,1e-15),
                           steps=FIT_STEPS)
    if residual.shape!=(32,2,8):
        raise AssertionError("fitted 1bit physical codebook wrong")
    # Physical FP16 round-trip BEFORE assigning the real full source bits.
    physical_centers=residual.astype("<f2").astype(np.float32)
    bitplane,metrics=assign_full_extra_bits(
        target,handle,physical_centers,code)
    patch=Patch(handle.name,tuple(handle.shape),
                physical_centers,np.frombuffer(bitplane,dtype=np.uint8).copy())
    return patch,{**metrics,
                  "training_FIT_original_weight_groups":len(indices),
                  "entire_original_trained_matrix_encoded":True,
                  "no_validation_or_QA_in_codebook_FIT":True,
                  "K32_5bit_source_codewords_UNCHANGED":True}


def train_block(teacher,original_cb59,block):
    cb=CB59(original_cb59)
    prefix=f"model.layers.{block}."
    keys=sorted(x for x in cb.handles
                if x.startswith(prefix) and x.endswith(".weight"))
    if len(keys)!=EXPECTED_BLOCK_MATRICES:
        raise ValueError("original packed layer must have seven trained 2D operators")
    modules={}
    fitting={}
    for key in keys:
        module=teacher.get_submodule(key[:-len(".weight")])
        if not isinstance(module,torch.nn.Linear):
            raise ValueError("not the original teacher linear operator "+key)
        weight=module.weight.detach().cpu().numpy()
        patch,metrics=fit_one(weight,cb.handles[key])
        modules[key]=patch
        fitting[key]=metrics
    return modules,fitting
