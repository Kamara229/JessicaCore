"""All replacements are complete same-byte native 5-bit blocks; zero new correction bits.

The baseline and earlier v92 physical source remain immutable. Other candidate
operators use different TRAIN-only activation-sensitive codebook reconstructions.
Selection is by actual task correctness, not weight RMSE or teacher KL.
"""
from __future__ import annotations
import hashlib
import numpy as np
from direct_codebook_v59 import CB59
from v92.encode import reencode_8d
from v92.archive import inplace_replace
from v95.spec import TARGET

def variants(original_bytes,v92_bytes,original_pretrained_weight,features):
    old=CB59(original_bytes)
    handle=old.handles[TARGET+".weight"]
    w=np.asarray(original_pretrained_weight,dtype=np.float32)
    if tuple(w.shape)!=handle.shape:raise AssertionError("pretrained exact original block geometry changed")
    fit=features["FIT"]
    verify=features["FIT_INTERNAL_HOLDOUT"]
    # Neither features nor variant selection ever read independent TRAIN_VALID.
    methods={
        "uniform":(np.ones_like(fit),np.ones_like(verify)),
        "activation_sqrt":(np.sqrt(fit/fit.mean()),np.sqrt(verify/verify.mean())),
        "activation_winsor":(np.clip(fit,*np.percentile(fit,[15,85])),
                              np.clip(verify,*np.percentile(verify,[15,85]))),
    }
    sources={"CB59_LOCKED":original_bytes,"v92_LOCKED":v92_bytes}
    details={}
    for name,(train_diag,inner_diag) in methods.items():
        candidate=reencode_8d(w,handle,train_diag,inner_diag,iterations=3)
        rebuilt,meta=inplace_replace(original_bytes,TARGET+".weight",candidate)
        if len(rebuilt)!=len(original_bytes):raise AssertionError("loss of frozen compression ratio")
        sources[name]=rebuilt
        details[name]={
            "native_complete_model_archive_bytes":len(rebuilt),
            "additional_paid_bytes":0,
            "candidate_SHA256":hashlib.sha256(rebuilt).hexdigest(),
            "full_block_same_native_K32V8_kernel":True,
            "all_unchanged_packed_layers_byte_identical":meta["all_other_modules_byte_unchanged"],
            "only_fit_internal_holdout_weight_geometry":True,
            "local_reconstruction_index_not_used_to_rank_models":True}
    # Dictionary-key uniqueness says nothing about archive contents.
    # Record physical SHA duplicates, preserving an auditable candidate list;
    # the FIT runner can reuse identical results without rerunning the model.
    first_by_sha={}
    for name,blob in sources.items():
        digest=hashlib.sha256(blob).hexdigest()
        details.setdefault(name,{})["content_SHA256"]=digest
        if digest in first_by_sha:
            details[name]["physical_duplicate_of"]=first_by_sha[digest]
        else:
            first_by_sha[digest]=name
    for name,data in sources.items():
        if len(data)!=len(original_bytes):raise AssertionError("source archive length grew")
        details.setdefault(name,{"candidate_SHA256":hashlib.sha256(data).hexdigest(),
            "native_complete_model_archive_bytes":len(data),"additional_paid_bytes":0})
    return sources,details
