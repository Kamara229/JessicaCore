"""Exact materialization of CB59 TRAINED matrix values, diagnostic only.

This dense tensor is NOT the original pretrained FP32 source; it merely
represents EXACTLY the same compressed 5bit indices/centers/row scales in
an ordinary PyTorch nn.Linear and nn.Embedding. It is never serialized
as an alternative low-bpp checkpoint and cannot be deployed at claimed
sub-0.7 bits. Native production execution stays packed.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForCausalLM
from direct_codebook_v59 import CB59,unpack5
from binary_geometry_v64 import MODEL,MODEL_SHA
from .protocol import EXPECTED_MATRICES,EXPECTED_BYTE_SIZE,check


def exact_dense_matrix(handle):
    rows,dim=handle.shape
    if rows<=0 or dim%8:
        raise ValueError("incompatible original compressed trained matrix")
    count=rows*(dim//8)
    ids=unpack5(handle.packed.tobytes(),count)
    if len(ids)!=count or ids.min()<0 or ids.max()>31:
        raise ValueError("broken original 5bit group index decoder")
    recovered=(handle.centroids[ids].reshape(rows,dim)
               *handle.row_scales[:,None]).astype(np.float32)
    if recovered.shape!=handle.shape or not np.isfinite(recovered).all():
        raise ValueError("invalid decompressed source diagnostic")
    return np.ascontiguousarray(recovered,dtype=np.float32)


def build_unpaid_reference(blob):
    check()
    if len(blob)!=EXPECTED_BYTE_SIZE:
        raise ValueError("invalid original complete native source length")
    archive=CB59(blob)
    desc=archive.meta["tensor_descriptors"]
    matrix_paths=[k for k,v in desc.items() if v["dtype"]=="CB59_K32V8"]
    if len(matrix_paths)!=EXPECTED_MATRICES:
        raise ValueError("not all original pretrained 2D trained operators")
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    # Must not leave any of the randomly initialized source parameters.
    actual_state=set(model.state_dict().keys())
    expected_state=set(archive.meta["full_original_state_keys"])
    if actual_state!=expected_state:
        raise ValueError(f"source config/state mismatch: missing={sorted(expected_state-actual_state)[:8]}, extra={sorted(actual_state-expected_state)[:8]}")
    with torch.no_grad():
        for path in sorted(matrix_paths):
            parent,_,name=path.rpartition(".")
            owner=model.get_submodule(parent)
            parameter=getattr(owner,name)
            if not isinstance(parameter,torch.nn.Parameter) or parameter.ndim!=2:
                raise ValueError("decompress source into wrong module "+path)
            h=archive.handles[path]
            if tuple(parameter.shape)!=h.shape:
                raise ValueError("source trained operator shape changed")
            source=exact_dense_matrix(h)
            parameter.copy_(torch.from_numpy(source))
        for path in sorted(archive.raw):
            # Aliases of original 1D parameters are processed too, if any.
            parent,_,name=path.rpartition(".")
            owner=model.get_submodule(parent)
            parameter=getattr(owner,name)
            if not isinstance(parameter,torch.nn.Parameter):
                raise ValueError("source trained 1D tensor not parameter "+path)
            source=archive.raw[path]
            if tuple(parameter.shape)!=source.shape:
                raise ValueError("raw source 1D weight geometry changed")
            parameter.copy_(torch.from_numpy(np.array(source,copy=True)))
    # Tied token embedding and LM head must use the same physical tensor.
    if model.model.embed_tokens.weight.data_ptr()!=model.lm_head.weight.data_ptr():
        model.tie_weights()
    if model.model.embed_tokens.weight.data_ptr()!=model.lm_head.weight.data_ptr():
        raise AssertionError("unpaid reference broke original tied source")
    return model,{
        "UNPAID_DECOMPRESSION_ONLY_FOR_DEBUG":True,
        "dense_source_is_NOT_original_FP32_teacher":True,
        "dense_source_has_exact_CB59_codewords_not_original_weights":True,
        "full_recovered_trained_2D_matrices":len(matrix_paths),
        "original_physically_stored_bytes":len(blob),
        "source_TIED_embedding_head_one_storage":True,
        "never_deploy_or_count_dense_diagnostic_as_0p7bpp":True,
    }
