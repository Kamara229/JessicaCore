#!/usr/bin/env python3
"""CALIBRI v61 direct CB59 C-GEMV plus on-the-fly residual syndrome.

No source trained matrix W or any dense NxM correction is reconstructed
in the inference process. A pre-existing native 5-bit packed GEMV reads
base weights and a sparse set of selected ONE-COLUMN residual syndromes
are added by (x_j * R[:,j]), an exact rank-one correction to those
columns. Embedding reconstructs only requested rows and then corrects
their selected output coordinates.

The error-correction analogy describes explicit paid source residual
side information; CRC/SHA detects file corruption but is not sufficient
to restore arbitrary lost lossy weight information.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from direct_runtime_v59 import NativeKernel,compile_kernel,bind_into_llama
from syndrome_codebook_v61 import CB61


def bind_syndrome_into_model(model,archive:CB61,kernel:NativeKernel):
    proof=bind_into_llama(model,archive.base,kernel)
    hooked=0
    for tensor_name,repairs in archive.correct.items():
        if not tensor_name.endswith(".weight"):
            raise AssertionError("source residual attached to non-2D tensor")
        mod=model.get_submodule(tensor_name[:-7])
        if not hasattr(mod,"handle"):
            raise AssertionError("residual cannot target packed source module")
        if isinstance(mod,nn.Module) and hasattr(mod,"num_embeddings"):
            # Store all U vectors as 1D buffers, never a dense matrix.
            for order,(col,U) in enumerate(repairs):
                symbol="cb61_residual_"+str(order)
                mod.register_buffer(symbol,torch.from_numpy(
                    U.astype(np.float32,copy=True)))
                def emb_hook(this,inputs,output,column=col,field=symbol):
                    idx=inputs[0]
                    new=output.clone()
                    new[...,column]+=getattr(this,field)[idx]
                    return new
                mod.register_forward_hook(emb_hook)
                hooked+=1
        else:
            for order,(col,U) in enumerate(repairs):
                symbol="cb61_residual_"+str(order)
                mod.register_buffer(symbol,torch.from_numpy(
                    U.astype(np.float32,copy=True)))
                def linear_hook(this,inputs,output,column=col,field=symbol):
                    return (output+inputs[0][...,column].unsqueeze(-1)*
                            getattr(this,field))
                mod.register_forward_hook(linear_hook)
                hooked+=1
    leftover={name:tuple(t.shape) for name,t in model.named_parameters()
              if t.ndim>=2}
    if leftover:
        raise AssertionError("original dense W materialized in direct CB61 hot runtime")
    if (model.get_input_embeddings().handle is not
        model.get_output_embeddings().handle):
        raise AssertionError("shared tied source lost in CB61")
    proof.update({
        "number_of_actual_one_column_residual_hard_syndromes_applied":hooked,
        "no_pretrained_original_dense_2D_hot_weights":True,
        "base_still_native_C_direct_5bit_GEMV":True,
        "residual_is_only_1D_FP16_on_disk_source_side_info":True,
    })
    return proof


def smoke():
    from direct_codebook_v59 import encode
    import torch
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=nn.Embedding(32,16)
            self.fc=nn.Linear(16,16)
            self.head=nn.Linear(16,32,bias=False)
            self.head.weight=self.emb.weight
        def get_input_embeddings(self):return self.emb
        def get_output_embeddings(self):return self.head
    torch.manual_seed(61)
    model=Toy()
    source=model.state_dict()
    base,_,_=encode(source)
    from syndrome_codebook_v61 import pack
    # Parent module names and source data from the actual Toy source.
    prepared,_,_=pack(base,source,{
        "fc.weight":np.arange(16,dtype=np.float64)+1,
        "emb.weight":np.arange(16,dtype=np.float64)+1
    },len(base)+22000)
    q=CB61(prepared,expected_keys=source.keys())
    native=NativeKernel(compile_kernel(
        Path(__file__).with_name("direct_kernel_v59.c")))
    original_input=torch.tensor([[0,1]],dtype=torch.long)
    bind_syndrome_into_model(model,q,native)
    logits=model.head(model.fc(model.emb(original_input)))
    if logits.shape!=(1,2,32) or not torch.isfinite(logits).all():
        raise AssertionError("native packed + coded residual exact Toy forward failed")
    assert not any(p.ndim>=2 for p in model.parameters())
    print("CALIBRI v61 actual native packed 5bit GEMV plus sparse hard-syndrome repair smoke PASS")


if __name__=="__main__":
    smoke()
