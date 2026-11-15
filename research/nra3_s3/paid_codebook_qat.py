"""NRA2 3.0 S3: train CB59 physically-paid centroids/scales, never dense export.

Research-only optimizer bridge. All 5-bit K32V8 indices remain packed and
fixed, every 2D tensor remains CB59, exact source byte length is preserved.
Dense weight views exist ONLY as differentiable temporary TRAIN tensors;
released CPU native inference consumes the same compact CB59 C operators.

This is a numerical TRAIN harness, NOT a Russian grammatical/semantic
training result. Real textual optimization is not enabled: it first needs
sealed historical2048+new TRAIN SHA allowlist and disjoint denied EVAL IDs.
No original CALIBRI, S3CI sixth, spent five EVALs or Jessica chat is accessed.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from direct_codebook_v59 import CB59, unpack5
from v176.codec import replace_fields


def _check(condition, reason):
    if not condition:
        raise ValueError("NRA3 S3 QAT: "+reason)


def _ste_fp16(x:torch.Tensor)->torch.Tensor:
    """Use actual FP16 storage values in TRAIN forward, keep gradients."""
    rounded=x.to(torch.float16).to(torch.float32)
    return x+(rounded-x).detach()


def train_sha_for(label: str)->str:
    if not isinstance(label,str) or not label.startswith("SYNTHETIC_NRA3_S3_TRAIN:"):
        raise ValueError("S3 permits synthetic TRAIN-only identity, never texts")
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SyntheticTrainingGuard:
    """Strict per-step SHA membership; no general TRAIN source accepted yet."""
    allowed_sha256: frozenset[str]
    forbidden_sha256: frozenset[str]

    def __post_init__(self):
        if not self.allowed_sha256 or self.allowed_sha256 & self.forbidden_sha256:
            raise ValueError("S3 training source overlap or empty allowlist")
        all_keys=self.allowed_sha256|self.forbidden_sha256
        if any(len(k)!=64 or any(c not in "0123456789abcdef" for c in k)
               for k in all_keys):
            raise ValueError("S3 malformed immutable SHA identity")

    def admit_optimizer_step(self,examples:tuple[str,...])->None:
        if not examples or len(examples)!=len(set(examples)):
            raise ValueError("S3 duplicates or empty training examples")
        for label in examples:
            digest=train_sha_for(label)
            if digest in self.forbidden_sha256 or digest not in self.allowed_sha256:
                raise ValueError("S3 optimizer tried unadmitted synthetic TRAIN hash")


class CB59PaidCodebook(nn.Module):
    """One physically coded 2D operator, with FP16 STE centroids and row scales."""
    def __init__(self,source:CB59,name:str):
        super().__init__()
        if name not in source.handles:
            raise ValueError("S3 cannot train non-CB59/non-2D source operator")
        h=source.handles[name]
        rows,cols=h.shape
        _check(cols%8==0 and rows>=1,"unsupported 8D code shape")
        codes=unpack5(h.packed.tobytes(),rows*(cols//8))
        self.register_buffer("codes",torch.from_numpy(codes.astype(np.int64)
                                                   .reshape(rows,cols//8).copy()))
        self.register_buffer("base_scales",torch.from_numpy(
            np.array(h.row_scales,copy=True)).float())
        self.centroids=nn.Parameter(torch.from_numpy(
            np.array(h.centroids,copy=True)).float())
        self.log_row_gain=nn.Parameter(torch.zeros(rows,dtype=torch.float32))
        self.original_packed_codes=bytes(h.packed)
        self.name=name
        self.rows=rows
        self.cols=cols
        if not bool((self.base_scales>0).all()):
            raise ValueError("S3 invalid zero source row scale")

    def fields(self):
        center=_ste_fp16(self.centroids)
        gain=self.log_row_gain.clamp(-4.0,4.0).exp()
        row=_ste_fp16(self.base_scales*gain)
        if not bool((row>0).all()) or not bool(torch.isfinite(center).all()):
            raise ValueError("S3 nonfinite codebook values")
        return center,row

    def training_weight(self):
        center,row=self.fields()
        return center[self.codes].reshape(self.rows,self.cols)*row[:,None]

    def forward(self,x):
        if (not isinstance(x,torch.Tensor) or x.dtype!=torch.float32
                or x.device.type!="cpu" or x.shape[-1]!=self.cols):
            raise ValueError("S3 quantized step requires CPU float32 input")
        return F.linear(x,self.training_weight())

    def save_paid_fields(self)->tuple[np.ndarray,np.ndarray]:
        centers,row=self.fields()
        return (np.ascontiguousarray(centers.detach().cpu().numpy(),
                                    dtype=np.float16),
                np.ascontiguousarray(row.detach().cpu().numpy(),
                                    dtype=np.float16))


def repack_paid_single_operator(original:bytes, name:str,
                                trained:CB59PaidCodebook)->bytes:
    """Rewrites ONLY pre-existing FP16 fields via Calibri v176 paid codec.

    Validates full physical source, unmodified exact packed index bytes,
    all other original source weights, meta, and overall file size.
    """
    _check(isinstance(original,bytes),"physical source bytes required")
    source=CB59(original)
    _check(name==trained.name and name in source.handles,
           "exact trained tensor identity drift")
    handle=source.handles[name]
    _check(handle.shape==(trained.rows,trained.cols) and
           handle.packed.tobytes()==trained.original_packed_codes,
           "original source code IDs changed unexpectedly")
    centers,scales=trained.save_paid_fields()
    values={}
    for key,desc in source.meta["tensor_descriptors"].items():
        if desc["dtype"]=="CB59_K32V8":
            src=source.handles[key]
            values[key]={
                "centroids": centers if key==name else src.centroids,
                "row_scales": scales if key==name else src.row_scales,
            }
        elif desc["dtype"]=="f4":
            values[key]={"raw":source.raw[key]}
        else:
            raise ValueError("S3 unsupported physical source tensor")
    # Verified codec updates ONLY already-present source byte segments,
    # including actual checksum rewrite, and preserves all original indices.
    physical=replace_fields(source,values)
    _check(len(physical)==len(original),"physical byte budget grew")
    restored=CB59(physical,expected_keys=source.meta["full_original_state_keys"])
    _check(restored.meta==source.meta,"manifest changed")
    for key in source.meta["tensor_descriptors"]:
        if key in source.handles:
            before=source.handles[key];after=restored.handles[key]
            _check(before.packed.tobytes()==after.packed.tobytes(),
                   "discrete index change forbidden")
            if key!=name:
                _check(np.array_equal(before.centroids,after.centroids) and
                       np.array_equal(before.row_scales,after.row_scales),
                       "unselected original matrix drift")
        elif not np.array_equal(source.raw[key],restored.raw[key]):
            raise ValueError("S3 protected original vector drift")
    _check(np.allclose(restored.handles[name].centroids,centers.astype(np.float32),
                       atol=0,rtol=0) and
           np.allclose(restored.handles[name].row_scales,scales.astype(np.float32),
                       atol=0,rtol=0),"FP16 physical codebook roundtrip error")
    return physical


@dataclass(frozen=True)
class SyntheticNumericalReport:
    before_train_mse:float
    after_train_mse:float
    after_physical_reload_mse:float
    actual_source_bytes_before:int
    actual_source_bytes_after:int
    source_sha_before:str
    source_sha_after:str
    optimizer_steps:int
    changed_centroids_only:bool
    no_real_language_training:bool=True
    output_gate:int=0
    candidate_accepted:bool=False


def train_one_synthetic_codebook(physical_cb59:bytes, operator_name:str,
                                 x_train:torch.Tensor,
                                 y_train:torch.Tensor,
                                 guard:SyntheticTrainingGuard,
                                 source_label:str,
                                 *,steps:int=60,lr:float=.04):
    """Genuine gradient steps update packed-source codebook, not dense release.

    TRAIN x/y must be synthetic tensors prepared outside, never gold from
    reserved EVAL/QA. This function cannot access raw textual data at all.
    """
    _check(1<=steps<=1000 and 0<lr<=0.25,"unsafe optimizer schedule")
    _check(x_train.device.type=="cpu" and
           x_train.dtype==torch.float32 and
           y_train.dtype==torch.float32 and y_train.device.type=="cpu" and
           x_train.ndim==2 and y_train.ndim==2 and
           x_train.shape[0]==y_train.shape[0],"synthetic TRAIN tensor ABI")
    source=CB59(physical_cb59)
    m=CB59PaidCodebook(source,operator_name).cpu().train()
    _check(x_train.shape[1]==m.cols and y_train.shape[1]==m.rows,
           "TRAIN matrix dimensions differ from physical source")
    opt=torch.optim.Adam(m.parameters(),lr=lr)
    with torch.no_grad():
        before=float(F.mse_loss(m(x_train),y_train))
    for step in range(steps):
        guard.admit_optimizer_step((source_label,))
        opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(m(x_train),y_train)
        _check(bool(torch.isfinite(loss)),"nonfinite TRAIN loss")
        loss.backward()
        for p in m.parameters():
            _check(p.grad is not None and bool(torch.isfinite(p.grad).all()),
                   "missing QAT codebook gradient")
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step()
    with torch.no_grad():
        after=float(F.mse_loss(m(x_train),y_train))
    saved=repack_paid_single_operator(physical_cb59,operator_name,m)
    reloaded=CB59PaidCodebook(CB59(saved),operator_name).cpu().eval()
    with torch.no_grad():
        last=float(F.mse_loss(reloaded(x_train),y_train))
    _check(math.isfinite(last),"physical reload changed to invalid source")
    report=SyntheticNumericalReport(
        before_train_mse=before,
        after_train_mse=after,
        after_physical_reload_mse=last,
        actual_source_bytes_before=len(physical_cb59),
        actual_source_bytes_after=len(saved),
        source_sha_before=hashlib.sha256(physical_cb59).hexdigest(),
        source_sha_after=hashlib.sha256(saved).hexdigest(),
        optimizer_steps=steps,
        changed_centroids_only=False,
    )
    return saved,report
