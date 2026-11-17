"""OSCAR v0.1: independent actual FP32 -> CB59 K32V8 conversion.

Two modes:
  cold: original scratch deterministic codebooks (never assumes a pretrained
        source will retain quality).
  inherited: original *trained* FP16 dictionaries/row scales from a prior,
        but EVERY 5bit group index is computed afresh from dense FP32 values.
        Original index bytes are NEVER read during forward quantization.

This codec is architecture agnostic (compatible tensor names/shapes in the
prior) and supports any decoder depth. It does NOT magically remap 30 layers
to 12 or preserve quality of independently pretrained FP32.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import struct
import zlib
from typing import Mapping

import numpy as np
import torch

from direct_codebook_v59 import CB59, HEADER, encode, pack5, unpack5

VECTOR=8
GROUP_BITS=5
K=32
BLOCK_ROWS=128


def sha(b: bytes)->str:
    return hashlib.sha256(b).hexdigest()


def _original_state(source: Mapping[str, torch.Tensor]):
    if not isinstance(source,Mapping) or not source:
        raise ValueError("OSCAR requires explicit nonempty original FP32 state_dict")
    for key,value in source.items():
        if (not isinstance(key,str) or not isinstance(value,torch.Tensor)
            or value.dtype!=torch.float32 or value.ndim not in (1,2)
            or not torch.isfinite(value).all()):
            raise ValueError("OSCAR requires finite original FP32 1D/2D tensors: "+str(key))
    return source


def _nearest(v: np.ndarray, centers: np.ndarray, *, chunk=512):
    """Bounded strict Euclidean 8D FP32 nearest-centroid search.

    Explicit row-level 32×8 floating differences match the preexisting
    trained CALIBRI codebook inverse, avoiding dot-product cancellation
    when decoder weights lie exactly at centroids.
    """
    vectors=np.asarray(v,dtype=np.float32)
    c=np.asarray(centers,dtype=np.float32)
    if vectors.ndim!=2 or vectors.shape[1]!=8 or c.shape!=(32,8):
        raise ValueError("OSCAR requires 8D source vectors and 32x8 codebooks")
    out=np.empty((len(vectors),),dtype=np.uint8)
    for start in range(0,len(vectors),chunk):
        z=vectors[start:start+chunk]
        diff=z[:,None,:]-c[None,:,:]
        dist=np.sum(diff*diff,axis=2,dtype=np.float32)
        if not np.isfinite(dist).all():
            raise ValueError("OSCAR nonfinite original FP32 weighted distance")
        out[start:start+len(z)]=np.argmin(dist,axis=1).astype(np.uint8)
    return out


def _inherited_matrix(fp32: np.ndarray, handle, *, tolerance=None):
    rows,cols=handle.shape
    x=np.asarray(fp32,dtype=np.float32)
    if (x.shape!=(rows,cols) or cols%VECTOR or
        not np.all(np.isfinite(x))):
        raise ValueError("OSCAR FP32 tensor incompatible with inherited codebook")
    centers=handle.centroids.astype(np.float32)
    scales=handle.row_scales.astype(np.float32)
    groups=cols//VECTOR
    if (scales.shape!=(rows,) or not np.isfinite(scales).all()
        or np.any(scales<=0) or centers.shape!=(K,VECTOR)):
        raise ValueError("OSCAR cannot use corrupt inherited row scales or dictionary")
    # No usage of original h.packed in assignment, only a byte LENGTH check.
    packed_parts=[]
    sqerr=0.
    energy=0.
    max_abs=0.
    num_codes=0
    for start in range(0,rows,BLOCK_ROWS):
        end=min(start+BLOCK_ROWS,rows)
        f=np.repeat(scales[start:end],groups)
        block=x[start:end].reshape(-1,VECTOR)
        projected=block/f[:,None]
        ids=_nearest(projected,centers)
        reconstructed=centers[ids]*f[:,None]
        difference=reconstructed.astype(np.float64)-block.astype(np.float64)
        sqerr+=float(np.square(difference).sum())
        energy+=float(np.square(block.astype(np.float64)).sum())
        max_abs=max(max_abs,float(np.abs(difference).max(initial=0)))
        packed_parts.append(pack5(ids))
        num_codes+=len(ids)
    packed=b"".join(packed_parts)
    # 43M and 135M widths guaranteed multiple of 8. For arbitrary matrices
    # pad index serialization at matrix scope only; internal blocks of 128
    # rows are byte aligned. This codec intentionally follows CB59 layout.
    if (len(packed)!=len(handle.packed) or
        num_codes!=rows*cols//VECTOR or not math.isfinite(sqerr)):
        raise AssertionError("OSCAR original physical code-stream ABI mismatch")
    if tolerance is not None and max_abs>tolerance:
        raise AssertionError("OSCAR mirror inversion exceeds declared tolerance")
    return packed,{
        "weight_max_abs":max_abs,
        "weight_sse":sqerr,
        "source_energy":energy,
        "relative_rmse":math.sqrt(sqerr/max(energy,1e-30)),
        "number_of_5bit_code_groups":num_codes,
    }


def _verify_prior(original, prior:CB59):
    canonical=prior.meta["tensor_descriptors"]
    if set(original)!=set(prior.meta["full_original_state_keys"]):
        raise ValueError("OSCAR original pretrained state tensor keys mismatch learned dictionary")
    total=0
    for name in canonical:
        tensor=original[name]
        shape=tuple(canonical[name]["shape"])
        if tuple(tensor.shape)!=shape:
            raise ValueError("OSCAR FP32 tensor shape incompatibility: "+name)
        total+=int(tensor.numel())
    if total!=prior.meta["original_unique_parameter_count"]:
        raise AssertionError("OSCAR model unique-parameter count drift")
    for alias,key in prior.meta["source_alias_keys"].items():
        a=original[alias]
        b=original[key]
        # Important: some transformers emit equal-valued but NOT tied
        # FP32 tensors. Explicit fail: alias semantics matter to inference.
        if (a.shape!=b.shape or not torch.equal(a,b)
            or a.untyped_storage().data_ptr()!=b.untyped_storage().data_ptr()):
            raise ValueError("OSCAR source FP32 broke original tied parameter alias: "+alias)
    return sorted(n for n,v in canonical.items() if v["dtype"]=="CB59_K32V8")


def _patch_original(prior: CB59, packed_code: Mapping[str,bytes],
                    updated_raw: Mapping[str,bytes])->bytes:
    """Replace only paid original source segments; same header and metadata.

    Never embeds a second decoded copy of original tensor weights or
    inherited original 5bit group indices.
    """
    modified=bytearray(prior.blob)
    _,_,meta_len,_=HEADER.unpack_from(modified)
    body=HEADER.size+meta_len
    descriptor=prior.meta["tensor_descriptors"]
    if set(packed_code)!={n for n in descriptor if descriptor[n]["dtype"]=="CB59_K32V8"}:
        raise ValueError("OSCAR must assign every original 2D operator")
    if set(updated_raw)!={n for n in descriptor if descriptor[n]["dtype"]=="f4"}:
        raise ValueError("OSCAR must explicitly account every original FP32 1D vector")
    for name in sorted(descriptor):
        dtype=descriptor[name]["dtype"]
        segment="packed" if dtype=="CB59_K32V8" else "raw"
        data=packed_code[name] if segment=="packed" else updated_raw[name]
        offset,length=descriptor[name]["parts"][segment]
        if len(data)!=length:
            raise AssertionError("OSCAR cannot omit or oversize physical source segment")
        modified[body+offset:body+offset+length]=data
    modified[-36:-32]=struct.pack("<I",zlib.crc32(modified[:-36]))
    modified[-32:]=hashlib.sha256(modified[:-32]).digest()
    fresh=bytes(modified)
    reread=CB59(fresh)
    if (len(fresh)!=len(prior.blob)
        or reread.meta!=prior.meta
        or set(reread.handles)!=set(prior.handles)):
        raise AssertionError("OSCAR physical source alias or metadata corruption")
    for name in packed_code:
        a,b=prior.handles[name],reread.handles[name]
        if (not np.array_equal(a.row_scales,b.row_scales)
            or not np.array_equal(a.centroids,b.centroids)):
            raise AssertionError("OSCAR inherited trained centroid or row scale modified")
    return fresh


@dataclass
class OscarResult:
    mode:str
    raw_cb59:bytes
    source_unique_params:int
    layer_count:int | None
    original_was_derived_from_calibri:bool | None
    matrices:int=0
    code_groups:int=0
    stats:dict=field(default_factory=dict)
    @property
    def sha256(self):
        return sha(self.raw_cb59)
    @property
    def physical_bytes(self):
        return len(self.raw_cb59)
    @property
    def bpp(self):
        return 8*len(self.raw_cb59)/self.source_unique_params
    def summary(self):
        return {
            "algorithm":"OSCAR-v0.1",
            "source":self.mode,
            "source_real_FP32_only":True,
            "original_was_derived_from_calibri":self.original_was_derived_from_calibri,
            "tied_original_state_preserved":True,
            "source_unique_params":self.source_unique_params,
            "decoder_layers":self.layer_count,
            "full_original_2D_matrices":self.matrices,
            "all_5bit_vector_codes_assigned_from_FP32":True,
            "original_historical_5bit_codes_copied":False,
            "code_groups":self.code_groups,
            "full_source_sha256":self.sha256,
            "full_original_CB59_bytes":self.physical_bytes,
            "original_core_only_bpp":self.bpp,
            "model_language_quality_established":False,
            "independent_FP32_pretraining_parity_established":False,
            "no_NRA_added":True,
            "individual_original_fp32_matrix_statistics":self.stats,
        }


class OscarConverter:
    """One reusable non-architecture-specific quantizer.

    Modes:
      inherited: use a previously *trained* quantization dictionary, but
                 compute NEW indices from actual FP32, validate every alias.
      cold: use original standalone K32V8 source-only K-means from FP32,
            for any compatible architecture; pretrained parity NOT presumed.
    """
    def __init__(self,*,mode:str,prior:CB59 | None=None,
                 tolerance:float | None=None):
        if mode not in ("inherited","cold"):
            raise ValueError("OSCAR v0.1 supports inherited and cold only")
        if (mode=="inherited") != (prior is not None):
            raise ValueError("OSCAR inherited mode needs a real CB59 dictionary prior")
        self.mode=mode
        self.prior=prior
        self.tolerance=tolerance

    def convert(self,original_fp32: Mapping[str,torch.Tensor],
                *,layer_count:int|None=None,
                derived_from_calibri:bool|None=None)->OscarResult:
        source=_original_state(original_fp32)
        if self.mode=="cold":
            cb,meta,diag=encode(source)
            parsed=CB59(cb,expected_keys=source.keys())
            count=sum(d["dtype"]=="CB59_K32V8"
                      for d in meta["tensor_descriptors"].values())
            codes=sum(int(np.prod(parsed.handles[n].shape))//8
                      for n,d in meta["tensor_descriptors"].items()
                      if d["dtype"]=="CB59_K32V8")
            return OscarResult(
                mode="cold",raw_cb59=cb,
                source_unique_params=meta["original_unique_parameter_count"],
                layer_count=layer_count,
                original_was_derived_from_calibri=derived_from_calibri,
                matrices=count,code_groups=codes,
                stats={"original_FP32_K32V8_source_only_RMSE":
                       diag["full_source_weighted_relative_RMSE"]})
        prior=self.prior
        matrices=_verify_prior(source,prior)
        codes={}
        diag={}
        groups=0
        for name in matrices:
            tensor=source[name].detach().cpu().numpy().astype(np.float32,copy=False)
            packed,record=_inherited_matrix(tensor,prior.handles[name],
                                            tolerance=self.tolerance)
            codes[name]=packed
            diag[name]=record
            groups+=record["number_of_5bit_code_groups"]
        vectors={}
        for name,d in prior.meta["tensor_descriptors"].items():
            if d["dtype"]=="f4":
                vectors[name]=source[name].detach().cpu().numpy().astype("<f4").tobytes()
        raw=_patch_original(prior,codes,vectors)
        return OscarResult(
            mode="inherited",raw_cb59=raw,
            source_unique_params=prior.meta["original_unique_parameter_count"],
            layer_count=layer_count,
            original_was_derived_from_calibri=derived_from_calibri,
            matrices=len(matrices),code_groups=groups,stats=diag)
