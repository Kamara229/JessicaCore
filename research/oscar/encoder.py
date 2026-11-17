"""Oscar: standalone offline FP32 -> fully physical CALIBRI CB59 converter.

Oscar accepts genuinely external FP32 state dictionaries, with or without a
pretrained CALIBRI CB59 dictionary prior. The prior contributes ONLY row
scales/centroid basis in warm modes, never old packed indices or 1D state.
The decoder at inference needs ONLY the newly produced CB59 binary; no
teacher, original source, sidecar or prior is hidden in its dependencies.

The neural 12-layer CALIBRI Green Tunnel and other learned mechanisms belong
to a SEPARATE, explicitly paid physical inference package. This codec does
not claim that raw FP32 re-quantization alone preserves model quality.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping
import zlib

import numpy as np
import torch

from direct_codebook_v59 import CB59, HEADER, encode, pack5, unpack5

GROUP = 8
CODEBOOK = 32
MODE_COLD = "cold"
MODE_LOCKED = "prior_locked"
MODE_ADAPT = "prior_adapt"
ALLOWED_MODES = (MODE_COLD, MODE_LOCKED, MODE_ADAPT)
MAX_BPP = .7


@dataclass(frozen=True)
class OscarResult:
    blob: bytes
    report: dict

    def authenticated_decoder(self) -> CB59:
        if hashlib.sha256(self.blob).hexdigest() != self.report["source_sha256"]:
            raise AssertionError("OSCAR full physical source SHA mismatch")
        return CB59(self.blob)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fp32_source_digest(state: Mapping[str,torch.Tensor],
                        canonical: list[str]) -> str:
    h = hashlib.sha256()
    for name in canonical:
        tensor = state[name].detach().cpu().contiguous()
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Oscar expects finite genuine FP32 model tensors: " + name)
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().astype("<f4", copy=False).tobytes())
    return h.hexdigest()


def _exact_nearest(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Raw float32 8D L2 argmin, same deterministic tie-break as original S7B.

    This intentionally is NOT CB59's snapped integer K-means distance.
    Re-quantizing a pretrained model using the wrong group metric was a
    real S7A quality failure. Keep chunk memory bounded to [1024,32,8].
    """
    if vectors.ndim!=2 or vectors.shape[1]!=GROUP or centroids.shape!=(CODEBOOK,GROUP):
        raise ValueError("Oscar original CB59 vector 8D/K32 geometry mismatch")
    chosen = np.empty((vectors.shape[0],), dtype=np.uint8)
    for start in range(0,len(vectors),1024):
        stop = min(start+1024,len(vectors))
        diff = (vectors[start:stop,None,:]-centroids[None,:,:]).astype(np.float32)
        dist = np.einsum("nki,nki->nk",diff,diff,optimize=True)
        if not np.isfinite(dist).all():
            raise ValueError("Oscar nonfinite FP32 codebook distance")
        chosen[start:stop] = np.argmin(dist,axis=1).astype(np.uint8)
    return chosen


def _trained_prior_refit(w:np.ndarray, scales:np.ndarray, prior_centroids:np.ndarray,
                         *,sample_count:int,steps:int,anchor:float):
    """Fit FP16 dictionary to truly NEW FP32 matrix; old indices never read."""
    m,n=w.shape
    groups=n//GROUP
    total=m*groups
    indices=np.linspace(0,total-1,min(sample_count,total),dtype=np.int64)
    r=indices//groups
    g=indices%groups
    v=(w[r[:,None],g[:,None]*GROUP+np.arange(GROUP)[None,:]] /
       scales[r,None]).astype(np.float32)
    if not np.isfinite(v).all():
        raise ValueError("Oscar source contains nonfinite normalized samples")
    centers=prior_centroids.astype(np.float32).copy()
    # Stay in a small neighborhood of the successfully TRAINED dictionaries;
    # this is a prior, not inherited dense teacher inference parameters.
    for _ in range(steps):
        codes=_exact_nearest(v,centers)
        count=np.bincount(codes,minlength=CODEBOOK)
        acc=np.zeros((CODEBOOK,GROUP),dtype=np.float64)
        np.add.at(acc,codes,v.astype(np.float64))
        occupied=count>0
        means=centers.copy()
        means[occupied]=(acc[occupied]/count[occupied,None]).astype(np.float32)
        centers=(anchor*prior_centroids+(1.-anchor)*means).astype("<f2").astype(np.float32)
        if not np.isfinite(centers).all():
            raise ValueError("Oscar FP16 new matrix center became nonfinite")
    return centers


def _encode_against_dictionary(w:np.ndarray, scales:np.ndarray,
                               centroids:np.ndarray, *,
                               block_rows:int=128):
    if w.ndim!=2 or w.shape[1]%GROUP:
        raise ValueError("Oscar 2D original model shape incompatible with CB59 K32/V8")
    m,n=w.shape
    groups=n//GROUP
    if scales.shape!=(m,) or centroids.shape!=(CODEBOOK,GROUP):
        raise ValueError("Oscar learned dictionary prior shape mismatch")
    if np.any(scales<=0) or not np.isfinite(scales).all():
        raise ValueError("Oscar learned row scales are not positive/finite")
    codes=np.empty((m*groups,),dtype=np.uint8)
    sse=0.
    energy=0.
    max_error=0.
    for start in range(0,m,block_rows):
        stop=min(start+block_rows,m)
        chunk=w[start:stop].reshape(-1,GROUP)
        scale=np.repeat(scales[start:stop],groups).astype(np.float32)
        normalized=(chunk/scale[:,None]).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("Oscar nonfinite original model FP32 group")
        ids=_exact_nearest(normalized,centroids)
        codes[start*groups:stop*groups]=ids
        reconstruct=centroids[ids]*scale[:,None]
        diff=(chunk-reconstruct).astype(np.float64)
        sse+=float(np.sum(diff*diff))
        energy+=float(np.sum(chunk.astype(np.float64)**2))
        max_error=max(max_error,float(np.max(np.abs(diff))))
    if not energy>0:
        raise ValueError("Oscar cannot encode entirely zero original matrix")
    return codes,{"relative_FP32_weight_RMSE":math.sqrt(sse/energy),
                  "squared_FP32_weight_error":sse,
                  "original_FP32_energy":energy,
                  "max_abs_FP32_weight_error":max_error,
                  "original_8weight_group_count":m*groups}


def _ensure_compatible(prior:CB59,source:CB59):
    # New state is ALWAYS serialized independently: original model's 1D,
    # full source aliases, and source SHA must come from the FP32 input.
    a=prior.meta
    b=source.meta
    for k in ("full_original_state_keys","trained_unique_keys","source_alias_keys"):
        if a[k]!=b[k]:
            raise ValueError("Oscar prior's original state/aliases are not compatible with FP32")
    if a["original_unique_parameter_count"]!=b["original_unique_parameter_count"]:
        raise ValueError("Oscar prior and original FP32 unique parameter count differ")
    for name,desc in b["tensor_descriptors"].items():
        prior_desc=a["tensor_descriptors"].get(name)
        if (prior_desc is None or prior_desc["shape"]!=desc["shape"]
            or prior_desc["dtype"]!=desc["dtype"]):
            raise ValueError("Oscar prior matrix geometry must match new FP32: "+name)


def encode_fp32(state: Mapping[str,torch.Tensor], *,
                mode:str=MODE_COLD,
                prior_blob:bytes|None=None,
                adapt_samples:int=4096,adapt_steps:int=5,
                prior_anchor:float=0.15) -> OscarResult:
    """Convert an independent *complete* FP32 state_dict to authenticated CB59.

    Oscar can only convert into an existing same-geometry transformer state.
    It does not reshape 30-layer pretrained weights into 12 layers. Independent
    model quality requires later distillation & held-out comparisons.

    * cold: source-only 32x8 5bit fit, no pretrained CALIBRI required.
    * prior_locked: use learned prior's FP16 centers AND scales, infer ALL
      fivebit codes from new FP32 input (S7B lossless same-source control).
    * prior_adapt: start dictionary with learned prior; fresh input-specific
      FP16 row scales + updated 32x8 centers, re-infer ALL codes.
    """
    if mode not in ALLOWED_MODES:
        raise ValueError("Unknown Oscar conversion mode: "+str(mode))
    if mode!=MODE_COLD and prior_blob is None:
        raise ValueError("Oscar warm mode requires a complete physical CB59 dictionary prior")
    if not (isinstance(adapt_samples,int) and adapt_samples>=32 and
            isinstance(adapt_steps,int) and 1<=adapt_steps<=100 and
            0<=prior_anchor<=1):
        raise ValueError("Invalid Oscar deterministic TRAIN-free source fitting hyperparameters")
    if not state:
        raise ValueError("Oscar FP32 state cannot be empty")
    # Always start with the NEW FP32 input. Never use old prior's 1D/aliases.
    cold_blob,_,cold_diagnostics=encode(state)
    original=CB59(cold_blob)
    canonical=list(original.meta["trained_unique_keys"])
    source_sha=_fp32_source_digest(state,canonical)
    if mode==MODE_COLD:
        return OscarResult(cold_blob,{
            "algorithm":"Oscar","version":0,
            "mode":mode,
            "source_sha256":_sha(cold_blob),
            "FP32_input_full_state_sha256":source_sha,
            "complete_core_bytes":len(cold_blob),
            "original_unique_parameters":original.meta["original_unique_parameter_count"],
            "core_only_bpp":8*len(cold_blob)/original.meta["original_unique_parameter_count"],
            "matrix_count":sum(x["dtype"]=="CB59_K32V8" for x in
                               original.meta["tensor_descriptors"].values()),
            "FP32_RMSE":cold_diagnostics["full_source_weighted_relative_RMSE"],
            "all_new_1d_from_source":True,
            "prior_used":False,
            "fp32_source_is_from_pretrained_independent_model_UNVERIFIED":True,
            "full_inference_with_GT_and_sidecar_unverified":True,
            "model_quality_proven":False,
        })
    prior=CB59(prior_blob)
    _ensure_compatible(prior,original)
    # Fresh archive is structurally identical to user's actual input state.
    # Rewrite only physical matrix packed codes, FP16 row scales & K32/V8.
    raw=bytearray(cold_blob)
    _,_,mlen,_=HEADER.unpack_from(raw)
    payload_start=HEADER.size+mlen
    matrix_stats={}
    changes=0
    for name in canonical:
        desc=original.meta["tensor_descriptors"][name]
        if desc["dtype"]!="CB59_K32V8":
            continue
        w=state[name].detach().cpu().numpy().astype(np.float32,copy=False)
        p=prior.handles[name]
        f=original.handles[name]
        if mode==MODE_LOCKED:
            scales=p.row_scales.copy()
            centers=p.centroids.copy()
        else:
            scales=f.row_scales.copy()
            centers=_trained_prior_refit(w,scales,p.centroids,
                      sample_count=adapt_samples,steps=adapt_steps,
                      anchor=prior_anchor)
        code,stats=_encode_against_dictionary(w,scales,centers)
        packed=pack5(code)
        if len(packed)!=len(f.packed):
            raise AssertionError("Oscar physical 5bit group count or padded width drift")
        changed_parts={
           "packed":packed,
           "row_scales":scales.astype("<f2").tobytes(),
           "centroids":centers.astype("<f2").tobytes(),
        }
        for part,value in changed_parts.items():
            pos,length=desc["parts"][part]
            if len(value)!=length:
                raise AssertionError("Oscar changing source 5bit group geometry: "+part)
            raw[payload_start+pos:payload_start+pos+length]=value
        stats["shape"]=list(w.shape)
        stats["actual_prior_dictionary_used"]=True
        stats["prior_row_scales_reused"]=mode==MODE_LOCKED
        stats["FP32_input_specific_row_scales"]=mode==MODE_ADAPT
        stats["old_prior_5bit_indices_read_to_encode"]=False
        matrix_stats[name]=stats
        changes+=len(code)
    raw[-36:-32]=struct.pack("<I",zlib.crc32(raw[:-36]))
    raw[-32:]=hashlib.sha256(raw[:-32]).digest()
    completed=bytes(raw)
    decoded=CB59(completed)
    if (decoded.meta!=original.meta or len(completed)!=len(cold_blob)
        or sum(x["dtype"]=="CB59_K32V8" for x in
                decoded.meta["tensor_descriptors"].values())!=len(matrix_stats)):
        raise AssertionError("Oscar failed authentic FULL CB59 physical reserialize")
    # 1D vectors are ALL from the supplied independent FP32 input; not prior.
    for name in canonical:
        if name in original.raw and not np.array_equal(
            decoded.raw[name],original.raw[name]):
            raise AssertionError("Oscar failed to preserve new FP32 1D tensors")
    if "lm_head.weight" in decoded.handles:
        if (decoded.handles["lm_head.weight"] is not
            decoded.handles["model.embed_tokens.weight"]):
            raise AssertionError("Oscar full original pretrained tied embedding lost")
    sse=sum(x["squared_FP32_weight_error"] for x in matrix_stats.values())
    energy=sum(x["original_FP32_energy"] for x in matrix_stats.values())
    report={
        "algorithm":"Oscar","version":0,"mode":mode,
        "source_sha256":_sha(completed),
        "FP32_input_full_state_sha256":source_sha,
        "prior_SHA256":_sha(prior_blob),
        "complete_core_bytes":len(completed),
        "original_unique_parameters":original.meta["original_unique_parameter_count"],
        "core_only_bpp":8*len(completed)/original.meta["original_unique_parameter_count"],
        "complete_2D_matrices":len(matrix_stats),
        "new_original_8weight_groups_encoded":changes,
        "all_old_packed_prior_indices_excluded":True,
        "all_new_1d_from_FP32_source":True,
        "FP32_RMSE":math.sqrt(sse/max(energy,1e-30)),
        "actual_all_matrix_weight_errors":matrix_stats,
        "FP16_centroids_and_scales_physically_serialized":True,
        "all_full_source_aliases_and_CRC_SHA_restored":True,
        "complete_model_GT_sidecar_and_runtime_bpp_must_be_checked_separately":True,
        "full_original_trained_FP32_language_quality_proven":False,
        "independent_FP32_source_provenance_requires_external_checkpoint":True,
    }
    return OscarResult(completed,report)


def to_cb61(source:bytes)->bytes:
    """Generic OSCAR CB61 zlib envelope for any original 12- or 30-layer CB59.

    Historical S87 loader accepts only its fixed 36M physical size, which is
    intentional. Oscar's standalone packer must also accept 135M physical
    sources without silently changing 135M model geometry.
    """
    parsed=CB59(source)
    if parsed.blob!=source:
        raise AssertionError("Oscar authenticated full source mismatch")
    header=struct.Struct("<8sQI32s")
    packed=(header.pack(b"CB61ZL01",len(source),zlib.crc32(source),
                        bytes.fromhex(_sha(source)))+
            zlib.compress(source,level=9))
    magic,n,crc,sha=header.unpack_from(packed)
    if magic!=b"CB61ZL01" or n!=len(source):
        raise AssertionError("Oscar CB61 source size/header mismatch")
    d=zlib.decompressobj()
    unpacked=d.decompress(packed[header.size:],n+1)
    if (not d.eof or d.unused_data or d.unconsumed_tail or
        unpacked!=source or zlib.crc32(unpacked)!=crc or
        hashlib.sha256(unpacked).digest()!=sha):
        raise AssertionError("Oscar CB61 full physical CRC/SHA roundtrip corrupt")
    CB59(unpacked)
    return packed
