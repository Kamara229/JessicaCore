"""Full original-operator SAME-BYTE recoder using actual pretrained FP32 weights.

All 2D original matrices retain K32x8, 5bit per vector, same FP16 row
scales, exactly the same 512B per-matrix codebook and original manifest.
The native CB59 C inference kernel is unchanged. This does NOT attach an
oracle model or uncharged residual, prompt-conditioned output prior, or GT.

Unlike the old 4096-vector global sampling of v59, fit 32768 source groups
per matrix with Lloyd centroids. Hypothesis, NOT claimed answer improvement.
"""
from __future__ import annotations
import hashlib,struct,zlib
import numpy as np
from direct_codebook_v59 import (
    CB59,HEADER,pack5,_nearest_integer_128)
from v137.fit import lloyd
from v117.protocol import ORIGINAL_CB59_SHA,MAX_SOURCE_BYTES
from .protocol import FIT_SAMPLES_PER_MATRIX,LLOYD_PASSES,CHUNK_GROUPS,check


def fit_one(weight,old_handle):
    source=np.asarray(weight,dtype=np.float32)
    if source.shape!=old_handle.shape or source.ndim!=2 or source.shape[1]%8:
        raise ValueError("changed trained matrix geometry")
    if not np.isfinite(source).all():
        raise ValueError("source original trained FP32 includes nonfinite weights")
    rows,cols=old_handle.shape
    groups=cols//8
    count=rows*groups
    if count%8:
        raise ValueError("must pack exactly same byte count without padding")
    vectors=source.reshape(count,8)
    scales=np.asarray(old_handle.row_scales,dtype=np.float32)
    if np.any(scales<=0) or not np.isfinite(scales).all():
        raise ValueError("invalid original row scales")
    sample_ids=np.linspace(0,count-1,min(count,FIT_SAMPLES_PER_MATRIX),dtype=np.int64)
    sample=scales[sample_ids//groups]
    points=(vectors[sample_ids]/sample[:,None]).astype("<f2").astype(np.float32)
    original=np.asarray(old_handle.centroids,dtype=np.float32)
    fitted=lloyd(points,original,passes=LLOYD_PASSES)
    fitted=fitted.astype("<f2").astype(np.float32)
    if fitted.shape!=(32,8) or not np.isfinite(fitted).all():
        raise ValueError("wrong physical source codebook dimensions")
    result=[]
    distortion=0.
    src_energy=0.
    for start in range(0,count,CHUNK_GROUPS):
        end=min(count,start+CHUNK_GROUPS)
        s=scales[np.arange(start,end,dtype=np.int64)//groups]
        v=vectors[start:end]
        normalized=(v/s[:,None]).astype("<f2").astype(np.float32)
        codes=_nearest_integer_128(normalized,fitted).astype(np.uint8)
        result.append(pack5(codes))
        approx=fitted[codes]*s[:,None]
        error=v.astype(np.float64)-approx.astype(np.float64)
        distortion+=float(np.sum(error*error,dtype=np.float64))
        src_energy+=float(np.sum(v.astype(np.float64)**2,dtype=np.float64))
    packed=b"".join(result)
    if len(packed)!=len(old_handle.packed):
        raise ValueError("source-only refit changed 5bit original group bytes")
    return fitted.astype("<f2").tobytes(),packed,{
        "trained_FP32_source_groups":int(len(sample_ids)),
        "reencoded_5bit_groups":int(count),
        "source_weight_relative_RMSE":float(np.sqrt(distortion/max(src_energy,1e-30))),
        "same_exact_original_matrix_bytes":True,
    }


def reencode_full(original,teacher):
    check()
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("unapproved old source: do not rewrite model families")
    archive=CB59(original)
    raw=bytearray(original)
    magic,version,manifest_len,body_len=HEADER.unpack_from(original)
    if magic!=b"CB59" or version!=1:
        raise ValueError("incompatible base physical source")
    body_start=HEADER.size+manifest_len
    full_state=teacher.state_dict()
    changed=[]
    source_records={}
    allowed=[]
    # Use unique canonical trained names only. Tied embedding and lm_head
    # remain ONE physical 2D dictionary and shared handle in CPU runtime.
    for name in sorted(archive.meta["tensor_descriptors"]):
        info=archive.meta["tensor_descriptors"][name]
        if info["dtype"]!="CB59_K32V8":
            continue
        tensor=full_state.get(name)
        if tensor is None:
            raise ValueError("pretrained source matrix not found: "+name)
        weight=tensor.detach().cpu().float().numpy()
        centers,codes,stats=fit_one(weight,archive.handles[name])
        if len(centers)!=32*8*2:
            raise AssertionError("centroid payload no longer FP16 32x8")
        for part,value in (("packed",codes),("centroids",centers)):
            offset,length=info["parts"][part]
            if length!=len(value):
                raise ValueError("new source codebook payload would increase bpp")
            pos=body_start+offset
            old=bytes(raw[pos:pos+length])
            raw[pos:pos+length]=value
            allowed.append((pos,pos+length))
            if old!=value:
                changed.append((name,part))
        source_records[name]=stats
    if len(source_records)<200:
        raise AssertionError("not full original 30-layer pretrained model")
    # Verify no original 1D, row scales, aliases or shape metadata changed.
    original_body=original[:-36]
    new_body=bytes(raw[:-36])
    cursor=0
    for low,high in sorted(allowed):
        if new_body[cursor:low]!=original_body[cursor:low]:
            raise AssertionError("unaccounted source metadata/scale modification")
        cursor=high
    if new_body[cursor:]!=original_body[cursor:]:
        raise AssertionError("original raw source changed outside codewords/centers")
    raw[-36:-32]=struct.pack("<I",zlib.crc32(new_body))
    raw[-32:]=hashlib.sha256(raw[:-32]).digest()
    candidate=bytes(raw)
    if len(candidate)!=len(original) or len(candidate)>MAX_SOURCE_BYTES:
        raise ValueError("CB170 introduced an unpaid physical bit")
    if hashlib.sha256(candidate).hexdigest()==ORIGINAL_CB59_SHA:
        raise ValueError("no source codebook actually updated")
    verified=CB59(candidate)
    if verified.meta!=archive.meta:
        raise AssertionError("new source silently changed tied aliases or manifest")
    for name in source_records:
        if verified.handles[name].shape!=archive.handles[name].shape:
            raise AssertionError("different source layer weight shape")
    if verified.handles["model.embed_tokens.weight"] is not verified.handles["lm_head.weight"]:
        raise ValueError("unpaid duplicate embedding/head")
    return candidate,{
        "original_base_bytes":len(original),
        "new_complete_source_bytes":len(candidate),
        "exact_same_physical_bpp":len(candidate)==len(original),
        "original_immutable_source_sha":ORIGINAL_CB59_SHA,
        "new_same_byte_source_sha":hashlib.sha256(candidate).hexdigest(),
        "number_original_2D_matrices_reencoded":len(source_records),
        "number_code_and_dictionary_segments_changed":len(changed),
        "only_original_source_FP32_weights_for_refit":True,
        "no_gold_answers_or_logit_optimizers_or_KL_NLL":True,
        "per_matrix_source_weight_diagnostic_NOT_QUALITY":source_records,
    }
