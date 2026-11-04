"""Frozen original source, no teacher logits, train hierarchical 1-bit residual."""
import hashlib
import numpy as np
from direct_codebook_v59 import unpack5
from v137.protocol import select,groups
from v139.refinement import train
from v139.protocol import RESIDUAL_BITS
from .codec import pack_bits

FROZEN_V139_SHA="a22dd9c69fe32d09d81aac6c36866fe1f9a46274cf0ac0e2b46d13419fe5b9bf"


def fit_from_original_pretrained(weight, handle):
    if weight.shape!=(49_152,576) or handle.shape!=weight.shape:
        raise ValueError("mismatched original tied source")
    total=weight.size//8
    if total!=RESIDUAL_BITS:
        raise ValueError("wrong number of tied 8D vector groups")
    sample,_=select(total)
    _,normalized,scales=groups(weight,handle.row_scales,sample)
    codes=unpack5(handle.packed.tobytes(),total)
    center=train(normalized,codes[sample],handle.centroids,
                 energy=scales.astype(np.float64)**2,steps=8)
    digest=hashlib.sha256(center.astype("<f2").tobytes()).hexdigest()
    if digest!=FROZEN_V139_SHA:
        raise AssertionError("v139 TRAIN fitted residual centroids changed: "+digest)
    return center,codes


def assign_full_extra_bits(weight,handle,centers,codes,*,batch_groups=16384):
    w=np.asarray(weight,dtype=np.float32)
    flat=w.reshape(-1,8)
    if len(flat)!=len(codes) or batch_groups<8 or batch_groups%8:
        raise ValueError("wrong 5-bit source assignment count")
    row_scales=np.asarray(handle.row_scales,dtype=np.float32)
    original=np.asarray(handle.centroids,dtype=np.float32)
    residual=np.asarray(centers,dtype=np.float32)
    if original.shape!=(32,8) or residual.shape!=(32,2,8):
        raise ValueError("wrong hierarchic codebook")
    bits=np.empty(len(codes),dtype=np.uint8)
    source_squared=old_squared=new_squared=0.0
    per_row=w.shape[1]//8
    for start in range(0,len(codes),batch_groups):
        stop=min(start+batch_groups,len(codes))
        scalars=row_scales[np.arange(start,stop,dtype=np.int64)//per_row]
        target=flat[start:stop]
        normalized=(target/scalars[:,None]).astype("<f2").astype(np.float32)
        code=codes[start:stop].astype(np.int64)
        base=original[code]
        gap=normalized-base
        pair=residual[code]
        distances=np.sum((gap[:,None,:]-pair)**2,axis=2,dtype=np.float64)
        bit=np.argmin(distances,axis=1).astype(np.uint8)
        bits[start:stop]=bit
        base_output=base*scalars[:,None]
        refined_output=(base+pair[np.arange(len(bit)),bit])*scalars[:,None]
        e0=(target.astype(np.float64)-base_output.astype(np.float64))**2
        e1=(target.astype(np.float64)-refined_output.astype(np.float64))**2
        source_squared+=float(np.sum(target.astype(np.float64)**2))
        old_squared+=float(np.sum(e0))
        new_squared+=float(np.sum(e1))
    physical=pack_bits(bits)
    return physical,{
        "full_source_embedding_RMSE_CB59":float(np.sqrt(old_squared/source_squared)),
        "full_source_embedding_RMSE_6bit":float(np.sqrt(new_squared/source_squared)),
        "full_source_onebit_counts":[int(np.count_nonzero(bits==i)) for i in (0,1)],
        "independent_full_source_groups":len(codes),
        "all_source_weights_counted":True,
    }
