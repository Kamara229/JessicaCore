"""Re-encode original 211 2D tensors at EXACT physical CB59 byte length.

Unlike fixed Euclidean source-RMSE v170, this trial uses diagonal
source-ACTIVATION input energy to weight source weight vector errors:
    approx E_x[(W - Q) x]^2.
The full block covariance and nonlinear teacher behavior are NOT solved,
and no pretraining parity is presumed. This changes physically stored
model code indices and centroids, NOT QA scoring or postprocess strings.
"""
from __future__ import annotations
import hashlib,struct,zlib
import numpy as np
from direct_codebook_v59 import CB59,HEADER,pack5
from v117.protocol import ORIGINAL_CB59_SHA,MAX_SOURCE_BYTES
from .protocol import WEIGHT_SAMPLE_GROUPS,PASSES,CHUNK_GROUPS,check


def weighted_nearest(points,centers,weights):
    x=np.asarray(points,dtype=np.float32)
    c=np.asarray(centers,dtype=np.float32)
    w=np.asarray(weights,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=8 or c.shape!=(32,8) or w.shape!=x.shape:
        raise ValueError("incorrect 8D anisotropic original source code")
    if not np.isfinite(x).all() or not np.isfinite(c).all() or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError("nonpositive or invalid source input energy")
    # Vectorized quadratic metric: one row of per-feature original-input
    # importance per 8 original weights; O(N*32*8), no 32x8xN tensor.
    costs=(np.sum(w*x*x,axis=1,dtype=np.float64)[:,None]
           -2.0*((w*x).astype(np.float64)@c.astype(np.float64).T)
           +w.astype(np.float64)@(c.astype(np.float64)**2).T)
    return np.argmin(costs,axis=1).astype(np.uint8)


def activation_fit_one(weight,handle,input_energy):
    trained=np.asarray(weight,dtype=np.float32)
    if trained.shape!=handle.shape or trained.ndim!=2 or trained.shape[1]%8:
        raise ValueError("wrong original source matrix")
    rows,columns=handle.shape
    importance=np.asarray(input_energy,dtype=np.float32)
    if importance.shape!=(columns,) or not np.isfinite(importance).all() or np.any(importance<=0):
        raise ValueError("not original feature-wise operator input energy")
    if not np.isfinite(trained).all():
        raise ValueError("not pretrained finite FP32 source")
    scales=np.asarray(handle.row_scales,dtype=np.float32)
    if len(scales)!=rows or np.any(scales<=0):
        raise ValueError("wrong pretrained row scales")
    n_group=columns//8
    total=rows*n_group
    if total%8:
        raise ValueError("group alignment would add hidden padding bits")
    vectors=trained.reshape(-1,8)
    feature=importance.reshape(n_group,8)
    ids=np.linspace(0,total-1,min(total,WEIGHT_SAMPLE_GROUPS),dtype=np.int64)
    samples=(vectors[ids]/scales[ids//n_group,None]).astype("<f2").astype(np.float32)
    weights=feature[ids%n_group].astype(np.float32).copy()
    median=float(np.median(scales))
    if median<=0:raise ValueError("zero row energy")
    row_factor=np.clip((scales[ids//n_group]/median)**2,0.1,10.0)
    weights*=row_factor[:,None]
    init=np.asarray(handle.centroids,dtype=np.float32)
    centers=init.copy()
    before=weighted_nearest(samples,init,weights)
    before_cost=float(np.sum(weights*(samples-init[before])**2,dtype=np.float64))
    # Each 8D codebook centroid is the diagonal-activation-weighted
    # conditional mean of original FP32 source vectors.
    for _ in range(PASSES):
        chosen=weighted_nearest(samples,centers,weights).astype(np.int64)
        sums=np.zeros((32,8),dtype=np.float64)
        denominators=np.zeros((32,8),dtype=np.float64)
        np.add.at(sums,chosen,samples.astype(np.float64)*weights.astype(np.float64))
        np.add.at(denominators,chosen,weights.astype(np.float64))
        occupied=denominators[:,0]>0
        centers[occupied]=(sums[occupied]/denominators[occupied]).astype("<f2").astype(np.float32)
        if not np.isfinite(centers).all():
            raise ValueError("infeasible original source codebook")
    centers=centers.astype("<f2").astype(np.float32)
    after=weighted_nearest(samples,centers,weights)
    after_cost=float(np.sum(weights*(samples-centers[after])**2,dtype=np.float64))
    # Descend in the actual source input-weighted loss; otherwise do not
    # replace an original operator without evidence. This is NOT an answer
    # quality gate, and no source labels or logits guide fitting.
    if after_cost>before_cost+1e-5*max(before_cost,1):
        centers=init.copy()
        after_cost=before_cost
    chunks=[]
    for start in range(0,total,CHUNK_GROUPS):
        stop=min(total,start+CHUNK_GROUPS)
        groups=np.arange(start,stop,dtype=np.int64)
        s=scales[groups//n_group]
        v=(vectors[start:stop]/s[:,None]).astype("<f2").astype(np.float32)
        w=feature[groups%n_group]
        codes=weighted_nearest(v,centers,w)
        chunks.append(pack5(codes))
    packed=b"".join(chunks)
    if len(packed)!=len(handle.packed):
        raise ValueError("activation-aware recode grew original 5bit payload")
    return centers.astype("<f2").tobytes(),packed,{
        "matrix_input_energy_features":len(importance),
        "samples_FIT_original_pretrained_FP32_weights":len(ids),
        "reencoded_original_weight_groups":total,
        "FIT_activation_weighted_vector_cost_before":before_cost,
        "FIT_activation_weighted_vector_cost_after":after_cost,
        "exact_original_5bit_storage":True,
        "no_heldout_QA_or_teacher_output_targets":True,
    }


def recode(original,teacher,source_activation_energy):
    check()
    if hashlib.sha256(original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("activation-aware experiment must start from pristine original source")
    cb=CB59(original)
    full=teacher.state_dict()
    packed=bytearray(original)
    _,_,mlen,_=HEADER.unpack_from(original)
    base=HEADER.size+mlen
    fit={}
    allowed=[]
    actually_changed=0
    for name,desc in sorted(cb.meta["tensor_descriptors"].items()):
        if desc["dtype"]!="CB59_K32V8":continue
        key=name[:-len(".weight")]
        operator="lm_head" if key=="model.embed_tokens" else key
        if operator not in source_activation_energy:
            raise ValueError("original FP32 activation not captured for "+operator)
        if name not in full:
            raise ValueError("original FP32 pretrained source missing "+name)
        source=full[name].detach().cpu().float().numpy()
        table,codes,stat=activation_fit_one(
            source,cb.handles[name],source_activation_energy[operator])
        for segment,value in (("packed",codes),("centroids",table)):
            offset,size=desc["parts"][segment]
            if len(value)!=size:
                raise ValueError("changed original physical matrix bit budget")
            first=base+offset
            if packed[first:first+size]!=value:
                actually_changed+=1
            packed[first:first+size]=value
            allowed.append((first,first+size))
        fit[name]=stat
    if len(fit)!=211:
        raise ValueError("not all 211 original full pretrained 2D matrices re-encoded")
    old=original[:-36]
    new=bytes(packed[:-36])
    cursor=0
    for start,stop in sorted(allowed):
        if new[cursor:start]!=old[cursor:start]:
            raise ValueError("noncodebook original trained state changed")
        cursor=stop
    if new[cursor:]!=old[cursor:]:
        raise ValueError("original physical tensor changed outside 5bit/centroid fields")
    packed[-36:-32]=struct.pack("<I",zlib.crc32(new))
    packed[-32:]=hashlib.sha256(packed[:-32]).digest()
    result=bytes(packed)
    if len(result)!=len(original) or len(result)>MAX_SOURCE_BYTES:
        raise ValueError("unpaid new neural bits")
    checked=CB59(result)
    if checked.meta!=cb.meta:
        raise ValueError("changed original alias/manifest")
    if checked.handles["lm_head.weight"] is not checked.handles["model.embed_tokens.weight"]:
        raise AssertionError("tied original FP32 pretrained model duplicated")
    return result,{
        "source_2D_operator_count":len(fit),
        "actual_5bit_or_FP16_codebook_segments_changed":actually_changed,
        "source_original_bytes":len(original),
        "new_complete_persisted_bytes":len(result),
        "exact_same_all_source_bits":len(original)==len(result),
        "candidate_sha256":hashlib.sha256(result).hexdigest(),
        "activation_weighted_fit_diagnostic_not_ANSWER":fit,
        "NO_quality_score_from_KL_NLL":True,
    }
