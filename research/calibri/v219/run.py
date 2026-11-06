"""CALIBRI v219 — source-only 5-bit REPRESENTATION trials for all 211 tensors.

The old CB59 32x8 K-means codebook uses a tiny 4096-vector initialization.
v217 fixed-index dictionary LS slightly lowered source reconstruction
error, but impaired QA; v218 partitioned only tied vocabulary decreased
head SSE 5.6% yet yielded 0/16. Instead compare two new all-tensor
quantizers: full K32V8 iterative group index reassignment, and product
K4xK8 over independent 4D subspaces (5 bits total). Include full-K32
head-only as an ablation for compatibility.

Key constraints:
 * NO QA/TRAIN/VALID/DEV/TEST used to design per-matrix quantizer or
   choose its source checkpoints: pretrained original FP32 weights ONLY.
 * The original authenticated native CB59 135M structure, packed 5bit
   groups, row scales, alias keys, raw 1D tensors, and 32x8 FP16 codebook
   geometry are preserved. No extra storage, no dense runtime, no C changes.
 * Product code indices refer to 32 CARTESIAN-product combinations of
   4 learned 4D values and 8 other 4D values. The expanded 32x8 table
   is actually stored in the original physical FP16 codebook, not
   reconstructed by an unaccounted decoder.
 * All source candidates SHA-sealed BEFORE any full-answer QA.
 * PUBLIC DEV16 is heavily reused, not independent. Don't select a
   production/checkpoint winner from public DEV scores.
 * Entire experiment research-only. No LIVE/main/Jessica/Render/APK.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59,unpack5
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v174.intervene import extract_original
from v174.protocol import freeze_cases
from v174.run import save
from v169.sources import all_cases
from v185.codec import replace_discrete
from v186.run import speed_pairs
from v196 import protocol as p
from v196.run import newly_frozen_public_dev_16
from v212.run import SOURCE_V206_SHA256

VOCAB="model.embed_tokens.weight"
VECTOR=8
K=32
N_MATRICES=211
SAMPLE=4096
INIT_ITER=5
FULL_ITER=2
BLOCK=32768
CANDIDATES=("fullK32_all","product4x8_all","fullK32_head_only")


def nearest(x,centers):
    # Batch-limited: at most 32768x32 temporary scores.
    x=np.asarray(x,dtype=np.float32)
    c=np.asarray(centers,dtype=np.float32)
    if x.ndim!=2 or c.ndim!=2 or x.shape[1]!=c.shape[1]:
        raise AssertionError("Invalid pretrained vector/center geometry")
    score=np.sum(c*c,axis=1)[None,:]-2.0*(x@c.T)
    return np.argmin(score,axis=1).astype(np.uint8)


def centroid_refit(vectors,scales,ids,centers,passes):
    """True FP32-original source weighted least squares on all groups."""
    out=np.asarray(centers,dtype=np.float32).copy()
    k,dim=out.shape
    if k<2 or k>32 or vectors.shape!=(len(ids),dim):
        raise AssertionError("Incomplete authentic original FP32 source groups")
    for _ in range(passes):
        sums=np.zeros((k,dim),dtype=np.float64)
        den=np.zeros(k,dtype=np.float64)
        for start in range(0,len(vectors),BLOCK):
            last=min(start+BLOCK,len(vectors))
            a=vectors[start:last]
            weights=np.asarray(scales[start:last],dtype=np.float32)
            chosen=nearest(a,out)
            sq=weights.astype(np.float64)**2
            den+=np.bincount(chosen,weights=sq,minlength=k)
            for d in range(dim):
                sums[:,d]+=np.bincount(
                    chosen,weights=a[:,d].astype(np.float64)*sq,
                    minlength=k)
        ok=den>1e-20
        out[ok]=(sums[ok]/den[ok,None]).astype(np.float32)
        out=out.astype(np.float16).astype(np.float32)
    return out


def train_subcode(vectors,weight,k):
    n=len(vectors)
    sample_idx=np.linspace(0,n-1,min(SAMPLE,n),dtype=np.int64)
    sample=vectors[sample_idx]
    start_idx=np.linspace(0,len(sample)-1,k,dtype=np.int64)
    centers=sample[start_idx].astype(np.float16).astype(np.float32)
    # Small initialization is deterministic and based on weights only.
    for _ in range(INIT_ITER):
        labels=nearest(sample,centers)
        for j in range(k):
            subset=sample[labels==j]
            if len(subset):
                centers[j]=subset.mean(axis=0)
        centers=centers.astype(np.float16).astype(np.float32)
    return centroid_refit(vectors,weight,None,centers,FULL_ITER)


def fit_matrix(handle,weight):
    m,n=handle.shape
    if n%VECTOR or tuple(weight.shape)!=(m,n):
        raise AssertionError("Wrong original 135M source geometry")
    scales=handle.row_scales.astype(np.float32)
    if np.any(scales<=0) or not np.isfinite(scales).all():
        raise AssertionError("Quantized source scales invalid")
    groups=n//VECTOR
    original=np.asarray(weight,dtype=np.float32).reshape(-1,VECTOR)
    repeats=np.repeat(scales,groups)
    normal=original/repeats[:,None]
    if not np.isfinite(normal).all():
        raise AssertionError("Original trained source weights contain NaN")
    old=unpack5(handle.packed.tobytes(),len(normal))
    before_centers=handle.centroids.astype(np.float32)
    def squared_error(codes,centers):
        # Full original weight SSE including frozen original physical row scaling.
        total=0.
        for start in range(0,len(normal),BLOCK):
            end=min(start+BLOCK,len(normal))
            r=(normal[start:end]-centers[codes[start:end]])*repeats[start:end,None]
            total+=float(np.sum(r.astype(np.float64)**2))
        return total
    original_sse=squared_error(old,before_centers)

    # Option A: standard K32V8 but reassign ALL 5-bit codes over original
    # weights and exact source-only FP16-centroid refits, not tiny QA steps.
    k32=train_subcode(normal,repeats,K)
    out32=np.empty(len(normal),dtype=np.uint8)
    for i in range(0,len(normal),BLOCK):
        out32[i:i+BLOCK]=nearest(normal[i:i+BLOCK],k32)
    k32_sse=squared_error(out32,k32)

    # Option B: two independent 4D codebooks with K4 and K8. This packs
    # a CARTESIAN basis (4*8=32) without changing the existing C decoder.
    left=train_subcode(normal[:,:4],repeats,4)
    right=train_subcode(normal[:,4:],repeats,8)
    product_centers=np.concatenate((
        np.repeat(left,8,axis=0),
        np.tile(right,(4,1))
    ),axis=1).astype("<f2").astype(np.float32)
    if product_centers.shape!=(K,VECTOR):
        raise AssertionError("Not a literal original physical K32V8 dictionary")
    product_ids=np.empty(len(normal),dtype=np.uint8)
    for i in range(0,len(normal),BLOCK):
        end=min(i+BLOCK,len(normal))
        a=nearest(normal[i:end,:4],left)
        b=nearest(normal[i:end,4:],right)
        product_ids[i:end]=(a*8+b).astype(np.uint8)
    product_sse=squared_error(product_ids,product_centers)
    return {
        "old_ids":old,"old_centers":before_centers,
        "k32_ids":out32,"k32_centers":k32,
        "product_ids":product_ids,"product_centers":product_centers,
        "original_source_SSE":original_sse,
        "fullK32_source_SSE":k32_sse,
        "product4x8_source_SSE":product_sse,
        "actual_FP32_weights_fitted":m*n
    }


def run(source,v161,v193,out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_V206_SHA256:
        raise AssertionError("Expected signed authentic v206 model")
    archive=CB59(raw)
    descriptors=archive.meta["tensor_descriptors"]
    names=[name for name in sorted(descriptors)
           if descriptors[name]["dtype"]=="CB59_K32V8"]
    if len(names)!=N_MATRICES or VOCAB not in names or (
        archive.handles["lm_head.weight"] is not archive.handles[VOCAB]):
        raise AssertionError("Original 211 tied K32V8 source incomplete")
    teacher=original_teacher().cpu().eval()
    state=teacher.state_dict()
    details={}
    edits={}
    for idx,name in enumerate(names):
        fitted=fit_matrix(archive.handles[name],
                          state[name].detach().cpu().numpy())
        details[name]={k:fitted[k] for k in (
            "original_source_SSE","fullK32_source_SSE",
            "product4x8_source_SSE","actual_FP32_weights_fitted")}
        edits[name]={k:fitted[k] for k in (
            "old_ids","old_centers","k32_ids","k32_centers",
            "product_ids","product_centers")}
        if (idx+1)%20==0 or name==VOCAB:
            print(json.dumps({"stage":"v219_relearn_codeword_source_only",
                "completed_original_135m_tensors":idx+1,
                "all_tensors":N_MATRICES,
                "name":name,
                "old_SSE":round(details[name]["original_source_SSE"],2),
                "k32_SSE":round(details[name]["fullK32_source_SSE"],2),
                "pq_SSE":round(details[name]["product4x8_source_SSE"],2)}),flush=True)
    del teacher,state
    totals={label:sum(row[label] for row in details.values())
            for label in ("original_source_SSE","fullK32_source_SSE",
                          "product4x8_source_SSE")}
    report={
        "protocol":"CALIBRI_v219_full_original_pretrained_source_5bit_K32_vs_product4x8",
        "original_135m_unique_parameters":p.FROZEN_SOURCE_135M_UNIQUE,
        "SOURCE_v206_sha256":SOURCE_V206_SHA256,
        "original_trained_2D_matrices":len(names),
        "relearned_real_FP32_weight_count":sum(x["actual_FP32_weights_fitted"] for x in details.values()),
        "global_SOURCE_SSE":totals,
        "per_matrix_original_FP32_source_only_error":details,
        "precommitted_arms_BEFORE_DEV":list(CANDIDATES),
        "no_QA_TRAIN_VALID_DEV_TEST_used_to_train_5bit_codes":True,
        "same_original_5bit_index_len_FP16_dictionary_shape_native_kernel":True,
        "no_new_full_model_bytes":True,
        "no_MAIN_LIVE_Jessica_Render_Supabase_APK":True,
        "source_arms":{}
    }
    for arm in CANDIDATES:
        source_objects={}
        for name,tensor in edits.items():
            if arm.startswith("fullK32"):
                if arm=="fullK32_all" or name==VOCAB:
                    codes=tensor["k32_ids"];cent=tensor["k32_centers"]
                else:
                    codes=tensor["old_ids"];cent=tensor["old_centers"]
            elif arm=="product4x8_all":
                codes=tensor["product_ids"];cent=tensor["product_centers"]
            else:raise AssertionError("Unknown precommitted arm")
            source_objects[name]=SimpleNamespace(
                codes=torch.from_numpy(np.array(codes,copy=True)),
                centroids=torch.from_numpy(np.array(cent,copy=True)))
        packed,seal=replace_discrete(archive,source_objects,strict=False)
        if len(packed)!=p.SOURCE_BYTES or seal["complete_bpp"]>0.700:
            raise AssertionError("No source rate increase permitted")
        if seal["sha256"]==SOURCE_V206_SHA256:
            raise AssertionError("New representation did not change v206 source")
        name="CALIBRI_v219_"+arm+"_all_paid_5bit.cb59"
        (out/name).write_bytes(packed)
        report["source_arms"][arm]={
            "file":name,"source_SHA256":seal["sha256"],
            "physical_bpp":seal["complete_bpp"],
            "5bit_codes_CHANGED":seal["actual_changed_5bit_group_indices"],
            "FP16_codebook_cells_CHANGED":seal["actually_changed_FP16_codebook_cells"],
            "sealed_before_QA":True
        }
    save(out/"SEALED_v219_product_quantizer_and_K32_BEFORE_ANY_QA.json",report)
    print(json.dumps({"stage":"v219_signed_full_physical_before_QA",
        "precommitted":report["source_arms"],
        "original_to_K32_source_SSE_ratio":
            totals["fullK32_source_SSE"]/max(totals["original_source_SSE"],1.e-20),
        "original_to_product_source_SSE_ratio":
            totals["product4x8_source_SSE"]/max(totals["original_source_SSE"],1.e-20)}),flush=True)

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    initial=extract_original(Path(v161))
    if len(initial.blob)!=p.SOURCE_BYTES:
        raise AssertionError("Missing original 5bit native speed control")
    with TemporaryDirectory(dir=out) as td:
        root=Path(td)
        for name in ("CB59","CB206",*CANDIDATES):(root/name).mkdir()
        (root/"CB59"/"source.cb59").write_bytes(initial.blob)
        cb59,_,_=make_model("baseline",SimpleNamespace(
            original=root/"CB59"/"source.cb59",folder=root/"CB59"))
        cb206,_,_=make_model("baseline",SimpleNamespace(
            original=Path(source),folder=root/"CB206"))
        models={"CB206":cb206}
        for arm in CANDIDATES:
            new,_,_=make_model("baseline",SimpleNamespace(
                original=out/report["source_arms"][arm]["file"],
                folder=root/arm))
            if any(p.ndim>=2 for p in new.parameters()):
                raise AssertionError("Unpaid dense original weights in C inference")
            models[arm]=new
        qa=newly_frozen_public_dev_16(models,tok,v193,out)
        if qa["scores"]["CB206"]["machine_confirmed"]!=3:
            raise AssertionError("Authentic original v206 3/16 baseline drift")
        speed={}
        for arm in CANDIDATES:
            measured=speed_pairs({"CB59":cb59,"CB186":models[arm]},
                                 tok,freeze_cases(all_cases()))
            speed[arm]=measured["median_speed_ratio_vs_CB59"]
    result={
        "completed":True,
        "original_FP32_135m_publicDEV_correct":qa["scores"]["FP32_existing_135M"]["machine_confirmed"],
        "baseline_v206_publicDEV_correct":qa["scores"]["CB206"]["machine_confirmed"],
        "v219_publicDEV16_scores":{
            arm:{"correct":qa["scores"][arm]["machine_confirmed"],
                 "wrong":qa["scores"][arm]["machine_recognized_wrong"],
                 "ungraded":qa["scores"][arm]["ungraded_not_known_wrong"],
                 "bpp":report["source_arms"][arm]["physical_bpp"],
                 "native_CPU_speed_CB59":speed[arm]}
            for arm in CANDIDATES},
        "source_errors":totals,
        "real_F32_original_weight_ONLY_fitted_codebooks":True,
        "public_DEV16_exploratory_REPEATED_NOT_independent_TEST":True,
        "no_QA_dependent_architecture_or_checkpoint_selection":True,
        "no_MAIN_LIVE_Jessica_Render_Supabase_APK":True
    }
    save(out/"v219_REAL_native_K32_vs_product4x8_QA_CPU.json",result)
    print(json.dumps({"stage":"v219_real_native_source_QA",**result}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("v206",type=Path)
    ap.add_argument("v161",type=Path)
    ap.add_argument("v193",type=Path)
    ap.add_argument("output",type=Path)
    opts=ap.parse_args()
    run(opts.v206,opts.v161,opts.v193,opts.output)
