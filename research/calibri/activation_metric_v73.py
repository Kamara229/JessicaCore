#!/usr/bin/env python3
"""CALIBRI v73: activation-covariance-aware 8D codec at ZERO extra storage bits.

Replace selected ORIGINAL 5-bit K32V8 indices using only WikiText TRAIN
teacher module inputs. Compare diagonal vs full 8x8 covariance with fixed
shrinkage. Select per-tensor improvement on TRAIN layer responses only.
Never use TEST data in encoding/selection. Rebuild exact full 135M source,
CRC/SHA, native C kernel and complete tied model unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,math,struct,time,zlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import HfApi
from direct_codebook_v59 import CB59,HEADER,pack5,unpack5
from quality import DATASET,load_token_ids
import binary_geometry_v64 as v64

MODEL=v64.MODEL
MODEL_SHA=v64.MODEL_SHA
N=v64.EXPECTED_UNIQUE
TRAIN_TOKENS=256
ROWS_PER_TENSOR=256
SHRINKAGE=0.25
TRAIN_IMPROVEMENT_THRESHOLD=0.005
ORIGINAL_BYTES=11229303
METHODS=("baseline","diag","full8")

def train_activations(model,modules,ids):
    samples={k:[] for k in modules}
    handles=[]
    for name,mod in modules.items():
        def capture(m,inp,*,key=name):
            samples[key].append(inp[0].detach().reshape(-1,inp[0].shape[-1]).cpu().numpy().astype(np.float32))
        handles.append(mod.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            for j in range(0,len(ids),64):
                model(input_ids=ids[j:j+64][None],use_cache=False)
    finally:
        for h in handles:h.remove()
    result={k:np.concatenate(v,axis=0) for k,v in samples.items()}
    if not all(x.shape[0]==TRAIN_TOKENS for x in result.values()):
        raise AssertionError("missing teacher TRAIN activation inputs")
    return result

def sample_rows(m):
    return np.unique(np.linspace(0,m-1,ROWS_PER_TENSOR,dtype=np.int64))

def reconstruct_rows(codes, h, rows):
    m,n=h.shape
    vecs=h.centroids[codes[rows]].reshape(len(rows),n)
    return vecs*h.row_scales[rows,None]

def norm_response_error(X,W,Q):
    # Restrict to teacher TRAIN observations and frozen selected rows.
    target=X@W.T
    pred=X@Q.T
    return float(np.linalg.norm((target-pred).astype(np.float64))/
                 max(np.linalg.norm(target.astype(np.float64)),1e-30))

def reassign(W,h,X,rows,method):
    """Optimize q_g with quadratic source response metric on trained X_g.

    Pairwise 8D covariance differs from v66's whole-row residual feedback:
    v73 makes independent vector assignments under an empirical Gram matrix,
    keeping the stored K32 FP16 dictionary/FP16 per-row scales FIXED.
    Cross-group covariance intentionally omitted; meaningful negative control.
    """
    m,n=W.shape
    G=n//8
    base=unpack5(h.packed.tobytes(),m*G).reshape(m,G)
    result=base.copy()
    if method=="baseline":return result
    C=h.centroids.astype(np.float64)
    scales=h.row_scales[rows].astype(np.float64)
    if np.any(scales<=0) or C.shape!=(32,8):raise AssertionError("bad original packed codebook")
    for g in range(G):
        Xa=X[:,g*8:(g+1)*8].astype(np.float64)
        cov=(Xa.T@Xa)/len(Xa)
        if method=="diag":
            cov=np.diag(np.diag(cov))
        elif method=="full8":
            cov=(1-SHRINKAGE)*cov+SHRINKAGE*np.diag(np.diag(cov))
        else:raise ValueError(method)
        # Preserve class K32 and all source row scale metadata unchanged.
        original=W[rows,g*8:(g+1)*8].astype(np.float64)/scales[:,None]
        diff=original[:,None,:]-C[None,:,:]
        costs=np.einsum("rki,ij,rkj->rk",diff,cov,diff,optimize=True)
        result[rows,g]=np.argmin(costs,axis=1).astype(np.uint8)
    if np.any(result[np.setdiff1d(np.arange(m),rows)]!=
              base[np.setdiff1d(np.arange(m),rows)]):
        raise AssertionError("modified unsampled source rows")
    return result

def toy():
    rng=np.random.default_rng(73)
    m,n=256,32
    scales=np.ones(m,dtype=np.float32)
    centers=rng.normal(size=(32,8)).astype(np.float32)*0.05
    from types import SimpleNamespace
    src=rng.integers(0,32,size=(m,n//8),dtype=np.uint8)
    h=SimpleNamespace(shape=(m,n),centroids=centers,row_scales=scales,
                      packed=np.frombuffer(pack5(src.reshape(-1)),dtype=np.uint8))
    W=centers[src].reshape(m,n)+rng.normal(size=(m,n)).astype(np.float32)*0.01
    X=rng.normal(size=(256,n)).astype(np.float32)
    rows=sample_rows(m)
    for method in METHODS:
        q=reassign(W,h,X,rows,method)
        assert q.shape==src.shape and len(pack5(q.reshape(-1)))==len(h.packed)
        assert np.array_equal(unpack5(pack5(q.reshape(-1)),q.size).reshape(q.shape),q)
    print("CALIBRI v73 toy 8D covariance K32 5bit invariant PASS",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--source",type=Path)
    ap.add_argument("--candidate",type=Path)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    toy()
    if args.smoke:return
    if args.source is None or args.candidate is None or args.output is None:
        ap.error("source, candidate, output required")
    start=time.monotonic()
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    original=args.source.read_bytes()
    cb=CB59(original)
    if len(original)!=ORIGINAL_BYTES or cb.meta["original_unique_parameter_count"]!=N:
        raise AssertionError("wrong genuine physical original source")
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).eval()
    modules=v64.choose_module_names(model)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    revision=HfApi().dataset_info(DATASET).sha
    train=load_token_ids(tok,"train",revision,TRAIN_TOKENS)
    # No TEST / VALID data is ever opened by this stage.
    activations=train_activations(model,modules,train)
    _,_,mlen,_=HEADER.unpack_from(original)
    offset=HEADER.size+mlen
    blob=bytearray(original)
    evidence=[]
    edits=0
    for name,mod in modules.items():
        key=name+".weight"
        h=cb.handles[key]
        W=mod.weight.detach().cpu().numpy().astype(np.float32)
        m,n=W.shape
        if tuple(h.shape)!=(m,n):raise AssertionError("source tensor mismatch")
        rows=sample_rows(m)
        X=activations[name]
        src=unpack5(h.packed.tobytes(),m*n//8).reshape(m,n//8)
        variants={k:reassign(W,h,X,rows,k) for k in METHODS}
        scores={k:norm_response_error(X,W[rows],reconstruct_rows(v,h,rows))
                for k,v in variants.items()}
        # Selection strictly on TRAIN module input/output objective, baseline
        # with a fixed 0.5% improvement gate avoids meaningless code churn.
        best=min(METHODS,key=lambda k:(scores[k],METHODS.index(k)))
        choice=best if scores[best]<scores["baseline"]*(1-TRAIN_IMPROVEMENT_THRESHOLD) else "baseline"
        selected=variants[choice]
        written=pack5(selected.reshape(-1))
        part=cb.meta["tensor_descriptors"][key]["parts"]["packed"]
        off,length=part
        if len(written)!=length or length!=len(h.packed):
            raise AssertionError("physical packed length changed")
        blob[offset+off:offset+off+length]=written
        changes=int(np.count_nonzero(src!=selected))
        edits+=changes
        evidence.append({"tensor":key,"sample_rows":ROWS_PER_TENSOR,
                         "trained_source_shape":[m,n],"methods":METHODS,
                         "train_response_rel_errors":scores,"TRAIN_chosen":choice,
                         "changed_5bit_codes":changes,
                         "TRAIN_only_selection_threshold":TRAIN_IMPROVEMENT_THRESHOLD})
        print(json.dumps({"tensor":key,"TRAIN_choice":choice,
                          "original_train_err":round(scores["baseline"],5),
                          "chosen_train_err":round(scores[choice],5),
                          "changed_codes":changes}),flush=True)
    prefix=bytes(blob[:-36])
    crc=struct.pack("<I",zlib.crc32(prefix))
    with_crc=prefix+crc
    candidate=with_crc+hashlib.sha256(with_crc).digest()
    assert len(candidate)==len(original)==ORIGINAL_BYTES
    parsed=CB59(candidate)
    if set(parsed.handles)!=set(cb.handles) or set(parsed.raw)!=set(cb.raw):
        raise AssertionError("incomplete original real source/aliases")
    if parsed.meta!=cb.meta:raise AssertionError("source changed manifest/metadata")
    assert math.floor(.700*N/8)>=len(candidate)
    args.candidate.parent.mkdir(parents=True,exist_ok=True)
    args.candidate.write_bytes(candidate)
    if hashlib.sha256(args.candidate.read_bytes()).hexdigest()!=hashlib.sha256(candidate).hexdigest():
        raise AssertionError("saved complete physical candidate SHA mismatch")
    audit={"experiment":"CALIBRI v73 pretrained TRAIN-covariance 8D 5bit K32 KNN selection",
           "source_model":MODEL,"pinned_pretrained_revision":MODEL_SHA,
           "original_unique_parameter_count":N,
           "source_full_archive_bytes_original":len(original),
           "candidate_complete_archive_bytes":len(candidate),
           "candidate_bpw":8*len(candidate)/N,
           "TRAIN_corpus_revision":revision,
           "TRAIN_token_sha256":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
           "TRAIN_sample_count":TRAIN_TOKENS,
           "source_sha256":hashlib.sha256(original).hexdigest(),
           "candidate_sha256":hashlib.sha256(candidate).hexdigest(),
           "new_metadata_bytes":0,
           "new_runtime_kernels_or_decoder_structures":0,
           "total_changed_5bit_symbols":edits,
           "all_four_modules_selection_TRAIN_only":True,
           "TEST_never_opened_in_training":True,
           "real_source_tensor_results":evidence,
           "whole_model_heldout_quality_NOT_measured_in_this_stage":True,
           "elapsed_seconds":time.monotonic()-start}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"stage":"v73_genuine_pretrained_train_only_8D",
                      "bytes":len(candidate),"bpw":round(8*len(candidate)/N,9),
                      "changed_codes":edits,"native_original_5bit_ABI":True}),flush=True)

if __name__=="__main__":main()
