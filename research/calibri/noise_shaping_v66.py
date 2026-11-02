#!/usr/bin/env python3
"""CALIBRI v66: functional quantization noise shaping without extra index bits.

A first transfer of analysis-by-synthesis / error feedback from signal codecs:
reassign existing 5-bit K32V8 indices using REAL TRAIN teacher activations
so per-row layer-output residual is reduced. The complete 135M CB59 physical
archive is rebuilt with identical dimensions, codebook bytes and bit length.
No original full pretrained matrix is part of the packed runtime archive.
Only 64 rows in each of four preregistered matrices are modified.
"""
from __future__ import annotations
import argparse, hashlib, json, math, struct, time, zlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

from direct_codebook_v59 import CB59, HEADER, pack5, unpack5
from quality import DATASET, load_token_ids
import binary_geometry_v64 as v64

MODEL=v64.MODEL
MODEL_SHA=v64.MODEL_SHA
N=v64.EXPECTED_UNIQUE
TRAIN_TOKENS=128
VALID_TOKENS=128
ROW_SAMPLE=64
ROUNDS=2
REGULARIZER=0.10  # predeclared, not chosen after heldout results


def representative_rows(m):
    # Deterministic, independent of source trained values and test data.
    if m < ROW_SAMPLE:
        raise AssertionError("intended real layer unexpectedly too short")
    rows=np.unique(np.linspace(0,m-1,ROW_SAMPLE,dtype=np.int64))
    if len(rows)!=ROW_SAMPLE:
        raise AssertionError("sample rows must be unique")
    return rows


def row_signal_fit(W, h, X_train, rows, regularizer):
    """Exact single-row 5bit per-8D code coordinate descent.

    For group g, hold other codes fixed, evaluate all original K32
    centroids against TRAIN output y=xW.T, and select lowest-output-
    error candidate with source error penalty for generalization.
    Feed the resulting residual to the NEXT group -- a bounded
    analysis-by-synthesis / noise shaping analogy, no extra side bits.
    """
    X=np.ascontiguousarray(X_train,dtype=np.float32)
    trained=np.ascontiguousarray(W,dtype=np.float32)
    m,n=trained.shape
    G=n//8
    if n%8 or h.shape!=(m,n):
        raise AssertionError("invalid K32V8 real matrix")
    centroids=np.asarray(h.centroids,dtype=np.float32)
    scales=np.asarray(h.row_scales,dtype=np.float32)
    if len(scales)!=m or centroids.shape!=(32,8):
        raise AssertionError("physical row metadata mismatch")
    codes=unpack5(h.packed.tobytes(),m*G).reshape(m,G)
    patched=codes.copy()
    # Efficient grouped code signal dictionary computed ONCE per layer,
    # not trained or dependent on heldout texts.
    T=len(X)
    proj=np.empty((G,T,32),dtype=np.float32)
    group_var=np.empty(G,dtype=np.float64)
    for g in range(G):
        block=X[:,g*8:g*8+8]
        proj[g]=block@centroids.T
        group_var[g]=float(np.mean(block.astype(np.float64)**2))
    if not np.isfinite(proj).all():
        raise FloatingPointError("nonfinite teacher TRAIN projections")
    target=X@trained[rows].T
    edits=0
    before=[]
    after=[]
    for i,row in enumerate(rows):
        scale=np.float32(scales[row])
        raw=trained[row].reshape(G,8).astype(np.float32)
        centers=centroids
        # Source-consistency penalty is ZERO on identical centroids.
        if regularizer:
            norms=np.sum((raw[:,None,:]-scale*centers[None,:,:])**2,axis=2)
            source_penalty=(regularizer*T*group_var[:,None]*norms).astype(np.float32)
        else:
            source_penalty=None
        current=patched[row].copy()
        predicted=scale*np.sum(proj[np.arange(G),:,current],axis=0)
        residual=target[:,i]-predicted
        before.append(float(np.dot(residual,residual)))
        for pass_id in range(ROUNDS):
            groups=range(G) if pass_id%2==0 else range(G-1,-1,-1)
            for g in groups:
                v=scale*proj[g]
                # residual excluding currently chosen contribution:
                excluding=residual+v[:,int(current[g])]
                # Incremental quadratic comparison, same exact source K32.
                error=(
                    np.einsum("tc,tc->c",v,v,dtype=np.float64)
                    -2.0*(excluding.astype(np.float64)@v.astype(np.float64))
                )
                if source_penalty is not None:
                    error+=source_penalty[g]
                best=int(np.argmin(error))
                residual=excluding-v[:,best]
                current[g]=best
        patched[row]=current
        after.append(float(np.dot(residual,residual)))
        edits+=int(np.count_nonzero(codes[row]!=current))
    return patched, {
        "selected_rows":len(rows),
        "index_edits":edits,
        "modified_source_fraction":edits/max(1,len(rows)*G),
        "selected_row_TRAIN_squared_output_error_before":float(np.sum(before)),
        "selected_row_TRAIN_squared_output_error_after":float(np.sum(after)),
        "regularizer":regularizer,
        "binaries_per_vector":5,
        "rounds":ROUNDS,
    }


def matrix_row_metric(original_w,handle,changed_codes,X,rows):
    m,n=original_w.shape
    G=n//8
    s=handle.row_scales[rows]
    c=handle.centroids[changed_codes[rows]].reshape(len(rows),n)
    q=s[:,None]*c
    original=original_w[rows]
    target=X@original.T
    predicted=X@q.T
    return float(np.linalg.norm(target-predicted)/
                 max(1e-20,float(np.linalg.norm(target))))


def toy():
    np.random.seed(66)
    m,n=64,32
    # Real-looking source codebooks and independent training/validation
    # to catch code-length corruption and unexpected overfitting.
    centroids=np.random.randn(32,8).astype(np.float32)*0.1
    scale=np.ones(m,dtype=np.float32)
    src=np.random.randint(0,32,(m,n//8),dtype=np.uint8)
    class Handle:
        shape=(m,n)
        row_scales=scale
        centroids=centroids
        packed=np.frombuffer(pack5(src.reshape(-1)),dtype=np.uint8)
    W=(centroids[src].reshape(m,n)*scale[:,None] +
       np.random.randn(m,n).astype(np.float32)*0.01)
    X=np.random.randn(32,n).astype(np.float32)
    rows=representative_rows(m)
    update,report=row_signal_fit(W,Handle,X,rows,0.)
    assert update.shape == src.shape
    assert len(pack5(update.reshape(-1)))==len(Handle.packed)
    assert report["selected_row_TRAIN_squared_output_error_after"] <= (
        report["selected_row_TRAIN_squared_output_error_before"]+1e-2)
    print("CALIBRI v66 toy deterministic K32 index-only TRAIN noise shaping, exact 5bit size PASS",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--candidate",type=Path)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    toy()
    if args.smoke:
        return
    if not (args.source and args.output and args.candidate):
        ap.error("source, output and candidate required")
    start=time.monotonic()
    original_blob=args.source.read_bytes()
    cb=CB59(original_blob)
    if cb.meta["original_unique_parameter_count"]!=N:
        raise AssertionError("not original trained pinned complete 135M physical source")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    original=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).eval()
    modules=v64.choose_module_names(original)
    dataset_rev=HfApi().dataset_info(DATASET).sha
    train=load_token_ids(tok,"train",dataset_rev,TRAIN_TOKENS)
    valid=load_token_ids(tok,"validation",dataset_rev,VALID_TOKENS)
    inputs=v64.real_inputs(original,modules,train,valid)
    manifest=cb.meta
    _,_,meta_size,_=HEADER.unpack_from(original_blob)
    body_offset=HEADER.size+meta_size
    mutated=bytearray(original_blob)
    evidence=[]
    aggregate_edits=0
    for name,module in modules.items():
        key=name+".weight"
        handle=cb.handles[key]
        W=module.weight.detach().cpu().numpy().astype(np.float32)
        m,n=W.shape
        rows=representative_rows(m)
        part=manifest["tensor_descriptors"][key]["parts"]["packed"]
        off,size=part
        original_codes=unpack5(handle.packed.tobytes(),m*(n//8)).reshape(m,n//8)
        Xtr=inputs[name]["train"].cpu().numpy()
        Xv=inputs[name]["valid"].cpu().numpy()
        fixed,info=row_signal_fit(W,handle,Xtr,rows,REGULARIZER)
        # Completely separate diagnostic unconstrained alternative λ=0:
        # never use VALID to choose lambda or final encoded fixed archive.
        unregularized,info0=row_signal_fit(W,handle,Xtr,rows,0.)
        updated_packed=pack5(fixed.reshape(-1))
        if len(updated_packed)!=size or size!=handle.packed.nbytes:
            raise AssertionError("same-sized original 5-bit source broke")
        if not np.array_equal(unpack5(updated_packed,m*(n//8)).reshape(m,n//8),fixed):
            raise AssertionError("modified true physical indices don't roundtrip")
        mutated[body_offset+off:body_offset+off+size]=updated_packed
        if np.any(fixed[np.setdiff1d(np.arange(m),rows)] !=
                  original_codes[np.setdiff1d(np.arange(m),rows)]):
            raise AssertionError("edited rows outside preregistered subset")
        metrics={
            "tensor":key,
            "shape":[m,n],
            "true_source_original_pack5_bytes":size,
            "true_source_after_pack5_bytes":len(updated_packed),
            "source_5bit_bytes_added":0,
            "selected_rows":rows.tolist(),
            "fixed_regularizer_0_10":info,
            "diagnostic_regularizer_0":info0,
            "CB59_selected_rows_TRAIN_error":matrix_row_metric(W,handle,original_codes,Xtr,rows),
            "CB59_selected_rows_VALID_error":matrix_row_metric(W,handle,original_codes,Xv,rows),
            "index_shaped_selected_rows_TRAIN_error":matrix_row_metric(W,handle,fixed,Xtr,rows),
            "index_shaped_selected_rows_VALID_error":matrix_row_metric(W,handle,fixed,Xv,rows),
            "unregularized_selected_rows_VALID_diagnostic_error":matrix_row_metric(W,handle,unregularized,Xv,rows),
        }
        metrics["VALID_ratio_shaped_to_original"]=(
            metrics["index_shaped_selected_rows_VALID_error"]/
            max(metrics["CB59_selected_rows_VALID_error"],1e-20))
        evidence.append(metrics)
        aggregate_edits+=info["index_edits"]
        print(json.dumps({
            "tensor":key,"changed_codes":info["index_edits"],
            "TRAIN_original":round(metrics["CB59_selected_rows_TRAIN_error"],4),
            "TRAIN_shaped":round(metrics["index_shaped_selected_rows_TRAIN_error"],4),
            "VALID_original":round(metrics["CB59_selected_rows_VALID_error"],4),
            "VALID_shaped":round(metrics["index_shaped_selected_rows_VALID_error"],4),
            "VALID_ratio":round(metrics["VALID_ratio_shaped_to_original"],4)}),
              flush=True)
    # Physically rebuild complete unchanged-length K32V8 archive, checksum
    # and hash both recomputed. No hidden codebook, predictor or corrector.
    unprotected=bytes(mutated[:-36])
    crc=struct.pack("<I",zlib.crc32(unprotected))
    protected=unprotected+crc
    candidate=protected+hashlib.sha256(protected).digest()
    if len(candidate)!=len(original_blob):
        raise AssertionError("physical entire source archive size drift")
    parsed=CB59(candidate)
    if set(parsed.handles)!=set(cb.handles) or set(parsed.raw)!=set(cb.raw):
        raise AssertionError("candidate incomplete trained 135M state")
    for name in modules:
        key=name+".weight"
        if len(parsed.handles[key].packed)!=len(cb.handles[key].packed):
            raise AssertionError("modified packed source length mismatch")
    changed=bytearray(candidate);changed[len(candidate)//2]^=1
    try:
        CB59(bytes(changed))
    except ValueError:
        pass
    else:
        raise AssertionError("new source corruption protection invalid")
    ratios=[x["VALID_ratio_shaped_to_original"] for x in evidence]
    audit={
        "experiment":"CALIBRI v66 actual 135M full CB59-format source-only bit-neutral, TRAIN functional error feedback index reassignment",
        "predeclared_index_shaping_regularizer":REGULARIZER,
        "source_model":MODEL,
        "original_trained_revision":MODEL_SHA,
        "original_CB59_sha256":cb.hash(),
        "new_complete_CB59_format_candidate_sha256":hashlib.sha256(candidate).hexdigest(),
        "source_file_physical_bytes_original":len(original_blob),
        "complete_candidate_source_bytes":len(candidate),
        "all_physical_5bit_index_code_bytes_unchanged":True,
        "same_CB59_native_packed_kernel_ABI":True,
        "new_candidate_bpw_original_unique_count":8*len(candidate)/N,
        "full_source_under_0_700":len(candidate)<=math.floor(.700*N/8),
        "pretrained_teacher_state_is_NOT_in_archive":True,
        "genuine_WikiText_dataset_revision":dataset_rev,
        "TRAIN_token_SHA256":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
        "VALID_token_SHA256":hashlib.sha256(valid.numpy().tobytes()).hexdigest(),
        "heldout_VALID_not_used_to_fit_index_stream":True,
        "total_reassigned_5bit_symbols":aggregate_edits,
        "selected_row_count_each_of_four_modules":ROW_SAMPLE,
        "four_tensor_mean_selected_row_VALID_error_ratio":sum(ratios)/len(ratios),
        "real_source_tensor_results":evidence,
        "independent_teach_free_live_model_execution_NOT_yet_tested":True,
        "no_entire_model_perplexity_accuracy_or_speed_claim":True,
        "duration_seconds":time.monotonic()-start,
    }
    args.candidate.parent.mkdir(parents=True,exist_ok=True)
    args.candidate.write_bytes(candidate)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"real_matrices":len(modules),
                      "original_complete_bytes":len(original_blob),
                      "new_complete_bytes":len(candidate),
                      "total_reassigned_5bit_symbols":aggregate_edits,
                      "mean_VALID_selected_row_ratio":round(sum(ratios)/len(ratios),5),
                      "physical_full_model_rate_verified":True,
                      "entire_model_accuracy_measured":False}),flush=True)


if __name__=="__main__":
    main()
