#!/usr/bin/env python3
"""CALIBRI v65: same true tensor-local bytes, source-only ITQ geometry ablation.

v64 demonstrated mixed REAL validation: q-proj improved, two MLP worsened,
and 3 MLP underused a CB59 per-tensor payload due to r<=192. v65 fixes
this confound by r<=256 and adds an original source-only iterative
orthogonal Procrustes alignment. No LLM deployment/quality claims.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

import binary_geometry_v64 as v64
from direct_codebook_v59 import CB59
from quality import DATASET, load_token_ids

ITQ_ITERS=8
SEED=65
RANK_CAP=256
TRAIN_TOKENS=v64.TRAIN_TOKENS
VALID_TOKENS=v64.HELDOUT_TOKENS


def source_only_rotations(U,S,V,seed=SEED):
    """A truly learned SVD latent rotation with NO tokens or labels.

    Given A=U sqrt(S), B=V sqrt(S), R orthogonal preserves A B^T
    continuously. Alternating binary sign assignment and orthogonal
    Procrustes minimize [A;B]R distance to a {+1,-1} hypercube.
    THIS IS a simplified original ITQ-style baseline, not a reproduction
    of LittleBit-2's joint iterative quantization training protocol.
    """
    A=torch.cat([U*torch.sqrt(S)[None,:],V*torch.sqrt(S)[None,:]],dim=0)
    r=len(S)
    rng=torch.Generator(device="cpu").manual_seed(seed)
    start=torch.randn(r,r,generator=rng)
    R,_=torch.linalg.qr(start)
    rotations={"identity":torch.eye(r),"fixed_random_orthogonal":R.clone()}
    objective=[]
    for i in range(ITQ_ITERS):
        bits=v64.sign_nonzero(A @ R)
        M=A.T @ bits
        left,sing,right=torch.linalg.svd(M,full_matrices=False)
        R=left @ right
        residual=torch.linalg.vector_norm(A @ R-v64.sign_nonzero(A @ R))
        objective.append(float(residual))
    rotations["source_only_procrustes_ITQ"]=R
    return rotations,objective


def candidate(W,scales,Xtrain,Xvalid,U,S,V,R,method,fit_mode):
    left=U*torch.sqrt(S)[None,:]
    right=V*torch.sqrt(S)[None,:]
    P=v64.sign_nonzero(left@R)
    Q=v64.sign_nonzero(right@R)
    gains=v64.fit_gains(P,Q,scales,W,Xtrain,fit_mode)
    decoded=((P*gains[None,:])@Q.T)*scales[:,None]
    m,n=W.shape;r=len(S)
    packed_P=np.packbits((P.numpy()>0).reshape(-1),bitorder="little")
    packed_Q=np.packbits((Q.numpy()>0).reshape(-1),bitorder="little")
    packed_scale=gains.to(torch.float16).numpy().tobytes()
    packed_rows=scales.to(torch.float16).numpy().tobytes()
    source_bytes=len(packed_P)+len(packed_Q)+len(packed_scale)+len(packed_rows)+64
    if source_bytes!=v64.physical_factor_payload(m,n,r):
        raise AssertionError("actual factor encoding differs from declared source")
    return {
        "method":method+"+"+fit_mode,
        "actual_local_factor_payload_bytes":source_bytes,
        "actual_local_factor_bpw":8*source_bytes/(m*n),
        "real_weight_relative_RMSE":v64.source_rmse(W,decoded),
        "real_TRAIN_response_relative_RMSE":v64.relative_response_error(W,decoded,Xtrain),
        "unseen_VALID_response_relative_RMSE":v64.relative_response_error(W,decoded,Xvalid),
    }


def matrix_trial(name,module,handle,inputs):
    W=module.weight.detach().cpu().float()
    m,n=W.shape
    scales=torch.from_numpy(handle.row_scales.copy()).float()
    old=v64.cb59_weight(handle)
    original_payload=int(handle.packed.nbytes)+2*m+2*32*8
    rank=v64.choose_rank(m,n,original_payload)
    if rank>min(m,n,RANK_CAP):
        raise AssertionError("rank beyond declared cap")
    # Explicit frozen random seed before source-only rank-reduced SVD.
    torch.manual_seed(SEED)
    U,S,V=torch.svd_lowrank(W/scales[:,None],q=min(rank+8,min(m,n)),niter=1)
    U,S,V=U[:,:rank],S[:rank],V[:,:rank]
    rotations,fit_objective=source_only_rotations(U,S,V)
    candidates=[]
    for name_rot,R in rotations.items():
        for fit in ("source_weight_LS","train_activation_LS"):
            candidates.append(candidate(W,scales,inputs["train"],inputs["valid"],
                                        U,S,V,R,name_rot,fit))
    for c in candidates:
        if c["actual_local_factor_payload_bytes"] > original_payload:
            raise AssertionError("bit rate cheat: binary factors exceed true CB59 payload")
    selected=min(candidates,key=lambda c:c["real_TRAIN_response_relative_RMSE"])
    cb59_valid=v64.relative_response_error(W,old,inputs["valid"])
    cb59_train=v64.relative_response_error(W,old,inputs["train"])
    record={
        "tensor":name+".weight",
        "matrix_shape":[m,n],
        "unique_weights":int(W.numel()),
        "chosen_rank_source_and_rate_only":rank,
        "CB59_original_local_physical_bytes":original_payload,
        "candidate_local_physical_bytes":selected["actual_local_factor_payload_bytes"],
        "CB59_TRAIN_error":cb59_train,
        "CB59_heldout_VALID_error":cb59_valid,
        "source_only_ITQ_geometry_errors":fit_objective,
        "six_methods":candidates,
        "TRAIN_only_selected_method":selected["method"],
        "TRAIN_selected_VALID_error":selected["unseen_VALID_response_relative_RMSE"],
        "TRAIN_selected_to_CB59_VALID_error_ratio":selected["unseen_VALID_response_relative_RMSE"]/max(cb59_valid,1e-15),
        "same_full_model_bpw_not_claimed":True,
    }
    print(json.dumps({"tensor":record["tensor"],"r":rank,
                      "source_baseline":original_payload,
                      "factor_bytes":record["candidate_local_physical_bytes"],
                      "CB59_VALID":round(cb59_valid,4),
                      "chosen":selected["method"],
                      "factor_VALID":round(record["TRAIN_selected_VALID_error"],4),
                      "ratio":round(record["TRAIN_selected_to_CB59_VALID_error_ratio"],4),
                      "ITQ_first_last_source_sign_error":[
                          round(fit_objective[0],3),round(fit_objective[-1],3)]}),flush=True)
    return record


def fixture():
    torch.manual_seed(SEED)
    A=torch.randn(32,24)
    U,S,V=torch.svd_lowrank(A,q=8,niter=1)
    transforms,errors=source_only_rotations(U,S,V)
    assert len(transforms)==3 and len(errors)==ITQ_ITERS
    assert all(torch.max(abs(R.T@R-torch.eye(len(S))))<1e-4 for R in transforms.values())
    assert errors[-1] <= errors[0]+1e-3
    print("CALIBRI v65 toy source-only alternating Procrustes converges, rotations orthonormal PASS",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    # Makes v64.choose_rank saturate the REAL original tensor budget
    # rather than cap all wider MLP matrices at 192.
    v64.RANK_CAP=RANK_CAP
    fixture()
    if args.smoke:
        return
    if args.source is None or args.output is None:
        ap.error("real CB59 source and output required")
    st=time.monotonic()
    cb=CB59(args.source.read_bytes())
    if cb.meta["original_unique_parameter_count"]!=v64.EXPECTED_UNIQUE:
        raise AssertionError("source is not the pinned 135M original")
    dataset_revision=HfApi().dataset_info(DATASET).sha
    tokenizer=AutoTokenizer.from_pretrained(v64.MODEL,revision=v64.MODEL_SHA)
    teacher=AutoModelForCausalLM.from_pretrained(
        v64.MODEL,revision=v64.MODEL_SHA,torch_dtype=torch.float32).eval()
    modules=v64.choose_module_names(teacher)
    train=load_token_ids(tokenizer,"train",dataset_revision,TRAIN_TOKENS)
    valid=load_token_ids(tokenizer,"validation",dataset_revision,VALID_TOKENS)
    inputs=v64.real_inputs(teacher,modules,train,valid)
    tested=[matrix_trial(k,m,cb.handles[k+".weight"],inputs[k])
            for k,m in modules.items()]
    ratios=[t["TRAIN_selected_to_CB59_VALID_error_ratio"] for t in tested]
    result={
        "experiment":"CALIBRI v65 real trained 135M source-only latent ITQ & local-bit-budget saturation, actual teacher TRAIN/VALID inputs",
        "public_trained_model":v64.MODEL,
        "model_sha":v64.MODEL_SHA,
        "source_CB59_SHA256":hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "WikiText2_dataset_revision":dataset_revision,
        "TRAIN_128_sha256":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
        "VALID_128_sha256":hashlib.sha256(valid.numpy().tobytes()).hexdigest(),
        "four_matrix_mean_selected_VALID_error_ratio_CB59":sum(ratios)/len(ratios),
        "bit_budget_saturated_where_rank256_available":True,
        "original_teacher_TEST_data_did_not_fit_candidates":True,
        "individual_tensor_rate_physical":True,
        "one_full_135m_model_archive_under_0_7_NOT_built":True,
        "native_bitpacked_factor_GEMV_NOT_built":True,
        "no_model_task_accuracy_or_speed_claim":True,
        "matrix_results":tested,
        "elapsed_seconds":time.monotonic()-st,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"real_matrices":len(tested),"mean_VALID_ratio":round(sum(ratios)/len(ratios),5),
                      "elapsed_seconds":round(result["elapsed_seconds"],1)}),flush=True)


if __name__=="__main__":
    main()
