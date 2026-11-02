#!/usr/bin/env python3
"""CALIBRI v64: real weight / real TRAIN activation matrix pilot.

Compare original v59 K32V8 index-codebook reconstruction to a same-rate
sub-1-bit binary latent-factor alternative inspired by LittleBit/NanoQuant.
This is a local tensor diagnostic, NOT a complete archived/inference model.
No TEST labels, no full-model output quality or speed assertions.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

from direct_codebook_v59 import CB59, unpack5
from quality import DATASET, load_token_ids

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_SHA = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
EXPECTED_UNIQUE = 134515008
TRAIN_TOKENS = 128
HELDOUT_TOKENS = 128
WINDOW = 64
RANK_CAP = 192  # predetermined computational budget, NOT the rate ceiling
SEED = 64


def log(x):
    print(json.dumps(x, ensure_ascii=False), flush=True)


def relative_response_error(w, q, x):
    source = x @ w.T
    estimated = x @ q.T
    return float(torch.linalg.vector_norm(source-estimated)/
                 torch.linalg.vector_norm(source).clamp_min(1e-15))


def source_rmse(w, q):
    return float(torch.linalg.vector_norm(w-q)/
                 torch.linalg.vector_norm(w).clamp_min(1e-15))


def physical_factor_payload(m, n, r):
    # P(m,r),Q(n,r) sign bits, r FP16 gains and m FP16 original row scales;
    # fixed 64 B tensor header budget (not a full-file container).
    return (m*r+7)//8 + (n*r+7)//8 + 2*r + 2*m + 64


def choose_rank(m, n, source_payload):
    ranks = [r for r in range(8, min(m,n,RANK_CAP)+1, 8)
             if physical_factor_payload(m,n,r) <= source_payload]
    if not ranks:
        raise AssertionError("no rank <= source payload")
    return ranks[-1]


def choose_module_names(model):
    selected = {}
    named = {k: v for k,v in model.named_modules() if isinstance(v, nn.Linear)}
    n_layers = model.config.num_hidden_layers
    # Cover distinct geometries and both early and late processing.
    names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.%d.mlp.gate_proj" % (n_layers//2),
        "model.layers.%d.mlp.up_proj" % (n_layers-1),
    ]
    for name in names:
        if name not in named:
            raise AssertionError("expected real pretrained module missing: "+name)
        selected[name] = named[name]
    return selected


def real_inputs(model, modules, train, valid):
    buffer = {k: {"train":[], "valid":[]} for k in modules}
    phase = ["train"]
    handles = []
    for name, mod in modules.items():
        def hook(m, inputs, *, key=name):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu().float()
            buffer[key][phase[0]].append(x)
        handles.append(mod.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            for which, ids in (("train", train), ("valid", valid)):
                phase[0]=which
                for start in range(0,len(ids),WINDOW):
                    model(input_ids=ids[start:start+WINDOW][None], use_cache=False)
    finally:
        for h in handles:
            h.remove()
    out = {}
    for name in modules:
        out[name] = {part: torch.cat(buffer[name][part], dim=0)
                     for part in ("train", "valid")}
        assert out[name]["train"].shape[0] == TRAIN_TOKENS
        assert out[name]["valid"].shape[0] == HELDOUT_TOKENS
    return out


def cb59_weight(handle):
    m,n = handle.shape
    codes = unpack5(handle.packed.tobytes(), m*(n//8))
    q = handle.centroids[codes].reshape(m,n)*handle.row_scales[:,None]
    if q.shape != (m,n) or not np.isfinite(q).all():
        raise AssertionError("source CB59 reconstruction problem")
    return torch.from_numpy(np.asarray(q, dtype=np.float32).copy())


def sign_nonzero(x):
    # A zero is encoded as +1, not as an imaginary free ternary bit.
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def fit_gains(P, Q, scales, W, X, method):
    """A=r×m and B=r×n binary factors, independent r FP16 gains.

    P and Q are m×r/n×r in {-1,+1}. For TRAIN activation LS use
    min_g || X W.T - (X Q) diag(g) (P*scale).T ||_F^2 .
    For weight LS use the analogous Frobenius error in weight space.
    """
    P64=(P.to(torch.float64)*scales[:,None].to(torch.float64))
    Q64=Q.to(torch.float64)
    W64=W.to(torch.float64)
    if method == "train_activation_LS":
        X64=X.to(torch.float64)
        T=X64 @ Q64
        Y=X64 @ W64.T
        gram=(T.T @ T)*(P64.T @ P64)
        rhs=torch.sum(T*(Y @ P64),dim=0)
    elif method == "source_weight_LS":
        gram=(P64.T @ P64)*(Q64.T @ Q64)
        rhs=torch.sum(P64*(W64 @ Q64),dim=0)
    else:
        raise ValueError(method)
    diagonal=gram.diag()
    ridge=1e-5*diagonal.mean().clamp_min(1.)
    gram.diagonal().add_(ridge)
    sol=torch.linalg.solve(gram,rhs).to(torch.float32)
    # Exactly same 16-bit serialization granularity used by CB59 metadata.
    encoded=sol.to(torch.float16)
    if not torch.isfinite(encoded).all():
        raise ValueError("binary factor gains not FP16 encodable")
    return encoded.float()


def binary_candidate(W, scales, Xtr, Xval, U, S, V, rotation, objective):
    r=S.numel()
    # Continuous SVD low-rank factorization; optional orthogonal latent
    # rotation preserves the continuous product but changes its sign geometry.
    left=U*torch.sqrt(S)[None,:]
    right=V*torch.sqrt(S)[None,:]
    if rotation == "random_orthogonal":
        g=torch.Generator(device="cpu").manual_seed(SEED)
        matrix=torch.randn((r,r),generator=g,dtype=torch.float32)
        rot,_=torch.linalg.qr(matrix)
        left=left @ rot
        right=right @ rot
    elif rotation != "identity":
        raise ValueError(rotation)
    P=sign_nonzero(left)
    Q=sign_nonzero(right)
    gains=fit_gains(P,Q,scales,W,Xtr,objective)
    # Diagnostic reconstruction only: native compressed binary inference
    # is NOT claimed. P and Q would be bitpacked with memory count checked.
    q=((P*gains[None,:])@Q.T)*scales[:,None]
    packed_left=np.packbits((P.numpy() > 0).reshape(-1),bitorder="little")
    packed_right=np.packbits((Q.numpy() > 0).reshape(-1),bitorder="little")
    assert len(packed_left)==(P.numel()+7)//8
    assert len(packed_right)==(Q.numel()+7)//8
    encoded_total=(len(packed_left)+len(packed_right)+
                   len(gains.to(torch.float16).numpy().tobytes())+
                   len(scales.to(torch.float16).numpy().tobytes())+64)
    m,n=W.shape
    if encoded_total!=physical_factor_payload(m,n,r):
        raise AssertionError("factor true bit-packing physical payload drift")
    result = {
        "method": rotation+"+"+objective,
        "local_tensor_rate_bytes": encoded_total,
        "local_effective_bpw": 8*encoded_total/W.numel(),
        "serialized_binary_factors_only": True,
        "factor_parameter_values": "+1/-1; 2 byte per-rank scale and original FP16 row scales",
        "source_weight_relative_RMSE": source_rmse(W,q),
        "real_TRAIN_activation_output_relative_RMSE": relative_response_error(W,q,Xtr),
        "unseen_WikiText_VALIDATION_activation_output_relative_RMSE": relative_response_error(W,q,Xval),
    }
    return result


def trial(name, mod, h, inputs):
    W=mod.weight.detach().cpu().float()
    m,n=W.shape
    if tuple(h.shape) != (m,n):
        raise AssertionError("trained source and CB59 shape mismatch")
    scales=torch.from_numpy(h.row_scales.copy()).float()
    baseline=cb59_weight(h)
    source_payload = int(h.packed.nbytes)+2*m+2*32*8
    r=choose_rank(m,n,source_payload)
    torch.manual_seed(SEED)
    # Finite single-pass *randomized* lowrank SVD on original W/scales.
    # Limit q to smaller singular dimension, do not evaluate on VALID
    # to fit codebook or choose rank. This is source-only initialization.
    normalized=W/scales[:,None]
    U,S,V=torch.svd_lowrank(normalized,q=min(r+8,min(m,n)),niter=1)
    U,S,V=U[:,:r],S[:r],V[:,:r]
    possibilities=[]
    for rotation in ("identity","random_orthogonal"):
        for objective in ("source_weight_LS","train_activation_LS"):
            possibilities.append(binary_candidate(W,scales,inputs["train"],inputs["valid"],
                                                  U,S,V,rotation,objective))
    # Training-only selection. Both unseen validation and original CB59
    # baseline are prohibited as selection signals.
    selected=min(possibilities,key=lambda x:x["real_TRAIN_activation_output_relative_RMSE"])
    baseline_metrics = {
        "source_weight_relative_RMSE": source_rmse(W,baseline),
        "real_TRAIN_activation_output_relative_RMSE": relative_response_error(W,baseline,inputs["train"]),
        "unseen_WikiText_VALIDATION_activation_output_relative_RMSE": relative_response_error(W,baseline,inputs["valid"]),
        "original_CB59_matrix_payload_bytes": source_payload,
        "original_CB59_local_effective_bpw": 8*source_payload/W.numel(),
    }
    record={
        "tensor": name+".weight",
        "shape": [m,n],
        "real_parameters": int(W.numel()),
        "factor_rank": r,
        "original_CB59_local_payload_budget_bytes": source_payload,
        "full_model_archive_budget_NOT_claimed": True,
        "baseline_CB59": baseline_metrics,
        "binary_candidate_options": possibilities,
        "candidate_selected_by_TRAIN_only": selected["method"],
        "selected_validation_error_ratio_vs_CB59": (
            selected["unseen_WikiText_VALIDATION_activation_output_relative_RMSE"] /
            max(1e-20, baseline_metrics["unseen_WikiText_VALIDATION_activation_output_relative_RMSE"])),
        "factor_rank_not_optimized_on_validation": True,
    }
    log({"tensor":name,"shape":[m,n],"r":r,
         "CB59_VALID_err":round(baseline_metrics["unseen_WikiText_VALIDATION_activation_output_relative_RMSE"],4),
         "factor_TRAIN_chosen":selected["method"],
         "factor_VALID_err":round(selected["unseen_WikiText_VALIDATION_activation_output_relative_RMSE"],4),
         "ratio":round(record["selected_validation_error_ratio_vs_CB59"],4)})
    return record


def fixture():
    torch.manual_seed(SEED)
    m,n,r=24,32,8
    W=torch.randn(m,n)/8
    s=torch.sqrt((W*W).mean(1)).to(torch.float16).float()
    X1=torch.randn(20,n)
    X2=torch.randn(16,n)
    U,S,V=torch.svd_lowrank(W/s[:,None],q=r+4,niter=1)
    results=[]
    for mode in ("identity","random_orthogonal"):
        for fit in ("source_weight_LS","train_activation_LS"):
            result=binary_candidate(W,s,X1,X2,U[:,:r],S[:r],V[:,:r],mode,fit)
            assert result["local_tensor_rate_bytes"]==physical_factor_payload(m,n,r)
            assert math.isfinite(result["source_weight_relative_RMSE"])
            results.append(result)
    assert len(results)==4
    print("CALIBRI v64 toy real bitpacked factor/gain, rotation, TRAIN-only fit PASS", flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    fixture()
    if args.smoke:
        return
    if args.source is None or args.output is None:
        ap.error("source and output required")
    start=time.monotonic()
    cb=CB59(args.source.read_bytes())
    if cb.meta["original_unique_parameter_count"] != EXPECTED_UNIQUE:
        raise AssertionError("not original complete v59 true 135M source")
    api=HfApi()
    data_revision=api.dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).eval()
    modules=choose_module_names(teacher)
    train=load_token_ids(tok,"train",data_revision,TRAIN_TOKENS)
    valid=load_token_ids(tok,"validation",data_revision,HELDOUT_TOKENS)
    assert hashlib.sha256(train.numpy().tobytes()).hexdigest()!=hashlib.sha256(valid.numpy().tobytes()).hexdigest()
    inputs=real_inputs(teacher,modules,train,valid)
    tensors=[trial(name,mod,cb.handles[name+".weight"],inputs[name])
             for name,mod in modules.items()]
    ratios=[z["selected_validation_error_ratio_vs_CB59"] for z in tensors]
    doc={
        "experiment":"CALIBRI v64 independent binary-factor pilot on ACTUAL pretrained weights and TRAIN/VALID teacher signals",
        "source_model":MODEL,
        "model_revision":MODEL_SHA,
        "dataset":DATASET,
        "dataset_revision":data_revision,
        "TRAIN_128_token_sha256":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
        "VALID_128_token_sha256":hashlib.sha256(valid.numpy().tobytes()).hexdigest(),
        "original_CB59_file_sha256":cb.hash(),
        "calibration_source_only_or_TRAIN":True,
        "validation_never_used_to_fit_factors_or_select_modes":True,
        "factor_local_payload_includes_packbits_gain_scales_and_64B_header":True,
        "physical_full_model_rate_verified_for_factor_alternative":False,
        "native_binary_factor_kernel_or_speed_verified":False,
        "full_LLM_quality_preservation_claim":False,
        "four_matrix_mean_selected_VALID_output_error_ratio_vs_CB59":sum(ratios)/len(ratios),
        "tensor_results":tensors,
        "elapsed_s":time.monotonic()-start,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+"\n")
    log({"tested_real_weights":len(tensors),"mean_VALID_ratio":round(doc["four_matrix_mean_selected_VALID_output_error_ratio_vs_CB59"],4),
         "real_run_seconds":round(doc["elapsed_s"],1), "direct_kernel":False})


if __name__=="__main__":
    main()
