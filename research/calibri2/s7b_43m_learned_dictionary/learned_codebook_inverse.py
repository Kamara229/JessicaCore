#!/usr/bin/env python3
"""S7B actual trained 12layer 43M FP32 mirror -> original codebook-prior CB61.

5bit indices are rebuilt solely from the DENSE FP32 weights + *trained*
original CB59 FP16 centroid dictionaries/row scales, NEVER from archived
stored index stream. Stored original 5bit indices are compared only AFTER
new source is fully packed. Source FP32 is derived from CALIBRI S142,
NOT an independently trained public 43M model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59,pack5,unpack5
from v185.codec import replace_discrete
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s7a_43m_fp32_transfer.actual_43m_fp32_conversion import (
    validate_parent,Stack,selected_S93_existing_train,new_CB61,digest,
    dump,log,BASE_36M,TOTAL_43M,MAX_BPP
)

S7B_MAX_MATRIX_ERROR=1e-4
SELECTED_8=8


def recover_one_matrix_from_FP32(w,h):
    """No input original code stream; assignment uses full learned FP16 centroids."""
    w=np.asarray(w,dtype=np.float32)
    m,n=h.shape
    if tuple(w.shape)!=(m,n) or n%8:
        raise AssertionError("External 43M decoder weights not compatible with original 12x320 5bit vectors")
    centers=h.centroids.astype(np.float32)
    scales=h.row_scales.astype(np.float32)
    groups=n//8
    recovered=[]
    error_max=0.0
    sqerr=0.0
    for start in range(0,m,128):
        stop=min(m,start+128)
        vectors=w[start:stop].reshape(-1,8)
        factors=np.repeat(scales[start:stop],groups)[:,None]
        normalized=vectors/factors
        # The original S93 source learned codebook is the only model prior.
        # No original archived indices used for assignment, not even ties.
        choices=[]
        for vstart in range(0,len(normalized),1024):
            x=normalized[vstart:vstart+1024]
            dif=(x[:,None,:]-centers[None,:,:]).astype(np.float32)
            distance=np.einsum("nki,nki->nk",dif,dif)
            choices.append(np.argmin(distance,axis=1).astype(np.uint8))
        ids=np.concatenate(choices,axis=0)
        decoded=centers[ids]*factors
        err=np.abs(decoded-vectors)
        error_max=max(error_max,float(np.max(err)))
        sqerr+=float(np.sum((decoded-vectors).astype(np.float64)**2))
        recovered.append(pack5(ids))
    rebuilt_packed=b"".join(recovered)
    if len(rebuilt_packed)!=len(h.packed) or not np.isfinite(sqerr):
        raise AssertionError("Packed source size or error corrupted original trained GT model")
    # Round-trip of source already in trained codebook span must be exact
    # to FP32 numerical tolerance. Doesn't prove outside-span generalization.
    if error_max>S7B_MAX_MATRIX_ERROR:
        raise AssertionError("S7B source trained-centroid inversion reconstructed FP32 with too large error")
    values=unpack5(rebuilt_packed,m*groups).astype(np.int64)
    return values,error_max,sqerr


def run(a):
    torch.manual_seed(43071217)
    torch.set_num_threads(2)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    parent,source,s93,side,bank,router,copy=validate_parent(a)
    audit={
       "stage":"S7B_actual_12layer_FP32_reconstruction_from_working_43M_then_paid_original_trained_centroid_inversion",
       "source_provenance":"pretrained-from-random CALIBRI S93 and S142",
       "original_FP32_is_same_source_dense_mirror":True,
       "independent_open_pretrained_FP32_43M_loaded":False,
       "input_trained_original_CB59_centroids_and_scales_used_as_prior":True,
       "old_archived_packed_indices_used_for_new_encoding":False,
       "original_43M_base_and_structural_additive":[BASE_36M,TOTAL_43M-BASE_36M],
       "same_original_S142_Green_Tunnel_and_other_S95_S98_mechanisms":True,
       "original_456plus128_J7_remains_immutable_attested_not_rechecked":True,
       "new_EVAL_or_QA_used":False,
       "no_NRA_and_no_LIVE_Jessica":True,
       "original_pretrained_FP32_independent_source_not_tested":True,
       "not_generalizable_to_135M_without_independent_FP32_and_distillation":True,
    }
    dump(out/"S7B_PRECOMMIT_BEFORE_DENSE_SOURCE_INVERSION.json",audit)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    rows=selected_S93_existing_train(tok,router)
    dense_model=s30.dense_from_paid(parent).eval().requires_grad_(False)
    matrices=sorted(k for k,d in parent.meta["tensor_descriptors"].items()
                    if d["dtype"]=="CB59_K32V8")
    if len(matrices)!=85:
        raise AssertionError("S7B original 12layer 36M CB59 85-matrix geometry changed")
    dense_state=dense_model.state_dict()
    recovered={}
    matrix_reports={}
    for idx,name in enumerate(matrices):
        h=parent.handles[name]
        new_codes,max_abs,sse=recover_one_matrix_from_FP32(
            dense_state[name].detach().cpu().numpy(),h)
        recovered[name]=SimpleNamespace(
            codes=torch.from_numpy(new_codes),
            centroids=torch.from_numpy(h.centroids.astype(np.float32).copy()))
        matrix_reports[name]={
            "shape":list(h.shape),
            "recovered_from_FP32_max_abs_weight_error":max_abs,
            "recovered_from_FP32_squared_error":sse,
            "new_5bit_groups":len(new_codes),
            "CODE_MATCH_ONLY_POSTHOC":None,
        }
        if (idx+1)%20==0:
            log("S7B_real_FP32_to_original_calibri_codebook_K32V8",done=idx+1,
                total=len(matrices),matrix=name,rmse=round(sse/max(1,len(new_codes)*8),12))
    reconstructed,seal=replace_discrete(parent,recovered,strict=False)
    rebuilt=CB59(reconstructed)
    if (len(reconstructed)!=len(source) or
        seal["all_2d_original_matrices"]!=85 or
        rebuilt.meta!=parent.meta or
        rebuilt.handles["model.embed_tokens.weight"] is not rebuilt.handles["lm_head.weight"]):
        raise AssertionError("Recovered FP32 43M source broke original 85 compressed matrices or alias")
    # Only after all new codes have been selected/persisted, check old codes.
    code_total=0;code_identical=0
    for name in matrices:
        old=parent.handles[name]
        num=old.shape[0]*(old.shape[1]//8)
        orig_code=unpack5(old.packed.tobytes(),num)
        fresh=recovered[name].codes.numpy()
        code_total+=num
        matched=int(np.count_nonzero(orig_code==fresh))
        code_identical+=matched
        matrix_reports[name]["CODE_MATCH_ONLY_POSTHOC"]=matched/num
    cb61=new_CB61(reconstructed)
    source_file=out/"S7B_REAL_S142_FP32_INVERTED_TRAINED_DICTIONARY_CB61.s61"
    source_file.write_bytes(cb61)
    teacher=Stack(parent,source,side,dense=True)
    candidate=Stack(rebuilt,reconstructed,side,dense=False)
    scored=[]
    try:
        for i,row in enumerate(rows):
            teacher_hidden,teacher_logits=teacher.layer_probe(
                tok,row["question"],row["feature"])
            candidate_hidden,candidate_logits=candidate.layer_probe(
                tok,row["question"],row["feature"])
            teacher_ids,teacher_meta=teacher.tokens(
                tok,row["question"],row["feature"],copy)
            actual_ids,copy_meta=candidate.tokens(
                tok,row["question"],row["feature"],copy)
            summary={
              "TRAIN_id":row["TRAIN_id"],
              "question_sha256":digest(row["question"].encode()),
              "original_FP32_43M_from_CALIBRI_ids":teacher_ids,
              "actual_FP32_to_original_dict_CB61_ids":actual_ids,
              "same_generated_full_plus_EOS":teacher_ids==actual_ids,
              "dense_FP32_original_S93_expected_gold":row["expected"],
              "full_correct_vs_original_gold":actual_ids==row["expected"],
              "max_next_vocab_logit_abs":
                  float(torch.max(torch.abs(teacher_logits-candidate_logits))),
              "all_original_12layer_GT_cosines":[
                  float(torch.nn.functional.cosine_similarity(x[None],y[None]).item())
                  for x,y in zip(teacher_hidden,candidate_hidden)],
              "full_original_hidden_L2":float(torch.linalg.vector_norm(
                  teacher_hidden[-1]-candidate_hidden[-1])),
              "FP32_mirror_text":tok.decode(teacher_ids,skip_special_tokens=True),
              "actual_reconstructed_source_text":tok.decode(actual_ids,skip_special_tokens=True),
              "neural_S98_copy_original":teacher_meta,
              "neural_S98_copy_new":copy_meta,
            }
            scored.append(summary)
            dump(out/"S7B_INCREMENTAL_ACTUAL_FP32_vs_LEARNED_DICT_KV_GT12.json",scored)
            log("S7B_actual_43M_FP32_dictionaries_full12_GT",
                prompt=i+1,full_match=summary["same_generated_full_plus_EOS"],
                hidden_L2=round(summary["full_original_hidden_L2"],7),
                logits_abs=round(summary["max_next_vocab_logit_abs"],7),
                answer=summary["actual_reconstructed_source_text"][:70])
    finally:
        teacher.close()
        candidate.close()
    package={
      "single_new_physically_reconstructed_CB61":source_file.stat().st_size,
      "one_actual_S142_trained_full_43M_sidecar":a.S142_FP16.stat().st_size,
      "original_S94_three_profile_bank":a.S94_bank.stat().st_size,
      "original_S95_prompt_only_router":a.S95_router.stat().st_size,
      "original_S98_second_token_copier":a.S98_copy_net.stat().st_size,
      "runtime_manifest":0,
    }
    report={
       **audit,
       "execution_completed":True,
       "input_real_FP32_base_parameters":BASE_36M,
       "reference_FP32_total_nominal_architecture_values":TOTAL_43M,
       "original_matrix_count":len(matrices),
       "actual_FP32_prior_inverted_codegroups":code_total,
       "old_code_indices_identical_POSTHOC":code_identical,
       "old_code_indices_agreement_POSTHOC":code_identical/code_total,
       "before_CB61_SHA":digest(a.S93_CB61.read_bytes()),
       "new_compiled_CB61_SHA":digest(cb61),
       "new_compiled_CB59_SHA":digest(reconstructed),
       "new_CB61_file":source_file.name,
       "original_source_and_recompiled_source_byte_identical":source==reconstructed,
       "actual_FP32_43M_mirror_vs_nearest_trained_dict_full_response_matches":
           sum(int(s["same_generated_full_plus_EOS"]) for s in scored),
       "actual_S93_original_TRAIN_debug_cases":len(scored),
       "original_gold_full_matches":sum(int(s["full_correct_vs_original_gold"]) for s in scored),
       "per_original_matrix_reconstructed_error":matrix_reports,
       "actual_all12_layer_GT_and_full_kv_output":scored,
       "actual_original_S126_Q2_full12_trained_gates_zero":True,
       "actual_S111_additional_donor_gates_absent_or_zero":True,
       "original_S87_local_GT3_and9_trainable":True,
       "all_physically_paid_component_bytes":package,
       "complete_effective_denominator_excluding_generated_6p88M_bpp":None,
       "all_inclusive_nominal_43p26M_bpp":None,
       "original_independent_FP32_43M_to_CALIBRI_conversion_achieved":False,
       "new_unseen_QA_or_test_opened":False,
       "model_accepted":False,
    }
    manifest=out/"S7B_REAL_12LAYER_FP32_to_CALIBRI_CODEBOOK_TRANSFER_REPORT.json"
    for _ in range(24):
        dump(manifest,report)
        manifestlen=manifest.stat().st_size
        if package["runtime_manifest"]==manifestlen:break
        package["runtime_manifest"]=manifestlen
        total=sum(package.values())
        report["all_inclusive_nominal_43p26M_bpp"]=8*total/TOTAL_43M
        report["complete_effective_denominator_excluding_generated_6p88M_bpp"]=8*total/BASE_36M
    else:
        raise AssertionError("No convergence in physically accounted 43M model manifest")
    dump(manifest,report)
    if (manifest.stat().st_size!=package["runtime_manifest"] or
        report["all_inclusive_nominal_43p26M_bpp"]>MAX_BPP):
        raise AssertionError("S7B physically persisted full real 43M source violates 0.7bpp")
    log("S7B_FINAL_REAL_43M_FP32_to_LEARNED_5bit_GT_SOURCE",
        complete_original_FP32_mirror_count=len(scored),
        matching_all_full_EOL=sum(s["same_generated_full_plus_EOS"] for s in scored),
        matching_original_5bit_groups=code_identical,
        total_5bit_groups=code_total,
        physical_nominal_bpp=report["all_inclusive_nominal_43p26M_bpp"],
        physical_excluding_6p88M_bpp=report[
            "complete_effective_denominator_excluding_generated_6p88M_bpp"],
        genuinely_independent_pretrained_FP32_compared=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for item in ("S93_CB61","S93_sidecar","S142_FP16","S94_bank",
                 "S94_report","S95_router","S95_report",
                 "S98_copy_net","S98_report","out"):
        p.add_argument(item,type=Path)
    run(p.parse_args())
