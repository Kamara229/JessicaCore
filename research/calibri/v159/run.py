"""Map hidden activation corruption along the full original 30-layer model.

The v157 robust gate FAILED after exploring v_proj: larger untouched
WikiText source NLL +0.0372, teacher KL +0.0404 and only 15/32
WikiText windows improved NLL. v158 found upstream v_proj input drift
2.7-2.9x larger than its local 5-bit weight quantization error,
and paid 1-bit source weight correction worsened overall operator RMSE.

The next research question is upstream: WHERE does the original CB59
architecture first develop substantial divergence vs original FP32?
Capture input embedding, outputs of EACH of 30 real pretrained transformer
blocks, and final norm on EXACTLY previous 4x64 wiki_fit/news_fit tokens,
identical across FP32, native CB59, and native CB59 plus physically paid
original v_proj-only correction. Stage-level drift deltas are correlative,
NOT proof that a block is uniquely responsible or permission to reallocate
source rate. There is no training and no fresh TEST/VALID exposure.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v154.arms import subset
from v158.run import pinned_fit

from .protocol import (CB153_SHA,MODEL_ARMS,FIT_DOMAINS,
                       EXPERIMENTAL_PAID_PATH,check_protocol)
from .evidence import verify_prior
from .trace import run_original_fit_trace
from .audit import compare_domain,summarize


def run(physical_archive,v157_json,v158_json,out):
    torch.set_num_threads(2)
    check_protocol()
    out.mkdir(parents=True,exist_ok=True)
    frozen_evidence=verify_prior(v157_json,v158_json)
    data=physical_archive.read_bytes()
    paid=unpack(data)
    if (paid.sha256!=CB153_SHA or
            hashlib.sha256(paid.original).hexdigest()!=ORIGINAL_CB59_SHA):
        raise ValueError("model archive must equal pinned immutable original CB153")
    if len(paid.patches)!=7 or EXPERIMENTAL_PAID_PATH not in paid.patches:
        raise ValueError("not original physically paid transformer layer6 v_proj")
    if frozen_evidence["original_paid_source_sha256"]!=paid.sha256:
        raise ValueError("original source or preceding measurement drifted")
    tokens,origin=pinned_fit()
    if tuple(tokens)!=FIT_DOMAINS:
        raise ValueError("training data from other sources unexpectedly present")
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("original pretrained teacher is not 30-layer SmolLM2")

    measured={}
    with TemporaryDirectory(dir=out) as scratch:
        root=Path(scratch)
        original=root/"original_CB59_direct_full_source.cb59"
        original.write_bytes(paid.original)
        student,metadata,_=make_model("baseline",SimpleNamespace(
            original=original,folder=root))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("original pretrained trained 2D matrix rehydrated")
        if len(student.model.layers)!=30:
            raise ValueError("not same 30-layer native packed source")
        native_embedding=student.model.embed_tokens
        native_head=student.lm_head
        if native_embedding.handle is not native_head.handle:
            raise ValueError("original tied embeddings unexpectedly copied")

        for domain in FIT_DOMAINS:
            original_teacher_hidden=run_original_fit_trace(teacher,tokens[domain])
            original_CB59_hidden=run_original_fit_trace(student,tokens[domain])
            with install_paid_block(student,subset(paid,(EXPERIMENTAL_PAID_PATH,)),root):
                onebit_vproj_hidden=run_original_fit_trace(student,tokens[domain])
            if student.model.embed_tokens is not native_embedding:
                raise ValueError("paid source unexpectedly changed input embedding")
            if student.lm_head is not native_head:
                raise ValueError("paid original operator changed shared output head")
            if any(p.ndim>=2 for p in student.parameters()):
                raise ValueError("dense original weights leaked into candidate")
            measured[domain]=compare_domain(
                original_teacher_hidden,original_CB59_hidden,onebit_vproj_hidden)
        del student

    del teacher
    findings=summarize(measured)
    report_data={
        "stage":"v159_ORIGINAL_30BLOCK_REAL_ACTIVATION_DRIFT_MAP",
        "full_physical_original_source_CB153_SHA256":paid.sha256,
        "whole_original_paid_source_0p7_bpp":report(data),
        "prior_actual_v157_FAIL_and_v158_INPUT_DRIFT_diagnostics":frozen_evidence,
        "old_original_wiki_and_news_TRAIN_FIT_source_only":origin,
        "all_32_original_operator_stage_activation_comparisons_per_domain":True,
        "original_FP32_teacher_is_DIAGNOSTIC_reference_ONLY":True,
        "pretrained_model_weights_retrained_or_optimized":False,
        "original_native_CB59_5bit_and_original_paid_vproj_onebit_only":True,
        "model_arms":list(MODEL_ARMS),
        "teacher_relative_RMSE_COSINE_and_stage_delta":findings,
        "single_layer_effect_cannot_be_causally_identified_by_simple_stage_delta":True,
        "no_official_independent_v89_TEST_or_other_new_validation":True,
        "no_SciQ_BoolQ_or_semantic_recovery_claim":True,
        "no_LIVE_Jessica_Render_Supabase_APK_change":True,
    }
    filename=out/"v159_teacher_vs_0p67bpp_full_30block_hidden_drift.json"
    filename.write_text(json.dumps(report_data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report_data["stage"],
        "physical_source_bpp":report_data["whole_original_paid_source_0p7_bpp"]["all_in_bpp_unique_original"],
        "v157_robustness":"FAIL",
        "most_correlated_upstream_divergence":findings["common_heaviest_5_RMSE_increments"],
        "model_emb_final_and_paid_layer6_drift":{
            d:{k:measured[d][k] for k in (
                "input_tied_embedding_original_CB59_RMSE",
                "original_block6_post_output_RMSE",
                "paid_vproj_block6_post_output_RMSE",
                "native_CB59_final_norm_RMSE",
                "paid_vproj_final_norm_RMSE",
                "paid_vproj_changes_drift_at_block6",
                "paid_vproj_changes_drift_after_final_norm")}
            for d in FIT_DOMAINS},
        "report":str(filename),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("complete_CB153_archive",type=Path)
    parser.add_argument("completed_real_v157_JSON",type=Path)
    parser.add_argument("completed_real_v158_JSON",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    run(a.complete_CB153_archive,a.completed_real_v157_JSON,
        a.completed_real_v158_JSON,a.out)
