"""Explain why native paid v_proj may lower NLL yet not restore meaning.

Only original, already-consumed v76 TRAIN FIT text drives these eight
paired 64-token full forward passes per arm. Freeze source CB153 with SHA,
verify exact original CB59, and measure FP32 operator on identical inputs
to separate upstream hidden-state drift from local quantization residual.
No new fit, validation, testing or physical bitstream changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import nn

from paid_lowrank_v76 import make_model,get_tokens
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v154.arms import subset

from .protocol import (FIT_DOMAINS,OPERATOR,KEY,CB153_SHA,
                       TOKENS_EACH,WIKI_REVISION,NEWS_REVISION,check)
from .capture import captured_fit
from .audit import decompose,summary


@torch.inference_mode()
def forward_local(module,inputs):
    y=module(inputs)
    if y.ndim!=2 or y.shape[0]!=TOKENS_EACH or not torch.isfinite(y).all():
        raise ValueError("original layer6 v_proj returns incorrect output shape")
    return y.detach().clone()


def pinned_fit():
    out={}
    origins={}
    for domain in FIT_DOMAINS:
        ids,origin=get_tokens(domain)
        expected=WIKI_REVISION if domain=="wiki_fit" else NEWS_REVISION
        if (origin.get("split")!="train" or origin.get("revision")!=expected or
                len(ids)!=TOKENS_EACH):
            raise ValueError("old original TRAIN FIT source revision or range changed")
        out[domain]=ids
        origins[domain]=origin
    return out,origins


def run(archive,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    blob=archive.read_bytes()
    source=unpack(blob)
    if source.sha256!=CB153_SHA or hashlib.sha256(source.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("not exact immutable original fully physically paid CB153 source")
    if KEY not in source.patches or len(source.patches)!=7:
        raise ValueError("paid original v_proj must exist among seven physically paid ops")
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("original reference architecture changed")
    original_operator=teacher.get_submodule(OPERATOR)
    if not isinstance(original_operator,nn.Linear):
        raise ValueError("must compare to genuine pretrained original v_proj")
    domains,provenance=pinned_fit()
    measurements={}
    with TemporaryDirectory(dir=out) as temporary:
        root=Path(temporary)
        file=root/"original_full_CB59.cb59"
        file.write_bytes(source.original)
        student,meta,_=make_model(
            "baseline",SimpleNamespace(original=file,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("pretrained 2D source is not packed")
        base_module=student.get_submodule(OPERATOR)
        for domain in FIT_DOMAINS:
            ids=domains[domain]
            x_teacher=captured_fit(teacher,ids)
            x_packed=captured_fit(student,ids)
            y_teacher=forward_local(original_operator,x_teacher)
            y_teacher_on_packed=forward_local(original_operator,x_packed)
            y_native_on_teacher=forward_local(base_module,x_teacher)
            y_native_on_packed=forward_local(base_module,x_packed)
            with install_paid_block(student,subset(source,(KEY,)),root):
                paid_op=student.get_submodule(OPERATOR)
                y_paid_on_teacher=forward_local(paid_op,x_teacher)
                y_paid_on_packed=forward_local(paid_op,x_packed)
            if student.get_submodule(OPERATOR) is not base_module:
                raise AssertionError("physical patch not fully restored")
            measurements[domain]=decompose(
                x_teacher,x_packed,y_teacher,y_teacher_on_packed,
                y_native_on_teacher,y_native_on_packed,
                y_paid_on_teacher,y_paid_on_packed)
        if any(x.ndim>=2 for x in student.parameters()):
            raise AssertionError("original 2D weights accidentally in packed model")

    del teacher
    finding=summary(measurements)
    result={
        "stage":"v158_ORIGINAL_LAYER6_VPROJ_WEIGHT_ERROR_VS_UPSTREAM_DRIFT",
        "original_pretrained_model":"SmolLM2-135M-Instruct",
        "exact_frozen_paid_CB153_sha256":source.sha256,
        "all_stored_paid_bits_still_counted":report(blob),
        "only_historical_original_TRAIN_FIT":provenance,
        "teacher_FP32_weights_used_for_ORACLE_diagnostic_NOT_deployment":True,
        "two_model_activations_captured_on_identical_source_ids":True,
        "same_reference_FP32_source_operator_reused_for_both_input_distributions":True,
        "no_new_model_FIT_validation_or_qa":True,
        "no_hidden_original_dense_source_in_inference_CBM":True,
        "source_weight_upstream_and_combined_relative_error":measurements,
        "findings":finding,
        "no_independent_v89_TEST":True,
        "no_semantics_instruction_or_quality_restoration_claim":True,
        "no_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    target=out/"v158_vproj_native_weight_vs_upstream_input_drift.json"
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "stage":result["stage"],
        "actual_original_source_physical_bpp":result["all_stored_paid_bits_still_counted"]["all_in_bpp_unique_original"],
        "operator_error_vs_input_error":finding,
        "report":str(target),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("physically_paid_original_CB153",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.physically_paid_original_CB153,args.out)
