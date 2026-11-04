"""Causal physical tied-input vs head correction, actual 0.6942bpp source.

Revisit original CB140 one-extra-bit tied token weight, which v144 selected
0 input / 0.5 head on a small VALID. v159 exposed original CB59 input
embedding activation RMSE ~0.72, with subsequent severe block0 drift.
We MUST NOT assume that locally improving tied embedding repairs hidden
states: compare real FP32 teacher activation trace and true heldout
gold next-token NLL/KL for fixed input/head interventions.

Source archive SHA immutable (11,672,747 bytes). All 4 arms charge full
original physical CB140 source: 0/0, 1/0, 0/1, 1/1. The 4-byte proposed
gain metadata is NOT actually serialized here, so no deployable updated
archive claim. No training, no validation-selected hyperparameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from time import perf_counter

import torch
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v140.codec import unpack,report
from v144.operators import Paired
from v144.run import get_native_logits
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v158.run import pinned_fit
from v159.trace import run_original_fit_trace
from v159.audit import compare_domain
from v142.scoring import compare_logits

from .protocol import (ORIGINAL_V140_SHA,ARMS,GAINS,DOMAINS,WINDOW,
                       WINDOWS,check)
from .evidence import verify_v159
from .sources import fresh_valid
from .audit import summarize


@torch.inference_mode()
def old_fit_embedding_and_early_drift(teacher,student,paired):
    data,provenance=pinned_fit()
    matched={}
    if tuple(data)!=("wiki_fit","news_fit"):
        raise ValueError("previous original TRAIN source FIT was changed")
    for domain,ids in data.items():
        teacher_trace=run_original_fit_trace(teacher,ids)
        paired.configure(0.,0.)
        baseline=run_original_fit_trace(student,ids)
        paired.configure(1.,0.)
        improved=run_original_fit_trace(student,ids)
        measures=compare_domain(teacher_trace,baseline,improved)
        matched[domain]={
            "same_true_teacher_and_source_tokens_for_both_inputs":True,
            "original_5bit_embedding_relative_RMSE":
                measures["input_tied_embedding_original_CB59_RMSE"],
            "paid_6bit_embedding_relative_RMSE":
                measures["full_CB153_paid_vproj_teacher_vs_packed_drift_by_source_stage"]["input_embedding"]["relative_RMSE"]
                if "full_CB153_paid_vproj_teacher_vs_packed_drift_by_source_stage" in measures
                else measures["full_CB153_paid_vproj_teacher_vs_packed_drift_by_source_stage"]["input_embedding"]["relative_RMSE"],
            "actual_input_delta_at_block0":
                measures["paid_vproj_minus_native_CB59_teacher_RMSE_by_stage"]["block_00_output"],
            "actual_input_delta_at_block9":
                measures["paid_vproj_minus_native_CB59_teacher_RMSE_by_stage"]["block_09_output"],
            "actual_input_delta_at_block28":
                measures["paid_vproj_minus_native_CB59_teacher_RMSE_by_stage"]["block_28_output"],
            "actual_input_delta_at_final_norm":
                measures["paid_vproj_changes_drift_after_final_norm"],
            "post_input_operator_drift_by_all_32_stages":
                measures["paid_vproj_minus_native_CB59_teacher_RMSE_by_stage"],
            "source_only_not_new_VALID":True,
        }
    paired.configure(0.,0.)
    return matched,provenance


@torch.inference_mode()
def new_natural_text(teacher,student,paired):
    """Open completely untouched source after fixed gains are sealed."""
    samples,provenance=fresh_valid()
    measured={arm:{domain:[] for domain in DOMAINS} for arm in ARMS}
    runtimes={arm:{domain:0.0 for domain in DOMAINS} for arm in ARMS}
    for domain in DOMAINS:
        tokens=samples[domain]
        for idx in range(WINDOWS):
            ids=tokens[idx*WINDOW:(idx+1)*WINDOW][None,:]
            teacher_logits=teacher(input_ids=ids,use_cache=False).logits.detach()
            logits_cache={}
            for input_gain in (0.,1.):
                t0=perf_counter()
                base,paid_delta=get_native_logits(
                    student,paired,ids,input_gain)
                source_seconds=perf_counter()-t0
                logits_cache[input_gain]=(base,paid_delta,source_seconds)
            for arm,(i,h) in zip(ARMS,GAINS):
                base,delta,src_sec=logits_cache[i]
                logits=base+h*delta
                row=compare_logits(teacher_logits,logits,ids)
                row.update({
                    "window_index":idx,
                    "original_input_gain":i,
                    "original_head_gain":h,
                    "gold_not_used_for_fit":True,
                })
                measured[arm][domain].append(row)
                runtimes[arm][domain]+=src_sec/2
    return measured,provenance,runtimes


def run(archive,v159_old_diagnostic,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    proof=verify_v159(v159_old_diagnostic)
    raw=archive.read_bytes()
    frozen=unpack(raw)
    if frozen.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not original full physical CB140 tied 6bit source")
    if hashlib.sha256(frozen.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original 5bit source sha changed")
    if len(raw)>MAX_SOURCE_BYTES:
        raise ValueError("original full 0.7bpp source budget breached")
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("original teacher architecture changed")
    with TemporaryDirectory(dir=out) as temp:
        directory=Path(temp)
        original=directory/"original_pretrained_direct.cb59"
        original.write_bytes(frozen.original)
        student,meta,_=make_model(
            "baseline",SimpleNamespace(original=original,folder=directory))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("source pretrained matrix loaded 2D")
        paired=Paired(student,frozen,directory)
        if paired.input.handle is not paired.head.handle:
            raise ValueError("paid source lost original tied 5+1bit code handle")

        # Pre-existing original FIT only: does the same *physical* 1bit
        # token correction reduce the original hidden-state drift?
        source_fit,fit_origins=old_fit_embedding_and_early_drift(
            teacher,student,paired)

        # Commit the input-only/head-only candidate set to disk BEFORE
        # touching the fresh WikiText/News token sources.
        seal=out/"v160_fixed_gains_BEFORE_fresh_source.json"
        seal.write_text(json.dumps({
            "old_v159_diagnostic":proof,
            "complete_physical_CB140_sha256":frozen.sha256,
            "bytes_complete_CB140":len(raw),
            "hypothetical_extra_2_FP16_gain_metadata_bytes":4,
            "gain_metadata_is_NOT_physically_serialized":True,
            "precommitted_arms_and_gains":dict(zip(ARMS,GAINS)),
            "preexisting_TRAIN_FIT_only_before_seal":True,
            "fresh_original_TRAIN_slice_NOT_opened_at_seal":True,
        },ensure_ascii=False,indent=2)+"\n")

        measured,valid_provenance,elapsed=new_natural_text(
            teacher,student,paired)
        paired.restore()
        if any(p.ndim>=2 for p in student.parameters()):
            raise AssertionError("original pretrained weights unexpectedly dense")
    del teacher
    quality=summarize(measured)
    result={
        "stage":"v160_ORIGINAL_PAID_TIED_EMBEDDING_VS_OUTPUT_CAUSAL_VALID",
        "original_frozen_complete_CB140_source_SHA256":frozen.sha256,
        "original_complete_CB59_source_SHA256":ORIGINAL_CB59_SHA,
        "all_source_physically_paid_CB140":report(raw),
        "actual_complete_source_bytes":len(raw),
        "actual_source_bpp":8*len(raw)/TOTAL_UNIQUE,
        "input_head_gain_metadata_UNSERIALIZED":True,
        "hypothetical_full_source_if_4_new_metadata_bytes":len(raw)+4,
        "same_existing_physical_source_for_every_arm":True,
        "prior_original_v159_hidden_drift":proof,
        "old_source_TRAIN_FIT_provenance":fit_origins,
        "matched_old_source_causal_activation_traces":source_fit,
        "gain_set_frozen_before_new_VALID":seal.name,
        "new_TRAIN_split_source_VALID":valid_provenance,
        "actual_matched_8x64_gold_next_token_rows":measured,
        "causal_input_vs_head_quality":quality,
        "shared_decoder_and_head_timing_NOT_full_autoregressive_speed":elapsed,
        "protected_original_v89_independent_TEST_not_opened":True,
        "no_teacher_or_gold_new_source_used_for_fitting":True,
        "no_valid_release_until_head_gain_tags_physically_saved":True,
        "no_verified_semantic_QA_or_coherent_response_quality":True,
        "no_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    path=out/"v160_tied_embedding_head_causal_source.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":result["stage"],"all_in_original_bpp":result["actual_source_bpp"],
        "original_FIT_activation_drift":source_fit,
        "new_source_VALID_NLL_KL_comparisons":quality[
            "paired_per_domain_causal_input_head_effects"],
        "input_only_pass_both":quality[
            "paid_input_ONLY_improves_NLL_and_KL_on_both_new_domains"],
        "report":str(path),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_complete_CB140",type=Path)
    p.add_argument("actual_completed_v159_JSON",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.original_complete_CB140,args.actual_completed_v159_JSON,args.out)
