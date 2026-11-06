"""CALIBRI v217 — exact original-weight source recovery inside the existing bits.

v216 one-bit vocabulary residual improved tied embedding weight SSE by 9.1%
but degraded public DEV QA to 2/16 and native speed to 59%. Additional logit
sidecars v213-v215 all hurt QA. The source itself is under-represented:
CB59 initializes each 32x8 codebook from a small fixed 4096-group sample,
then the v206 student changes those centroids through a tiny QA TRAIN suite.

This source-only candidate refits EACH of 211 already-paid FP16 codebooks to
ALL authentic pretrained FP32 weight vectors assigned to its existing 5-bit
codes. The globally optimal least-squares centroid conditional on the fixed
codes is sum(row_scale * original_weight) / sum(row_scale**2).
We preserve the original byte-exact group indices, scales, aliases, raw 1D
weights, metadata and CPU C ABI. Physical size and rate do NOT increase.

Prespecified three genuine physical checkpoint ablations:
  1. only tied input/output embedding/head codebook refit;
  2. all other 210 attention/MLP codebooks refit;
  3. all 211 codebooks refit.
All three are decided by FP32 source weight reconstruction only, NOT QA,
TRAIN answers, public DEV16, VALID, or held-out TEST. Each source and
its hash are sealed before ANY answers are generated. None is chosen as
a winner on public DEV; historical DEV16 is heavily reused and cannot be
declared independent. No main/LIVE/Jessica/Render/Supabase/APK edits.
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

VOCAB_MATRIX = "model.embed_tokens.weight"
CANDIDATE_MODES = ("head_only","body_only","full_source")
NUM_MATRICES = 211
WEIGHT_SSE_RELATIVE_ACCEPT_FLOOR = 1.e-12


def all_assigned_centroid_least_squares(handle,raw):
    """Sufficient statistics over ALL genuine FP32 source vectors.

    The method is deterministic conditional on physical v206 assignments
    and scales; unlike sampling 4096 groups it uses the full original tensor.
    """
    w=np.asarray(raw,dtype=np.float32)
    rows,cols=handle.shape
    if (w.shape!=(rows,cols) or cols%8 or
            not np.isfinite(w).all()):
        raise AssertionError("Invalid authentic pretrained matrix")
    groups=cols//8
    codes=unpack5(handle.packed.tobytes(),rows*groups)
    scales=handle.row_scales.astype(np.float32)
    if not np.isfinite(scales).all() or np.any(scales<=0):
        raise AssertionError("Pretrained packed row scales invalid")
    row_repeated=np.repeat(scales,groups)
    den=np.bincount(codes,weights=(row_repeated*row_repeated),
                    minlength=32).astype(np.float64)
    original=w.reshape(-1,8)
    num=np.empty((32,8),dtype=np.float64)
    for dimension in range(8):
        num[:,dimension]=np.bincount(
            codes, weights=original[:,dimension]*row_repeated,
            minlength=32)
    squared_original=float(np.sum(w.astype(np.float64)**2))
    before=handle.centroids.astype(np.float32)
    updated=before.copy()
    occupied=den>1.e-20
    updated[occupied]=(num[occupied]/den[occupied,None]).astype(np.float32)
    updated=updated.astype("<f2").astype(np.float32)
    if not np.isfinite(updated).all():
        raise AssertionError("Actual FP16 physical codebook overflow")
    def exact_source_squared_error(centers):
        energy=float(np.sum(den*np.sum(centers.astype(np.float64)**2,axis=1)))
        cross=float(2*np.sum(centers.astype(np.float64)*num))
        return max(0.0,squared_original+energy-cross)
    loss_before=exact_source_squared_error(before)
    loss_after=exact_source_squared_error(updated)
    improved=loss_after < loss_before*(1.-WEIGHT_SSE_RELATIVE_ACCEPT_FLOOR)
    # Improve original full-matrix FP32 reconstruction only. Never a QA
    # acceptance gate or instruction-answer-guided 5-bit mutation.
    result=updated if improved else before
    return result,{
        "squared_source_error_original":loss_before,
        "squared_source_error_refit":loss_after,
        "accepted_source_error_only":bool(improved),
        "FP16_cells_changed":int(np.count_nonzero(
            result.astype("<f2")!=before.astype("<f2"))),
        "original_group_indices_unchanged":True,
        "actual_groups_used":rows*groups,
        "original_FP32_vectors_consulted_not_QA":True,
    },codes


def run(v206_path,v161_path,v193_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(217)
    torch.set_num_threads(2)
    original_bytes=Path(v206_path).read_bytes()
    if hashlib.sha256(original_bytes).hexdigest()!=SOURCE_V206_SHA256:
        raise AssertionError("Not signed true 135M v206 source")
    original_archive=CB59(original_bytes)
    descriptions=original_archive.meta["tensor_descriptors"]
    matrix_names=[name for name in sorted(descriptions)
                  if descriptions[name]["dtype"]=="CB59_K32V8"]
    if len(matrix_names)!=NUM_MATRICES or VOCAB_MATRIX not in matrix_names:
        raise AssertionError("Authentic 211-matrix compressed model missing")
    if original_archive.handles["lm_head.weight"] is not original_archive.handles[VOCAB_MATRIX]:
        raise AssertionError("Original shared embedding/head source broken")
    teacher=original_teacher().cpu().eval()
    fp32_state=teacher.state_dict()
    if set(fp32_state)!=set(original_archive.meta["full_original_state_keys"]):
        raise AssertionError("Not pinned original FP32 tensor names")
    all_sources={}
    report={}
    changed=0
    for index,name in enumerate(matrix_names):
        original_weight=fp32_state[name].detach().cpu().numpy()
        repaired,stats,codes=all_assigned_centroid_least_squares(
            original_archive.handles[name],original_weight)
        old=original_archive.handles[name].centroids
        if stats["accepted_source_error_only"]:
            changed+=1
        all_sources[name]={
            "old":old.astype(np.float32).copy(),
            "new":repaired.astype(np.float32).copy(),
            "codes":codes.astype(np.uint8).copy(),
        }
        report[name]=stats
        if (index+1)%25==0 or name==VOCAB_MATRIX:
            print(json.dumps({"stage":"v217_all_source_vectors",
                "done":index+1,"total":NUM_MATRICES,
                "changed_dictionary_count":changed,
                "head_seen":name==VOCAB_MATRIX}),flush=True)
    del teacher,fp32_state
    if changed<=0:
        raise AssertionError("No source-only FP16 dictionary update")
    summary={
        "method":"weighted_LS_original_FP32_source_all_assigned_groups_fixed_real_5bit_indices",
        "exact_signed_original_v206_SHA256":SOURCE_V206_SHA256,
        "original_unique_parameters":p.FROZEN_SOURCE_135M_UNIQUE,
        "original_211_packed_matrices":NUM_MATRICES,
        "actual_reconstructed_group_count":sum(
            x["actual_groups_used"] for x in report.values()),
        "updated_dictionaries":changed,
        "total_original_source_SSE_before":sum(
            x["squared_source_error_original"] for x in report.values()),
        "total_refit_source_SSE_after":sum(
            x["squared_source_error_refit"] if x["accepted_source_error_only"]
            else x["squared_source_error_original"] for x in report.values()),
        "matrix_diagnostics":report,
        "candidate_ablations_precommitted":list(CANDIDATE_MODES),
        "NO_TRAIN_DEV_VALID_TEST_QA_answers_used_to_rebuild_or_select":True,
        "no_new_physical_weights_or_extra_bitstream":True,
        "no_live_or_Jessica_changes":True,
        "physical_sources":{}
    }
    save(out/"v217_weight_source_error_ONLY_diagnostics.json",summary)
    models={}
    for mode in CANDIDATE_MODES:
        sources={}
        for name,fields in all_sources.items():
            should_refit=(
                mode=="full_source" or
                (mode=="head_only" and name==VOCAB_MATRIX) or
                (mode=="body_only" and name!=VOCAB_MATRIX))
            centers=fields["new"] if should_refit else fields["old"]
            sources[name]=SimpleNamespace(
                centroids=torch.from_numpy(centers.copy()),
                codes=torch.from_numpy(fields["codes"].copy()))
        archive,seal=replace_discrete(original_archive,sources,strict=False)
        if (seal["complete_bpp"]>0.700 or
                len(archive)!=p.SOURCE_BYTES or
                seal["sha256"]==SOURCE_V206_SHA256):
            raise AssertionError("Source unchanged or physical budget exceeded")
        name="CALIBRI_v217_"+mode+"_same_0p66784bpp.cb59"
        (out/name).write_bytes(archive)
        summary["physical_sources"][mode]={
            "file":name,"sha256":seal["sha256"],
            "physical_bpp":seal["complete_bpp"],
            "changed_indices":seal["actual_changed_5bit_group_indices"],
            "changed_paid_FP16_centroid_cells":seal["actually_changed_FP16_codebook_cells"],
            "sealed_before_any_generation":True
        }
    save(out/"SEALED_v217_three_original_weight_refits_BEFORE_ANY_QA.json",summary)
    print(json.dumps({"stage":"v217_physically_sealed_before_QA",
                      "changed_dictionary_count":changed,
                      "sse_ratio":summary["total_refit_source_SSE_after"]/
                        max(summary["total_original_source_SSE_before"],1.e-30),
                      "sources":summary["physical_sources"]}),flush=True)
    # This happens only AFTER SHA sealing and source-only selection.
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    control=extract_original(Path(v161_path))
    if len(control.blob)!=p.SOURCE_BYTES:
        raise AssertionError("Missing original native CB59 CPU baseline")
    with TemporaryDirectory(dir=out) as td:
        tmp=Path(td)
        for name in ("CB59","CB206",*CANDIDATE_MODES):
            (tmp/name).mkdir()
        (tmp/"CB59"/"source.cb59").write_bytes(control.blob)
        cb59,_,_=make_model("baseline",SimpleNamespace(
            original=tmp/"CB59"/"source.cb59",folder=tmp/"CB59"))
        cb206,_,_=make_model("baseline",SimpleNamespace(
            original=Path(v206_path),folder=tmp/"CB206"))
        native={"CB206":cb206}
        for mode in CANDIDATE_MODES:
            new,_,_=make_model("baseline",SimpleNamespace(
                original=out/summary["physical_sources"][mode]["file"],
                folder=tmp/mode))
            if any(weight.ndim>=2 for weight in new.parameters()):
                raise AssertionError("Unpaid runtime dense pretrained parameters")
            native["CB217_"+mode]=new
        qa=newly_frozen_public_dev_16(native,tok,v193_path,out)
        if qa["scores"]["CB206"]["machine_confirmed"]!=3:
            raise AssertionError("Signed v206 answers drift")
        cpu={}
        cases=freeze_cases(all_cases())
        for mode in CANDIDATE_MODES:
            name="CB217_"+mode
            result=speed_pairs({"CB59":cb59,"CB186":native[name]},tok,cases)
            cpu[name]=result["median_speed_ratio_vs_CB59"]
    outcome={
        "completed":True,"precommitted_physical_sources":summary["physical_sources"],
        "pure_FP32_weight_source_recovery_not_public_QA_fit":True,
        "fp32_correct_PUBLIC_DEV16":qa["scores"]["FP32_existing_135M"]["machine_confirmed"],
        "original_v206_correct_PUBLIC_DEV16":qa["scores"]["CB206"]["machine_confirmed"],
        "v217_correct_PUBLIC_DEV16":{
            "CB217_"+mode:qa["scores"]["CB217_"+mode]["machine_confirmed"]
            for mode in CANDIDATE_MODES},
        "v217_wrong_PUBLIC_DEV16":{
            "CB217_"+mode:qa["scores"]["CB217_"+mode]["machine_recognized_wrong"]
            for mode in CANDIDATE_MODES},
        "v217_ungraded_PUBLIC_DEV16":{
            "CB217_"+mode:qa["scores"]["CB217_"+mode]["ungraded_not_known_wrong"]
            for mode in CANDIDATE_MODES},
        "native_32_token_CPU_speed_ratio_vs_CB59":cpu,
        "original_SOURCE_sse_ratio_after_to_before":
            summary["total_refit_source_SSE_after"]/
            max(summary["total_original_source_SSE_before"],1.e-30),
        "public_DEV16_repeated_exploratory_not_independent_TEST":True,
        "no_dev_tuned_model_or_checkpoint_selected":True,
        "NO_MAIN_LIVE_Jessica_Render_Supabase_APK":True
    }
    save(out/"v217_REAL_NATIVE_all_211_same_bit_source_only_QA_CPU.json",outcome)
    print(json.dumps({"stage":"v217_real_native_source_only_QA",**outcome}),flush=True)
    return outcome


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("actual_v206_CB59",type=Path)
    ap.add_argument("original_CB59_baseline",type=Path)
    ap.add_argument("original_FP32_frozen_QA",type=Path)
    ap.add_argument("output",type=Path)
    x=ap.parse_args()
    run(x.actual_v206_CB59,x.original_CB59_baseline,
        x.original_FP32_frozen_QA,x.output)
