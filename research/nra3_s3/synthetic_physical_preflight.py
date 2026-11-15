"""NRA3 S3 physical synthetic QAT certificate; NEVER real grammar TRAIN.

Reproducibly creates a full 6,106,154-weight *random* NRA3 research source,
CB59 packs it, trains one existing grammar projection's FP16 codebook and
row scales to approximate a synthetic teacher on 64 generated vectors,
then physically repacks the full checkpoint WITHOUT changing code indices,
saves SHA-verifiable CB59 + S61 alternative, and tests actual native C output.

No source real Calibri weights, S3CI sixth, prior consumed EVALs, original
J7 protected questions, Russian text, Green Tunnel mutations or LIVE calls.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3NeuralArea
from research.nra3_s1.physical_cb59 import encode_area
from research.nra3_s1.entropy_envelope import store_lossless_cb59,restore_lossless_cb59
from research.nra3_s2.native_cpu import NativeCB59Kernel,NRA3NativePackedCPU
from research.nra3_s3.paid_codebook_qat import (
    SyntheticTrainingGuard,train_sha_for,train_one_synthetic_codebook,
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("S3 physical result must never overwrite a prior archive")
    torch.set_num_threads(1)
    torch.manual_seed(303077)
    original_model=NRA3NeuralArea().cpu().eval()
    source,metadata,_=encode_area(original_model)
    name="grammar.0.ffn.0.weight"
    teacher=original_model.state_dict()[name].detach().clone()
    x=torch.randn(64,teacher.shape[1],dtype=torch.float32)
    y=F.linear(x,teacher)
    label="SYNTHETIC_NRA3_S3_TRAIN:UNTRAINED_FULL_6M_GRAMMAR_FFN"
    gate=SyntheticTrainingGuard(frozenset({train_sha_for(label)}),frozenset())
    updated,report=train_one_synthetic_codebook(
        source,name,x,y,gate,label,steps=64,lr=.035)
    if not report.after_train_mse<report.before_train_mse:
        raise AssertionError("S3 synthetic QAT made no TRAIN numerical progress")
    if not (len(source)==len(updated) and
            CB59(updated).meta==CB59(source).meta):
        raise AssertionError("S3 full source CB59 physical contract failed")
    # Validate actual deployed CPU operator consumes codes, not teacher weights.
    args.out.mkdir(parents=True,exist_ok=True)
    lib=args.out/"lib_calibri_cb59_native_s3.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                    "research/calibri/direct_kernel_v59.c","-o",str(lib)],
                   check=True)
    kern=NativeCB59Kernel(lib)
    native=kern.matmul(CB59(updated).handles[name],x)
    final_mse=float(F.mse_loss(native,y))
    if abs(final_mse-report.after_physical_reload_mse)>1e-4:
        raise AssertionError("S3 physical native C differs from QAT roundtrip")
    native_full=NRA3NativePackedCPU(updated,original_model.config,lib)
    with torch.inference_mode():
        result=native_full(torch.randn(1,1,320))
    if (native_full.unique_parameters!=6_106_154 or
            native_full.packed_matrix_count()!=43 or
            not bool(torch.isfinite(result.relations).all()) or
            len(native_full.state_dict())!=0):
        raise AssertionError("S3 full 6M native physical inference failed")
    packed_file=args.out/"NRA3_S3_UNTRAINED_BASE_SYNTHETIC_QAT_PAID_CB59.cb59"
    packed_file.write_bytes(updated)
    if packed_file.read_bytes()!=updated:
        raise AssertionError("S3 disk physical CB59 mismatch")
    envelope=store_lossless_cb59(updated)
    entropy_file=args.out/"NRA3_S3_UNTRAINED_BASE_SYNTHETIC_QAT_PAID_S61.n3s61"
    entropy_file.write_bytes(envelope)
    if restore_lossless_cb59(entropy_file.read_bytes())!=updated:
        raise AssertionError("S3 physical entropy reload changed CB59")
    summary={
        **asdict(report),
        "stage":"NRA2_3_S3_PHYSICALLY_PAID_SYNTHETIC_CODEBOOK_QAT_ONLY",
        "source_is_random_untrained_6M_neural_area":True,
        "synthetic_example_count":64,
        "trained_operator":name,
        "trainable_source_fields":"existing_FP16_K32x8_centroids_and_row_scales",
        "original_packed_matrix_count":43,
        "whole_area_unique_parameters":native_full.unique_parameters,
        "all_original_5bit_code_indices_unchanged":True,
        "native_C_trained_operator_mse":final_mse,
        "native_complete_6M_CPU_forward_executed":True,
        "native_matrix_dense_resident":False,
        "packed_archive_bytes":len(updated),
        "physical_CB59_sha256":hashlib.sha256(updated).hexdigest(),
        "packed_file":packed_file.name,
        "physical_s61_archive_bytes":len(envelope),
        "physical_s61_sha256":hashlib.sha256(envelope).hexdigest(),
        "physical_s61_file":entropy_file.name,
        "not_a_parent_GT_all_inclusive_bpp_certificate":True,
        "parent_43M_read_only_integration_verified":False,
        "real_616_J7_verified":False,
        "native_cpu_speed_ratio_vs_CB59_measured":False,
        "real_Russian_grammar_trained":False,
        "real_semantic_relational_abilities_trained":False,
        "independent_new_full_EOS_evaluation":False,
        "output_gate":0,
        "candidate_accepted":False,
    }
    path=args.out/"NRA3_S3_PHYSICAL_SYNTHETIC_QAT_REPORT.json"
    path.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
