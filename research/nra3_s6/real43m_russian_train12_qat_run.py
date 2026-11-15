"""S6: train directly from TRUE S5 43M CPU Russian TRAIN12 hidden onto paid CB59.

No synthetic hidden input can substitute for physically reloaded S5.
No evaluation of unseen Russian data; all shown improvement is in-sample
TRAIN only. Real Calibri + Green Tunnel unchanged; no Jessica chat tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config
from research.nra3_s1.entropy_envelope import (
    store_lossless_cb59,restore_lossless_cb59,
)
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s4.complete_prefix_training_contract import (
    train_only_multitask_loss,
)
from research.nra3_s6.physical_joint_grammar_qat import (
    S5_ORIGINAL_RUN,S5_SOURCE_SHA256,S194_FP16_SHA256,
    PhysicallyPaidNRA3QAT,load_pinned_real43m_train12,
    train_locked_12_real_hidden,
)


PARENT_S3_SHA256="b8257d28d127733a59bf32869d698983abe732f76222a6fbbe36b4c0b02d0d1f"
PARENT_S3_RUN=43696909386


def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s5-hidden",type=Path,required=True)
    p.add_argument("--s5-report",type=Path,required=True)
    p.add_argument("--s3-cb59",type=Path,required=True)
    p.add_argument("--s3-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("S6 never overwrites another physically paid TRAIN result")
    torch.set_num_threads(2)
    torch.manual_seed(306006)
    samples,s5=load_pinned_real43m_train12(args.s5_hidden,args.s5_report)
    source=args.s3_cb59.read_bytes()
    if sha(source)!=PARENT_S3_SHA256 or len(source)!=653798:
        raise ValueError("S6 physically pinned S3 full 6M CB59 source not loaded")
    s3=json.loads(args.s3_report.read_text())
    if (s3["whole_area_unique_parameters"]!=6_106_154 or
        s3["physical_CB59_sha256"]!=PARENT_S3_SHA256 or
        s3["packed_archive_bytes"]!=len(source) or
        s3["no_real_language_training"] is not True or
        s3["candidate_accepted"] is not False or
        s3["output_gate"]!=0):
        raise ValueError("S6 exact synthetic S3 prior parent SHA/ABI drift")
    cfg=NRA3Config()
    qat=PhysicallyPaidNRA3QAT(source,cfg)
    metrics=train_locked_12_real_hidden(qat,samples,steps=24,lr=.012)
    paid=qat.save_full_paid_physical_cb59()
    physical_counts=qat.physical_change_counts(paid)
    if (len(paid)!=len(source) or sha(paid)==sha(source)
            or not physical_counts["grammar_matrix_changes"]
            or not physical_counts["semantic_matrix_changes"]):
        raise AssertionError("S6 no exact physically paid grammar+semantics learning")
    # Exact native packed C inference of the *trained* complete 6M area;
    # no dense 2D matrix can be loaded into the runtime decoder.
    args.out.mkdir(parents=True,exist_ok=True)
    native_lib=args.out/"lib_calibri_cb59_s6_native_cpu.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                    "research/calibri/direct_kernel_v59.c","-o",str(native_lib)],
                   check=True)
    packed=NRA3NativePackedCPU(paid,cfg,native_lib)
    with torch.inference_mode():
        actual_losses=[
            float(train_only_multitask_loss(packed,row.hidden320.unsqueeze(0),
                                           row.row,row.aligned).total)
            for row in samples
        ]
    native_loss=sum(actual_losses)/len(actual_losses)
    if (abs(native_loss-metrics["after_train12_multitask_loss"])>0.03
            or not native_loss<metrics["before_train12_multitask_loss"]):
        raise AssertionError("S6 physically paid C CPU forward lost measured TRAIN result")
    if (len(list(packed.parameters()))!=0 or
            packed.native_cpu_weights_dense_resident or
            packed.output_gate!=0 or packed.candidate_accepted):
        raise AssertionError("S6 native source contract or output gate changed")
    archive=args.out/"NRA3_S6_43M_REAL_HIDDEN_RU_TRAIN12_PAID_CB59.cb59"
    archive.write_bytes(paid)
    if archive.read_bytes()!=paid or sha(archive.read_bytes())!=sha(paid):
        raise AssertionError("S6 real TRAIN CB59 physical disk reload drift")
    entropy=store_lossless_cb59(paid)
    envelope=args.out/"NRA3_S6_43M_REAL_HIDDEN_RU_TRAIN12_PAID_S61.n3s61"
    envelope.write_bytes(entropy)
    if restore_lossless_cb59(envelope.read_bytes())!=paid:
        raise AssertionError("S6 real TRAIN physical entropy archive fails exact reload")
    report={
      "stage":"NRA3_S6_ACTUAL_FROZEN_CALIBRI_43M_RUSSIAN_TRAIN12_PHYSICALLY_PAID_CB59_QAT",
      "real_parent_43M_source_frozen":True,
      "parent_S5_run_id":S5_ORIGINAL_RUN,
      "parent_S5_real43M_feature_sha256":S5_SOURCE_SHA256,
      "parent_S194_43M_FP16_sha256":S194_FP16_SHA256,
      "parent_S3_synthetic_paid_CB59_run_id":PARENT_S3_RUN,
      "parent_S3_CB59_sha256":PARENT_S3_SHA256,
      "parent_S4_real_Russian_UD_TRAIN12_sha256":
        s5["S4_RU_UD_TRAIN12_CANONICAL_SHA256"],
      "actual_Russian_TRAIN_only_sentences":len(samples),
      "actual_genuine_43M_layer8_hidden_used":True,
      "actual_original_43M_CPU_hidden_feature_file_sha256":
        s5["actual_fp16_feature_sha256"],
      "actual_original_616_J7_confirmed_at_capture":True,
      "no_real_parent_43M_or_GT_update":True,
      "true_physical_6M_packed_CPU_forward_completed":True,
      "true_native_C_cpu_training_source_mean_loss":native_loss,
      "prior_cb59_original_physical_bytes":len(source),
      "trained_cb59_new_physical_bytes":len(paid),
      "trained_cb59_new_physical_sha256":sha(paid),
      "trained_cb59_filename":archive.name,
      "trained_cb59_new_source_bpp":8*len(paid)/packed.unique_parameters,
      "trained_s61_new_physical_bytes":len(entropy),
      "trained_s61_new_sha256":sha(entropy),
      "trained_s61_filename":envelope.name,
      "trained_s61_new_source_bpp":8*len(entropy)/packed.unique_parameters,
      "all_original_43_matrix_5bit_indices_unchanged":True,
      "physical_1d_source_unchanged":True,
      "original_unique_new_NRA3_parameters":packed.unique_parameters,
      "grammar_semantic_joint_training_done":True,
      "relational_core_explicit_graph_training_done":False,
      "real_original_43M_full_J7_candidate_after_training_tested":False,
      "independent_heldout_Russian_grammar_accuracy_tested":False,
      "new_full_answer_EOS_generalization_tested":False,
      "matched_actual_CPU_speed_vs_CB59_measured":False,
      "complete_real_Calibri_GT_NRA_source_all_in_bpp_certified":False,
      "original_5_spent_EVAL128_untouched":True,
      "reserved_6th_S3CI_EVAL128_untouched":True,
      "result_is_only_TRAIN_fit_and_not_generalization":True,
      "output_gate":0,
      "candidate_accepted":False,
      **metrics,
      **physical_counts,
    }
    report_file=args.out/"NRA3_S6_REAL_HIDDEN_RUSSIAN_TRAIN12_PAID_QAT.json"
    report_file.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                           encoding="utf-8")
    print(json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
