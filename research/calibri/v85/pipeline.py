"""Orchestration facade: learn genuinely activation-routed packed operators.

Base inference: actual original CB59 native C, whole pretrained model.
Dynamic operators: 2 direct native C variants + tiny trained FP16 gate.
No original pretrained source weights or corpus labels in the gate.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from paid_lowrank_v76 import make_model
from v83.study import original_packed_source
from v84.sources import TARGETS, original_teacher
from v84.valid import choose_guarded_variant

from .physical import CB85, inspect_source_pair, create_packed
from .sources import collect,paired_model_activations
from .gate_train import native_dual_outputs,fit_activation_gate
from .validation import score,variant_sets


def run(original_path:Path,rewritten_v84_path:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(85)
    torch.set_num_threads(2)
    original=original_path.read_bytes()
    reformed=rewritten_v84_path.read_bytes()
    audit_original=original_packed_source(original_path)
    unmodified,alternative=inspect_source_pair(original,reformed,TARGETS)
    teacher=original_teacher()
    student,physical,_=make_model("baseline",SimpleNamespace(
        original=original_path,folder=out_dir))
    if not physical["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("trained original 2D dense weights leaked into student")
    if physical["complete_physical_SHA256"]!=audit_original["original_CB59_source_sha256"]:
        raise AssertionError("wrong native pretrained student source")

    # Only TRAIN data before the new candidate's FP16 params are frozen.
    train=collect(teacher,student,"train")
    gates={}
    gate_logs={}
    for path in TARGETS:
        x,target=paired_model_activations(train,path)
        native=student.get_submodule(path)
        alt=alternative.handles[path+".weight"].centroids
        a,b=native_dual_outputs(native,alt,x)
        (w,bias),measure=fit_activation_gate(x,target,a,b)
        gates[path]=(w,bias)
        gate_logs[path]=measure
        print(json.dumps({
            "stage":"v85_conditioned_native_operator_TRAIN_fitted",
            "target":path,
            "TRAIN_orig_error":round(measure[
                "original_student_TRAIN_relative_squared_error"],5),
            "TRAIN_gated_error":round(measure[
                "trained_gated_TRAIN_relative_squared_error"],5),
            "std_gate":round(measure["gate_std"],5)
        }),flush=True)

    physically_complete=create_packed(original,reformed,gates)
    loaded=CB85(physically_complete)
    stats=loaded.physical_stats()
    if stats["original_CB59_bytes"]!=len(original) or stats["routed_modules"]!=3:
        raise AssertionError("full physical source missing trained original operators")
    # Our FINAL physical FP16 source is now frozen. Do not train/select gate
    # parameters using any new VAL or TEST token.
    source_path=out_dir/"v85_RESEARCH_only_conditional_native_source.cb85"
    source_path.write_bytes(physically_complete)
    print(json.dumps({"stage":"v85_FROZEN_before_first_VALID",
                      "full_physical_source_SHA256":stats["source_SHA256"],
                      "actual_full_bpw":stats["all_in_bpw"],
                      "includes_all_actual_runtime_gate_weights":True}),flush=True)

    valid=collect(teacher,student,"valid")
    baseline=score(student,valid,loaded,())
    variants={}
    for name,paths in variant_sets(TARGETS):
        variants[name]=score(student,valid,loaded,paths)
        print(json.dumps({"stage":"v85_actual_native_conditional_VALID",
                          "variant":name,"mean_full_vocabulary_KL":
                              round(variants[name]["mean_KL"],6)}),flush=True)
    winner,go=choose_guarded_variant(baseline,variants)
    original_after=hashlib.sha256(original_path.read_bytes()).hexdigest()
    if original_after!=audit_original["original_CB59_source_sha256"]:
        raise AssertionError("old pretrained original source mutated")
    result={
      "study":"CALIBRI_v85_original_pretrained_operator_adaptive_rearchitecture",
      "experimental_type":"activation_conditioned_mixture_of_two_native_C_operators",
      "original_CB59_complete_source":audit_original,
      "fully_paid_new_adaptive_CB85":stats,
      "parent_codebook_rewrite_SHA":hashlib.sha256(reformed).hexdigest(),
      "train_only_split_provenance":{k:v["source"] for k,v in train.items()},
      "new_untouched_before_training_valid_provenance":{
          k:v["source"] for k,v in valid.items()},
      "train_fit_learned_independent_gates":gate_logs,
      "actual_original_native_CB59_VALID":baseline,
      "physical_CB85_variant_VALID":variants,
      "guarded_variant_selection":go,
      "valid_selected_compressed_module_set":
            list(variants[winner]["adaptive_operator_paths"]) if
            winner in variants else [],
      "no_change_to_the_pretrained_original_source_file":True,
      "only_original_TRAIN_validation_used_no_TEST":True,
      "research_only_original_Jessica_untouched":True,
      "not_end_to_end_quality_or_real_assistant_test":True,
      "fully_native_fused_adaptive_operator":False,
      "caveats":"Two original native C projections per dynamic operator, "
                "mixed with a Torch CPU input gate. 256 TRAIN tokens and "
                "128 distinct TRAIN-VALID tokens; no independently unseen TEST. "
                "Gate training targets teacher local outputs on actual packed "
                "student activations. All additional codebook/gates accounted. "
                "No evidence of broad language reasoning gains without new TEST."
    }
    report=out_dir/"v85_NATIVE_runtime_conditional_operator_TRAIN_VALID_audit.json"
    report.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
      "stage":"v85_REAL_native_C_conditional_original_TRAIN_VALID_completed",
      "full_bpw":round(stats["all_in_bpw"],9),
      "baseline_mean_KL":round(baseline["mean_KL"],6),
      "best_mean_KL":round(go["selected_mean_KL"],6),
      "research_VALID_go":go["newly_rewritten_source_valid_guard_pass"],
      "candidate":winner,
      "TEST_seen":False,
      "Jessica_changed":False,
    },ensure_ascii=False),flush=True)
    return result
