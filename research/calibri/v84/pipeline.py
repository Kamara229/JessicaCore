"""CALIBRI v84 orchestration facade for real source-level rewritten native operators.

Do NOT evaluate TEST, deploy, rewrite other model weights, or claim competence.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from direct_runtime_v59 import PackedLinear
from paid_lowrank_v76 import make_model, SOURCE_BYTES
from v83.study import original_packed_source

from .archive import rewrite_codebooks, manifest_audit
from .fit import CHECKPOINTS, Projection, train_codebook, local_relative_error
from .sources import TARGETS, original_teacher, capture_original, paired_inputs
from .valid import score_native_runtime, choose_guarded_variant


def run(original: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(84)
    original_bytes = original.read_bytes()
    meta = original_packed_source(original)
    teacher = original_teacher()
    student, packed_audit, _ = make_model(
        "baseline", SimpleNamespace(original=original, folder=out_dir))
    if (not packed_audit["zero_dense_pretrained_2D_hot_weights"] or
            packed_audit["complete_physical_SHA256"] !=
                meta["original_CB59_source_sha256"]):
        raise AssertionError("not real original complete native packed CB59 model")
    real = capture_original(teacher)
    baseline = score_native_runtime(student, real, original_bytes, changes={})
    print(json.dumps({"stage":"v84_pre_fit_genuine_baseline",
                      "valid_mean_KL":baseline["mean_KL"]}),flush=True)

    proposed = {}
    operators = {}
    for path in TARGETS:
        native = student.get_submodule(path)
        if not isinstance(native, PackedLinear):
            raise AssertionError("missing original native packed operator")
        op = Projection.from_native(path, native)
        inputs, targets = paired_inputs(real, ("wiki_fit","news_fit"),path)
        validate_x, validate_y = paired_inputs(
            real, ("wiki_valid","news_valid"), path)
        # Verify differentiable source construction agrees with the actual
        # direct-C codebook operator BEFORE rewriting any source bytes.
        with torch.inference_mode():
            index = min(4, len(inputs))
            native_outputs = native(inputs[:index])
            source_outputs = op.output(inputs[:index],op.base_codebook)
            relative_difference = float(
                (native_outputs-source_outputs).double().norm() /
                torch.clamp(native_outputs.double().norm(),min=1e-12))
            if relative_difference > 0.0002:
                raise AssertionError(
                    "TRAIN surrogate differs from native packed operator: "
                    + path + f" relative={relative_difference}")
        variants = train_codebook(op,inputs,targets)
        valid_baseline = local_relative_error(
            op,validate_x,validate_y,op.base_codebook.numpy())
        deltas = {}
        best_step = None
        best_local = valid_baseline
        for step in CHECKPOINTS:
            test_centers = variants[step]
            valid_relative = local_relative_error(
                op,validate_x,validate_y,test_centers)
            deltas[str(step)] = {
                "heldout_TRAIN_VALID_teacher_conditioned_relative_L2": valid_relative,
                "FP16_changed_centroid_scalars": int(np.count_nonzero(
                    test_centers != op.base_codebook.numpy()))}
            if valid_relative < best_local - 1e-6:
                best_local, best_step = valid_relative,step

        # Even if local improvement does not materialize, archive the
        # preregistered last checkpoint as an exploratory REWRITE, but
        # never count it as a selected quality-preserving source.
        chosen_step = best_step if best_step is not None else CHECKPOINTS[-1]
        proposed[path] = variants[chosen_step]
        operators[path] = {
            "train_examples": len(inputs),
            "valid_examples": len(validate_x),
            "native_vs_training_operator_relative_difference":
                relative_difference,
            "unmodified_local_relative_L2": valid_baseline,
            "best_local_valid_relative_L2": best_local,
            "best_local_valid_checkpoint": best_step,
            "physical_checkpoint_for_independent_full_model_VALID": chosen_step,
            "checkpoints": deltas,
            "local_VALID_improvement_confirmed": best_step is not None,
        }
        print(json.dumps({"stage":"v84_rewritten_real_training_codebook",
                          "module":path,"chosen_step":chosen_step,
                          "local_L2_before":round(valid_baseline,6),
                          "local_L2_after":round(best_local,6)}),flush=True)

    variants = {}
    changes = {}
    for path in TARGETS:
        changes[path] = proposed[path]
        variants[path] = score_native_runtime(
            student,real,original_bytes,{path:proposed[path]})
        print(json.dumps({"stage":"v84_true_full_student_valid",
                          "operator":path,
                          "KL":round(variants[path]["mean_KL"],6)}),
                          flush=True)
    variants["all_three_individually_rewritten"] = score_native_runtime(
        student,real,original_bytes,changes)
    best_name,selection = choose_guarded_variant(baseline,variants)
    # Always retain a fully physical same-byte rewrite as a research artifact.
    # Source is NOT approved unless selection passes two independent
    # TRAIN-VALID corpus guards, and STILL is never automatically deployed.
    trial = rewrite_codebooks(original_bytes, changes)
    trial_file = out_dir / "v84_research_only_three_operator_CB59.cb59"
    trial_file.write_bytes(trial)
    result = {
        "experiment":"CALIBRI_v84_genuine_pretrained_adaptive_operator_rewrite",
        "method":"TRAIN_fitted_FP16_native_CB59_centroid_reparameterization",
        "no_sidecar_and_no_EXTRA_bits":True,
        "same_complete_trained_source_physical_bytes":len(trial),
        "bits_per_original_unique_param":
            8*len(trial)/meta["original_unique_parameters"],
        "baseline_source_sha":meta["original_CB59_source_sha256"],
        "source_trial_audit":manifest_audit(original_bytes,trial),
        "original_teacher_pinned":meta["original_real_pretrained_model"],
        "targets":list(TARGETS),
        "local_operator_rewrite_metrics":operators,
        "source_TRAIN_VALID_dataset_provenance":{
            k:v["source"] for k,v in real.items()},
        "baseline_native_original_CB59_TRAIN_VALID":baseline,
        "source_rewritten_full_model_variants_TRAIN_VALID":variants,
        "validation_guard":selection,
        "candidate_winner_after_TRAIN_VALID":best_name,
        "candidate_physical_file_even_if_guard_fails":str(trial_file),
        "source_changed_during_RUN":False,
        "original_CB59_source_SHA_unchanged":
            hashlib.sha256(original.read_bytes()).hexdigest() ==
                    meta["original_CB59_source_sha256"],
        "not_newly_unseen_TEST_and_not_production_ready":True,
        "untouched_Jessica_main_Render_Supabase_APK":True,
        "base_packed_operator_remains_native_C":True,
        "caveats":"Only 256 TRAIN and 128 VALID teacher-forced tokens; "
                   "codebook changes require true native revalidation; "
                   "VALID selection is NOT independent TEST; local distortion "
                   "importance does not prove downstream task quality; "
                   "only K32V8 codebooks are rewritten, not transformer topology.",
    }
    if len(trial) != SOURCE_BYTES or not result["original_CB59_source_SHA_unchanged"]:
        raise AssertionError("original pretrained source bytes altered unexpectedly")
    report = out_dir/"v84_real_original_TRAIN_VALID_operator_rewrite_audit.json"
    report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "stage":"v84_REAL_native_operator_rewrite_TRAIN_VALID_audited",
        "rewritten_operator_count":len(TARGETS),
        "physical_bpw":round(result["bits_per_original_unique_param"],9),
        "selected_variant":best_name,
        "baseline_VALID_KL":round(baseline["mean_KL"],6),
        "best_VALID_KL":round(selection["selected_mean_KL"],6),
        "all_three_VALID_KL":round(variants[
            "all_three_individually_rewritten"]["mean_KL"],6),
        "research_go":selection["newly_rewritten_source_valid_guard_pass"],
        "TEST_touched":False,
        "production_untouched":True,
    }),ensure_ascii=False),flush=True)
    return result
