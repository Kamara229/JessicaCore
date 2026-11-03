"""Top-level CALIBRI v83 experiment facade: composition, no math internals.

This intentionally DOES NOT re-train / select / rewrite weights. The local
teacher-forced distortion atlas uses ONLY prior train split windows.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from .measure import MetricBook, PROJECTIONS, probe_one_window
from .study import original_packed_source, train_windows, summarize


def run(original: Path, output: Path) -> dict:
    torch.manual_seed(83)
    torch.set_num_threads(2)
    source = original_packed_source(original)
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_SHA, torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    packed, native_info, _ = make_model(
        "baseline", SimpleNamespace(original=original, folder=output.parent))
    if not native_info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("native model accidentally materialized original trained 2D")
    if native_info["complete_physical_SHA256"] != source["original_CB59_source_sha256"]:
        raise AssertionError("native runtime uses a different pretrained CB59 source")

    book = MetricBook()
    corpora = {}
    records = []
    for domain, provenance, windows in train_windows():
        corpora[domain] = provenance
        for i, window in enumerate(windows):
            witness = probe_one_window(teacher, packed, window["tokens"], book)
            records.append({"train_domain": domain, "window": i,
                            "original_token_window_sha256": window["sha256"],
                            **witness})
            print(json.dumps({
                "stage": "v83_TRUE_TRAIN_layer_probe",
                "domain": domain, "window": i, **witness
            }), flush=True)
    after = hashlib.sha256(original.read_bytes()).hexdigest()
    if after != source["original_CB59_source_sha256"]:
        raise AssertionError("original packed source unexpectedly modified")
    scoring = summarize(book)
    expected = len(teacher.model.layers) * len(PROJECTIONS)
    if scoring["measured_operators"] != expected:
        raise AssertionError("missing packed projection operator")
    if scoring["measured_layers"] != len(teacher.model.layers):
        raise AssertionError("missing full-model hidden state drift")
    if any(x["observations"] != len(records)
           for x in scoring["local_projection_metrics"].values()):
        raise AssertionError("one projection missing in one of the TRAIN windows")
    result = {
        "experiment": "CALIBRI_v83_actual_pretrained_135m_layer_error_atlas",
        "scope": "TRAIN_only_diagnostic_NOT_TEST_NOT_new_weights",
        "whole_pretrained_original": source,
        "runtime_audit": native_info,
        "corpora_train_provenance": corpora,
        "paired_true_source_windows": records,
        "experimental_outputs": scoring,
        "all_weights_frozen_and_unchanged": True,
        "no_floating_teacher_weights_in_native_student": True,
        "no_TEST_used_for_learning_selection_or_probe": True,
        "no_change_to_Jessica_main_Render_Supabase_APK": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "stage": "v83_ACTUAL_TRAIN_only_atlas_done",
        "compressed_original_bpw": round(source["original_CB59_physical_bpw"], 9),
        "teacher_forced_projection_count": scoring["measured_operators"],
        "paired_train_windows": len(records),
        "largest_local_error_modules": scoring[
            "highest_local_teacher_forced_error_energy"][:3],
        "new_model_trained": False,
        "TEST_touched": False
    }), flush=True)
    return result
