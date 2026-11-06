"""Post-seal continuation of CALIBRI v212 after the native speed API alias crash.

The 24 TRAIN-only recovery steps already completed and a physical CB59 source
was sealed BEFORE any VALID/DEV quality score.  Reuse that immutable source:
never retrain, never choose changes from public DEV, never touch LIVE.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v174.intervene import extract_original
from v174.protocol import freeze_cases
from v174.run import save
from v169.sources import all_cases
from v186.run import speed_pairs, validation_pairs
from v196.data import training_inputs
from v196.run import newly_frozen_public_dev_16
from v212.run import SOURCE_V206_SHA256
from v196 import protocol as p

# From the GitHub run log printed at the instant of physical sealing.
SEALED_V212_SHA256 = "612c1bc5f284b6b2561870f03e1ceb008764adf4130d0666c2c428bb7607719c"


def verify_sealed_files(v206_path, sealed_dir):
    sealed_dir = Path(sealed_dir)
    file = sealed_dir / "CALIBRI_v212_guarded_24step_135M.cb59"
    actual = file.read_bytes()
    sha = hashlib.sha256(actual).hexdigest()
    if sha != SEALED_V212_SHA256 or len(actual) != p.SOURCE_BYTES:
        raise AssertionError("Wrong or incomplete frozen v212 physical source")
    prior = Path(v206_path).read_bytes()
    if (len(prior) != len(actual) or
            hashlib.sha256(prior).hexdigest() != SOURCE_V206_SHA256):
        raise AssertionError("Wrong verified v206 source")
    seal = json.loads((sealed_dir / "SEALED_paid_source_before_untouched_QA.json").read_text())
    audit = json.loads((sealed_dir / "ACTUAL_guarded_TRAIN_24_step_audit.json").read_text())
    manifest = json.loads((sealed_dir / "PRECOMMITTED_TRAIN_ONLY_v212.json").read_text())
    if (seal["sha256"] != sha or seal["complete_bpp"] > 0.7 or
            not seal["v212_frozen_before_ANY_VALID_or_DEV"] or
            audit["actual_pilot_steps"] != 24 or len(audit["steps"]) != 24 or
            not audit["QA_and_VALID_never_seen_for_TRAIN_acceptance"] or
            not manifest["TUNING_USES_ONLY_TRAIN_not_public_DEV_or_VALID"]):
        raise AssertionError("Missing precommitted TRAIN-only seal or bpp budget")
    if not any(x["source_step_accepted"] for x in audit["steps"]):
        raise AssertionError("No TRAIN-accepted recovery steps in sealed pilot")
    return file, seal, audit


def run(v206_path, v161_path, v193_path, sealed_dir, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(200)
    torch.set_num_threads(2)
    file, seal, audit = verify_sealed_files(v206_path, sealed_dir)
    original = extract_original(Path(v161_path))
    if len(original.blob) != p.SOURCE_BYTES:
        raise AssertionError("Missing original physical CB59 control")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    _, _, valid_pairs, _ = training_inputs(tok)
    frozen_dev = freeze_cases(all_cases())
    with TemporaryDirectory(dir=out) as directory:
        tmp = Path(directory)
        for name in ("original", "v206", "v212"):
            (tmp / name).mkdir()
        (tmp / "original" / "source.cb59").write_bytes(original.blob)
        (tmp / "v206" / "source.cb59").write_bytes(Path(v206_path).read_bytes())
        cb59, _, _ = make_model("baseline", SimpleNamespace(
            original=tmp / "original" / "source.cb59", folder=tmp / "original"))
        cb206, _, _ = make_model("baseline", SimpleNamespace(
            original=tmp / "v206" / "source.cb59", folder=tmp / "v206"))
        cb212, _, _ = make_model("baseline", SimpleNamespace(
            original=file, folder=tmp / "v212"))
        if any(x.ndim >= 2 for x in cb212.parameters()):
            raise AssertionError("Native restored model loaded unpaid dense weights")
        # The v186 helper hardcodes its comparison key 'CB186'.
        # Map ONLY the argument key, then replace its obsolete output label.
        native_speed = speed_pairs({"CB59": cb59, "CB186": cb212}, tok, frozen_dev)
        for row in native_speed["paired_32_token_runs"]:
            row["wall_seconds"]["CB212"] = row["wall_seconds"].pop("CB186")
            row["speed_ratio_CB212_vs_CB59"] = row.pop("speed_ratio_CB186_vs_CB59")
        native_speed["real_native_source_version"] = "CB212"
        native_speed["post_seal_no_retraining"] = True
        save(out / "real_matched_native_32_token_CPU_vs_CB59.json", native_speed)

        valid = validation_pairs({"CB206": cb206, "GT212": cb212}, tok, valid_pairs)
        save(out / "HELDOUT_v206_VALID_pair_comparison_AFTER_sealing.json", valid)
        dev_16 = newly_frozen_public_dev_16(
            {"CB206": cb206, "GT212": cb212}, tok, v193_path, out)

    result = {
        "completed": True,
        "resumed_from_immutable_seal_without_retraining": True,
        "original_135m_unique_parameters": p.FROZEN_SOURCE_135M_UNIQUE,
        "source_SHA256": seal["sha256"],
        "whole_original_source_bpp": seal["complete_bpp"],
        "train_accepted_steps": audit["accepted_steps_before_full_TRAIN_guard"],
        "final_guard_rolled_back": audit["all_pilot_mutations_rolled_back_on_full_TRAIN_fail"],
        "native_CPU_speed_ratio_CB59": native_speed["median_speed_ratio_vs_CB59"],
        "original_FP32_16_machine_correct": dev_16["scores"]["FP32_existing_135M"]["machine_confirmed"],
        "previous_v206_16_machine_correct": dev_16["scores"]["CB206"]["machine_confirmed"],
        "new_v212_16_machine_correct": dev_16["scores"]["GT212"]["machine_confirmed"],
        "new_v212_complete_answers": dev_16["native_compact_actual_QA"]["GT212"],
        "public_DEV_diagnostic_not_training_selection": True,
        "FP32_parity_not_preclaimed": True,
        "Jessica_tests_not_run": True,
        "LIVE_unchanged": True
    }
    save(out / "v212_real_NATIVE_SOURCE_QA_summary.json", result)
    print(json.dumps({"stage": "v212_postseal_native_scoring", **{
        k: v for k, v in result.items() if k != "new_v212_complete_answers"
    }}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("v206_source", "original_v161", "original_v193_fp32_json", "sealed_v212_dir", "output"):
        ap.add_argument(name, type=Path)
    x = ap.parse_args()
    run(x.v206_source, x.original_v161, x.original_v193_fp32_json, x.sealed_v212_dir, x.output)
