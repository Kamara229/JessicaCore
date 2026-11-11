"""S136 real-archive research gate: attest S135 46.70M physical stage3 input.

Usage (ONLY after successful S135 GPU run):
python -m calibri2.s136.attest_stage3_entry \
    S135_NEXT_STAGE_46M_PHYSICAL_GROWTH_REPORT.json \
    S135_GROWN_46M_ZERO_NEW_CHANNELS.safetensors \
    S132_GROWN_43M_ZERO_NEW_CHANNELS.safetensors \
    S136_REAL_STAGE3_ENTRY_ATTESTATION.json

No training; no hidden QA, S118, Jessica LIVE, Render, Supabase or APK.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibri2.s136.verify_stage3_entry import (
    verified_s135_stage3_entry,
)


def attest_stage3(
    *, report:Path, physical:Path, parent:Path, output:Path,
) -> dict:
    """Verify one complete actual S135 archive before any 46.7M GPU train."""
    if any(not isinstance(p,Path) or not p.is_file()
           for p in (report,physical,parent)):
        raise AssertionError("S136 requires all three persisted S135/S132 input files")
    paths=[x.resolve() for x in (report,physical,parent)]
    if len(set(paths))!=3 or output.resolve() in set(paths):
        raise AssertionError("S136 cannot attest an aliased or overwritten source")
    manifest=verified_s135_stage3_entry(report,physical,parent)
    if (manifest.get("completed_s135_J7_cycles")!=5
        or manifest.get("stage3_new_24_coefficients_exact_zero") is not True
        or manifest.get("stage1_frozen_24_coefficients_unchanged") is not True
        or manifest.get("stage2_new_24_coefficients_retained") is not True
        or manifest.get("original_S101_Q2_tensors_immutable") is not True
        or manifest.get("model_accepted") is not False):
        raise AssertionError("S136 S135 report violates protected stage3 entry")

    proof={
        **manifest,
        "kind":"S136_REAL_PHYSICAL_SOURCE_ATTESTATION_NO_TRAINING",
        "S135_report_name":report.name,
        "S135_physical_name":physical.name,
        "S132_previous_physical_name":parent.name,
        "five_real_J7_required":True,
        "no_456_128_heldout_QA_used":True,
        "no_live_services_modified":True,
        "S136_stage3_GPU_training_started":False,
        "research_only":True,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(proof,indent=2,ensure_ascii=False,sort_keys=True),
        encoding="utf-8",
    )
    return proof


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s135_report",type=Path)
    parser.add_argument("s135_grown_physical",type=Path)
    parser.add_argument("s132_parent_physical",type=Path)
    parser.add_argument("attestation_json",type=Path)
    args=parser.parse_args()
    proof=attest_stage3(
        report=args.s135_report,physical=args.s135_grown_physical,
        parent=args.s132_parent_physical,output=args.attestation_json,
    )
    print(json.dumps({
        "event":"S136_REAL_46M_STAGE3_RESEARCH_ENTRY_VERIFIED",
        "rank":proof["stage3_rank"],
        "architectural_core_params":proof["stage3_architecture_params"],
        "old_456_full":proof["old_456_full"],
        "new_128_full":proof["new_128_full"],
        "physical_bpp":proof["physical_bpp"],
        "CPU_ratio":proof["CPU_speed_ratio"],
        "five_J7_audited":proof["completed_s135_J7_cycles"],
        "source_sha256":proof["physical_sha256"],
        "research_only":True,
        "GPU_stage3_started":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
