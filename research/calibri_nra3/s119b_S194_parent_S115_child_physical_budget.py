"""NRA3 S119b: physical S115 neural-child vs S194 historical all-in rate audit.

No model forward, no training, no GOLD, no S118 scoring and no performance
claim. S194 historical bpp used 43,261,760 logical-architecture denominator;
S119a deliberately uses 36,380,480 unique actually TRAINED compressed-core
weights as a conservative cross-stage denominator. THESE ARE NOT INTERCHANGEABLE.
Convert historical ratio to historical *charged byte-equivalent* before adding
the real, SHA-verified full S115 FP16 child, then report BOTH rates.

S194 JSON is an independently completed CPU/J7 source ATTESTATION; it is not
a complete per-file loader inventory. Accordingly even a <=.7 projection is
never a new all-in physical certification. Unknown loader/code duplication or
unpaid adapters only REDUCE available headroom, never increase it.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct

from research.calibri_nra3.s119a_hard_four_goals import (
    BPP_MAX, SOURCE_UNIQUE_TRAINED_WEIGHTS, packed_budget_bytes,
    GateFailure, check,
)

S194_FILE = "S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json"
S115_FILE = "NRA3_S115_SYNTAGRUS_NC_NEW8_BOUNDED_TRUST_53751_IN_CALIBRI_FP16.safetensors"
S115_REPORT = "NRA3_S115_SYNTAGRUS_NC_NEW8_TRAIN_ONLY_NO_DEV_REPORT.json"
REPORT = "NRA3_S119B_PARENT_ACCOUNTING_AND_PHYSICAL_S115_HEADROOM.json"
S115_EXPECTED_SHA256 = "42e0bb4e81cb4ea199e4ee8b5fccd939ab4550b7c6269af44ede6a77462d795c"
S194_REFERENCE_DENOM = 43_261_760
OLD_NRA3_N = 53_743
FULL_NRA3_N = 53_751


def parse_safetensors(raw: bytes) -> dict:
    """Validate complete safetensors payload without torch or untrusted pickle."""
    check(len(raw) >= 32, "short S115 FP16 safetensors")
    header_n = struct.unpack("<Q", raw[:8])[0]
    check(10 < header_n <= 1_000_000 and 8 + header_n <= len(raw),
          "invalid safetensors header length")
    header = json.loads(raw[8:8 + header_n])
    check(isinstance(header, dict), "malformed safetensors")
    payload_n = len(raw) - 8 - header_n
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    check(len(entries) >= 8, "not a full neural child")
    offsets = []
    total = 0
    for name, info in entries.items():
        check(name.startswith("model.nra3_neural_weights.") and
              info["dtype"] == "F16", "not a canonical full FP16 in-Calibri child")
        shape = info["shape"]
        check(type(shape) is list and all(type(d) is int and d > 0 for d in shape),
              "bad FP16 shape")
        elements = 1
        for d in shape:
            elements *= d
        total += elements
        start, end = info["data_offsets"]
        check(type(start) is int and type(end) is int and
              start >= 0 and end - start == elements * 2 and end <= payload_n,
              "truncated or overlapping tensor data offsets")
        offsets.append((start,end))
    offsets.sort()
    cursor = 0
    for start,end in offsets:
        check(start == cursor, "safetensors holes or overlap")
        cursor = end
    check(cursor == payload_n and total == FULL_NRA3_N,
          "S115 neural checkpoint must have exactly 53,751 FP16 params")
    trust = entries.get("model.nra3_neural_weights.structural_trust_gate")
    check(trust is not None and trust["shape"] == [8],
          "eight structural trust gates missing")
    return {"neural_fp16_parameters": total,
            "old_neural_parameters": OLD_NRA3_N,
            "additional_trust_weights": 8,
            "data_bytes": payload_n,
            "file_bytes": len(raw),
            "tensor_count": len(entries)}


def historical_projection(*, parent_bpp: float, physical_child_bytes: int,
                          metadata_and_new_code_bytes: int = 0) -> dict:
    check(type(parent_bpp) in (float,int) and
          0 < parent_bpp <= BPP_MAX and
          type(physical_child_bytes) is int and physical_child_bytes > 0 and
          type(metadata_and_new_code_bytes) is int and metadata_and_new_code_bytes >= 0,
          "invalid parent historical rate or added physical bytes")
    historical_charged_equivalent = float(parent_bpp) * S194_REFERENCE_DENOM / 8
    planned_extra = physical_child_bytes + metadata_and_new_code_bytes
    projected_equivalent = historical_charged_equivalent + planned_extra
    strict_cap = packed_budget_bytes()
    legacy_cap = BPP_MAX * S194_REFERENCE_DENOM / 8
    strict_rate = projected_equivalent * 8 / SOURCE_UNIQUE_TRAINED_WEIGHTS
    legacy_rate = projected_equivalent * 8 / S194_REFERENCE_DENOM
    return {
        "s194_reported_parent_bpp_on_historical_43261760_denominator": parent_bpp,
        "s194_historical_denominator_unique_trained_core_NOT_verified": True,
        "s194_reference_denominator": S194_REFERENCE_DENOM,
        "s119a_conservative_trained_core_denominator": SOURCE_UNIQUE_TRAINED_WEIGHTS,
        "s194_inferred_charged_parent_byte_equivalent_NOT_physical_file_inventory":
            historical_charged_equivalent,
        "s115_actual_full_FP16_child_file_bytes": physical_child_bytes,
        "additional_new_runtime_code_and_metadata_bytes": metadata_and_new_code_bytes,
        "projected_legacy_bpp": legacy_rate,
        "projected_conservative_bpp": strict_rate,
        "legacy_allowable_max_bytes": legacy_cap,
        "strict_max_bytes": strict_cap,
        "strict_projected_headroom_bytes": strict_cap - projected_equivalent,
        "legacy_projected_headroom_bytes": legacy_cap - projected_equivalent,
        "candidate_passes_conservative_rate_projection": strict_rate <= BPP_MAX,
        "candidate_passes_historical_rate_projection": legacy_rate <= BPP_MAX,
        "candidate_accepted": False,
        "activation_allowed": False,
        "S119b_physically_inventory_complete": False,
        "S119b_real_CPU_speed_measured": False,
        "S119b_candidate_J7_full616_replayed": False,
        "S119b_independent_quality_gain_scored": False,
    }


def audit(s194: Path, s115: Path, s115_report: Path) -> dict:
    check(s194.name == S194_FILE and s115.name == S115_FILE and
          s115_report.name == S115_REPORT, "unexpected artifact filenames")
    baseline = json.loads(s194.read_text(encoding="utf8"))
    child_meta = json.loads(s115_report.read_text(encoding="utf8"))
    check(baseline.get("stage") ==
          "S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER" and
          baseline.get("all_616_exact_real_CPU_generated_tokens_captured") is True and
          baseline.get("physical_FP16_reload_verified") is True and
          baseline.get("model_accepted") is False and
          baseline.get("candidate_weight_update_executed") is False,
          "S194 historical CPU J7 baseline provenance is not genuine")
    j7 = baseline.get("J7Retention584_frozen_baseline", {})
    check(isinstance(j7,dict) and
          j7.get("old_full_eos_correct") == 456 and
          j7.get("new_full_eos_correct") == 128,
          "S194 real 456+128 CPU retention evidence missing")
    raw = s115.read_bytes()
    actual_sha = sha256(raw).hexdigest()
    check(actual_sha == S115_EXPECTED_SHA256 ==
          child_meta["S115_trained_IN_CALIBRI_53751_FP16_child_SHA256"] and
          child_meta["ALL_PREVIOUS_53743_NRA2_mini8_memory16_router12_structural_FP16_EXACT"] is True and
          child_meta["S115_ONLY_NEW_EIGHT_neural_trust_weights_trained"] is True and
          child_meta["output_gate"] == 0 and
          child_meta["candidate_accepted"] is False and
          child_meta["NONCOMMERCIAL_research_only_NO_model_licensing_commercial_use"] is True,
          "S115 actual trained neural child SHA or research provenance drift")
    child = parse_safetensors(raw)
    check(baseline["physical_bpp_including_S194_source"] <= BPP_MAX,
          "historical S194 parent exceeds its own reported cap")
    projection = historical_projection(
        parent_bpp=baseline["physical_bpp_including_S194_source"],
        physical_child_bytes=child["file_bytes"])
    return {"stage":"NRA3_S119B_REAL_S115_CHILD_VS_S194_BASELINE_RATE_AUDIT",
            "S194_report_sha256":sha256(s194.read_bytes()).hexdigest(),
            "S115_report_sha256":sha256(s115_report.read_bytes()).hexdigest(),
            "S115_full_child_SHA256":actual_sha,
            "S115_fp16_verified":child,
            "historical_parent_has_REAL_original616_baseline_held":True,
            "historical_parent_does_NOT_supply_NRA3_candidate616":True,
            "S118_B_GOLD_unopened_by_this_job":True,
            "NRA3_training_or_weight_mutation_executed":False,
            "CC_BY_NC_SA_4_0_noncommercial_research_only":True,
            "budget_projection":projection,
            "decision":"BUDGET_BLOCKED_NO_ACTIVATION" if
                not projection["candidate_passes_conservative_rate_projection"]
                else "PROVISIONAL_RATE_PASS_STILL_REQUIRES_LOADER_INVENTORY_CPU_J7_QUALITY",
            "accepted":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s194",type=Path,required=True)
    p.add_argument("--s115",type=Path,required=True)
    p.add_argument("--s115-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    check(not args.out.exists(),"never overwrite S119b physical audit")
    result=audit(args.s194,args.s115,args.s115_report)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119B_PHYSICAL_CHILD_BUDGET_AUDIT "+
          json.dumps({"decision":result["decision"],
                      "legacy_bpp":result["budget_projection"]["projected_legacy_bpp"],
                      "conservative_bpp":result["budget_projection"]["projected_conservative_bpp"],
                      "headroom":result["budget_projection"]["strict_projected_headroom_bytes"],
                      "s115_bytes":result["S115_fp16_verified"]["file_bytes"],
                      "candidate_accepted":False},sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
