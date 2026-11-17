"""S3V-LEDGER partial-evidence validator. Research only; never infers omitted roles.

Usage: python -m research.nra2_bridge.s3v_partial_ledger_audit SOURCE.json EXPECTED_SHA256
No network, model, labels lookup, inference, training, or output-gate changes.
This is NOT a full 64-row ledger or a new candidate score.
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

N = 64
STAGE = "NRA2_S3V_PRECOMMITTED_S3T64_SOFT_VS_HARD_FROZEN_NATIVE_CPU_TRIAL"
S3T_SHA = "034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3"
S3O_SHA = "40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"


def audit(raw: bytes, expected_sha256: str) -> dict:
    actual = hashlib.sha256(raw).hexdigest()
    if (not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
        or actual != expected_sha256.lower()):
        raise ValueError("FAIL CLOSED: raw source JSON physical SHA256 mismatch")
    # Reject ambiguous JSON objects: Python normally accepts duplicate keys and
    # silently keeps the last value, weakening provenance/gate checks.
    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("FAIL CLOSED: duplicate JSON key: " + key)
            obj[key] = value
        return obj

    # Python's JSON decoder accepts NaN/Infinity by default although they are
    # not valid JSON. Reject them before inspecting provenance or aggregates.
    def reject_nonfinite(token):
        raise ValueError("FAIL CLOSED: non-finite JSON constant: " + token)

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("FAIL CLOSED: non-finite JSON numeric overflow")
        return value

    report = json.loads(raw, object_pairs_hook=unique_pairs,
                        parse_constant=reject_nonfinite, parse_float=finite_float)
    if not isinstance(report, dict):
        raise ValueError("FAIL CLOSED: source report must be an object")
    if (report.get("stage") != STAGE
        or report.get("S3T_prior_commitment_sha256") != S3T_SHA
        or report.get("S3O_frozen_adapter_sha256") != S3O_SHA
        or report.get("original_616_S194_J7_hashes_preserved_by_S3U") is not True
        or report.get("original_584_retained_full_EOS_unchanged") is not True
        or report.get("candidate_accepted") is not False
        or type(report.get("output_gate")) not in (int, float)
        or report["output_gate"] != 0
        or report.get("new_training_or_parameter_tuning") is not False):
        raise ValueError("FAIL CLOSED: report provenance or zero gate mismatch")
    metrics = report["all_and_family_yes_no_accuracy"]["all"]
    if (type(metrics.get("count")) is not int or metrics["count"] != N
        or type(metrics.get("yes")) is not int or metrics["yes"] != 32
        or type(metrics.get("no")) is not int or metrics["no"] != 32):
        raise ValueError("FAIL CLOSED: preregistered population mismatch")
    records = report["disagreement_or_error_cases"]
    if not isinstance(records, list) or len(records) > N:
        raise ValueError("FAIL CLOSED: invalid sparse record collection")
    seen = set()
    for r in records:
        if not isinstance(r, dict):
            raise ValueError("FAIL CLOSED: sparse record must be an object")
        i = r.get("index")
        if type(i) is not int or not 0 <= i < N or i in seen:
            raise ValueError("FAIL CLOSED: duplicate/out-of-range case index")
        seen.add(i)
        if (type(r.get("soft_correct")) is not bool
            or type(r.get("hard_correct")) is not bool
            or type(r.get("source_correct")) is not bool
            or type(r.get("target_correct")) is not bool
            or not isinstance(r.get("prompt_sha256"), str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", r["prompt_sha256"]) is None):
            raise ValueError("FAIL CLOSED: malformed observed record")
        # Producer emits all incorrect answers. Any both-correct record must
        # have an answer disagreement, which is impossible with fixed binary truth.
        if r["soft_correct"] and r["hard_correct"]:
            raise ValueError("FAIL CLOSED: record contradicts S3V producer predicate")
    for field in ("soft_correct", "hard_correct", "correct_source", "correct_destination"):
        value = metrics.get(field)
        if type(value) is not int or not 0 <= value <= N:
            raise ValueError("FAIL CLOSED: invalid aggregate count")
    soft_errors = sum(not r["soft_correct"] for r in records)
    hard_errors = sum(not r["hard_correct"] for r in records)
    if (soft_errors != N - metrics["soft_correct"]
        or hard_errors != N - metrics["hard_correct"]):
        raise ValueError("FAIL CLOSED: sparse record / aggregate inconsistency")
    unknown = N - len(seen)
    # Cross-check producer-side role aggregates against the sparse observations.
    # This checks internal consistency, NOT independent reproduction of predictions.
    for role_field, aggregate_field in (
        ("source_correct", "correct_source"),
        ("target_correct", "correct_destination"),
    ):
        total = metrics.get(aggregate_field)
        observed = sum(r[role_field] for r in records)
        if type(total) is not int or not observed <= total <= observed + unknown:
            raise ValueError("FAIL CLOSED: producer role aggregate outside sparse bounds")
    # Producer's original family and yes/no aggregates must partition all 64
    # cases; do not invent missing per-row roles or order-specific accuracies.
    slices = report["all_and_family_yes_no_accuracy"]
    yes_slice, no_slice = slices.get("true_yes"), slices.get("true_no")
    if not isinstance(yes_slice, dict) or not isinstance(no_slice, dict):
        raise ValueError("FAIL CLOSED: missing original truth partition")
    for field in ("count", "correct_source", "correct_destination",
                  "soft_correct", "hard_correct"):
        if (type(yes_slice.get(field)) is not int
            or type(no_slice.get(field)) is not int
            or not 0 <= yes_slice[field] <= 32
            or not 0 <= no_slice[field] <= 32
            or yes_slice[field] + no_slice[field] != metrics[field]):
            raise ValueError("FAIL CLOSED: producer yes/no aggregates inconsistent")
    # S3V producer emits four fixed families, each split by truth. Validate
    # the full partition without reconstructing any omitted individual roles.
    families = ("chain", "fork", "cycle_sink", "diamond")
    additive = ("count", "yes", "no", "soft_correct", "hard_correct",
                "correct_source", "correct_destination")
    family_slices = []
    for family in families:
        parent = slices.get("family_" + family)
        fy, fn = slices.get("family_" + family + "_yes"), slices.get("family_" + family + "_no")
        if not all(isinstance(v, dict) for v in (parent, fy, fn)):
            raise ValueError("FAIL CLOSED: missing S3V family/truth partition")
        for key, obj, expected in ((family, parent, 16),
                                   (family + "_yes", fy, 8),
                                   (family + "_no", fn, 8)):
            if type(obj.get("count")) is not int or obj["count"] != expected:
                raise ValueError("FAIL CLOSED: family/truth population mismatch: " + key)
            for field in additive[1:]:
                v = obj.get(field)
                if type(v) is not int or not 0 <= v <= expected:
                    raise ValueError("FAIL CLOSED: malformed family aggregate: " + key)
        if fy["yes"] != 8 or fy["no"] != 0 or fn["yes"] != 0 or fn["no"] != 8:
            raise ValueError("FAIL CLOSED: family truth balance mismatch")
        for field in additive:
            if fy[field] + fn[field] != parent[field]:
                raise ValueError("FAIL CLOSED: family yes/no aggregates inconsistent")
        family_slices.append((parent, fy, fn))
    for field in additive:
        if sum(row[0][field] for row in family_slices) != metrics[field]:
            raise ValueError("FAIL CLOSED: family totals inconsistent")
        if sum(row[1][field] for row in family_slices) != yes_slice[field]:
            raise ValueError("FAIL CLOSED: family yes totals inconsistent")
        if sum(row[2][field] for row in family_slices) != no_slice[field]:
            raise ValueError("FAIL CLOSED: family no totals inconsistent")
    def bounds(field):
        correct = sum(bool(r[field]) for r in records)
        return {"observed_correct": correct, "unknown": unknown,
                "lower": correct / N, "upper": (correct + unknown) / N}
    joint = sum(r["source_correct"] and r["target_correct"] for r in records)
    return {"source_json_sha256": actual, "recorded": len(seen),
            "omitted_role_status": "unknown", "omitted": unknown,
            "source_bounds": bounds("source_correct"),
            "target_bounds": bounds("target_correct"),
            "joint_role_bounds": {"observed_correct": joint, "unknown": unknown,
                                  "lower": joint / N, "upper": (joint + unknown) / N},
            "full_64_row_ledger_verified": False,
            "independent_source_j7_digest_recomputed": False,
            "candidate_accepted": False, "output_gate": 0}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: s3v_partial_ledger_audit SOURCE.json EXPECTED_RAW_SHA256")
    result = audit(Path(sys.argv[1]).read_bytes(), sys.argv[2])
    print(json.dumps(result, sort_keys=True, indent=2))
