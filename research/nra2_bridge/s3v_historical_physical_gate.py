"""Research-only physical provenance gate for historical S3V JSON.

This is NOT a 64-row attestation. Requires an independently reviewed manifest;
does not fetch artifacts, score, train, or expose labels. No runtime side effects.
"""
import hashlib
import json
from pathlib import Path


def _pairs_unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("FAIL CLOSED: duplicate JSON key")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError("FAIL CLOSED: nonfinite JSON number")


def _load(raw):
    return json.loads(raw, object_pairs_hook=_pairs_unique,
                      parse_constant=_reject_constant,
                      parse_float=lambda s: float(s))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_physical_gate(producer_bytes, manifest_bytes, artifact_zip_bytes=None):
    """Verify historical producer physical bytes against independent manifest.

    Manifest must be externally reviewed and pinned; this function cannot
    establish that independence. It deliberately returns BLOCKED, not ATTESTED.
    """
    manifest = _load(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("manifest_reviewed") is not True:
        raise ValueError("FAIL CLOSED: independently reviewed manifest required")
    required = ("run_id", "run_attempt", "workflow_path", "head_sha",
                "artifact_id", "artifact_name", "zip_sha256",
                "json_member_path", "json_sha256")
    if any(not manifest.get(k) for k in required):
        raise ValueError("FAIL CLOSED: missing artifact provenance")
    if type(manifest["run_id"]) is not int or type(manifest["run_attempt"]) is not int:
        raise ValueError("FAIL CLOSED: invalid run identifiers")
    if type(manifest["artifact_id"]) is not int:
        raise ValueError("FAIL CLOSED: invalid artifact identifier")
    for key in ("head_sha", "zip_sha256", "json_sha256"):
        length = 40 if key == "head_sha" else 64
        value = manifest[key]
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("FAIL CLOSED: malformed immutable hash")
    if artifact_zip_bytes is None:
        raise ValueError("FAIL CLOSED: original ZIP bytes required")
    if _sha(artifact_zip_bytes) != manifest["zip_sha256"]:
        raise ValueError("FAIL CLOSED: ZIP physical SHA mismatch")
    if _sha(producer_bytes) != manifest["json_sha256"]:
        raise ValueError("FAIL CLOSED: JSON physical SHA mismatch")
    producer = _load(producer_bytes)
    if not isinstance(producer, dict):
        raise ValueError("FAIL CLOSED: invalid historical producer JSON")
    gate = producer.get("output_gate")
    if type(gate) is not float or gate != 0.0:
        raise ValueError("FAIL CLOSED: historical float gate mismatch")
    if producer.get("candidate_accepted") is not False:
        raise ValueError("FAIL CLOSED: candidate acceptance mismatch")
    return {
        "status": "BLOCKED_PENDING_INDEPENDENT_64_ROW_AND_J7_ATTESTATION",
        "producer_json_sha256": _sha(producer_bytes),
        "producer_zip_sha256": _sha(artifact_zip_bytes),
        "historical_producer_gate_type": "float",
        "full_64_row_ledger_verified": False,
        "independent_source_j7_digest_recomputed": False,
        "candidate_accepted": False,
        "output_gate": 0,
    }


if __name__ == "__main__":
    raise SystemExit("Research library only; no standalone attestation without reviewed manifest.")
