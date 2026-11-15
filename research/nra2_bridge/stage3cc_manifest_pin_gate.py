"""S3CC research-only external manifest pin gate; never trains or exposes labels.

The expected manifest SHA must come from an independently sealed CPU capture,
not from the optimizer input or from this function's output.
"""
import hashlib
import json
import re
from research.nra2_bridge.stage3cb_physical_row_binding import FIELDS, verify_rows

HEX64 = re.compile(r"[0-9a-f]{64}\Z")

def canonical_bytes(rows):
    if not isinstance(rows, list) or len(rows) != 2240:
        raise ValueError("expected exactly 2240 ordered training rows")
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(FIELDS):
            raise ValueError("invalid manifest schema")
        if any(not isinstance(row[k], str) or HEX64.fullmatch(row[k]) is None for k in FIELDS):
            raise ValueError("invalid digest")
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")

def verify_pinned_manifest(sealed_rows, physical_rows, externally_pinned_sha256):
    if not isinstance(externally_pinned_sha256, str) or HEX64.fullmatch(externally_pinned_sha256) is None:
        raise ValueError("missing/invalid independent manifest pin")
    actual_sha = hashlib.sha256(canonical_bytes(sealed_rows)).hexdigest()
    if actual_sha != externally_pinned_sha256:
        raise ValueError("independent manifest SHA mismatch")
    if physical_rows is sealed_rows:
        raise ValueError("expected/actual manifest alias forbidden")
    verify_rows(sealed_rows, physical_rows)
    return True

def self_test():
    keys = list(FIELDS)
    rows = [{k: f"{i * len(keys) + j + 1:064x}" for j, k in enumerate(keys)} for i in range(2240)]
    pin = hashlib.sha256(canonical_bytes(rows)).hexdigest()  # fixture only; never runtime pin
    physical = [dict(row) for row in rows]
    assert verify_pinned_manifest(rows, physical, pin)
    bad = [dict(row) for row in rows]
    bad[4], bad[5] = bad[5], bad[4]
    for expected, actual, digest in (
        (rows, bad, pin),
        (rows, physical, "0" * 64),
        (rows, rows, pin),
    ):
        try:
            verify_pinned_manifest(expected, actual, digest)
        except ValueError:
            continue
        raise AssertionError("negative control accepted")
    print("S3CC pinned-manifest gate self-test PASS; fixture only, no physical CPU capture")

if __name__ == "__main__":
    self_test()
