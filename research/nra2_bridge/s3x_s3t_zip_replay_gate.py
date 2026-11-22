"""Fail-closed physical S3T ZIP -> canonical prereg replay, research-only.

Usage: python -m research.nra2_bridge.s3x_s3t_zip_replay_gate original.zip
No model, gold, training, or filesystem writes. Does not independently attest
the GitHub Actions download channel or original tokenizer identity.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

from research.nra2_bridge.s3x_s3t_commitment_replay_gate import verify

PINNED_ZIP_SHA256 = "41c3c67966d14443bd1011521b6d197e9dc22839b8581561a68f4cb6a1932cb1"
PINNED_JSON_SHA256 = "537d0470"  # prefix only: secondary check, never an attestation by itself
PINNED_RUN = 43527496158
PINNED_ARTIFACT_ID = 14085702245
PINNED_SOURCE_HEAD = "c7c8b84d1247ee6939d7c5d894f364eea40d0f2e"


def verify_zip(raw):
    if hashlib.sha256(raw).hexdigest() != PINNED_ZIP_SHA256:
        raise ValueError("original S3T ZIP physical SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos = z.infolist()
        if len(infos) != 1:
            raise ValueError("expected exactly one JSON member")
        info = infos[0]
        if (info.is_dir() or not info.filename.endswith(".json")
                or info.filename.startswith("/")
                or ".." in info.filename.split("/")
                or info.flag_bits & 1
                or info.file_size > 1_000_000
                or info.compress_size > 1_000_000):
            raise ValueError("unsafe S3T ZIP member")
        data = z.read(info)
    if not hashlib.sha256(data).hexdigest().startswith(PINNED_JSON_SHA256):
        raise ValueError("S3T JSON secondary digest prefix mismatch")
    result = verify(data)
    result.update({
        "physical_zip_sha256_verified": True,
        "pinned_run_id": PINNED_RUN,
        "pinned_artifact_id": PINNED_ARTIFACT_ID,
        "pinned_source_head": PINNED_SOURCE_HEAD,
        "github_origin_independently_attested": False,
        "output_gate": 0,
        "candidate_accepted": False,
    })
    return result


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m research.nra2_bridge.s3x_s3t_zip_replay_gate S3T.zip")
    print(json.dumps(verify_zip(Path(sys.argv[1]).read_bytes()), sort_keys=True))


if __name__ == "__main__":
    main()
