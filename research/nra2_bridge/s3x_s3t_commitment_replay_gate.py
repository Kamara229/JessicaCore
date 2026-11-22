"""Read-only S3T prereg commitment replay. No model, scoring, or gold output.

Usage: python -m research.nra2_bridge.s3x_s3t_commitment_replay_gate /path/to/original-S3T.json
The supplied JSON MUST be extracted independently from pinned GitHub Actions
run 43527496158 artifact 14085702245. This script does not attest ZIP origin.
"""
import hashlib
import json
import sys
from pathlib import Path

from research.nra2_bridge.stage3t_unseen_wording_prereg import build_rows, commitment

PINNED_COMMITMENT = "034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3"
EXPECTED_RUN = 43527496158
EXPECTED_ARTIFACT = 14085702245


def strict_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify(artifact_bytes):
    # Independently regenerate the locked prompt texts using the pinned
    # source implementation; never infer the commitment from the ZIP hash.
    generated = commitment(build_rows())
    if generated["commitment_sha256"] != PINNED_COMMITMENT:
        raise ValueError("pinned S3T generator/commitment mismatch")
    supplied = json.loads(artifact_bytes.decode("utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(supplied, dict):
        raise ValueError("S3T artifact must be a JSON object")
    claimed = supplied.get("commitment_sha256")
    if claimed != PINNED_COMMITMENT:
        raise ValueError("artifact commitment field mismatch")
    unsigned = {k: v for k, v in supplied.items() if k != "commitment_sha256"}
    if hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != PINNED_COMMITMENT:
        raise ValueError("artifact canonical unsigned commitment mismatch")
    if canonical_bytes(supplied) != canonical_bytes(generated):
        raise ValueError("artifact differs from pinned deterministic S3T generation")
    if len(generated["question_sha256"]) != 64 or len(generated["query_char_offsets"]) != 64:
        raise ValueError("incomplete prereg metadata")
    return {
        "source_generator_replay": True,
        "canonical_commitment_verified": True,
        "ordered_prompt_sha_verified": True,
        "ordered_char_offsets_verified": True,
        "physical_zip_origin_verified": False,
        "native_tokenizer_abi_verified": False,
        "original_616_j7_verified": False,
        "output_gate": 0,
        "candidate_accepted": False,
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m research.nra2_bridge.s3x_s3t_commitment_replay_gate S3T.json")
    # Only read the supplied file; do not load model weights or write results.
    result = verify(Path(sys.argv[1]).read_bytes())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
