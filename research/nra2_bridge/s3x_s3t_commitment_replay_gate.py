"""Read-only S3T prereg commitment replay. No model, scoring, or gold output.

Usage: python -m research.nra2_bridge.s3x_s3t_commitment_replay_gate /path/to/original-S3T.json
The supplied JSON MUST be extracted independently from pinned GitHub Actions
run 43527496158 artifact 14085702245. This script does not attest ZIP origin.
"""
import hashlib
import json
import re
import sys
from pathlib import Path
import importlib

PINNED_DEPENDENCY_GIT_BLOB_SHA1 = {
    "stage3m_role_balanced_graph_data.py": "468bf5bf1e09e61c00822f70230593d3efc3eac7",
    "stage3g_factorized_graph_controls.py": "ac4866d18820661887863e3febfa9c931acbdf20",
    "stage3q_independent_soft_hard_prereg.py": "3111609c99069f320d0eff6bf68ae5e47e3ca7e1",
}
PINNED_GENERATOR_GIT_BLOB_SHA1 = "26dd55bcdd61aaf855750ac0d35d9deaef10706b"
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


def independent_query_offsets(question):
    """Re-derive literal query spans without calling the S3T generator helper."""
    q_end = question.find("?")
    if q_end < 0 or question.find("?", q_end + 1) != -1:
        raise ValueError("ambiguous S3T question boundary")
    delimiter = question.rfind(". ", 0, q_end)
    if delimiter < 0:
        raise ValueError("missing S3T query delimiter")
    start = delimiter + 2
    hits = list(re.finditer(r"(?<![A-Za-z0-9_])R([0-5])(?![A-Za-z0-9_])", question[start:q_end]))
    if len(hits) != 2 or hits[0].group(1) == hits[1].group(1):
        raise ValueError("S3T query must have two distinct node mentions")
    spans = [[start + m.start(), start + m.end()] for m in hits]
    for span, match in zip(spans, hits):
        if question[span[0]:span[1]] != match.group(0):
            raise ValueError("S3T literal span slice mismatch")
    return {
        "clause": [start, q_end + 1],
        "mention_spans": spans,
        "text_order_nodes": [int(m.group(1)) for m in hits],
    }


def verify(artifact_bytes):
    # Reject untrusted/malformed artifacts before importing any research modules.
    # This preflight has no model access and does not inspect candidate labels.
    supplied = json.loads(artifact_bytes.decode("utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(supplied, dict):
        raise ValueError("S3T artifact must be a JSON object")
    claimed = supplied.get("commitment_sha256")
    if claimed != PINNED_COMMITMENT:
        raise ValueError("artifact commitment field mismatch")
    unsigned = {k: v for k, v in supplied.items() if k != "commitment_sha256"}
    if hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != PINNED_COMMITMENT:
        raise ValueError("artifact canonical unsigned commitment mismatch")
    # Independently regenerate the locked prompt texts using the pinned
    # source implementation; never infer the commitment from the ZIP hash.
    # Hash the source file BEFORE importing/executing the generator module.
    # This gate is not a sandbox: execute only within a trusted research checkout.
    source_path = Path(__file__).resolve().with_name("stage3t_unseen_wording_prereg.py")
    source = source_path.read_bytes()
    git_blob = hashlib.sha1(b"blob " + str(len(source)).encode("ascii") + b"\0" + source).hexdigest()
    if git_blob != PINNED_GENERATOR_GIT_BLOB_SHA1:
        raise ValueError("S3T generator source differs from pinned GitHub blob")
    # Fail closed before import: transitive research source files can execute
    # Python during import. Pin their physical Git blobs, not just the generator.
    for filename, expected_blob in PINNED_DEPENDENCY_GIT_BLOB_SHA1.items():
        dependency_path = source_path.with_name(filename)
        dependency = dependency_path.read_bytes()
        actual_blob = hashlib.sha1(
            b"blob " + str(len(dependency)).encode("ascii") + b"\0" + dependency
        ).hexdigest()
        if actual_blob != expected_blob:
            raise ValueError("S3T dependency source differs from pinned GitHub blob: " + filename)
    # Refuse cached modules: importlib otherwise reuses unverified code.
    pinned_module_names = [
        "research.nra2_bridge." + name[:-3]
        for name in PINNED_DEPENDENCY_GIT_BLOB_SHA1
    ] + ["research.nra2_bridge.stage3t_unseen_wording_prereg"]
    if any(name in sys.modules for name in pinned_module_names):
        raise ValueError("pinned S3T module already loaded before verified import")
    generator_module = importlib.import_module("research.nra2_bridge.stage3t_unseen_wording_prereg")
    if Path(generator_module.__file__).resolve() != source_path:
        raise ValueError("S3T generator resolved to unexpected module path")
    # Verify imported dependency provenance as well as on-disk bytes: a custom
    # importer or altered search path must not substitute another module.
    for filename in PINNED_DEPENDENCY_GIT_BLOB_SHA1:
        module_name = "research.nra2_bridge." + filename[:-3]
        loaded = sys.modules.get(module_name)
        expected_path = source_path.with_name(filename)
        if loaded is None or not getattr(loaded, "__file__", None):
            raise ValueError("pinned dependency was not imported: " + module_name)
        if Path(loaded.__file__).resolve() != expected_path:
            raise ValueError("pinned dependency resolved to unexpected path: " + module_name)
    # Rehash source bytes after import to detect import-time source mutation.
    # This is a provenance guard, not a Python execution sandbox.
    for filename, expected_blob in {
        "stage3t_unseen_wording_prereg.py": PINNED_GENERATOR_GIT_BLOB_SHA1,
        **PINNED_DEPENDENCY_GIT_BLOB_SHA1,
    }.items():
        current = source_path.with_name(filename).read_bytes()
        current_blob = hashlib.sha1(
            b"blob " + str(len(current)).encode("ascii") + b"\0" + current
        ).hexdigest()
        if current_blob != expected_blob:
            raise ValueError("pinned source changed during import: " + filename)
    # Recompute ordered prompt digests and literal offsets from the exact
    # generated texts, independently of the generator's commitment builder.
    # No model forward, scoring, or label disclosure is performed here.
    rows = generator_module.build_rows()
    if len(rows) != 64:
        raise ValueError("S3T replay did not produce exactly 64 prompts")
    physical_prompt_sha = [
        hashlib.sha256(row.question.encode("utf-8")).hexdigest() for row in rows
    ]
    physical_offsets = [independent_query_offsets(row.question) for row in rows]
    generated = generator_module.commitment(rows)
    if generated.get("question_sha256") != physical_prompt_sha:
        raise ValueError("S3T regenerated prompt text SHA mismatch")
    if generated.get("query_char_offsets") != physical_offsets:
        raise ValueError("S3T regenerated literal query offsets mismatch")
    if generated["commitment_sha256"] != PINNED_COMMITMENT:
        raise ValueError("pinned S3T generator/commitment mismatch")
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
