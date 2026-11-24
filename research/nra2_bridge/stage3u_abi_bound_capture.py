"""S3U ABI-bound replay: pin physical S3T, bind actual prefill IDs to S3O masks.

Research-only, read-only. Runs the existing S3U CPU/J7 capture unchanged, but
records the exact input_ids passed to the frozen CPU forward and binds the
same IDs to native offset mappings and S3O query masks. No training, fitting,
candidate scoring, FP32 teacher use, or protected writes.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

PINNED_S3T_ZIP_SHA256 = "41c3c67966d14443bd1011521b6d197e9dc22839b8581561a68f4cb6a1932cb1"
PINNED_S3T_JSON_SHA256 = "537d0470912c1dbd1c630f3488661464a3aa42ed643349e00fb58db8391919f9"
PINNED_S3T_COMMITMENT = "034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3"
PINNED_S3T_RUN = 43527496158
PINNED_S3T_ARTIFACT = 14085702245
PINNED_S3T_HEAD = "c7c8b84d1247ee6939d7c5d894f364eea40d0f2e"
SOURCE_BLOBS = {
    "research/nra2_bridge/stage3t_unseen_wording_prereg.py": "26dd55bcdd61aaf855750ac0d35d9deaef10706b",
    "research/nra2_bridge/stage3m_role_balanced_graph_data.py": "468bf5bf1e09e61c00822f70230593d3efc3eac7",
    "research/nra2_bridge/stage3g_factorized_graph_controls.py": "ac4866d18820661887863e3febfa9c931acbdf20",
    "research/nra2_bridge/stage3q_independent_soft_hard_prereg.py": "3111609c99069f320d0eff6bf68ae5e47e3ca7e1",
}
S3T_JSON_NAME = "NRA2_S3T_NEW_GRAMMAR64_PRECOMMITTED_NO_SCORE.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError("duplicate S3T JSON key")
        result[key] = value
    return result


def git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def verify_s3t_physical_source():
    token = os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or repo != "Kamara229/JessicaCore":
        raise AssertionError("S3U requires the repository-scoped read-only GitHub token")
    archive = Path("/tmp/S3T_ORIGINAL_PHYSICAL_ARTIFACT.zip")
    url = (
        "https://api.github.com/repos/" + repo
        + "/actions/artifacts/" + str(PINNED_S3T_ARTIFACT) + "/zip"
    )
    subprocess.run([
        "curl", "-fsSL",
        "-H", "Authorization: Bearer " + token,
        "-H", "Accept: application/vnd.github+json",
        url, "-o", str(archive),
    ], check=True)
    raw_zip = archive.read_bytes()
    zip_sha = sha256_bytes(raw_zip)
    if zip_sha != PINNED_S3T_ZIP_SHA256:
        raise AssertionError("physical S3T ZIP SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise AssertionError("S3T artifact must contain exactly one file")
        info = infos[0]
        if (info.filename != S3T_JSON_NAME or info.is_dir()
                or info.flag_bits & 1 or info.file_size > 1_000_000):
            raise AssertionError("unsafe or unexpected S3T ZIP member")
        inner = zf.read(info)
    inner_sha = sha256_bytes(inner)
    if inner_sha != PINNED_S3T_JSON_SHA256:
        raise AssertionError("physical S3T JSON SHA256 mismatch")
    obj = json.loads(inner.decode("utf-8"), object_pairs_hook=strict_pairs)
    if obj.get("commitment_sha256") != PINNED_S3T_COMMITMENT:
        raise AssertionError("S3T commitment field mismatch")
    unsigned = {k: v for k, v in obj.items() if k != "commitment_sha256"}
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    canonical_sha = sha256_bytes(canonical)
    if canonical_sha != PINNED_S3T_COMMITMENT:
        raise AssertionError("S3T canonical unsigned commitment mismatch")
    extracted = Path("inputs/nra2_stage3t") / S3T_JSON_NAME
    if not extracted.is_file() or extracted.read_bytes() != inner:
        raise AssertionError("actions/download-artifact bytes differ from pinned physical S3T JSON")
    if (obj.get("questions") != 64
            or len(obj.get("question_sha256", [])) != 64
            or len(obj.get("query_char_offsets", [])) != 64):
        raise AssertionError("S3T preregistration manifest cardinality mismatch")

    source_hashes = {}
    for path, expected in SOURCE_BLOBS.items():
        content = subprocess.check_output(["git", "show", "HEAD:" + path])
        actual = git_blob_sha1(content)
        if actual != expected:
            raise AssertionError("pinned S3T producer/dependency source drift: " + path)
        source_hashes[path] = actual

    audit = {
        "stage": "NRA2_S3U_S3T_PHYSICAL_PROVENANCE_ONLY",
        "s3t_run_id": PINNED_S3T_RUN,
        "s3t_artifact_id": PINNED_S3T_ARTIFACT,
        "s3t_source_head": PINNED_S3T_HEAD,
        "s3t_zip_sha256": zip_sha,
        "s3t_inner_json_sha256": inner_sha,
        "s3t_canonical_commitment_sha256": canonical_sha,
        "s3t_commitment_sha256": PINNED_S3T_COMMITMENT,
        "s3t_generator_and_dependencies_git_blob_sha1": source_hashes,
        "physical_zip_verified": True,
        "physical_json_verified": True,
        "canonical_commitment_verified": True,
        "source_generator_pinned": True,
        "literal_prompt_texts_in_original_manifest": False,
        "output_gate": 0,
        "candidate_accepted": False,
    }
    audit_path = Path("inputs/nra2_stage3t") / "S3T_PHYSICAL_ARTIFACT_AUDIT.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return obj, audit


def tokenizer_backend_sha256(tok) -> str:
    if not getattr(tok, "is_fast", False) or not hasattr(tok, "backend_tokenizer"):
        raise AssertionError("S3U requires the original fast tokenizer backend")
    return sha256_bytes(tok.backend_tokenizer.to_str().encode("utf-8"))


def main():
    # Physical artifact and exact producer source are verified BEFORE imports
    # of the S3T generator or any model/capture module.
    prereg, physical_audit = verify_s3t_physical_source()

    from nra2_bridge.stage3t_unseen_wording_prereg import (
        build_rows, commitment, query_char_offsets,
    )
    rows = build_rows()
    prompts = [row.question for row in rows]
    prompt_hashes = [sha256_bytes(q.encode("utf-8")) for q in prompts]
    generated = commitment(rows)
    offsets = [query_char_offsets(q) for q in prompts]
    if (generated != prereg
            or prompt_hashes != prereg["question_sha256"]
            or offsets != prereg["query_char_offsets"]):
        raise AssertionError("S3T prompt text reconstruction differs from physical preregistration")
    expected_hashes = set(prompt_hashes)
    if len(expected_hashes) != 64:
        raise AssertionError("S3T prompt identities are not unique")

    import torch
    from safetensors.torch import load_file, save_file
    from calibri2.s160 import real_parallel_slots_j7 as s160
    import research.nra2_bridge.stage3j_token_entity_alignment as s3j
    import research.nra2_bridge.stage3o_neural_query_roles as s3o

    original_prompt_tokens = s160.prompt_tokens
    original_validate = s3j.validate_without_gold
    original_build_query_mask = s3o.build_query_text_mask

    captured_ids = {}
    actual_tokenizer = {}
    aligned_holder = {}
    qmask_holder = {}

    def capture_prompt_tokens(tok, question):
        ids = list(original_prompt_tokens(tok, question))
        digest = sha256_bytes(question.encode("utf-8"))
        if digest in expected_hashes:
            fingerprint = tokenizer_backend_sha256(tok)
            if not actual_tokenizer:
                actual_tokenizer.update({
                    "object": tok,
                    "backend_sha256": fingerprint,
                    "name_or_path": str(getattr(tok, "name_or_path", "")),
                    "commit_hash": str(getattr(tok, "init_kwargs", {}).get("_commit_hash", "")),
                })
            elif tok is not actual_tokenizer["object"]:
                raise AssertionError("S3U graph prefills did not use one native tokenizer instance")
            elif fingerprint != actual_tokenizer["backend_sha256"]:
                raise AssertionError("S3U native tokenizer backend changed during capture")
            if digest in captured_ids:
                raise AssertionError("duplicate S3T native prefill input capture")
            captured_ids[digest] = ids
        return ids

    def validate_with_actual_prefill_ids(tok, questions, prefill_id_sequences=None):
        if prefill_id_sequences is not None:
            raise AssertionError("S3U wrapper owns the actual prefill input-ID binding")
        hashes = [sha256_bytes(q.encode("utf-8")) for q in questions]
        if set(hashes) != expected_hashes or len(hashes) != 64:
            raise AssertionError("S3U tokenizer validation prompt order/identity mismatch")
        if set(captured_ids) != expected_hashes:
            raise AssertionError("S3U actual CPU prefill input IDs incomplete")
        if tokenizer_backend_sha256(tok) != actual_tokenizer.get("backend_sha256"):
            raise AssertionError("S3U offset tokenizer differs from actual CPU prefill tokenizer")
        ordered_ids = [captured_ids[h] for h in hashes]
        aligned = original_validate(tok, questions, prefill_id_sequences=ordered_ids)
        aligned_holder["value"] = aligned
        aligned_holder["tokenizer"] = tok
        return aligned

    def capture_s3o_query_mask(tok, questions, alignment, sequence_length):
        qmask = original_build_query_mask(tok, questions, alignment, sequence_length)
        qmask_holder["value"] = qmask
        return qmask

    s160.prompt_tokens = capture_prompt_tokens
    s3j.validate_without_gold = validate_with_actual_prefill_ids
    s3o.build_query_text_mask = capture_s3o_query_mask
    try:
        from research.nra2_bridge import stage3u_newgrammar64_real_cpu_capture as s3u
        s3u.main()
    finally:
        s160.prompt_tokens = original_prompt_tokens
        s3j.validate_without_gold = original_validate
        s3o.build_query_text_mask = original_build_query_mask

    if set(captured_ids) != expected_hashes or len(captured_ids) != 64:
        raise AssertionError("S3U failed to capture all 64 actual CPU prefill input-ID sequences")
    if "value" not in aligned_holder or "value" not in qmask_holder:
        raise AssertionError("S3U S3O alignment/query mask was not produced")
    aligned = aligned_holder["value"]
    native_tok = aligned_holder["tokenizer"]
    qmask = qmask_holder["value"]
    native_sha = tokenizer_backend_sha256(native_tok)
    if native_sha != actual_tokenizer["backend_sha256"]:
        raise AssertionError("S3U actual and offset tokenizer backend SHA mismatch")

    out = Path(sys.argv[-1])
    token_file = out / "NRA2_S3U_REAL_CPU_GRAPH64_ALL_TOKENS_FP16.safetensors"
    feature_file = out / "NRA2_S3U_REAL_43M_GRAPH64_HIDDEN320_FP16.safetensors"
    report_file = out / "NRA2_S3U_64_INDEPENDENT_REAL_CPU_AUDIT.json"
    token_data = load_file(str(token_file), device="cpu")
    token_hidden = token_data["token_hidden320"]
    token_mask = token_data["token_mask"].to(torch.bool)
    token_lengths = token_data["token_lengths"].to(torch.int64)
    if token_hidden.shape[0] != 64 or token_hidden.shape[-1] != 320:
        raise AssertionError("S3U original FP16 token feature shape changed")
    maximum = int(token_hidden.shape[1])
    input_ids = torch.zeros((64, maximum), dtype=torch.int64)
    input_id_mask = torch.zeros((64, maximum), dtype=torch.bool)
    for i, question in enumerate(prompts):
        ids = captured_ids[prompt_hashes[i]]
        length = int(token_lengths[i])
        if len(ids) != length or not 0 < length <= maximum:
            raise AssertionError("S3U actual prefill IDs differ from captured hidden-token length")
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.int64)
        input_id_mask[i, :length] = True
    if not torch.equal(input_id_mask, token_mask):
        raise AssertionError("S3U input-ID mask differs from actual CPU hidden-token mask")

    from research.nra2_bridge.stage3j_token_entity_alignment import content_in_chat
    offset_mapping = torch.zeros((64, maximum, 2), dtype=torch.int32)
    content_starts = torch.zeros((64,), dtype=torch.int32)
    query_clause_spans = torch.zeros((64, 2), dtype=torch.int32)
    mention_char_spans = torch.zeros((64, 2, 2), dtype=torch.int32)
    mention_token_start_end = torch.zeros((64, 2, 2), dtype=torch.int32)

    for i, (question, alignment) in enumerate(zip(prompts, aligned)):
        declared = prereg["query_char_offsets"][i]
        observed = query_char_offsets(question)
        if declared != observed:
            raise AssertionError("S3U committed literal character offsets changed")
        full_chat, content_start = content_in_chat(native_tok, question)
        mapped = native_tok(
            full_chat, return_offsets_mapping=True, add_special_tokens=False
        )
        mapped_ids = list(mapped["input_ids"])
        offsets_i = [tuple(pair) for pair in mapped["offset_mapping"]]
        if mapped_ids != captured_ids[prompt_hashes[i]] or len(offsets_i) != int(token_lengths[i]):
            raise AssertionError("S3U tokenizer IDs/offsets are not the exact native prefill sequence")
        content_starts[i] = int(content_start)
        query_clause_spans[i] = torch.tensor(observed["clause"], dtype=torch.int32)
        for j, pair in enumerate(offsets_i):
            if len(pair) != 2 or any(type(v) is not int for v in pair):
                raise AssertionError("S3U tokenizer returned malformed offset mapping")
            offset_mapping[i, j] = torch.tensor(pair, dtype=torch.int32)

        q_start, q_end = observed["clause"]
        expected_query_positions = [
            j for j, (left, right) in enumerate(offsets_i)
            if left < content_start + q_end and right > content_start + q_start
        ]
        actual_query_positions = torch.where(qmask.query_mask[i])[0].tolist()
        if expected_query_positions != actual_query_positions:
            raise AssertionError("S3U S3O query mask differs from exact native offset mapping")

        query_mentions = [
            m for m in alignment["mentions"]
            if q_start <= m.char_start and m.char_end <= q_end - 1
        ]
        if (len(query_mentions) != 2
                or [[m.char_start, m.char_end] for m in query_mentions] != observed["mention_spans"]
                or [m.node for m in query_mentions] != observed["text_order_nodes"]):
            raise AssertionError("S3U exact two query mentions do not match preregistered text spans")
        if qmask.node_order[i].tolist() != observed["text_order_nodes"]:
            raise AssertionError("S3U S3O node order differs from preregistered mention order")

        used = set()
        for k, mention in enumerate(query_mentions):
            start, end = mention.char_start, mention.char_end
            mention_char_spans[i, k] = torch.tensor((start, end), dtype=torch.int32)
            expected_positions = [
                j for j, (left, right) in enumerate(offsets_i)
                if left < content_start + end and right > content_start + start
            ]
            actual_positions = list(mention.token_positions)
            anchor_positions = torch.where(qmask.anchor_mask[i, k])[0].tolist()
            if (not expected_positions or expected_positions != actual_positions
                    or expected_positions != anchor_positions
                    or expected_positions != list(range(expected_positions[0], expected_positions[-1] + 1))
                    or any(p in used for p in expected_positions)
                    or any(p >= int(token_lengths[i]) or not bool(token_mask[i, p])
                           or not bool(qmask.query_mask[i, p]) for p in expected_positions)):
                raise AssertionError("S3U exact native two-mention token span ABI mismatch")
            used.update(expected_positions)
            mention_token_start_end[i, k] = torch.tensor(
                (expected_positions[0], expected_positions[-1] + 1), dtype=torch.int32
            )
        if set(torch.where(qmask.anchor_mask[i].any(dim=0))[0].tolist()) != used:
            raise AssertionError("S3U S3O anchor union differs from the two literal mentions")

    # Persist the exact prompts as text-only provenance; no labels are included.
    prompt_text_file = out / "NRA2_S3U_S3T_PROMPTS_EXACT_NO_LABELS.json"
    prompt_text_payload = {
        "stage": "NRA2_S3U_EXACT_S3T_PROMPTS_NO_LABELS",
        "s3t_commitment_sha256": PINNED_S3T_COMMITMENT,
        "prompt_sha256_in_order": prompt_hashes,
        "prompts": prompts,
    }
    prompt_text_file.write_text(
        json.dumps(prompt_text_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    # safetensors load_file may retain mmap-backed storage. Never truncate
    # the source file while any of its tensors are still mapped.
    original_token_file_sha256 = sha256_file(token_file)
    token_data = {k: v.detach().cpu().contiguous().clone() for k, v in token_data.items()}
    token_data["input_ids"] = input_ids.clone()
    token_data["input_id_mask"] = input_id_mask.clone()
    staged_token_file = token_file.with_suffix(".staged.safetensors")
    save_file(token_data, str(staged_token_file))
    token_reloaded = load_file(str(staged_token_file), device="cpu")
    if set(token_reloaded) != set(token_data):
        raise AssertionError("S3U token/ID physical reload key mismatch")
    for k, v in token_data.items():
        actual = token_reloaded[k]
        if (actual.dtype != v.dtype or actual.shape != v.shape
                or not torch.equal(actual, v)
                or not torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                                   v.reshape(-1).view(torch.uint8))):
            raise AssertionError(f"S3U token/ID physical reload mismatch: {k}")
    if sha256_file(token_file) != original_token_file_sha256:
        raise AssertionError("S3U original token file changed before atomic replacement")
    staged_sha256 = sha256_file(staged_token_file)
    os.replace(staged_token_file, token_file)
    if sha256_file(token_file) != staged_sha256:
        raise AssertionError("S3U atomic token file replacement SHA mismatch")

    abi_tensors = {
        "input_ids": input_ids.contiguous(),
        "input_id_mask": input_id_mask.contiguous(),
        "offset_mapping": offset_mapping.contiguous(),
        "query_mask": qmask.query_mask.to(torch.bool).cpu().contiguous(),
        "anchor_mask": qmask.anchor_mask.to(torch.bool).cpu().contiguous(),
        "left_context_mask": qmask.left_context_mask.to(torch.bool).cpu().contiguous(),
        "right_context_mask": qmask.right_context_mask.to(torch.bool).cpu().contiguous(),
        "node_order": qmask.node_order.to(torch.int64).cpu().contiguous(),
        "relative_positions": qmask.relative_positions.to(torch.float32).cpu().contiguous(),
        "token_lengths": token_lengths.to(torch.int64).contiguous(),
        "question_content_char_start": content_starts.contiguous(),
        "query_clause_char_spans": query_clause_spans.contiguous(),
        "mention_char_spans": mention_char_spans.contiguous(),
        "mention_token_start_end": mention_token_start_end.contiguous(),
    }
    sidecar_file = out / "NRA2_S3U_QUERY_ABI_EVIDENCE.safetensors"
    save_file(abi_tensors, str(sidecar_file))
    abi_reloaded = load_file(str(sidecar_file), device="cpu")
    if set(abi_reloaded) != set(abi_tensors) or any(
            not torch.equal(abi_reloaded[k], v) for k, v in abi_tensors.items()):
        raise AssertionError("S3U query ABI sidecar physical reload mismatch")

    physical_audit_out = out / "S3T_PHYSICAL_ARTIFACT_AUDIT.json"
    physical_audit_out.write_text(
        json.dumps(physical_audit, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "stage": "NRA2_S3U_ABI_BOUND_EVIDENCE_ONLY",
        "output_gate": 0,
        "candidate_accepted": False,
        "candidate_scored": False,
        "training_steps": 0,
        "s3t_zip_sha256": physical_audit["s3t_zip_sha256"],
        "s3t_inner_json_sha256": physical_audit["s3t_inner_json_sha256"],
        "s3t_commitment_sha256": PINNED_S3T_COMMITMENT,
        "s3t_source_head": PINNED_S3T_HEAD,
        "s3t_source_git_blob_sha1": physical_audit["s3t_generator_and_dependencies_git_blob_sha1"],
        "prompt_sha256_in_order": prompt_hashes,
        "prompt_text_file_sha256": sha256_file(prompt_text_file),
        "tokenizer_name_or_path": actual_tokenizer["name_or_path"],
        "tokenizer_commit_hash": actual_tokenizer["commit_hash"],
        "tokenizer_backend_sha256": actual_tokenizer["backend_sha256"],
        "actual_prefill_ids_bound_to_native_tokenizer": True,
        "actual_prefill_ids_bound_to_S3O_mask": True,
        "original_token_file_sha256": original_token_file_sha256,
        "token_file_sha256": sha256_file(token_file),
        "sidecar_sha256": sha256_file(sidecar_file),
        "tensor_schema": {
            key: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for key, value in abi_tensors.items()
        },
    }
    manifest_file = out / "NRA2_S3U_QUERY_ABI_MANIFEST.json"
    manifest_file.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    report_file = out / "NRA2_S3U_64_INDEPENDENT_REAL_CPU_AUDIT.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    report.update({
        "S3T_physical_zip_sha256": physical_audit["s3t_zip_sha256"],
        "S3T_inner_json_sha256": physical_audit["s3t_inner_json_sha256"],
        "S3T_canonical_commitment_verified": True,
        "S3T_prompt_texts_reconstructed_from_pinned_source": True,
        "S3T_generator_and_dependency_git_blob_sha1": physical_audit["s3t_generator_and_dependencies_git_blob_sha1"],
        "S3T_prompt_text_file_sha256": sha256_file(prompt_text_file),
        "native_prefill_input_ids_bound_to_S3O_mask": True,
        "native_tokenizer_backend_sha256": actual_tokenizer["backend_sha256"],
        "query_abi_sidecar_sha256": sha256_file(sidecar_file),
        "query_abi_manifest_sha256": sha256_file(manifest_file),
        "S3T_physical_audit_sha256": sha256_file(physical_audit_out),
        "full_token_fp16_sha256": sha256_file(token_file),
        "query_abi_independent_reinspection_ready": True,
        "output_gate": 0,
        "training_steps": 0,
        "candidate_scored": False,
        "candidate_accepted": False,
        "physical_reload_bitwise_equal": True,
    })
    report_file.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "stage": "NRA2_S3U_ABI_BOUND_EVIDENCE_ONLY",
        "s3t_zip_sha256": physical_audit["s3t_zip_sha256"],
        "s3t_inner_json_sha256": physical_audit["s3t_inner_json_sha256"],
        "s3t_commitment_sha256": PINNED_S3T_COMMITMENT,
        "prompt_count": len(prompts),
        "native_prefill_input_ids_bound_to_S3O_mask": True,
        "tokenizer_backend_sha256": actual_tokenizer["backend_sha256"],
        "token_feature_shape": list(token_hidden.shape),
        "token_file_sha256": sha256_file(token_file),
        "query_abi_sidecar_sha256": sha256_file(sidecar_file),
        "output_gate": 0,
        "candidate_accepted": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
