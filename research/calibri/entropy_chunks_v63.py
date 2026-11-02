#!/usr/bin/env python3
"""CALIBRI v63: lossless entropy and restartable chunks on REAL CB59 archive.

Only the v59 CB59 5-bit source index bytes are transformed. This does not
change the quantized weights or train/evaluate the language model. CB63
is an offline candidate container, NOT a native direct-inference runtime.
All metadata, chunk directories, CRC and SHA are physically counted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib
import numpy as np

HEADER = struct.Struct("<4sBII")
ENTRY = struct.Struct("<IIIb3x")   # relative data offset, stored, raw, method
FOOTER = 36
SOURCE_MAGIC = b"CB59"
CANDIDATE_MAGIC = b"CB63"
ROWS = (64, 256, 1024)
N_UNIQUE = 134515008
MAX_BPW = 0.700


def verified(data: bytes, magic: bytes):
    if len(data) < HEADER.size + FOOTER:
        raise ValueError("archive truncated")
    got_magic, version, meta_len, body_len = HEADER.unpack_from(data)
    if got_magic != magic or version != 1:
        raise ValueError("bad magic/version")
    if meta_len > 5_000_000 or body_len > 300_000_000:
        raise ValueError("unreasonable segment size")
    if HEADER.size + meta_len + body_len + FOOTER != len(data):
        raise ValueError("archive physical length mismatch")
    if zlib.crc32(data[:-FOOTER]) != struct.unpack_from("<I", data, len(data)-FOOTER)[0]:
        raise ValueError("CRC32 mismatch")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("SHA256 mismatch")
    meta_raw = data[HEADER.size:HEADER.size+meta_len]
    meta = json.loads(meta_raw)
    body = memoryview(data)[HEADER.size+meta_len:-FOOTER]
    return meta, meta_raw, body


def parts_in_source_order(meta):
    items = []
    for name, desc in meta["tensor_descriptors"].items():
        for part, value in desc["parts"].items():
            offset, size = value
            if not isinstance(offset, int) or not isinstance(size, int):
                raise ValueError("noninteger source segment")
            if offset < 0 or size < 0:
                raise ValueError("negative source segment")
            items.append((offset, size, name, part))
    items.sort()
    cursor = 0
    for offset, size, name, part in items:
        if offset != cursor:
            raise ValueError("source segment overlap or gap")
        cursor += size
    return items, cursor


def source_ok(blob):
    meta, raw, body = verified(blob, SOURCE_MAGIC)
    if meta["vector_length"] != 8 or meta["bits_per_code"] != 5 or meta["codebook_entries"] != 32:
        raise ValueError("not CB59 K32V8 input")
    if meta["original_unique_parameter_count"] != N_UNIQUE and not meta.get("test_fixture"):
        raise ValueError("not real 135M source")
    parts, size = parts_in_source_order(meta)
    if size != len(body):
        raise ValueError("manifest body extent mismatch")
    return meta, raw, body, parts


def assemble(magic, meta_raw, body):
    head = HEADER.pack(magic, 1, len(meta_raw), len(body))
    raw = head + meta_raw + bytes(body)
    crc = struct.pack("<I", zlib.crc32(raw))
    protected = raw + crc
    return protected + hashlib.sha256(protected).digest()


def matrix_dims(meta, name, byte_count):
    desc = meta["tensor_descriptors"][name]
    m, n = desc["shape"]
    if n % 8 or m <= 0:
        raise ValueError("non-K32V8 matrix dimensions")
    groups = n // 8
    expected = (m*groups*5+7)//8
    if expected != byte_count:
        raise ValueError("nonstandard source group padding; cannot losslessly chunk")
    return m, groups


def get_indices(packed, count):
    raw = np.frombuffer(packed, dtype=np.uint8)
    bit = np.arange(count, dtype=np.uint64)*np.uint64(5)
    byte = (bit >> np.uint64(3)).astype(np.int64)
    shift = (bit & np.uint64(7)).astype(np.uint16)
    low = raw[byte].astype(np.uint16)
    high = np.zeros(count, dtype=np.uint16)
    valid = byte+1 < len(raw)
    high[valid] = raw[byte[valid]+1]
    return (((low | (high << 8)) >> shift) & 31).astype(np.uint8)


def entropy(codes):
    if len(codes) == 0:
        return 0.0
    c = np.bincount(codes, minlength=32).astype(np.float64)
    p = c[c > 0]/len(codes)
    return float(-np.sum(p*np.log2(p)))


def conditional_entropy(codes):
    if len(codes) < 2:
        return 0.0
    pairs = (codes[:-1].astype(np.int32)*32 + codes[1:].astype(np.int32))
    counts = np.bincount(pairs, minlength=1024).reshape(32, 32).astype(np.float64)
    by_prev = np.sum(counts, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = np.where(counts > 0, counts * np.log2(by_prev/counts), 0.)
    return float(np.sum(conditional)/(len(codes)-1))


def packed_chunks(source, nrows, groups, chunk_rows):
    n = len(source)
    for start in range(0, nrows, chunk_rows):
        end = min(nrows, start + chunk_rows)
        first_code = start * groups
        last_code = end * groups
        if first_code % 8:
            raise AssertionError("non-byte-aligned restart boundary")
        low = first_code * 5 // 8
        high = (last_code * 5 + 7) // 8
        if high > n or low >= high:
            raise ValueError("bad restart boundary")
        yield source[low:high].tobytes()


def encode_packed_chunked(source, nrows, groups, chunk_rows):
    table = bytearray()
    content = bytearray()
    count = 0
    for block in packed_chunks(source, nrows, groups, chunk_rows):
        compressed = zlib.compress(block, 9)
        method = 1 if len(compressed) < len(block) else 0
        stored = compressed if method else block
        table += ENTRY.pack(len(content), len(stored), len(block), method)
        content += stored
        count += 1
    return struct.pack("<I", count) + bytes(table) + bytes(content), count


def candidate_from_cb59(source_blob, chunk_rows):
    meta, meta_raw, old_body, source_parts = source_ok(source_blob)
    body = bytearray()
    total_indices = 0
    chunks = 0
    for off, n, name, part in source_parts:
        raw = old_body[off:off+n]
        if part == "packed":
            m, g = matrix_dims(meta, name, n)
            transformed, n_chunks = encode_packed_chunked(raw, m, g, chunk_rows)
            body.extend(transformed)
            chunks += n_chunks
            total_indices += m * g
        else:
            body.extend(raw)
    return assemble(CANDIDATE_MAGIC, meta_raw, body), {
        "restart_rows": chunk_rows,
        "chunk_records": chunks,
        "quantized_index_count": total_indices,
        "original_source_bytes": len(source_blob),
        "candidate_complete_source_bytes": HEADER.size + len(meta_raw) + len(body) + FOOTER,
        "saved_bytes": len(source_blob)-(HEADER.size+len(meta_raw)+len(body)+FOOTER),
    }


def decode_cb63(candidate):
    meta, meta_raw, body = verified(candidate, CANDIDATE_MAGIC)
    parts, original_body_bytes = parts_in_source_order(meta)
    out = bytearray()
    cursor = 0
    randomly_retrieved = 0
    for _, expected, name, part in parts:
        if part != "packed":
            if cursor+expected > len(body):
                raise ValueError("truncated unchanged source metadata")
            out.extend(body[cursor:cursor+expected])
            cursor += expected
            continue
        m, groups = matrix_dims(meta, name, expected)
        if cursor + 4 > len(body):
            raise ValueError("truncated chunk header")
        count = struct.unpack_from("<I", body, cursor)[0]
        if count <= 0 or count > m:
            raise ValueError("invalid restart count")
        table_start = cursor+4
        start = table_start+count*ENTRY.size
        if start > len(body):
            raise ValueError("invalid restart table")
        total_data = 0
        chunks = []
        for j in range(count):
            offset, stored, raw_size, method = ENTRY.unpack_from(body, table_start+j*ENTRY.size)
            if offset != total_data or method not in (0, 1) or stored <= 0 or raw_size <= 0:
                raise ValueError("invalid restart descriptor")
            total_data += stored
            chunks.append((offset, stored, raw_size, method))
        if start+total_data > len(body):
            raise ValueError("truncated compressed chunk data")
        for off, n, raw_len, method in chunks:
            stored = bytes(body[start+off:start+off+n])
            if method:
                chunk = zlib.decompress(stored)
            else:
                chunk = stored
            if len(chunk) != raw_len:
                raise ValueError("reconstructed index chunk size mismatch")
            out.extend(chunk)
            randomly_retrieved += 1
        cursor = start + total_data
    if cursor != len(body) or len(out) != original_body_bytes:
        raise ValueError("candidate extra/missing bytes")
    restored = assemble(SOURCE_MAGIC, meta_raw, out)
    if len(restored) != HEADER.size+len(meta_raw)+original_body_bytes+FOOTER:
        raise ValueError("incorrect original length")
    return restored, randomly_retrieved


def audit_indices(source_blob):
    meta, _, body, parts = source_ok(source_blob)
    n_codes = 0
    h_bits = 0.0
    hc_bits = 0.0
    scores = []
    for off, n, name, part in parts:
        if part != "packed":
            continue
        m, g = matrix_dims(meta, name, n)
        codes = get_indices(body[off:off+n], m*g)
        h = entropy(codes)
        hc = conditional_entropy(codes)
        n_codes += len(codes)
        h_bits += len(codes)*h
        hc_bits += len(codes)*hc
        scores.append({
            "tensor": name,
            "codes": len(codes),
            "marginal_entropy_bits_per_5bit_code": round(h, 6),
            "lag1_conditional_entropy_bits_per_code": round(hc, 6),
            "packed_index_bytes": n,
        })
    if n_codes == 0:
        raise ValueError("no true packed source weights")
    return {
        "packed_matrix_count": len(scores),
        "total_5bit_indices": n_codes,
        "source_only_marginal_H_bits_per_index": h_bits/n_codes,
        "source_only_lag1_conditional_H_bits_per_index": hc_bits/n_codes,
        "best_case_marginal_entropy_bytes_NO_METADATA": math.ceil(h_bits/8),
        "theoretical_conditional_entropy_bytes_NO_CONTEXT_TABLES": math.ceil(hc_bits/8),
        "total_original_packed_index_bytes": sum(s["packed_index_bytes"] for s in scores),
        "least_entropy_tensors": sorted(scores, key=lambda x:x["marginal_entropy_bits_per_5bit_code"])[:12],
        "highest_entropy_tensors": sorted(scores, key=lambda x:-x["marginal_entropy_bits_per_5bit_code"])[:12],
        "interpretation": "H is an optimistic rate bound only; context/dictionary tables, restart index and decoding cost are NOT included.",
    }


def fixture():
    # Synthetic CB59 archive without torch, network, or trained-weight claims.
    counts = 24*8
    sequence = np.asarray([i % 7 for i in range(counts)], dtype=np.uint8)
    bits = 0
    for idx, val in enumerate(sequence):
        bits |= int(val) << (idx*5)
    packed = bits.to_bytes((counts*5+7)//8, "little")
    scales = b"\x00\x3c"*24
    centroids = b"\x00\x00"*(32*8)
    raw1d = b"\x00"*16
    data = packed+scales+centroids+raw1d
    manifest = {
        "test_fixture": True,
        "vector_length": 8, "bits_per_code": 5, "codebook_entries": 32,
        "original_unique_parameter_count": counts*8+4,
        "tensor_descriptors": {
            "toy.weight": {"shape": [24, 64], "dtype": "CB59_K32V8",
                "parts": {"packed": [0, len(packed)],
                          "row_scales": [len(packed), len(scales)],
                          "centroids": [len(packed)+len(scales), len(centroids)]}},
            "toy.bias": {"shape": [4], "dtype": "f4",
                "parts": {"raw": [len(packed)+len(scales)+len(centroids), len(raw1d)]}},
        },
    }
    meta = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    original = assemble(SOURCE_MAGIC, meta, data)
    assert np.array_equal(get_indices(memoryview(packed), counts), sequence)
    for rows in ROWS:
        candidate, summary = candidate_from_cb59(original, rows)
        back, chunks = decode_cb63(candidate)
        assert back == original and chunks == summary["chunk_records"]
        broken = bytearray(candidate)
        broken[len(broken)//2] ^= 1
        try:
            decode_cb63(bytes(broken))
        except ValueError:
            pass
        else:
            raise AssertionError("corruption undetected")
    assert entropy(sequence) < 3
    print("CALIBRI v63 synthetic CB59->CB63->identical CB59, random chunk, integrity, code entropy PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    fixture()
    if args.smoke:
        return
    if args.source is None or args.output is None or args.candidate is None:
        parser.error("source/output/candidate required for real data")
    source = args.source.read_bytes()
    audit = audit_indices(source)
    options = []
    winner = None
    for rows in ROWS:
        candidate, summary = candidate_from_cb59(source, rows)
        recovered, chunk_count = decode_cb63(candidate)
        if recovered != source or chunk_count != summary["chunk_records"]:
            raise AssertionError("actual CB59 physical source changed by chunk compression")
        summary["exact_roundtrip_CB59_sha256"] = hashlib.sha256(recovered).hexdigest()
        summary["candidate_full_bpw"] = 8*len(candidate)/N_UNIQUE
        summary["within_max_0_7_total_physical_bytes"] = len(candidate) <= math.floor(MAX_BPW*N_UNIQUE/8)
        options.append(summary)
        if winner is None or len(candidate) < len(winner[0]):
            winner = (candidate, summary)
    args.candidate.parent.mkdir(parents=True, exist_ok=True)
    args.candidate.write_bytes(winner[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "v63 lossless-source-only chunks; NO functional quality claim",
        "model_revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "candidate_sha256": hashlib.sha256(winner[0]).hexdigest(),
        "actual_original_complete_bpw": 8*len(source)/N_UNIQUE,
        "exact_0_700_max_bytes": math.floor(MAX_BPW*N_UNIQUE/8),
        "zlib_version": zlib.ZLIB_VERSION,
        "entropy_analysis": audit,
        "real_physical_restart_configs": options,
        "best_restart_rows_by_complete_archive_bytes": winner[1]["restart_rows"],
        "native_CB63_execution_implemented": False,
        "whole_original_weight_matrix_reconstruction_required_for_this_offline_audit": False,
        "original_CB59_source_roundtrip_bit_identical": True,
        "important_limit": "Candidate requires bounded decompression of requested index chunks; runtime memory, speed and direct C kernel NOT yet tested.",
    }
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({
        "original_bytes": len(source),
        "best_candidate_bytes": len(winner[0]),
        "best_saved_bytes": winner[1]["saved_bytes"],
        "best_rows": winner[1]["restart_rows"],
        "marginal_H": round(audit["source_only_marginal_H_bits_per_index"], 4),
        "conditional_H": round(audit["source_only_lag1_conditional_H_bits_per_index"], 4),
        "real_roundtrip": True,
        "native_CB63": False,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
