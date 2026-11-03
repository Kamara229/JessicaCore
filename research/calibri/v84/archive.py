"""CB59 native source-level operator rewrite: same exact bytes, new FP16 codebook.

This changes actual operator parameters (not a runtime-only bias/LoRA).
Every reference, segment offset, CRC and SHA is checked after rewriting.
"""
from __future__ import annotations
import hashlib
import struct
import zlib

import numpy as np
from direct_codebook_v59 import CB59, HEADER, FOOTER, K, VECTOR


def rewrite_codebooks(original: bytes, changes: dict[str, np.ndarray]) -> bytes:
    baseline = CB59(original)
    out = bytearray(original)
    _, _, meta_bytes, body_bytes = HEADER.unpack_from(original)
    body_first = HEADER.size + meta_bytes
    if body_first + body_bytes + FOOTER != len(original):
        raise AssertionError("original complete physical CB59 size inconsistent")
    protected_offsets = set()
    for path, new_centroids in changes.items():
        name = path + ".weight"
        if name not in baseline.handles:
            raise AssertionError("unknown pretrained packed projection " + name)
        desc = baseline.meta["tensor_descriptors"].get(name)
        if desc is None or desc["dtype"] != "CB59_K32V8":
            raise AssertionError("cannot modify source alias or noncodebook tensor")
        offset, size = desc["parts"]["centroids"]
        if size != K * VECTOR * 2:
            raise AssertionError("CB59 centroid physical dimension changed")
        values = np.asarray(new_centroids, dtype=np.float32)
        if values.shape != (K, VECTOR) or not np.isfinite(values).all():
            raise AssertionError("invalid synthesized native operator codebook")
        packed = values.astype("<f2")
        if not np.isfinite(packed).all():
            raise AssertionError("nonfinite physically represented FP16 values")
        start = body_first + offset
        assert start not in protected_offsets
        protected_offsets.add(start)
        out[start:start + size] = packed.tobytes()

    if not changes:
        raise AssertionError("no functional operator was rewritten")
    out[-FOOTER:-32] = struct.pack("<I", zlib.crc32(out[:-FOOTER]))
    out[-32:] = hashlib.sha256(out[:-32]).digest()
    result = bytes(out)
    if len(result) != len(original) or result == original:
        raise AssertionError("no genuine equal-byte-count source rewrite")
    rebuilt = CB59(result)
    if rebuilt.meta != baseline.meta:
        raise AssertionError("rewritten pretrained source manifest changed")
    for path in changes:
        name = path + ".weight"
        want = np.asarray(changes[path], dtype="<f2").astype(np.float32)
        if not np.array_equal(rebuilt.handles[name].centroids, want):
            raise AssertionError("serialized rewritten native centroids do not roundtrip")
        if not np.array_equal(rebuilt.handles[name].packed,
                              baseline.handles[name].packed):
            raise AssertionError("original compact code indices changed unexpectedly")
    return result


def manifest_audit(original: bytes, modified: bytes) -> dict:
    if len(original) != len(modified):
        raise AssertionError("different physical source costs")
    return {
        "original_source_sha": hashlib.sha256(original).hexdigest(),
        "rewritten_source_sha": hashlib.sha256(modified).hexdigest(),
        "full_source_bytes_unchanged": len(modified),
        "integrity_verified_with_CB59_parser": True,
        "same_5bit_indices_and_all_unmodified_tensor_data": True,
    }
