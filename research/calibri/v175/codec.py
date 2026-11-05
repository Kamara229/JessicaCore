"""Rewrite existing 5-bit indices only, preserving all other source fields."""
import hashlib
import json
import struct
import zlib
import numpy as np
from direct_codebook_v59 import CB59, HEADER, pack5, unpack5
from .feedback import fit_indices
from .protocol import SOURCE_BYTES, SOURCE_SHA, calibration_key


def replace_indices(archive, replacements):
    required = {n for n, d in archive.meta["tensor_descriptors"].items()
                if d["dtype"] == "CB59_K32V8"}
    if set(replacements) != required:
        raise ValueError("Incomplete or duplicated matrix assignments")
    original = archive.blob
    raw = bytearray(original)
    _, _, mlen, _ = HEADER.unpack_from(original)
    start = HEADER.size + mlen
    ranges = []
    for name in sorted(required):
        h = archive.handles[name]
        ids = np.asarray(replacements[name])
        if ids.shape != (h.shape[0], h.shape[1] // 8) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Invalid source index geometry/type")
        if np.any(ids < 0) or np.any(ids > 31):
            raise ValueError("Index outside existing K32 codebook")
        value = pack5(ids.astype(np.uint8).reshape(-1))
        offset, length = archive.meta["tensor_descriptors"][name]["parts"]["packed"]
        if len(value) != length:
            raise ValueError("Index stream would change physical byte budget")
        low, high = start + offset, start + offset + length
        raw[low:high] = value
        ranges.append((low, high))
    cursor = 0
    for low, high in sorted(ranges):
        if raw[cursor:low] != original[cursor:low]:
            raise AssertionError("Unaccounted modification")
        cursor = high
    if raw[cursor:-36] != original[cursor:-36]:
        raise AssertionError("Changed source outside stored indices")
    raw[-36:-32] = struct.pack("<I", zlib.crc32(raw[:-36]))
    raw[-32:] = hashlib.sha256(raw[:-32]).digest()
    result = bytes(raw)
    verified = CB59(result)
    if len(result) != len(original) or verified.meta != archive.meta:
        raise AssertionError("Archive budget or metadata changed")
    return result


def recode(archive, model, moments):
    if len(archive.blob) != SOURCE_BYTES or archive.hash() != SOURCE_SHA:
        raise ValueError("Not immutable source")
    state = model.state_dict()
    replacements, diagnostics = {}, {}
    names = sorted(n for n, d in archive.meta["tensor_descriptors"].items()
                   if d["dtype"] == "CB59_K32V8")
    if len(names) != 211:
        raise ValueError("Incomplete original model")
    for i, name in enumerate(names):
        h = archive.handles[name]
        old = unpack5(h.packed.tobytes(), h.shape[0] * h.shape[1] // 8).reshape(h.shape[0], -1)
        codes, stat = fit_indices(state[name].detach().numpy(), h, old,
                                  moments[calibration_key(name)], tied="embed_tokens" in name)
        replacements[name] = codes
        diagnostics[name] = stat
        print(json.dumps({"matrix": i + 1, "name": name, **stat}), flush=True)
    blob = replace_indices(archive, replacements)
    return blob, diagnostics
