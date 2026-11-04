"""Assemble a complete immutable CB59 candidate using fixed-size native index fields."""
import hashlib
import struct
import zlib

import numpy as np
from direct_codebook_v59 import CB59, HEADER, pack5, unpack5
from v86.provenance import ORIGINAL_CB59_SHA
from .feedback import recode_rows


def rewrite(source, teacher, bases):
    if hashlib.sha256(source).hexdigest() != ORIGINAL_CB59_SHA:
        raise ValueError("expected the frozen CB59 archive")
    archive = CB59(source)
    payload = HEADER.size + HEADER.unpack_from(source)[2]
    result = bytearray(source)
    metrics = {}
    for name, descriptor in archive.meta["tensor_descriptors"].items():
        if descriptor["dtype"] != "CB59_K32V8":
            continue
        handle = archive.handles[name]
        rows, width = handle.shape
        groups = width // 8
        weights = teacher.get_parameter(name).detach().cpu().numpy()
        original = unpack5(handle.packed.tobytes(), rows * groups).reshape(rows, groups)
        indices = np.empty_like(original)
        before, after = 0.0, 0.0
        for start in range(0, rows, 512):
            sl = slice(start, start + 512)
            target = weights[sl] / handle.row_scales[sl, None]
            codes, old_loss, new_loss = recode_rows(
                target, handle.centroids, original[sl], bases[name])
            indices[sl] = codes
            before += old_loss
            after += new_loss
        offset, length = descriptor["parts"]["packed"]
        stream = pack5(indices)
        if len(stream) != length:
            raise ValueError(f"index stream changed size for {name}")
        result[payload + offset:payload + offset + length] = stream
        metrics[name] = {"changed_codes": int(np.count_nonzero(indices != original)),
                         "objective_before": before, "objective_after": after}
    result[-36:-32] = struct.pack("<I", zlib.crc32(result[:-36]))
    result[-32:] = hashlib.sha256(result[:-32]).digest()
    candidate = bytes(result)
    rebuilt = CB59(candidate)
    if (len(candidate) != len(source) or rebuilt.meta != archive.meta
            or rebuilt.handles["lm_head.weight"] is not rebuilt.handles["model.embed_tokens.weight"]):
        raise AssertionError("physical rate, manifest or tied source changed")
    for name, descriptor in archive.meta["tensor_descriptors"].items():
        for field, (offset, length) in descriptor["parts"].items():
            if field != "packed":
                segment = slice(payload + offset, payload + offset + length)
                if source[segment] != candidate[segment]:
                    raise AssertionError(f"unexpected change to {name}/{field}")
    return candidate, metrics
