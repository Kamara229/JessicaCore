"""CB177: one physically complete mixed 4/5/6-bit matrix archive.

Groups still represent eight FP32 weights, but their stored index widths
and codebook dimensions differ by matrix. Every additional codebook byte,
new index bit, manifest byte, checksum and alias is charged to the cap.
No trained dense 2D tensor or FP32 teacher is needed by the reader.
"""
import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from direct_codebook_v59 import CB59, HEADER, row_scales, _source_sample, _nearest_integer_128
from .protocol import MAX_BYTES, TOTAL_UNIQUE, allocation

MAGIC = b"C177"
FOOTER = 36
VECTOR = 8
ITERATIONS = 5


def pack_indices(indices, bits):
    if bits not in (4, 5, 6):
        raise ValueError("Unsupported physical index width")
    values = np.asarray(indices, dtype=np.uint8).reshape(-1)
    if len(values) == 0 or np.any(values >= (1 << bits)):
        raise ValueError("Invalid code indices")
    n = len(values)
    result = np.zeros((n * bits + 7) // 8, dtype=np.uint8)
    offsets = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    byte = (offsets // 8).astype(np.int64)
    shift = (offsets % 8).astype(np.uint16)
    wide = values.astype(np.uint16) << shift
    np.bitwise_or.at(result, byte, (wide & 255).astype(np.uint8))
    carry = (shift + bits) > 8
    if np.any(carry):
        np.bitwise_or.at(result, byte[carry] + 1, (wide[carry] >> 8).astype(np.uint8))
    return result.tobytes()


def unpack_indices(packed, count, bits):
    if bits not in (4, 5, 6) or count <= 0 or len(packed) != (count * bits + 7) // 8:
        raise ValueError("Invalid variable-bit packed source")
    data = np.frombuffer(packed, dtype=np.uint8)
    offsets = np.arange(count, dtype=np.uint64) * np.uint64(bits)
    byte = (offsets // 8).astype(np.int64)
    shift = (offsets % 8).astype(np.uint16)
    wide = data[byte].astype(np.uint16)
    carry = (shift + bits) > 8
    wide[carry] |= data[byte[carry] + 1].astype(np.uint16) << 8
    return ((wide >> shift) & ((1 << bits) - 1)).astype(np.uint8)


def fit_codebook(source, k):
    if k not in (16, 64):
        raise ValueError("Only four/six-bit source refits are permitted")
    samples = np.asarray(source, dtype="<f2").astype(np.float32)
    if samples.ndim != 2 or samples.shape[1] != VECTOR or len(samples) < k:
        raise ValueError("Insufficient weight samples")
    centers = samples[np.linspace(0, len(samples)-1, k, dtype=np.int64)].copy()
    for _ in range(ITERATIONS):
        chosen = _nearest_integer_128(samples, centers)
        counts = np.bincount(chosen, minlength=k)
        updates = np.zeros((k, VECTOR), dtype=np.float64)
        np.add.at(updates, chosen, samples.astype(np.float64))
        occupied = counts > 0
        centers[occupied] = (updates[occupied] / counts[occupied, None]).astype("<f2").astype(np.float32)
    snapped = centers.astype("<f2")
    if not np.isfinite(snapped).all():
        raise ValueError("Nonfinite codebook")
    return snapped


def quantize_matrix(weight, bits):
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % VECTOR or not np.isfinite(w).all():
        raise ValueError("Invalid original trained dense matrix")
    m, n = w.shape
    groups_per_row = n // VECTOR
    scales = row_scales(w)
    centers = fit_codebook(_source_sample(w, scales), 1 << bits)
    centers32 = centers.astype(np.float32)
    ids = np.empty(m * groups_per_row, dtype=np.uint8)
    sq_error, sq_weight = 0.0, 0.0
    for first in range(0, m, 128):
        last = min(m, first+128)
        part = w[first:last].reshape(-1, VECTOR)
        factors = np.repeat(scales[first:last].astype(np.float32), groups_per_row)
        normalized = (part / factors[:, None]).astype("<f2").astype(np.float32)
        chosen = _nearest_integer_128(normalized, centers32).astype(np.uint8)
        ids[first*groups_per_row:last*groups_per_row] = chosen
        reconstructed = centers32[chosen] * factors[:, None]
        sq_error += float(np.sum(np.square((part-reconstructed).astype(np.float64))))
        sq_weight += float(np.sum(np.square(part.astype(np.float64))))
    packed = pack_indices(ids, bits)
    if not np.array_equal(unpack_indices(packed, len(ids), bits), ids):
        raise AssertionError("Variable-bit logical index roundtrip")
    return {"packed": packed, "row_scales": scales.astype("<f2").tobytes(),
            "centroids": centers.astype("<f2").tobytes()}, {
            "relative_weight_RMSE": math.sqrt(sq_error / max(sq_weight, 1e-30)),
            "groups": int(len(ids)), "bits": bits,
            "fitted_source_only": True}


def encode(original, teacher_state, arm):
    if not isinstance(original, CB59):
        raise TypeError("Expected validated original CB59")
    if original.meta["original_unique_parameter_count"] != TOTAL_UNIQUE:
        raise AssertionError("Unexpected original unique parameter count")
    if set(teacher_state) != set(original.meta["full_original_state_keys"]):
        raise ValueError("Original teacher source incomplete")
    _, _, old_mlen, _ = HEADER.unpack_from(original.blob)
    old_body = memoryview(original.blob)[HEADER.size+old_mlen:-FOOTER]
    body = bytearray()
    new_desc, measurements = {}, {}
    for name in sorted(original.meta["tensor_descriptors"]):
        d = original.meta["tensor_descriptors"][name]
        shape = tuple(d["shape"])
        if d["dtype"] == "CB59_K32V8":
            bits = allocation(arm, name)
            if tuple(teacher_state[name].shape) != shape:
                raise ValueError("Original FP32 geometry mismatch " + name)
            if bits == 5:
                payload = {part: bytes(old_body[off:off+size])
                           for part, (off, size) in d["parts"].items()}
                measure = {"bits": 5, "fitted_source_only": False,
                           "exact_CB59_matrix_fields": True}
            else:
                weight = teacher_state[name].detach().cpu().numpy()
                payload, measure = quantize_matrix(weight, bits)
            counts = 1 << bits
            if len(payload["packed"]) != (shape[0] * shape[1] // VECTOR * bits + 7) // 8:
                raise AssertionError("Variable-width index cost mismatch")
            if len(payload["centroids"]) != counts * VECTOR * 2:
                raise AssertionError("Unpaid dictionary data")
            kind = "CB177_VARIABLE"
            new_desc[name] = {"shape": list(shape), "dtype": kind, "bits": bits,
                              "codebook_entries": counts, "parts": {}}
            measurements[name] = measure
        else:
            if d["dtype"] != "f4":
                raise ValueError("Unsupported source tensor")
            payload = {"raw": bytes(old_body[s:s+n]) for s,n in [d["parts"]["raw"]]}
            new_desc[name] = {"shape": list(shape), "dtype": "f4", "parts": {}}
        for part in ("packed", "row_scales", "centroids") if "packed" in payload else ("raw",):
            block = payload[part]
            new_desc[name]["parts"][part] = [len(body), len(block)]
            body.extend(block)
    meta = dict(original.meta)
    meta["format"] = "CALIBRI v177 complete matrix-variable 4/5/6-bit source"
    meta["version"] = 177
    meta["bits_per_code"] = None
    meta["codebook_entries"] = None
    meta["tensor_descriptors"] = new_desc
    meta["physical_bit_allocation"] = arm
    meta["quality_not_guaranteed"] = True
    manifest = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf8")
    if len(manifest) > 5_000_000 or len(body) > 300_000_000:
        raise AssertionError("Source parser byte bounds")
    start = HEADER.pack(MAGIC, 1, len(manifest), len(body))
    pre_crc = start + manifest + bytes(body)
    protected = pre_crc + struct.pack("<I", zlib.crc32(pre_crc))
    blob = protected + hashlib.sha256(protected).digest()
    if len(blob) > MAX_BYTES:
        raise ValueError("Real archive exceeds 0.7 bpp: " + str(len(blob)))
    reader = CB177(blob)
    if reader.hash() != hashlib.sha256(blob).hexdigest():
        raise AssertionError("Physical archive checksum mismatch")
    return blob, {"arm": arm, "sha256": reader.hash(), "bytes": len(blob),
                  "bpp": 8*len(blob)/TOTAL_UNIQUE, "max_bytes": MAX_BYTES,
                  "source_matrix_stats": measurements,
                  "stored_unique_matrices": len(measurements),
                  "no_unpaid_sidecars": True}


@dataclass
class VariableHandle:
    name: str
    shape: tuple
    packed: np.ndarray
    row_scales: np.ndarray
    centroids: np.ndarray
    bits: int


class CB177:
    """Integrity-verified, compact, alias-aware reader (no pretrained 2D FP32)."""
    def __init__(self, blob):
        if not isinstance(blob, bytes) or len(blob) < HEADER.size + FOOTER or len(blob) > MAX_BYTES:
            raise ValueError("Invalid physical v177 source bytes")
        magic, ver, mlen, blen = HEADER.unpack_from(blob)
        if magic != MAGIC or ver != 1 or mlen > 5_000_000 or blen > 300_000_000:
            raise ValueError("Invalid variable-bit source header")
        if HEADER.size + mlen + blen + FOOTER != len(blob):
            raise ValueError("Invalid source allocation")
        if hashlib.sha256(blob[:-32]).digest() != blob[-32:] or zlib.crc32(blob[:-36]) != struct.unpack_from("<I",blob,len(blob)-36)[0]:
            raise ValueError("Source CRC/SHA mismatch")
        self.blob = blob
        self.meta = json.loads(blob[HEADER.size:HEADER.size+mlen].decode("utf8"))
        desc = self.meta["tensor_descriptors"]
        if sorted(desc) != self.meta["trained_unique_keys"]:
            raise ValueError("Unexpected unique parameters")
        if self.meta["physical_bit_allocation"] not in ("ADAPT_A", "ADAPT_B"):
            raise ValueError("Unknown compressed format arm")
        body = memoryview(blob)[HEADER.size+mlen:-FOOTER]
        self.handles, self.raw = {}, {}
        end = 0
        for name in sorted(desc):
            d = desc[name]
            shape = tuple(d["shape"])
            fields = ("packed","row_scales","centroids") if d["dtype"] == "CB177_VARIABLE" else ("raw",)
            if set(d["parts"]) != set(fields):
                raise ValueError("Wrong physical fields " + name)
            data = {}
            for part in fields:
                offset, length = d["parts"][part]
                if offset != end or length <= 0 or end+length > len(body):
                    raise ValueError("Missing, overlapping, reordered source segment")
                data[part] = body[offset:offset+length]
                end += length
            if d["dtype"] == "CB177_VARIABLE":
                bits = d["bits"]
                if bits != allocation(self.meta["physical_bit_allocation"], name) or d["codebook_entries"] != 1 << bits or len(shape) != 2 or shape[1] % 8:
                    raise ValueError("Non-precommitted index width")
                rows, cols = shape
                if len(data["packed"]) != (rows*cols//8*bits+7)//8 or len(data["row_scales"]) != rows*2 or len(data["centroids"]) != (1<<bits)*8*2:
                    raise ValueError("Incorrect compact field size")
                scales = np.frombuffer(data["row_scales"], dtype="<f2").astype(np.float32)
                centers = np.frombuffer(data["centroids"], dtype="<f2").astype(np.float32).reshape(1<<bits,8)
                if not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(centers).all():
                    raise ValueError("Corrupt variable-bit codebook/scales")
                self.handles[name] = VariableHandle(name,shape,np.frombuffer(data["packed"],dtype=np.uint8),scales,centers,bits)
            elif d["dtype"] == "f4":
                vector = np.frombuffer(data["raw"],dtype="<f4")
                if len(vector) != int(np.prod(shape,dtype=np.int64)):
                    raise ValueError("Corrupt raw vector size")
                self.raw[name] = vector.reshape(shape)
            else:
                raise ValueError("Unsupported matrix kind")
        if end != len(body):
            raise ValueError("Unaccounted physical payload")
        for alias, canonical in self.meta["source_alias_keys"].items():
            if alias in self.handles or alias in self.raw:
                raise ValueError("Duplicate physical tied source")
            if canonical in self.handles:
                self.handles[alias] = self.handles[canonical]
            elif canonical in self.raw:
                self.raw[alias] = self.raw[canonical]
            else:
                raise ValueError("Unresolved original alias")
        if set(self.handles) | set(self.raw) != set(self.meta["full_original_state_keys"]):
            raise ValueError("Incomplete native model source")

    def hash(self):
        return hashlib.sha256(self.blob).hexdigest()
