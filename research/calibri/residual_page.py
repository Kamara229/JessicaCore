"""Small CPU reference: ternary page + mandatory indexed FP16 corrections.

Synthetic-test container only. No GPU kernel, signature, model manifest, or
production loader. CRC detects accidental damage, not malicious replacement.
"""
import struct
import zlib
import numpy as np
from index import encode, decode

HEADER = '<4sHHIII'
HEADER_BYTES = struct.calcsize(HEADER)


def pack_page(weight, activation_rms, zero_fraction, correction_fraction):
    w = np.asarray(weight, dtype=np.float64)
    rms = np.asarray(activation_rms, dtype=np.float64)
    if (w.ndim != 2 or min(w.shape) < 1 or w.size > 65535
            or not np.isfinite(w).all() or rms.shape != (w.shape[1],)
            or not np.isfinite(rms).all() or np.any(rms < 0)
            or not 0 <= zero_fraction <= 1 or not 0 <= correction_fraction <= 1):
        raise ValueError('invalid page or calibration parameters')
    threshold = np.quantile(np.abs(w), zero_fraction, axis=1, keepdims=True)
    q = np.where(np.abs(w) > threshold, np.sign(w), 0).astype(np.int8)
    levels = np.zeros((len(w), 2), dtype='<f2')
    for i in range(len(w)):
        for j, sign in enumerate((-1, 1)):
            values = w[i, q[i] == sign]
            if len(values):
                levels[i, j] = values.mean()
    if not np.isfinite(levels).all():
        raise ValueError('FP16 level overflow')
    base = np.where(q < 0, levels[:, 0:1], np.where(q > 0, levels[:, 1:2], 0.))
    residual = w - base
    # Diagonal activation score is only a ranking heuristic. Candidate quality
    # is measured afterward on complete activation vectors, including cross terms.
    score = np.abs(residual) * rms[None, :]
    k = int(w.size * correction_fraction)
    chosen = np.argsort(-score.ravel(), kind='stable')[:k]
    chosen = np.sort(chosen).astype('<u2')
    values = residual.ravel()[chosen].astype('<f2')
    if not np.isfinite(values).all():
        raise ValueError('FP16 residual overflow')
    stream = encode(q)
    payload = stream + levels.tobytes() + chosen.tobytes() + values.tobytes()
    return struct.pack(HEADER, b'CR12', *w.shape, len(stream), k,
                       zlib.crc32(payload)) + payload


def unpack_page(blob):
    if len(blob) < HEADER_BYTES:
        raise ValueError('truncated page')
    magic, rows, cols, nbase, k, crc = struct.unpack_from(HEADER, blob)
    payload = blob[HEADER_BYTES:]
    if (magic != b'CR12' or not rows or not cols or rows * cols > 65535
            or k > rows * cols or len(payload) != nbase + rows * 4 + k * 4
            or zlib.crc32(payload) != crc):
        raise ValueError('invalid page framing or checksum')
    q = decode(payload[:nbase])
    if q.shape != (rows, cols):
        raise ValueError('base shape differs')
    levels = np.frombuffer(payload, '<f2', count=2 * rows, offset=nbase).reshape(rows, 2)
    indices = np.frombuffer(payload, '<u2', count=k, offset=nbase + rows * 4)
    values = np.frombuffer(payload, '<f2', count=k, offset=nbase + rows * 4 + 2 * k)
    if (not np.isfinite(levels).all() or not np.isfinite(values).all()
            or (k and (indices[-1] >= rows * cols
                       or np.any(np.diff(indices.astype(np.int64)) <= 0)))):
        raise ValueError('invalid levels or residual indices')
    out = np.where(q < 0, levels[:, 0:1], np.where(q > 0, levels[:, 1:2], 0.)).astype(np.float64)
    out.ravel()[indices] += values
    return out
