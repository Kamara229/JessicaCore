"""Rewrite only the already-paid FP16 codebooks/scales and FP32 vectors."""
import hashlib
import struct
import zlib
import numpy as np
from direct_codebook_v59 import CB59, HEADER


def replace_fields(archive, values):
    desc = archive.meta['tensor_descriptors']
    if set(values) != set(desc):
        raise ValueError('Incomplete source parameters')
    raw = bytearray(archive.blob)
    _, _, mlen, _ = HEADER.unpack_from(raw)
    start = HEADER.size + mlen
    for name, d in desc.items():
        expected = {'centroids', 'row_scales'} if d['dtype'] == 'CB59_K32V8' else {'raw'}
        if set(values[name]) != expected:
            raise ValueError('Only paid continuous fields can change')
        for part, array in values[name].items():
            dtype = '<f4' if part == 'raw' else '<f2'
            value = np.asarray(array, dtype=dtype)
            shape = (32, 8) if part == 'centroids' else ((d['shape'][0],) if part == 'row_scales' else tuple(d['shape']))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid saved field shape/values')
            if part == 'row_scales' and np.any(value <= 0):
                raise ValueError('Invalid saved positive scales')
            offset, size = d['parts'][part]
            data = value.tobytes()
            if len(data) != size:
                raise ValueError('Changed source allocation')
            raw[start+offset:start+offset+size] = data
    raw[-36:-32] = struct.pack('<I', zlib.crc32(raw[:-36]))
    raw[-32:] = hashlib.sha256(raw[:-32]).digest()
    result = bytes(raw)
    saved = CB59(result)
    if len(result) != len(archive.blob) or saved.meta != archive.meta:
        raise AssertionError('Changed source byte budget or metadata')
    for name, d in desc.items():
        if d['dtype'] == 'CB59_K32V8' and not np.array_equal(saved.handles[name].packed, archive.handles[name].packed):
            raise AssertionError('Unplanned code-index modification')
    return result


def export(archive, model, sources):
    values = {}
    for name, d in archive.meta['tensor_descriptors'].items():
        if d['dtype'] == 'CB59_K32V8':
            centers, scales = sources[name].fields()
            values[name] = {'centroids': centers.detach().numpy(), 'row_scales': scales.detach().numpy()}
        else:
            parent, _, leaf = name.rpartition('.')
            values[name] = {'raw': getattr(model.get_submodule(parent), leaf).detach().numpy()}
    return replace_fields(archive, values)

