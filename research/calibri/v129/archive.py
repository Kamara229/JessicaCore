"""CB129 has an explicit format tag and a deterministic 60-slot role layout.

All learned numbers live inside the integrity-protected archive. The input
embedding reuses the same U and V stored for the output correction.
"""
import hashlib
import struct
import zlib
import numpy as np
from v117.archive import pinned
from v117.protocol import SOURCE_BYTES
from v118.archive import as16
from v118.protocol import VOCAB, HEAD_DIM, factor_data_bytes
from .protocol import layout

MAGIC = b"CB129QKV"
SIZE = len(MAGIC) + SOURCE_BYTES + factor_data_bytes() + 36


def shapes():
    for name, rows, cols, rank in layout(True):
        yield name, ((rows, rank), (rank, cols))
    yield "lm_head", ((VOCAB, 2), (2, HEAD_DIM), (VOCAB,))


def pack(base, blocks, head):
    pinned(base)
    if set(blocks) != {name for name, *_ in layout(True)}:
        raise ValueError("unexpected operator set")
    factors = blocks | {"lm_head": head}
    parts = []
    for name, sizes in shapes():
        for a, shape in zip(factors[name], sizes, strict=True):
            parts.append(as16(a, shape).tobytes())
    prefix = MAGIC + base + b"".join(parts)
    crc = prefix + struct.pack("<I", zlib.crc32(prefix))
    raw = crc + hashlib.sha256(crc).digest()
    if len(raw) != SIZE:
        raise ValueError("unexpected physical size")
    return raw


class Source:
    def __init__(self, raw):
        if (len(raw) != SIZE or not raw.startswith(MAGIC)
                or hashlib.sha256(raw[:-32]).digest() != raw[-32:]
                or zlib.crc32(raw[:-36]) != struct.unpack_from("<I", raw, len(raw) - 36)[0]):
            raise ValueError("invalid CB129 archive")
        offset = len(MAGIC) + SOURCE_BYTES
        self.base = raw[len(MAGIC):offset]
        pinned(self.base)
        self.factors = {}
        for name, sizes in shapes():
            parts = []
            for shape in sizes:
                count = int(np.prod(shape))
                a = np.frombuffer(raw, dtype="<f2", count=count, offset=offset).reshape(shape)
                if not np.isfinite(a).all():
                    raise ValueError(f"nonfinite factors for {name}")
                parts.append(a)
                offset += 2 * count
            self.factors[name] = tuple(parts)
        if offset != len(raw) - 36:
            raise ValueError("unaccounted factor bytes")

    def role(self, name):
        return self.factors[name]

    def head(self):
        return self.factors["lm_head"]
