"""NRA3 S1: lossless CB61-style entropy envelope over paid CB59 archive.

Shares Calibri S61's mechanism (zlib-compressed CB59, one startup decode).
Never invents zero-cost weights or changes the CB59 codewords; physically
stored complete bytes, SHA, CRC and decompression are verified.
No direct packed C inference, training, or acceptance is claimed.
"""
from __future__ import annotations
import hashlib
import struct
import zlib

from direct_codebook_v59 import CB59

MAGIC=b"N3CB61A1"
HEADER=struct.Struct("<8sQI32s")
MAX_RESTORED=16_000_000


def store_lossless_cb59(raw:bytes) -> bytes:
    if not isinstance(raw,bytes):
        raise ValueError("CB61 input must be physical bytes")
    source=CB59(raw)
    if source.physical_bytes()!=len(raw) or len(raw)>MAX_RESTORED:
        raise ValueError("CB61 source physical bounds invalid")
    packed=zlib.compress(raw,level=9)
    header=HEADER.pack(MAGIC,len(raw),zlib.crc32(raw),hashlib.sha256(raw).digest())
    return header+packed


def restore_lossless_cb59(packed:bytes) -> bytes:
    if not isinstance(packed,bytes) or len(packed)<HEADER.size+1:
        raise ValueError("CB61 truncated physical envelope")
    magic,length,crc,digest=HEADER.unpack_from(packed)
    if magic!=MAGIC or length<1 or length>MAX_RESTORED:
        raise ValueError("CB61 invalid header, magic or length")
    decoder=zlib.decompressobj()
    try:
        raw=decoder.decompress(packed[HEADER.size:],length+1)
    except zlib.error as e:
        raise ValueError("CB61 compressed bytes corrupted") from e
    if (decoder.unconsumed_tail or not decoder.eof or decoder.unused_data
            or len(raw)!=length or zlib.crc32(raw)!=crc
            or hashlib.sha256(raw).digest()!=digest):
        raise ValueError("CB61 restored bytes drift, overflow or trailing data")
    CB59(raw)
    return raw
