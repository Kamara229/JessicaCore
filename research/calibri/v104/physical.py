"""Full physically paid CB104 = re-coded CB59 source + 8 bank centroids.

Explicitly counts ALL extra banks, header, CRC/SHA. CB59 internal
single-codebook head indices are INVALID on their own and the archive
must be loaded only by the dedicated banked native kernel.
"""
from __future__ import annotations
import hashlib,struct,zlib
import numpy as np
from direct_codebook_v59 import CB59
from binary_geometry_v64 import EXPECTED_UNIQUE
from .spec import BANKS,K,V,VOCAB,DIM

HEADER=struct.Struct("<8sIIII")
MAGIC=b"CLB104\0\0"
DIGEST=36
EXTRA_BYTES=BANKS*K*V*2
BASE_BYTES=11229303
TOTAL_BYTES=HEADER.size+BASE_BYTES+EXTRA_BYTES+DIGEST
BPW=8*TOTAL_BYTES/EXPECTED_UNIQUE
assert BPW<=0.700 and EXTRA_BYTES==4096

def pack(base,bank_centroids):
    cb=CB59(base)
    head=cb.handles["lm_head.weight"]
    if (head is not cb.handles["model.embed_tokens.weight"]
        or head.shape!=(VOCAB,DIM) or len(base)!=BASE_BYTES):
        raise AssertionError("must contain entire original tied head and model source")
    z=np.asarray(bank_centroids,dtype="<f2")
    if z.shape!=(BANKS,K,V) or not np.isfinite(z).all():
        raise AssertionError("bank codebook shapes, dtype or contents invalid")
    body=HEADER.pack(MAGIC,BASE_BYTES,BANKS,K,V)+base+z.tobytes()
    with_crc=body+struct.pack("<I",zlib.crc32(body))
    blob=with_crc+hashlib.sha256(with_crc).digest()
    if len(blob)!=TOTAL_BYTES:
        raise AssertionError("full disk archive is not actually paid")
    return blob

def unpack(blob):
    if len(blob)!=TOTAL_BYTES:
        raise AssertionError("CB104 full physical archive incomplete")
    magic,base_size,banks,k,v=HEADER.unpack_from(blob)
    if (magic,base_size,banks,k,v)!=(MAGIC,BASE_BYTES,BANKS,K,V):
        raise AssertionError("unsupported fixed CB104 full-model source ABI")
    if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
        raise AssertionError("original CB104 complete CRC/SHA integrity failure")
    if zlib.crc32(blob[:-36])!=struct.unpack_from("<I",blob,len(blob)-36)[0]:
        raise AssertionError("CB104 source CRC mismatch")
    start=HEADER.size
    base=blob[start:start+BASE_BYTES]
    c=np.frombuffer(blob[start+BASE_BYTES:-36],dtype="<f2").copy().reshape(BANKS,K,V)
    parsed=CB59(base)
    if parsed.handles["lm_head.weight"] is not parsed.handles["model.embed_tokens.weight"]:
        raise AssertionError("shared original trained output/input codebook lost")
    if not np.isfinite(c).all():
        raise AssertionError("non-finite original FP16 bank centroids")
    return base,c

def result_audit(blob):
    base,c=unpack(blob)
    return {"complete_archive_bytes":len(blob),
       "all_in_bits_per_original_unique_parameter":8*len(blob)/EXPECTED_UNIQUE,
       "source_CB59_manifest_and_packed_tensors_bytes":len(base),
       "new_physically_stored_FP16_bank_codebooks_bytes":c.nbytes,
       "header_CRC_and_SHA_bytes":HEADER.size+DIGEST,
       "source_file_SHA256":hashlib.sha256(blob).hexdigest(),
       "all_trained_pretrained_2D_tensors_direct_native_not_dense":True}
