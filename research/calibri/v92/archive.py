"""Reassemble CB59 with one full operator rewritten, original length/ABI unchanged.
All other packed segments are byte-identical; CRC32 and SHA256 recalculated.
"""
from __future__ import annotations
import hashlib,struct,zlib
import numpy as np
from direct_codebook_v59 import CB59,HEADER,pack5

def inplace_replace(original_blob:bytes,name:str,packed_candidate):
    source=CB59(original_blob)
    desc=source.meta["tensor_descriptors"]
    if name not in desc or desc[name]["dtype"]!="CB59_K32V8":
        raise AssertionError("target must be one uniquely stored native compressed matrix")
    if name in source.meta["source_alias_keys"]:
        raise AssertionError("cannot rewrite alias rather than canonical source")
    m,n=desc[name]["shape"]
    data={"packed":pack5(packed_candidate["indices"]),
          "row_scales":packed_candidate["row_scales"].astype("<f2").tobytes(),
          "centroids":packed_candidate["centroids"].astype("<f2").tobytes()}
    ml=HEADER.unpack_from(original_blob)[2]
    offset=HEADER.size+ml
    replacement=bytearray(original_blob)
    changed_ranges=[]
    for segment in ("packed","row_scales","centroids"):
        off,logical_size=desc[name]["parts"][segment]
        if len(data[segment])!=logical_size:
            raise AssertionError("zero-extra-storage rewrite changed compressed matrix size")
        lo,hi=offset+off,offset+off+logical_size
        replacement[lo:hi]=data[segment]
        changed_ranges.append([lo,hi])
    replacement[-36:-32]=struct.pack("<I",zlib.crc32(replacement[:-36]))
    replacement[-32:]=hashlib.sha256(replacement[:-32]).digest()
    rebuilt=bytes(replacement)
    if len(rebuilt)!=len(original_blob):raise AssertionError("total physical archive size changed")
    after=CB59(rebuilt)
    if after.meta!=source.meta:raise AssertionError("manifest metadata changed")
    for key in desc:
        if key==name:continue
        if key in source.handles:
            a,b=source.handles[key],after.handles[key]
            if (not np.array_equal(a.packed,b.packed) or
                not np.array_equal(a.centroids,b.centroids) or
                not np.array_equal(a.row_scales,b.row_scales)):
                raise AssertionError("unselected packed source changed: "+key)
        else:
            if not np.array_equal(source.raw[key],after.raw[key]):
                raise AssertionError("unselected raw trained source changed: "+key)
    if len(data["packed"])!=m*n*5//64:
        raise AssertionError("5-bit/8 native geometry source changed")
    return rebuilt,{"base_SHA256":hashlib.sha256(original_blob).hexdigest(),
                    "candidate_SHA256":hashlib.sha256(rebuilt).hexdigest(),
                    "same_complete_source_length":len(rebuilt),
                    "native_ABI_unchanged":True,
                    "all_other_modules_byte_unchanged":True,
                    "rewritten_module":name,"rewritten_physical_segments":changed_ranges,
                    "extra_paid_bytes":0}
