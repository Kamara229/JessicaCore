"""Repack ONE already compressed tied input/output matrix at SAME byte count.

CB59's manifest, all other matrices, original unique count, and 5-bit/8 group
geometry are preserved byte-for-byte. Footer CRC/SHA are regenerated and the
independent CB59 parser verifies the resulting source. This is a new
experimentally re-encoded source, not a mutation of the frozen CB59/v119 file.
"""
import hashlib
import struct
import zlib
import numpy as np
from direct_codebook_v59 import CB59,HEADER,pack5

KEY="model.embed_tokens.weight"
TIED="lm_head.weight"


def checked_source(base, expected_shape=(49152,576)):
    model=CB59(base)
    if KEY not in model.meta["tensor_descriptors"]:
        raise ValueError("the canonical tied embedding is missing")
    if (model.handles.get(TIED) is not model.handles[KEY] or
            model.meta["source_alias_keys"].get(TIED)!=KEY):
        raise ValueError("input/output pretrained weight tying lost")
    if model.handles[KEY].shape!=expected_shape:
        raise ValueError("unexpected original embedding shape")
    return model


def encoded_all(weight,scales,centers,chunk_groups=4096, *, expected_shape=(49152,576)):
    """Stream full 3,538,944 source groups, never full dense CB59 weights."""
    from direct_codebook_v59 import _nearest_integer_128
    w=np.asarray(weight,dtype=np.float32)
    if w.ndim!=2 or w.shape!=expected_shape or len(scales)!=w.shape[0]:
        raise ValueError("teacher embedding source geometry changed")
    c=np.asarray(centers,dtype="<f2").astype(np.float32)
    if c.shape!=(32,8) or not np.isfinite(c).all() or not np.isfinite(w).all():
        raise ValueError("bad input codebook")
    if chunk_groups%8 or chunk_groups<8:
        raise ValueError("code groups must be byte-aligned")
    v=w.reshape(-1,8)
    per_row=w.shape[1]//8
    codes=[]
    for start in range(0,len(v),chunk_groups):
        stop=min(len(v),start+chunk_groups)
        rows=np.arange(start,stop,dtype=np.int64)//per_row
        scaled=(v[start:stop]/np.asarray(scales[rows],dtype=np.float32)[:,None])
        snapped=scaled.astype("<f2").astype(np.float32)
        indices=_nearest_integer_128(snapped,c).astype(np.uint8)
        codes.append(pack5(indices))
    packed=b"".join(codes)
    if len(packed)!=len(v)*5//8:
        raise ValueError("packed code bytes differ from original group count")
    return packed


def patch_one_matrix(base,centers,packed, *, expected_shape=(49152,576)):
    original=checked_source(base,expected_shape)
    magic,version,mlen,blen=HEADER.unpack_from(base)
    if magic!=b"CB59" or version!=1:
        raise ValueError("incompatible old format")
    body_start=HEADER.size+mlen
    desc=original.meta["tensor_descriptors"][KEY]
    expected={"packed":len(original.handles[KEY].packed),
              "centroids":32*8*2}
    values={"packed":bytes(packed),
            "centroids":np.asarray(centers,dtype="<f2").tobytes()}
    patched=bytearray(base)
    for name in expected:
        start,size=desc["parts"][name]
        if size!=expected[name] or len(values[name])!=size:
            raise ValueError("changing physical byte size is not allowed")
        patched[body_start+start:body_start+start+size]=values[name]
    old_body=base[:-36]
    new_body=bytes(patched[:-36])
    # Verify original source only changed inside the two permitted regions.
    if len(new_body)!=len(old_body):
        raise AssertionError("bitstream length changed")
    for key,(offset,size) in desc["parts"].items():
        if key in expected:continue
        if new_body[body_start+offset:body_start+offset+size]!=old_body[
                body_start+offset:body_start+offset+size]:
            raise AssertionError("noncodebook matrix data changed")
    segments=sorted((body_start+desc["parts"][n][0],
                     body_start+desc["parts"][n][0]+desc["parts"][n][1])
                    for n in expected)
    cursor=0
    for left,right in segments:
        if new_body[cursor:left]!=old_body[cursor:left]:
            raise AssertionError("unexpected byte change outside selected tensor")
        cursor=right
    if new_body[cursor:]!=old_body[cursor:]:
        raise AssertionError("unexpected raw tensor change")
    patched[-36:-32]=struct.pack("<I",zlib.crc32(new_body))
    patched[-32:]=hashlib.sha256(bytes(patched[:-32])).digest()
    result=bytes(patched)
    loaded=checked_source(result,expected_shape)
    if len(result)!=len(base) or loaded.meta!=original.meta:
        raise AssertionError("original full-model bit budget/manifest changed")
    if (loaded.handles[KEY].packed.tobytes()!=values["packed"] or
        not np.array_equal(loaded.handles[KEY].centroids,
                           np.asarray(centers,dtype="<f2").astype(np.float32))):
        raise AssertionError("independent decoder didn't preserve new codes")
    return result
