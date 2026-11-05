"""Exact-preserving packed 5-bit code and FP16 dictionary rewrite.

Only already-paid CB59 fields may change: source packed indices and
K32xV8 FP16 codebooks. All original row scales, 1D tensors, metadata
and tied aliases remain byte-identical. Same 11,229,303 file size.
"""
import hashlib,struct,zlib
import numpy as np
from direct_codebook_v59 import CB59,HEADER,pack5,unpack5
from . import protocol as p


def replace_discrete(archive, sources, strict=True):
    if not isinstance(archive,CB59):
        raise ValueError("Expected actual authenticated CB59")
    if strict and (archive.hash()!=p.SOURCE_SHA or
                   len(archive.blob)!=p.SOURCE_BYTES):
        raise ValueError("Unpinned original source")
    matrices={name for name,d in archive.meta["tensor_descriptors"].items()
              if d["dtype"]=="CB59_K32V8"}
    if set(sources)!=matrices:
        raise ValueError("Must rewrite every original 2D matrix")
    raw=bytearray(archive.blob)
    _,_,meta_size,_=HEADER.unpack_from(raw)
    start=HEADER.size+meta_size
    flips=0; changed_centroids=0
    for name in sorted(matrices):
        descriptor=archive.meta["tensor_descriptors"][name]
        entry=sources[name]
        shape=tuple(descriptor["shape"])
        target_count=shape[0]*(shape[1]//8)
        current=unpack5(archive.handles[name].packed.tobytes(),target_count)
        chosen=entry.codes.detach().cpu().numpy().astype(np.uint8)
        if len(chosen)!=target_count or np.any(chosen>31):
            raise ValueError("Wrong 5bit model index shape: "+name)
        flips+=int(np.count_nonzero(current!=chosen))
        blob=pack5(chosen)
        off,length=descriptor["parts"]["packed"]
        if len(blob)!=length:
            raise AssertionError("Physically unequal code stream")
        raw[start+off:start+off+length]=blob
        old_centroids=archive.handles[name].centroids
        trained=entry.centroids.detach().cpu().numpy().astype("<f2")
        if trained.shape!=(32,8) or not np.isfinite(trained).all():
            raise ValueError("Corrupt learned 8D codebook")
        changed_centroids+=int(np.count_nonzero(
            trained.astype(np.float32)!=old_centroids))
        off,length=descriptor["parts"]["centroids"]
        if len(trained.tobytes())!=length:
            raise AssertionError("Codebook metadata size unexpectedly changed")
        raw[start+off:start+off+length]=trained.tobytes()
    raw[-36:-32]=struct.pack("<I",zlib.crc32(raw[:-36]))
    raw[-32:]=hashlib.sha256(raw[:-32]).digest()
    result=bytes(raw)
    rebuilt=CB59(result)
    if len(result)!=len(archive.blob) or rebuilt.meta!=archive.meta:
        raise AssertionError("Source budget or original geometry drift")
    for name,d in archive.meta["tensor_descriptors"].items():
        if d["dtype"]=="f4":
            if not np.array_equal(archive.raw[name],rebuilt.raw[name]):
                raise AssertionError("Unpaid 1D vector mutation")
        else:
            if not np.array_equal(archive.handles[name].row_scales,
                                  rebuilt.handles[name].row_scales):
                raise AssertionError("Frozen row scale was rewritten")
    if rebuilt.handles["model.embed_tokens.weight"] is not rebuilt.handles["lm_head.weight"]:
        raise AssertionError("Original tied source alias broken")
    if strict and len(result)!=p.SOURCE_BYTES:
        raise AssertionError("v185 original complete physical byte budget drift")
    if len(result)>p.MAX_BYTES:
        raise AssertionError("Exceeded entire original parameter 0.7 cap")
    return result,{"bytes":len(result),
        "complete_bpp":8*len(result)/archive.meta["original_unique_parameter_count"],
        "sha256":hashlib.sha256(result).hexdigest(),
        "actual_changed_5bit_group_indices":flips,
        "actually_changed_FP16_codebook_cells":changed_centroids,
        "all_2d_original_matrices":len(matrices),
        "raw_1d_and_scales_unchanged":True,
        "no_unpaid_sidecar":True,
        "physically_paid":True}
