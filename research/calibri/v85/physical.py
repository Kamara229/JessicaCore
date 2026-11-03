"""CB85: complete physically accounted original CB59 plus tiny adaptive operators.

No weights hidden in runtime Python. Native CB59 original archive is included
in full, with the original SHA/CRC. Only NEW FP16 codebooks and per-operator
token-dependent FP16 gating vectors are additionally stored and counted.
"""
from __future__ import annotations
import hashlib
import json
import math
import struct
import zlib

import numpy as np
from direct_codebook_v59 import CB59, K, VECTOR
from paid_lowrank_v76 import SOURCE_BYTES
from binary_geometry_v64 import EXPECTED_UNIQUE

MAGIC=b"CLB85\0\0\0"
TAIL=struct.Struct("<8sII")
FOOTER=36


def inspect_source_pair(original: bytes, rewritten: bytes, paths: tuple[str,...]):
    a, b = CB59(original), CB59(rewritten)
    if len(original)!=SOURCE_BYTES or len(rewritten)!=SOURCE_BYTES:
        raise AssertionError("not complete original and rewritten CB59 physical sources")
    if a.meta!=b.meta:
        raise AssertionError("source topology changed in previous experiment")
    for name in sorted(a.handles):
        aa,bb=a.handles[name],b.handles[name]
        if not np.array_equal(aa.packed,bb.packed):
            raise AssertionError("v84 changed original 5bit index source")
        if not np.array_equal(aa.row_scales,bb.row_scales):
            raise AssertionError("v84 changed original row scales")
        if name not in tuple(path+".weight" for path in paths):
            if not np.array_equal(aa.centroids,bb.centroids):
                raise AssertionError("v84 changed unselected original tensor")
    if set(a.raw)!=set(b.raw):
        raise AssertionError("v84 modified original scalar tensor set")
    for name in a.raw:
        if not np.array_equal(a.raw[name],b.raw[name]):
            raise AssertionError("v84 changed original scalar tensor")
    return a,b


def create_packed(original: bytes, changed: bytes, gates: dict[str,tuple]) -> bytes:
    paths=tuple(sorted(gates))
    baseline,secondary=inspect_source_pair(original,changed,paths)
    extra=bytearray()
    manifest={"version":1,"source":"original_CB59_complete_and_untouched",
              "original_SHA256":hashlib.sha256(original).hexdigest(),
              "original_source_bytes":len(original),
              "model_original_unique_params":EXPECTED_UNIQUE,
              "routed_modules":{}}
    for path in paths:
        key=path+".weight"
        h=baseline.handles[key]
        t=secondary.handles[key]
        weights,bias=gates[path]
        w=np.asarray(weights,dtype="<f2")
        b=np.asarray(bias,dtype="<f2").reshape(-1)
        if w.shape!=(h.shape[1],) or b.shape!=(1,):
            raise AssertionError("wrong original native operator gate shape")
        if not (np.isfinite(w).all() and np.isfinite(b).all()):
            raise AssertionError("nonfinite actual physical gate")
        centroid=t.centroids.astype("<f2")
        if centroid.shape!=(K,VECTOR,):
            raise AssertionError("physically packed centroids malformed")
        segments={}
        for name,data in (("alt_codebook",centroid.tobytes()),
                          ("gate_weights",w.tobytes()),("gate_bias",b.tobytes())):
            segments[name]=[len(extra),len(data)]
            extra.extend(data)
        manifest["routed_modules"][path]={"shape":list(h.shape),
                                         "segments":segments}
    meta=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
    payload=original+bytes(extra)+meta+TAIL.pack(MAGIC,len(meta),len(extra))
    protected=payload+struct.pack("<I",zlib.crc32(payload))
    blob=protected+hashlib.sha256(protected).digest()
    if 8*len(blob)/EXPECTED_UNIQUE > 0.700:
        raise AssertionError("adaptive source exceeds complete physical budget")
    return blob


class CB85:
    def __init__(self, blob: bytes):
        if len(blob)<SOURCE_BYTES+TAIL.size+FOOTER:
            raise ValueError("incomplete paid 0.700 bpw model")
        if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
            raise ValueError("CB85 SHA256 integrity failed")
        if zlib.crc32(blob[:-36])!=struct.unpack_from("<I",blob,len(blob)-36)[0]:
            raise ValueError("CB85 CRC32 integrity failed")
        first=blob[:SOURCE_BYTES]
        self.original=CB59(first)
        tail_at=len(blob)-FOOTER-TAIL.size
        magic, meta_len,extra_len=TAIL.unpack_from(blob,tail_at)
        if magic!=MAGIC or meta_len<=0 or extra_len<=0:
            raise ValueError("CB85 invalid footer")
        begin_meta=tail_at-meta_len
        begin_extra=begin_meta-extra_len
        if begin_extra!=SOURCE_BYTES:
            raise ValueError("CB85 external weights unaccounted or malformed")
        self.meta=json.loads(blob[begin_meta:tail_at].decode("utf8"))
        if (self.meta["original_SHA256"]!=hashlib.sha256(first).hexdigest()
            or self.meta["model_original_unique_params"]!=EXPECTED_UNIQUE):
            raise ValueError("CB85 original model provenance changed")
        self.extra={}
        for path,section in self.meta["routed_modules"].items():
            m,n=section["shape"]
            if path+".weight" not in self.original.handles:
                raise ValueError("CB85 unknown source module")
            if tuple(self.original.handles[path+".weight"].shape)!=(m,n):
                raise ValueError("CB85 wrong original module geometry")
            def read(name):
                off,length=section["segments"][name]
                if off<0 or length<0 or off+length>extra_len:
                    raise ValueError("CB85 unbounded extras")
                return blob[begin_extra+off:begin_extra+off+length]
            centroids=np.frombuffer(read("alt_codebook"),dtype="<f2")
            gate=np.frombuffer(read("gate_weights"),dtype="<f2")
            bias=np.frombuffer(read("gate_bias"),dtype="<f2")
            if (centroids.size!=K*VECTOR or gate.size!=n or bias.size!=1 or
                not np.isfinite(centroids).all() or
                not np.isfinite(gate).all() or not np.isfinite(bias).all()):
                raise ValueError("CB85 invalid physical adaptive weights")
            self.extra[path]=(centroids.astype(np.float32).reshape(K,VECTOR),
                              gate.astype(np.float32),float(bias[0]))
        self.blob=blob

    def physical_stats(self):
        return {"total_source_bytes":len(self.blob),
                "all_in_bpw":8*len(self.blob)/EXPECTED_UNIQUE,
                "source_SHA256":hashlib.sha256(self.blob).hexdigest(),
                "original_CB59_bytes":SOURCE_BYTES,
                "routed_modules":len(self.extra),
                "includes_all_extra_codebooks_gates_header_SHA_CRC":True}
