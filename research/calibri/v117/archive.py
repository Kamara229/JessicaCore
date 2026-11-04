"""Append a strictly paid, rank-2 structural correction to frozen native CB59.

The initial adapter is ZERO. This is an integration test, NOT a quality gain.
"""
import hashlib,json,struct,zlib
import numpy as np
from .protocol import (SOURCE_BYTES,ORIGINAL_CB59_SHA,TOTAL_UNIQUE,
                       MAX_SOURCE_BYTES,RANK,HEADER,roles,declared_factor_bytes)
HEAD=struct.Struct("<8sII")
TAIL=36

def pinned(base):
    if len(base)!=SOURCE_BYTES or hashlib.sha256(base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("full original CB59 sha or length changed")

def encode(base,factors=None):
    pinned(base)
    data=bytearray()
    manifest=[]
    for name,rows,cols in roles():
        if factors is None:
            u=np.zeros((rows,RANK),dtype="<f2")
            v=np.zeros((RANK,cols),dtype="<f2")
        else:
            if name not in factors:raise ValueError("missing paid role "+name)
            u,v=factors[name]
            u=np.asarray(u,dtype="<f2");v=np.asarray(v,dtype="<f2")
        if u.shape!=(rows,RANK) or v.shape!=(RANK,cols):
            raise ValueError("invalid paid factor shape "+name)
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("non-finite correction")
        manifest.append([name,rows,cols,len(data)])
        data.extend(u.tobytes());data.extend(v.tobytes())
    if len(data)!=declared_factor_bytes():
        raise AssertionError("unpaid / duplicate rank-2 coefficients")
    info=json.dumps({"version":117,"unique":TOTAL_UNIQUE,
                     "source_sha256":ORIGINAL_CB59_SHA,"rank":RANK,
                     "layers":manifest},separators=(",",":")).encode()
    body=HEAD.pack(HEADER,len(info),len(data))+base+info+data
    crc=body+struct.pack("<I",zlib.crc32(body))
    result=crc+hashlib.sha256(crc).digest()
    if len(result)>MAX_SOURCE_BYTES:
        raise AssertionError("full checkpoint >0.7 bits/original parameter")
    return result

class Archive:
    def __init__(self,raw):
        self.raw=bytes(raw)
        if len(raw)<HEAD.size+SOURCE_BYTES+TAIL:raise ValueError("truncated")
        magic,mlen,dlen=HEAD.unpack_from(raw)
        if magic!=HEADER or mlen<10 or mlen>20000 or dlen!=declared_factor_bytes():
            raise ValueError("invalid v117 source layout")
        if len(raw)!=HEAD.size+SOURCE_BYTES+mlen+dlen+TAIL or len(raw)>MAX_SOURCE_BYTES:
            raise ValueError("incomplete or over-budget full source")
        if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
            raise ValueError("bad whole-source SHA")
        if zlib.crc32(raw[:-36])!=struct.unpack_from("<I",raw,len(raw)-36)[0]:
            raise ValueError("bad whole-source CRC")
        self.base=raw[HEAD.size:HEAD.size+SOURCE_BYTES]
        pinned(self.base)
        start=HEAD.size+SOURCE_BYTES
        meta=json.loads(raw[start:start+mlen])
        if (meta["version"]!=117 or meta["unique"]!=TOTAL_UNIQUE or
            meta["source_sha256"]!=ORIGINAL_CB59_SHA or meta["rank"]!=RANK):
            raise ValueError("changed original model or denominator")
        self.payload=memoryview(raw)[start+mlen:-TAIL]
        if len(meta["layers"])!=len(roles()):raise ValueError("missing roles")
        self.slices={}
        end=0
        for (name,rows,cols),rec in zip(roles(),meta["layers"]):
            if rec!=[name,rows,cols,end]:
                raise ValueError("invalid/missing sidecar role or byte offset")
            self.slices[name]=(end,rows,cols)
            end+=2*RANK*(rows+cols)
        if end!=len(self.payload):raise ValueError("missing paid bytes")
        for name in self.slices:
            u,v=self.factors(name)
            if not np.isfinite(u).all() or not np.isfinite(v).all():
                raise ValueError("non-finite factors")

    def factors(self,name):
        start,rows,cols=self.slices[name]
        a=2*rows*RANK
        b=2*cols*RANK
        u=np.frombuffer(self.payload[start:start+a],dtype="<f2").reshape(rows,RANK)
        v=np.frombuffer(self.payload[start+a:start+a+b],dtype="<f2").reshape(RANK,cols)
        return u,v

    def stats(self):
        return {"original_CB59_SHA_preserved":True,
                "full_original_unique_parameters":TOTAL_UNIQUE,
                "complete_source_bytes":len(self.raw),
                "native_CB59_base_bytes":len(self.base),
                "extra_all_in_bytes":len(self.raw)-len(self.base),
                "all_in_bpp":8*len(self.raw)/TOTAL_UNIQUE,
                "max_total_bytes":MAX_SOURCE_BYTES,
                "physical_all_roles":len(self.slices),
                "physical_factor_bytes":len(self.payload),
                "sha256":hashlib.sha256(self.raw).hexdigest()}
