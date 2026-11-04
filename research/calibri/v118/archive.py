"""All-inclusive physical source: original locked CB59 + roles + head."""
import hashlib,json,struct,zlib
import numpy as np
from v117.archive import pinned
from .protocol import (SOURCE_BYTES,ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES,
                       HEADER,VOCAB,HEAD_DIM,roles,factor_data_bytes)
HDR=struct.Struct("<8sII")
def as16(x,shape):
    a=np.asarray(x,dtype="<f2")
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError("bad paid tensor")
    return a
def pack(base,head=None,blocks=None):
    pinned(base)
    if blocks is not None and set(blocks)!={n for n,_,_ in roles()}:
        raise ValueError("missing operator")
    data=bytearray();layout=[]
    for name,rows,cols in roles():
        u,v=(np.zeros((rows,1)),np.zeros((1,cols))) if blocks is None else blocks[name]
        u=as16(u,(rows,1));v=as16(v,(1,cols))
        layout.append([name,rows,cols,len(data)])
        data.extend(u.tobytes());data.extend(v.tobytes())
    if head is None:
        u,v,b=np.zeros((VOCAB,2)),np.zeros((2,HEAD_DIM)),np.zeros(VOCAB)
    else:u,v,b=head
    u=as16(u,(VOCAB,2));v=as16(v,(2,HEAD_DIM));b=as16(b,(VOCAB,))
    off=len(data)
    for x in (u,v,b):data.extend(x.tobytes())
    if len(data)!=factor_data_bytes():raise AssertionError("unpaid factor stream")
    meta=json.dumps({"version":118,"original":ORIGINAL_CB59_SHA,
                     "unique":TOTAL_UNIQUE,"roles":layout,"head_at":off},
                    separators=(",",":")).encode()
    body=HDR.pack(HEADER,len(meta),len(data))+base+meta+data
    digest=body+struct.pack("<I",zlib.crc32(body))
    result=digest+hashlib.sha256(digest).digest()
    if len(result)>MAX_SOURCE_BYTES:raise AssertionError("above full 0.7bpp budget")
    return result
class Paid:
    def __init__(self,raw):
        if len(raw)<HDR.size+SOURCE_BYTES+36:raise ValueError("truncated")
        magic,mlen,dlen=HDR.unpack_from(raw)
        if (magic!=HEADER or not 10<=mlen<=20000 or dlen!=factor_data_bytes()
            or len(raw)!=HDR.size+SOURCE_BYTES+mlen+dlen+36 or len(raw)>MAX_SOURCE_BYTES):
            raise ValueError("bad physical all-in layout")
        if hashlib.sha256(raw[:-32]).digest()!=raw[-32:] or zlib.crc32(raw[:-36])!=struct.unpack_from("<I",raw,len(raw)-36)[0]:
            raise ValueError("whole-archive CRC or SHA mismatch")
        self.raw=bytes(raw)
        self.base=raw[HDR.size:HDR.size+SOURCE_BYTES]
        pinned(self.base)
        start=HDR.size+SOURCE_BYTES
        m=json.loads(raw[start:start+mlen])
        if (m.get("version")!=118 or m.get("original")!=ORIGINAL_CB59_SHA
            or m.get("unique")!=TOTAL_UNIQUE or len(m.get("roles",[]))!=60):
            raise ValueError("bad source model or denominator")
        self.data=memoryview(raw)[start+mlen:-36]
        self.layout={}
        off=0
        for (name,r,c),rec in zip(roles(),m["roles"]):
            if rec!=[name,r,c,off]:raise ValueError("role corruption")
            self.layout[name]=(off,r,c)
            off+=2*(r+c)
        if m.get("head_at")!=off or off+2*(VOCAB*2+2*HEAD_DIM+VOCAB)!=len(self.data):
            raise ValueError("unpaid head")
        self.head_at=off
        if not all(np.isfinite(x).all() for x in self.head()):raise ValueError("bad head values")
    def role(self,name):
        off,r,c=self.layout[name]
        u=np.frombuffer(self.data[off:off+2*r],dtype="<f2").reshape(r,1)
        v=np.frombuffer(self.data[off+2*r:off+2*(r+c)],dtype="<f2").reshape(1,c)
        return u,v
    def head(self):
        off=self.head_at;a=VOCAB*2*2;b=2*HEAD_DIM*2
        u=np.frombuffer(self.data[off:off+a],dtype="<f2").reshape(VOCAB,2)
        v=np.frombuffer(self.data[off+a:off+a+b],dtype="<f2").reshape(2,HEAD_DIM)
        bias=np.frombuffer(self.data[off+a+b:],dtype="<f2").reshape(VOCAB)
        return u,v,bias
    def stats(self):
        return {"all_in_bytes":len(self.raw),"original_CB59_bytes":len(self.base),
                "extra_physically_paid_bytes":len(self.raw)-len(self.base),
                "factor_bytes":len(self.data),"unique_original_weights":TOTAL_UNIQUE,
                "all_in_bpp":8*len(self.raw)/TOTAL_UNIQUE,
                "free_bytes_until_0_700":MAX_SOURCE_BYTES-len(self.raw),
                "original_frozen_SHA":ORIGINAL_CB59_SHA,
                "whole_archive_SHA":hashlib.sha256(self.raw).hexdigest()}
