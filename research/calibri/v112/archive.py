"""Real fully serialized 4-bank 5-bit operator archive; blockwise matvec.

The archive covers the SELECTED 24 operator weights, not embeddings,
norms, output head or the full LLM. End-to-end inference in pilot temporarily
reconstructs selected matrices to test their FUNCTION, never claims
compressed-resident full-model performance.
"""
from __future__ import annotations
import hashlib
import json
import struct
import zlib
import numpy as np
from .fit import BANKS, K, GROUP, encode_matrix, learn_banks
from .roles import bank_for

MAGIC = b"CLB112\0\0"
HEADER = struct.Struct("<8sII")
TRAILER = 36
CODEBOOK_BYTES = BANKS*K*GROUP*2

def pack5(values):
    data = np.asarray(values, dtype=np.uint8).ravel()
    if np.any(data >= 32):
        raise ValueError("5-bit index overflow")
    bit = np.arange(len(data), dtype=np.int64)*5
    out = np.zeros((len(data)*5+7)//8, dtype=np.uint8)
    for k in range(5):
        at = bit+k
        piece = ((data>>k)&1).astype(np.uint8) << (at%8).astype(np.uint8)
        np.bitwise_or.at(out, at//8, piece)
    return out.tobytes()

def unpack5_region(source, first, count):
    if first < 0 or count < 0:
        raise ValueError("invalid 5bit range")
    data = np.frombuffer(source, dtype=np.uint8)
    bit = np.arange(first, first+count, dtype=np.int64)*5
    byte = bit//8
    if len(byte) and byte[-1]>=len(data):
        raise ValueError("truncated 5bit input")
    low = data[byte].astype(np.uint16) >> (bit%8)
    other = data[np.minimum(byte+1, len(data)-1)].astype(np.uint16)
    high = other << (8-(bit%8))
    return ((low|high)&31).astype(np.uint8)

def create_archive(source, mode):
    books = learn_banks(source, mode)
    if books.shape != (BANKS,K,GROUP):
        raise AssertionError("wrong book dimensions")
    payload = bytearray(books.astype("<f2").tobytes())
    records = []
    for name,w in sorted(source.items()):
        row,col = w.shape
        bank = bank_for(mode,name)
        idx,scales = encode_matrix(w,books[bank])
        scbytes = scales.astype("<f2").tobytes()
        idxbytes = pack5(idx)
        record = {"name":name,"rows":int(row),"cols":int(col),"bank":int(bank),
                  "scale_at":len(payload),"scale_bytes":len(scbytes)}
        payload.extend(scbytes)
        record.update({"codes_at":len(payload),"codes_bytes":len(idxbytes),
                       "code_count":int(row*(col//GROUP))})
        payload.extend(idxbytes)
        records.append(record)
    metadata = {"schema":"CALIBRI-STRUCTURAL-v112-1","mode":mode.ljust(16),
                "group":GROUP,"K":K,"banks":BANKS,
                "selected_operator_count":len(records),
                "selected_weight_elements":int(sum(w.size for w in source.values())),
                "records":records}
    j = json.dumps(metadata,sort_keys=True,separators=(",",":")).encode()
    body = HEADER.pack(MAGIC,len(j),len(payload))+j+payload
    with_crc = body+struct.pack("<I",zlib.crc32(body))
    return with_crc+hashlib.sha256(with_crc).digest()

class Archive:
    def __init__(self,blob):
        self.bytes = bytes(blob)
        if len(blob)<HEADER.size+TRAILER+CODEBOOK_BYTES:
            raise ValueError("invalid truncated v112 source")
        magic,mlen,plen = HEADER.unpack_from(blob)
        if magic != MAGIC or mlen<10 or mlen>200000 or plen<CODEBOOK_BYTES:
            raise ValueError("unrecognized structural format")
        if HEADER.size+mlen+plen+TRAILER != len(blob):
            raise ValueError("invalid physical size")
        if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
            raise ValueError("bad whole-file SHA256")
        if zlib.crc32(blob[:-36]) != struct.unpack_from("<I",blob,len(blob)-36)[0]:
            raise ValueError("bad whole-file CRC32")
        self.meta = json.loads(blob[HEADER.size:HEADER.size+mlen])
        if self.meta["schema"]!="CALIBRI-STRUCTURAL-v112-1":
            raise ValueError("bad schema")
        self.payload = memoryview(blob)[HEADER.size+mlen:HEADER.size+mlen+plen]
        self.books = np.frombuffer(self.payload[:CODEBOOK_BYTES],
                                   dtype="<f2").reshape((BANKS,K,GROUP))
        self.records = {}
        last = CODEBOOK_BYTES
        count = 0
        for rec in self.meta["records"]:
            name=rec["name"]
            if name in self.records or not 0<=rec["bank"]<BANKS:
                raise ValueError("duplicate or invalid physical operator")
            rows,cols=rec["rows"],rec["cols"]
            if rows<=0 or cols<=0 or cols%GROUP:
                raise ValueError("bad matrix geometry")
            if rec["scale_at"]!=last or rec["scale_bytes"]!=rows*2:
                raise ValueError("scale layout corruption")
            last+=rec["scale_bytes"]
            groups=rows*cols//GROUP
            if rec["code_count"]!=groups or rec["codes_at"]!=last or rec["codes_bytes"]!=(groups*5+7)//8:
                raise ValueError("code stream mismatch")
            last+=rec["codes_bytes"]
            if bank_for(self.meta["mode"].strip(),name)!=rec["bank"]:
                raise ValueError("role/layout bank mismatch")
            self.records[name]=rec
            count+=rows*cols
        if last!=plen or count!=self.meta["selected_weight_elements"]:
            raise ValueError("unpaid or missing physical source bytes")
    def _parts(self,name):
        r=self.records[name]
        scale = np.frombuffer(self.payload[r["scale_at"]:r["scale_at"]+r["scale_bytes"]],dtype="<f2")
        codes = self.payload[r["codes_at"]:r["codes_at"]+r["codes_bytes"]]
        return r,scale,codes
    def decode_rows(self,name,first,stop):
        r,scale,codes=self._parts(name)
        if not 0<=first<=stop<=r["rows"]:
            raise ValueError("row range exceeds encoded matrix")
        g = r["cols"]//GROUP
        idx = unpack5_region(codes,first*g,(stop-first)*g)
        lookup=np.asarray(self.books[r["bank"]],dtype=np.float32)
        result=lookup[idx].reshape((stop-first,r["cols"]))
        return np.ascontiguousarray(result*scale[first:stop,None].astype(np.float32))
    def decode(self,name):
        r=self.records[name]
        return self.decode_rows(name,0,r["rows"])
    def direct(self,name,x,chunk=16):
        """Streaming selected GEMV with <=chunk source rows resident in scratch."""
        r=self.records[name]
        a=np.asarray(x,dtype=np.float32)
        if a.ndim!=2 or a.shape[1]!=r["cols"] or chunk<=0:
            raise ValueError("expected batch by in_features")
        out=np.empty((len(a),r["rows"]),dtype=np.float32)
        for start in range(0,r["rows"],chunk):
            w=self.decode_rows(name,start,min(start+chunk,r["rows"]))
            out[:,start:start+len(w)]=a @ w.T
        return out
    def summary(self):
        n=self.meta["selected_weight_elements"]
        return {"operator_count":len(self.records),"selected_elements":n,
                "whole_source_bytes":len(self.bytes),
                "selected_bits_per_weight":8*len(self.bytes)/n,
                "all_four_books_physically_present":True,
                "full_model_bpp_NOT_measured":True,
                "whole_source_sha256":hashlib.sha256(self.bytes).hexdigest()}
