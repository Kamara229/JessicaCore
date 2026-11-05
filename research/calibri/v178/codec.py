"""Physically paid column-conditioned 5-bit source; no test/QA in fitting.

Weights represented as W[i,j] = row_scale[i] * centroid[code[i,j//8], j%8]
                              * column_scale[j].
The column scale is stored once per matrix, not per row. Native matvec
scales activations before lookup and embedding scales decoded coordinates.
"""
import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from direct_codebook_v59 import CB59, HEADER, unpack5, _nearest_integer_128
from v177.codec import pack_indices, unpack_indices
from . import protocol as p

MAGIC = b"C178"
FOOTER = 36


def calculate_columns(weights, rows):
    """Source-only RMS equalizer. Exact inputs are teacher weights and paid row scales."""
    w = np.asarray(weights, dtype=np.float32)
    r = np.asarray(rows, dtype=np.float32)
    if w.ndim != 2 or r.shape != (w.shape[0],) or w.shape[1] % 8:
        raise ValueError("Bad source matrix")
    if not np.isfinite(w).all() or np.any(r <= 0) or not np.isfinite(r).all():
        raise ValueError("Invalid original source")
    accum = np.zeros(w.shape[1], dtype=np.float64)
    for start in range(0,len(w),128):
        part = w[start:start+128].astype(np.float64) / r[start:start+128,None]
        accum += np.sum(part * part,axis=0)
    rms = np.sqrt(accum/len(w))
    med = float(np.median(rms))
    if not np.isfinite(med) or med <= 1e-12:
        raise ValueError("Degenerate column RMS")
    c = np.clip(rms / med, *p.COLUMN_CLAMP).astype("<f2")
    if not np.isfinite(c).all() or np.any(c<=0):
        raise ValueError("Unrepresentable paid column multipliers")
    return c


def sample_source(weights, row_scales, column_scales):
    w=np.asarray(weights,dtype=np.float32)
    m,n=w.shape
    g=n//8
    groups=m*g
    positions=np.linspace(0,groups-1,min(groups,p.SAMPLES),dtype=np.int64)
    rows=positions//g
    block=positions%g
    cols=block[:,None]*8+np.arange(8)[None,:]
    sample=w[rows[:,None],cols]/row_scales[rows,None]/column_scales[cols].astype(np.float32)
    return sample.astype("<f2").astype(np.float32)


def fit_dictionary(source):
    x=np.asarray(source,dtype="<f2").astype(np.float32)
    if x.ndim!=2 or x.shape[1]!=8 or len(x)<p.ENTRIES:
        raise ValueError("Missing original pretrained samples")
    centers=x[np.linspace(0,len(x)-1,p.ENTRIES,dtype=np.int64)].copy()
    for _ in range(p.KMEANS_ITERATIONS):
        ix=_nearest_integer_128(x,centers)
        count=np.bincount(ix,minlength=p.ENTRIES)
        sums=np.zeros((p.ENTRIES,8),dtype=np.float64)
        np.add.at(sums,ix,x.astype(np.float64))
        occupied=count>0
        centers[occupied]=(sums[occupied]/count[occupied,None]).astype("<f2").astype(np.float32)
    result=centers.astype("<f2")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite trained codebook")
    return result


def quantize(weights, row_scales):
    w=np.asarray(weights,dtype=np.float32)
    rows=np.asarray(row_scales,dtype=np.float32)
    if w.ndim!=2 or w.shape[1]%8 or rows.shape!=(len(w),):
        raise ValueError("Original trained source shape invalid")
    m,n=w.shape
    g=n//8
    cols=calculate_columns(w,rows)
    center=fit_dictionary(sample_source(w,rows,cols))
    centers=center.astype(np.float32)
    codes=np.empty(m*g,dtype=np.uint8)
    old_sq=new_sq=energy=0.0
    for start in range(0,m,128):
        end=min(start+128,m)
        region=w[start:end]
        normalized=(region/rows[start:end,None]/cols.astype(np.float32)[None,:]).astype("<f2").astype(np.float32)
        ids=_nearest_integer_128(normalized.reshape(-1,8),centers).astype(np.uint8)
        codes[start*g:end*g]=ids
        recovered=(centers[ids].reshape(end-start,n)*
                   rows[start:end,None]*cols.astype(np.float32)[None,:])
        new_sq+=float(np.sum(np.square((region-recovered).astype(np.float64))))
        energy+=float(np.sum(np.square(region.astype(np.float64))))
    payload={"packed":pack_indices(codes,p.BITS),
             "row_scales":rows.astype("<f2").tobytes(),
             "centroids":center.tobytes(),
             "col_scales":cols.tobytes()}
    if len(payload["packed"])!=(m*g*p.BITS+7)//8 or len(payload["col_scales"])!=2*n:
        raise AssertionError("Invalid paid source length")
    if not np.array_equal(unpack_indices(payload["packed"],m*g,p.BITS),codes):
        raise AssertionError("Physical bit-index roundtrip")
    return payload,{"column_scale_min":float(cols.min()),"column_scale_max":float(cols.max()),
                    "column_scale_median":float(np.median(cols)),"source_weight_RMSE":
                    math.sqrt(new_sq/max(energy,1e-30)),
                    "no_QA_or_logits_used_in_fit":True}


def encode(original, teacher_state):
    if not isinstance(original,CB59) or original.hash()!=p.SOURCE_SHA:
        raise ValueError("Invalid immutable CB59 reference")
    if original.meta["original_unique_parameter_count"]!=p.TOTAL_UNIQUE:
        raise ValueError("Original unique-parameter denominator changed")
    if set(teacher_state)!=set(original.meta["full_original_state_keys"]):
        raise ValueError("Original teacher source missing or extra parameters")
    _,_,old_mlen,_=HEADER.unpack_from(original.blob)
    old_body=memoryview(original.blob)[HEADER.size+old_mlen:-FOOTER]
    body=bytearray()
    desc={}
    measures={}
    for name,d in sorted(original.meta["tensor_descriptors"].items()):
        shape=tuple(d["shape"])
        if d["dtype"]=="CB59_K32V8":
            handle=original.handles[name]
            if tuple(teacher_state[name].shape)!=shape:
                raise ValueError("Teacher pretrained matrix mismatch "+name)
            weight=teacher_state[name].detach().cpu().numpy()
            fields,measure=quantize(weight,handle.row_scales)
            old_ids=unpack5(handle.packed.tobytes(),shape[0]*shape[1]//8)
            previous=(handle.centroids[old_ids].reshape(shape)*handle.row_scales[:,None])
            old_rmse=float(np.linalg.norm((weight-previous).astype(np.float64)) /
                           max(np.linalg.norm(weight.astype(np.float64)),1e-15))
            measure["original_CB59_weight_RMSE"]=old_rmse
            measures[name]=measure
            desc[name]={"shape":list(shape),"dtype":"CB178_COLSCALE_K32V8",
                        "bits":5,"parts":{}}
        elif d["dtype"]=="f4":
            offset,size=d["parts"]["raw"]
            fields={"raw":bytes(old_body[offset:offset+size])}
            desc[name]={"shape":list(shape),"dtype":"f4","parts":{}}
        else:
            raise ValueError("Unexpected original trained parameter type")
        for part in ("packed","row_scales","centroids","col_scales") if "packed" in fields else ("raw",):
            block=fields[part]
            desc[name]["parts"][part]=[len(body),len(block)]
            body.extend(block)
    metadata=dict(original.meta)
    metadata.update({"format":"CALIBRI v178 shared 5-bit per matrix plus paid FP16 column multipliers",
                     "version":178,"tensor_descriptors":desc,
                     "quality_not_guaranteed":True,
                     "physical_column_scaling":"sqrt mean original normalized column energy clipped .25..4"})
    manifest=json.dumps(metadata,sort_keys=True,separators=(",",":")).encode("utf8")
    if len(manifest)>5_000_000 or len(body)>300_000_000:
        raise AssertionError("Manifest/source allocation bounds")
    prefix=HEADER.pack(MAGIC,1,len(manifest),len(body))+manifest+bytes(body)
    marked=prefix+struct.pack("<I",zlib.crc32(prefix))
    blob=marked+hashlib.sha256(marked).digest()
    if len(blob)>p.MAX_BYTES:
        raise AssertionError("Complete source exceeds physical <=0.7 bpp budget")
    reader=CB178(blob)
    if reader.hash()!=hashlib.sha256(blob).hexdigest():
        raise AssertionError("Saved data seal mismatch")
    return blob,{"bytes":len(blob),"bpp":8*len(blob)/p.TOTAL_UNIQUE,
                 "source_sha256":original.hash(),"candidate_sha256":reader.hash(),
                 "matrix_count":len(measures),"matrix_diagnostics":measures,
                 "actual_complete_source_not_sidecar":True}


@dataclass
class ColumnHandle:
    name: str
    shape: tuple
    packed: np.ndarray
    row_scales: np.ndarray
    centroids: np.ndarray
    col_scales: np.ndarray


class CB178:
    def __init__(self,blob):
        if not isinstance(blob,bytes) or len(blob)<HEADER.size+FOOTER or len(blob)>p.MAX_BYTES:
            raise ValueError("Invalid complete physical source")
        magic,v,mlen,blen=HEADER.unpack_from(blob)
        if magic!=MAGIC or v!=1 or mlen>5_000_000 or blen>300_000_000:
            raise ValueError("Invalid column-source format")
        if HEADER.size+mlen+blen+FOOTER!=len(blob):
            raise ValueError("Source physical length mismatch")
        if zlib.crc32(blob[:-36])!=struct.unpack_from("<I",blob,len(blob)-36)[0] or hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
            raise ValueError("Source integrity failure")
        self.blob=blob
        self.meta=json.loads(blob[HEADER.size:HEADER.size+mlen])
        if (self.meta["original_unique_parameter_count"]!=p.TOTAL_UNIQUE or
            self.meta["bits_per_code"]!=p.BITS or
            self.meta["codebook_entries"]!=p.ENTRIES or
            self.meta["vector_length"]!=p.GROUP):
            raise ValueError("Invalid archive tensor coding ABI")
        desc=self.meta["tensor_descriptors"]
        if sorted(desc)!=self.meta["trained_unique_keys"]:
            raise ValueError("Missing unique source tensor")
        body=memoryview(blob)[HEADER.size+mlen:-FOOTER]
        self.handles,self.raw={},{}
        end=0
        for name,d in sorted(desc.items()):
            shape=tuple(d["shape"])
            fields=("packed","row_scales","centroids","col_scales") if d["dtype"]=="CB178_COLSCALE_K32V8" else ("raw",)
            if set(fields)!=set(d["parts"]):
                raise ValueError("Incomplete/unexpected paid source fields")
            buffers={}
            for part in fields:
                offset,size=d["parts"][part]
                if offset!=end or size<=0 or end+size>len(body):
                    raise ValueError("Overlapping or invalid source byte offsets")
                buffers[part]=body[offset:offset+size]
                end+=size
            if d["dtype"]=="CB178_COLSCALE_K32V8":
                if len(shape)!=2 or shape[1]%8:
                    raise ValueError("Invalid 2D geometry")
                m,n=shape
                if (d["bits"]!=p.BITS or
                    len(buffers["packed"])!=(m*n//8*p.BITS+7)//8 or
                    len(buffers["row_scales"])!=2*m or
                    len(buffers["col_scales"])!=2*n or
                    len(buffers["centroids"])!=p.ENTRIES*8*2):
                    raise ValueError("Unpaid / missing preconditioned operator bytes")
                rows=np.frombuffer(buffers["row_scales"],dtype="<f2").astype(np.float32)
                cols=np.frombuffer(buffers["col_scales"],dtype="<f2").astype(np.float32)
                cent=np.frombuffer(buffers["centroids"],dtype="<f2").astype(np.float32).reshape(p.ENTRIES,8)
                if not all(np.isfinite(a).all() for a in (rows,cols,cent)) or np.any(rows<=0) or np.any(cols<=0):
                    raise ValueError("Invalid physical learned scales")
                self.handles[name]=ColumnHandle(name,shape,
                    np.frombuffer(buffers["packed"],dtype=np.uint8), rows,cent,cols)
            elif d["dtype"]=="f4":
                array=np.frombuffer(buffers["raw"],dtype="<f4")
                if len(array)!=int(np.prod(shape,dtype=np.int64)):
                    raise ValueError("Incomplete FP32 vector")
                self.raw[name]=array.reshape(shape)
            else:
                raise ValueError("Unrecognized original state type")
        if end!=len(body):
            raise ValueError("Extra uncharged source payload bytes")
        for alias,source in self.meta["source_alias_keys"].items():
            if alias in self.handles or alias in self.raw:
                raise ValueError("Duplicate tied source")
            if source in self.handles:
                self.handles[alias]=self.handles[source]
            elif source in self.raw:
                self.raw[alias]=self.raw[source]
            else:
                raise ValueError("Dangling source alias")
        if set(self.handles)|set(self.raw)!=set(self.meta["full_original_state_keys"]):
            raise ValueError("Missing trained tensor in compact archive")

    def hash(self):
        return hashlib.sha256(self.blob).hexdigest()
