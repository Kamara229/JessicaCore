"""Costed V2/V1 higher-rate controls establish where original model recovers.

NOT a <=0.7 bpp design. A deliberate diagnostic upper bound after CB59
failed at V8: same 5 bits per group, but 2 or 1 weight per group.
"""
import hashlib,json,struct,zlib
from dataclasses import dataclass
import numpy as np
from direct_codebook_v59 import CB59, HEADER,pack5,unpack5,_nearest_integer_128
from .protocol import K,BITS,GROUPS,SOURCE_SHA,TOTAL_UNIQUE

MAGIC=b"C181"
FOOTER=36

def assign(vectors,centers):
    a=np.asarray(vectors,dtype=np.float32)
    c=np.asarray(centers,dtype=np.float32)
    if a.shape[1]==1:
        order=np.argsort(c[:,0])
        ordered=c[order,0]
        mid=(ordered[1:]+ordered[:-1])/2
        result=np.searchsorted(mid,a[:,0],side="left")
        return order[result].astype(np.uint8)
    return _nearest_integer_128(a,c).astype(np.uint8)

def quantize(w,scale,width):
    w=np.asarray(w,dtype=np.float32)
    scale=np.asarray(scale,dtype=np.float32)
    m,n=w.shape
    if width not in (1,2) or n%width or scale.shape!=(m,) or np.any(scale<=0):
        raise ValueError("Unrepresentable source group")
    groups=n//width
    sample_indices=np.linspace(0,m*groups-1,min(4096,m*groups),dtype=np.int64)
    rows=sample_indices//groups
    starts=sample_indices%groups
    columns=starts[:,None]*width+np.arange(width)[None,:]
    samples=(w[rows[:,None],columns]/scale[rows,None]).astype("<f2").astype(np.float32)
    centers=samples[np.linspace(0,len(samples)-1,K,dtype=np.int64)].copy()
    for _ in range(5):
        ix=assign(samples,centers)
        cnt=np.bincount(ix,minlength=K)
        accum=np.zeros((K,width),dtype=np.float64)
        np.add.at(accum,ix,samples.astype(np.float64))
        good=cnt>0
        centers[good]=(accum[good]/cnt[good,None]).astype("<f2").astype(np.float32)
    centers=centers.astype("<f2")
    codes=np.empty(m*groups,dtype=np.uint8)
    total_error=0.; total_energy=0.
    for first in range(0,m,128):
        stop=min(m,first+128)
        original=w[first:stop]
        norm=(original/scale[first:stop,None]).reshape(-1,width)
        ids=assign(norm,centers.astype(np.float32))
        codes[first*groups:stop*groups]=ids
        decoded=(centers.astype(np.float32)[ids].reshape(stop-first,n)*
                 scale[first:stop,None])
        total_error+=float(np.square((original-decoded).astype(np.float64)).sum())
        total_energy+=float(np.square(original.astype(np.float64)).sum())
    data={"packed":pack5(codes),"row_scales":scale.astype("<f2").tobytes(),
          "centroids":centers.tobytes()}
    if not np.array_equal(unpack5(data["packed"],len(codes)),codes):
        raise AssertionError("Real 5-bit source index roundtrip")
    return data,{"relative_weight_RMSE":float(np.sqrt(total_error/max(total_energy,1e-30))),
                 "codes":len(codes),"group":width}

def encode(original,teacher_state,arm):
    if arm not in GROUPS or not isinstance(original,CB59) or original.hash()!=SOURCE_SHA:
        raise ValueError("Unpinned pretrained source or wrong arm")
    if set(teacher_state)!=set(original.meta["full_original_state_keys"]):
        raise ValueError("Incomplete teacher source")
    group=GROUPS[arm]
    _,_,mlen,_=HEADER.unpack_from(original.blob)
    old=memoryview(original.blob)[HEADER.size+mlen:-FOOTER]
    body=bytearray();desc={};diag={}
    for name,d in sorted(original.meta["tensor_descriptors"].items()):
        shape=tuple(d["shape"])
        if d["dtype"]=="CB59_K32V8":
            oldscale=original.handles[name].row_scales
            weight=teacher_state[name].detach().cpu().numpy()
            if weight.shape!=shape:raise ValueError("Teacher shape mismatch")
            parts,record=quantize(weight,oldscale,group)
            diag[name]=record
            dtype=f"CB181_K32V{group}"
        elif d["dtype"]=="f4":
            off,size=d["parts"]["raw"]
            parts={"raw":bytes(old[off:off+size])}
            dtype="f4"
        else:raise ValueError("Unsupported original dtype")
        desc[name]={"shape":list(shape),"dtype":dtype,"parts":{}}
        for k,payload in parts.items():
            desc[name]["parts"][k]=[len(body),len(payload)]
            body.extend(payload)
    meta=dict(original.meta)
    meta.update({"format":"CB181 deliberately high rate quality recovery control",
                 "tensor_descriptors":desc,"vector_length":group,
                 "calibri_recovery_oracle":True,"bits_per_code":5,
                 "quality_not_guaranteed":True})
    encoded=json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
    prefix=HEADER.pack(MAGIC,1,len(encoded),len(body))+encoded+bytes(body)
    signed=prefix+struct.pack("<I",zlib.crc32(prefix))
    blob=signed+hashlib.sha256(signed).digest()
    saved=CB181(blob)
    if saved.meta["vector_length"]!=group:
        raise AssertionError("Oracle source changed group size")
    return blob,{"sha256":saved.hash(),"bytes":len(blob),
          "bpp":8*len(blob)/TOTAL_UNIQUE,"arm":arm,"group":group,
          "original_sha256":SOURCE_SHA,"per_matrix":diag,
          "NOT_AN_UNDER_0p7_MODEL":True}

@dataclass
class Handle:
    name:str
    shape:tuple
    packed:np.ndarray
    row_scales:np.ndarray
    centroids:np.ndarray
    group:int

class CB181:
    def __init__(self,blob):
        if not isinstance(blob,bytes) or len(blob)<HEADER.size+FOOTER:
            raise ValueError("Invalid oracle archive")
        magic,ver,mlen,blen=HEADER.unpack_from(blob)
        if magic!=MAGIC or ver!=1 or mlen>5000000 or blen>200000000:
            raise ValueError("Invalid high-rate diagnostic")
        if HEADER.size+mlen+blen+FOOTER!=len(blob):
            raise ValueError("Truncated physical oracle")
        if zlib.crc32(blob[:-36])!=struct.unpack_from("<I",blob,len(blob)-36)[0] or hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
            raise ValueError("Physical source SHA/CRC fail")
        self.blob=blob
        self.meta=json.loads(blob[HEADER.size:HEADER.size+mlen])
        g=self.meta["vector_length"]
        if g not in (1,2) or self.meta["original_unique_parameter_count"]!=TOTAL_UNIQUE:
            raise ValueError("Unregistered source geometry")
        desc=self.meta["tensor_descriptors"]
        if sorted(desc)!=self.meta["trained_unique_keys"]:
            raise ValueError("Missing original matrix")
        body=memoryview(blob)[HEADER.size+mlen:-FOOTER]
        self.handles={};self.raw={}
        end=0
        for name,d in sorted(desc.items()):
            shape=tuple(d["shape"])
            fields=("packed","row_scales","centroids") if d["dtype"]==f"CB181_K32V{g}" else ("raw",)
            if set(fields)!=set(d["parts"]):raise ValueError("Missing field")
            buffers={}
            for k in fields:
                off,size=d["parts"][k]
                if off!=end or size<=0 or off+size>len(body):
                    raise ValueError("Invalid physical allocation")
                buffers[k]=body[off:off+size]
                end+=size
            if "packed" in fields:
                m,n=shape
                if n%g or len(buffers["packed"])!=(m*n//g*5+7)//8 or len(buffers["row_scales"])!=m*2 or len(buffers["centroids"])!=32*g*2:
                    raise ValueError("Incorrect physical 1/2D codebook layout")
                scales=np.frombuffer(buffers["row_scales"],dtype="<f2").astype(np.float32)
                centers=np.frombuffer(buffers["centroids"],dtype="<f2").astype(np.float32).reshape(32,g)
                if not np.isfinite(scales).all() or np.any(scales<=0) or not np.isfinite(centers).all():
                    raise ValueError("Invalid compressed values")
                self.handles[name]=Handle(name,shape,np.frombuffer(buffers["packed"],dtype=np.uint8),scales,centers,g)
            elif d["dtype"]=="f4":
                arr=np.frombuffer(buffers["raw"],dtype="<f4")
                if arr.size!=np.prod(shape):raise ValueError("Raw vector incomplete")
                self.raw[name]=arr.reshape(shape)
            else:raise ValueError("Unexpected dtype")
        if end!=len(body):raise ValueError("Trailing uncounted source")
        for alias,canon in self.meta["source_alias_keys"].items():
            if canon in self.handles:self.handles[alias]=self.handles[canon]
            elif canon in self.raw:self.raw[alias]=self.raw[canon]
            else:raise ValueError("Tied source alias missing")
        if (set(self.handles)|set(self.raw))!=set(self.meta["full_original_state_keys"]):
            raise ValueError("Incomplete pretrained source")
    def hash(self):return hashlib.sha256(self.blob).hexdigest()
