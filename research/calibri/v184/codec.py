"""Physically charged, authenticated complete CB59 + joint rank2 sidecar.

The source CB59 bytes and their CRC/SHA are preserved byte-identical.
Sidecar contains all 211 A matrices as INT8, 211 individual FP32
scales and just TWO shared FP16 B matrices. No dense model tensor,
unpaid adapter, or duplicated tied source is permitted.
"""
import hashlib,json,math,struct,zlib
import numpy as np
from direct_codebook_v59 import CB59
from . import protocol as p

MAGIC=b"CAL184R2"
HEAD=struct.Struct("<8sII")
FOOTER=36

def _matrix_alpha(alpha):
    a=np.asarray(alpha,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=p.RANK or not np.isfinite(a).all():
        raise ValueError("Invalid trained shared A")
    scale=max(float(np.max(np.abs(a)))/p.INT8_MAX,1.e-8/p.INT8_MAX)
    data=np.rint(a/scale).clip(-p.INT8_MAX,p.INT8_MAX).astype(np.int8)
    return data.tobytes(),struct.pack("<f",scale)

def encode(base,alphas,betas):
    if not isinstance(base,CB59) or base.hash()!=p.SOURCE_SHA:
        raise ValueError("Changed frozen original CB59 source")
    keys=sorted(name for name,d in base.meta["tensor_descriptors"].items()
                if d["dtype"]=="CB59_K32V8")
    if len(keys)!=211:
        raise ValueError("Expected 211 compressed 2D trained matrices")
    if set(alphas)!=set(keys) or set(betas)!={576,1536}:
        raise ValueError("Incomplete jointly trainable source")
    body=bytearray();desc={}
    for name in keys:
        shape=base.meta["tensor_descriptors"][name]["shape"]
        if len(shape)!=2:raise ValueError("Only 2D compressed weight sidecars")
        blob,scale=_matrix_alpha(alphas[name])
        if len(blob)!=shape[0]*p.RANK:
            raise ValueError("A rank or rows inconsistent with original matrix")
        desc[name]={"rows":shape[0],"cols":shape[1],"parts":{}}
        for part,payload in (("A_int8",blob),("A_scale_f32",scale)):
            desc[name]["parts"][part]=[len(body),len(payload)]
            body.extend(payload)
    shared={}
    for width in sorted(betas):
        beta=np.asarray(betas[width],dtype=np.float32)
        if beta.shape!=(p.RANK,width) or not np.isfinite(beta).all():
            raise ValueError("Wrong shared rank-two source width")
        quant=beta.astype("<f2")
        if not np.isfinite(quant.astype(np.float32)).all():
            raise ValueError("Shared rank factors overflow FP16")
        payload=quant.tobytes()
        shared[str(width)]={"shape":[p.RANK,width],
                            "parts":{"B_fp16":[len(body),len(payload)]}}
        body.extend(payload)
    meta={"format":"CALIBRI v184 paid shared lowrank matrix-source adapter",
          "original_cb59_sha256":p.SOURCE_SHA,
          "original_cb59_bytes":p.SOURCE_BYTES,
          "original_unique_parameter_count":p.TOTAL_UNIQUE,
          "rank":p.RANK,"shared_widths":[576,1536],
          "alpha_int8_per_row_per_matrix_scale_f32":True,
          "beta_fp16_cross_layer_shared":True,
          "original_211_matrix_descriptors":desc,
          "shared_basis":shared,
          "source_aliases_preserved_by_original_CB59":True,
          "tied_embedding_output_residual_shared":True,
          "training_quantization_aware":True,
          "all_payload_physically_counted":True}
    manifest=json.dumps(meta,sort_keys=True,separators=(",",":")).encode("utf8")
    tail=HEAD.pack(MAGIC,len(manifest),len(body))+manifest+body
    unsigned=base.blob+tail
    crc=struct.pack("<I",zlib.crc32(unsigned))
    signed=unsigned+crc
    data=signed+hashlib.sha256(signed).digest()
    if len(data)>p.MAX_BYTES:
        raise ValueError("Joint rank-two source exceeds complete physical <=0.7 budget")
    restored=Paid184(data)
    return data,{"complete_physical_bytes":len(data),
                 "complete_bpp":8*len(data)/p.TOTAL_UNIQUE,
                 "sha256":hashlib.sha256(data).hexdigest(),
                 "base_bytes":len(base.blob),"source_SHA256":p.SOURCE_SHA,
                 "all_211_A_int8_bytes":sum(len(x[0]) for x in
                    (_matrix_alpha(alphas[name]) for name in keys)),
                 "shared_B_FP16_bytes":sum(p.RANK*n*2 for n in betas),
                 "matrix_count":len(restored.alpha),
                 "physically_paid":True}

class Paid184:
    def __init__(self,data):
        if not isinstance(data,bytes) or len(data)<p.SOURCE_BYTES+HEAD.size+FOOTER or len(data)>p.MAX_BYTES:
            raise ValueError("Incomplete or overbudget physical source")
        base=data[:p.SOURCE_BYTES]
        self.base=CB59(base)
        if self.base.hash()!=p.SOURCE_SHA:
            raise ValueError("Original source tampered")
        tail=data[p.SOURCE_BYTES:]
        magic,mlen,blen=HEAD.unpack_from(tail)
        if magic!=MAGIC or mlen>300000 or blen>p.HEADROOM or mlen<=0:
            raise ValueError("Unrecognized or oversized native paid residual format")
        if len(tail)!=HEAD.size+mlen+blen+FOOTER:
            raise ValueError("Trailing uncounted or truncated adapter bytes")
        if zlib.crc32(data[:-FOOTER])!=struct.unpack_from("<I",data,len(data)-FOOTER)[0]:
            raise ValueError("Full physical source CRC mismatch")
        if hashlib.sha256(data[:-32]).digest()!=data[-32:]:
            raise ValueError("Full physical source SHA mismatch")
        self.meta=json.loads(tail[HEAD.size:HEAD.size+mlen])
        meta=self.meta
        if (meta["original_cb59_sha256"]!=p.SOURCE_SHA or
            meta["original_unique_parameter_count"]!=p.TOTAL_UNIQUE or
            meta["rank"]!=p.RANK or meta["shared_widths"]!=[576,1536]):
            raise ValueError("Unsealed initial source geometry")
        trained=set(name for name,d in self.base.meta["tensor_descriptors"].items()
                    if d["dtype"]=="CB59_K32V8")
        if len(trained)!=211 or set(meta["original_211_matrix_descriptors"])!=trained:
            raise ValueError("Missing individual A matrix")
        area=memoryview(tail)[HEAD.size+mlen:-FOOTER]
        self.alpha={};self.betas={}
        offset=0
        for name,desc in sorted(meta["original_211_matrix_descriptors"].items()):
            shape=tuple(self.base.meta["tensor_descriptors"][name]["shape"])
            if (desc["rows"],desc["cols"])!=shape:
                raise ValueError("Physical rank2 field shape mismatch")
            fields=desc["parts"]
            if set(fields)!={"A_int8","A_scale_f32"}:
                raise ValueError("Omitted matrix scale or data")
            off,size=fields["A_int8"]
            if off!=offset or size!=shape[0]*p.RANK:
                raise ValueError("Countable alpha bytes broken")
            a=np.frombuffer(area[off:off+size],dtype=np.int8).reshape(shape[0],p.RANK)
            offset+=size
            off,size=fields["A_scale_f32"]
            if off!=offset or size!=4:
                raise ValueError("Unpaid matrix scale")
            scale=struct.unpack_from("<f",area,off)[0]
            if not math.isfinite(scale) or scale<=0:raise ValueError("Bad source factor scale")
            offset+=4
            self.alpha[name]=(a,float(scale))
        for width in (576,1536):
            desc=meta["shared_basis"][str(width)]
            if desc["shape"]!=[p.RANK,width]:
                raise ValueError("Missing shared B input width")
            off,size=desc["parts"]["B_fp16"]
            if off!=offset or size!=p.RANK*width*2:
                raise ValueError("Uncounted shared basis")
            self.betas[width]=np.frombuffer(area[off:off+size],dtype="<f2").astype(np.float32).reshape(p.RANK,width)
            if not np.isfinite(self.betas[width]).all():
                raise ValueError("Corrupted shared B")
            offset+=size
        if offset!=len(area):raise ValueError("Unpaid physical fields")
        if self.base.handles["model.embed_tokens.weight"] is not self.base.handles["lm_head.weight"]:
            raise ValueError("Tied pretrained source not shared")
        self.data=data
    def hash(self):return hashlib.sha256(self.data).hexdigest()
    def adapter_for(self,handle):
        canonical=self.base.meta["source_alias_keys"].get(handle.name,handle.name)
        alpha,scale=self.alpha[canonical]
        beta=self.betas[handle.shape[1]]
        return alpha,scale,beta
