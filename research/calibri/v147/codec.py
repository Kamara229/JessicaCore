"""CB147 = immutable physical CB140 + 49152 signed 8-bit bias values.

Full 135M pretrained packed source, paid 1bit tied input/output residual,
precommitted half head gain, tiny learned vocab logit correction. No dense
pretrained source and no free sidecars in the size accounting.
"""
from dataclasses import dataclass
import hashlib
import math
import struct
import zlib
import numpy as np
from v140.codec import unpack as parse_cb140
from v141.run import ORIGINAL_V140_SHA
from v117.protocol import TOTAL_UNIQUE,MAX_SOURCE_BYTES

MAGIC=b"CLB147\0\0"
HEADER=struct.Struct("<8sHHfI")
COUNT=49152
GAIN_INPUT=0.0
GAIN_HEAD=.5
DIGEST=36


def pack(paid_cb140,bias_codes,scale):
    source=parse_cb140(paid_cb140)
    if source.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not the pinned original paid 0.694-bit source")
    q=np.asarray(bias_codes,dtype=np.int8)
    if q.shape!=(COUNT,):
        raise ValueError("all output vocabulary bias codes must be physical")
    scale=float(scale)
    if not math.isfinite(scale) or not scale>0:
        raise ValueError("bias step must be finite and positive")
    buffer=(HEADER.pack(MAGIC,1,0,GAIN_HEAD,COUNT)
            +bytes(paid_cb140)+struct.pack("<f",scale)+q.tobytes())
    prefix=buffer+struct.pack("<I",zlib.crc32(buffer))
    blob=prefix+hashlib.sha256(prefix).digest()
    if len(blob)>MAX_SOURCE_BYTES:
        raise ValueError("full paid output bias exceeds 0.7-bit cap")
    return blob


@dataclass(frozen=True)
class Physical:
    paid: object
    code: np.ndarray
    scale: float
    size: int
    sha256: str

    @property
    def bias(self):
        return self.code.astype(np.float32)*self.scale


def unpack(source):
    raw=bytes(source)
    if len(raw)<HEADER.size+COUNT+4+DIGEST:
        raise ValueError("incomplete CB147")
    magic,version,input_gain,output_gain,count=HEADER.unpack_from(raw)
    if (magic,version,input_gain,output_gain,count)!=(MAGIC,1,0,GAIN_HEAD,COUNT):
        raise ValueError("unrecognized gain or archive geometry")
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError("corrupt CB147 SHA")
    if zlib.crc32(raw[:-36])!=struct.unpack_from("<I",raw,len(raw)-36)[0]:
        raise ValueError("corrupt CB147 CRC")
    offset=HEADER.size
    paid_len=len(raw)-HEADER.size-COUNT-4-DIGEST
    paid=parse_cb140(raw[offset:offset+paid_len])
    if paid.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("CB140 anchor differs")
    offset+=paid_len
    scale=struct.unpack_from("<f",raw,offset)[0]
    q=np.frombuffer(raw[offset+4:offset+4+COUNT],dtype=np.int8).copy()
    if not math.isfinite(scale) or scale<=0 or len(q)!=COUNT:
        raise ValueError("bad output-scale representation")
    if len(raw)>MAX_SOURCE_BYTES:
        raise ValueError("above physical 0.7 bits per parameter")
    return Physical(paid,q,scale,len(raw),hashlib.sha256(raw).hexdigest())


def measured_rate(source):
    item=unpack(source)
    return {"complete_CB147_bytes":item.size,
            "complete_bpp_original_unique":8*item.size/TOTAL_UNIQUE,
            "headroom_below_0_7bpp_bytes":MAX_SOURCE_BYTES-item.size,
            "one_full_tied_CB140_physical_source_included":True,
            "int8_vocabulary_logit_prior_paid":COUNT,
            "calibration_head_gain_frozen":GAIN_HEAD}
