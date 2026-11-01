"""CALIBRI-HYBRID: automatically prefer CL8-I or the two-stage dictionary.

Stored format: b"CLHX" + one mode byte + self-describing CL8-I/CL7R or CL11.
Row quantization scale metadata is EXTERNAL in both branches and must be
accounted separately; this format does not convert FP weights into ternary.
The QMoE dictionary LUT is shared and needed only if a CL11 block is picked.
"""
import numpy as np
from index import encode as cl8_encode,decode as cl8_decode
from dual_codec import StaticPairDictionary,encode_grouped,decode_grouped

MAGIC=b"CLHX"
MODE_CL8=0
MODE_DUAL=1

def encode_hybrid(q,dictionary=None,group=64):
    q=np.asarray(q,dtype=np.int8)
    base=cl8_encode(q)
    if dictionary is None:
        # No proven dictionary means we can still write fully verifiable CL8.
        return MAGIC+bytes([MODE_CL8])+base, {"selected":"CL8","base_bytes":len(base),"dual_bytes":None}
    words,offs=dictionary.encode(q)
    dual,_=encode_grouped(words,offs,q.shape[1],group,"adaptive")
    if len(dual)<len(base):
        return MAGIC+bytes([MODE_DUAL])+dual, {"selected":"DUAL","base_bytes":len(base),"dual_bytes":len(dual)}
    return MAGIC+bytes([MODE_CL8])+base, {"selected":"CL8","base_bytes":len(base),"dual_bytes":len(dual)}

def decode_hybrid(blob,dictionary=None):
    if len(blob)<6 or blob[:4]!=MAGIC:raise ValueError("bad hybrid stream")
    tag=blob[4];data=blob[5:]
    if tag==MODE_CL8:return cl8_decode(data)
    if tag!=MODE_DUAL or dictionary is None:
        raise ValueError("unknown mode or missing required static dictionary")
    words,offsets,width=decode_grouped(data)
    return dictionary.decode(words,offsets,width)
