"""CALIBRI-HYBRID: automatically prefer CL8-I or the two-stage dictionary.

Stored format: b"CLHX" + one mode byte + self-describing CL8-I/CL7R or CL11.
Row quantization scale metadata is EXTERNAL in both branches and must be
accounted separately; this format does not convert FP weights into ternary.
The QMoE dictionary LUT is shared and needed only if a CL11 block is picked.
"""
import numpy as np
from index import encode as cl8_encode,decode as cl8_decode
from dual_codec import LUT_BYTES,StaticPairDictionary,encode_grouped,decode_grouped
from shared_cost import select_formats,nonnegative_integer

MAGIC=b"CLHX"
MODE_CL8=0
MODE_DUAL=1

def _ternary(q):
    q=np.asarray(q)
    if q.ndim!=2 or min(q.shape)<1 or not np.all(np.isin(q,[-1,0,1])):
        raise ValueError('nonempty ternary matrix required')
    return q.astype(np.int8)

def encode_hybrid(q,dictionary=None,group=64):
    """Standalone choice charges the entire shared LUT; bundle via API below."""
    q=_ternary(q)
    base=cl8_encode(q)
    if dictionary is None:
        # No proven dictionary means we can still write fully verifiable CL8.
        return MAGIC+bytes([MODE_CL8])+base, {"selected":"CL8","base_bytes":len(base),"dual_bytes":None}
    words,offs=dictionary.encode(q)
    dual,_=encode_grouped(words,offs,q.shape[1],group,"adaptive")
    if len(dual)+LUT_BYTES<len(base):
        return MAGIC+bytes([MODE_DUAL])+dual, {"selected":"DUAL","base_bytes":len(base),"dual_bytes":len(dual)}
    return MAGIC+bytes([MODE_CL8])+base, {"selected":"CL8","base_bytes":len(base),"dual_bytes":len(dual)}

def encode_hybrid_bundle(matrices,dictionary=None,group=64,external_bytes=0):
    """Choose across matrices, charging LUT once. Returns streams + byte audit.

    External bytes MUST include row scales and any enclosing model metadata.
    This does not serialize those external data or supply a GPU/ARM decoder.
    """
    external_bytes=nonnegative_integer(external_bytes)
    bases=[];duals=[]
    for matrix in matrices:
        q=_ternary(matrix)
        bases.append(cl8_encode(q))
        if dictionary is not None:
            words,offs=dictionary.encode(q)
            dual,_=encode_grouped(words,offs,q.shape[1],group,'adaptive')
            duals.append(dual)
        else:
            duals.append(bases[-1])
    audit=select_formats([len(b)+5 for b in bases],
                         [len(d)+5 for d in duals],LUT_BYTES)
    blobs=[MAGIC+bytes([MODE_DUAL if m=='DUAL' else MODE_CL8])
           +(d if m=='DUAL' else b)
           for b,d,m in zip(bases,duals,audit['modes'])]
    audit['external_bytes']=external_bytes
    audit['total_bytes']+=external_bytes
    audit['all_base_bytes']+=external_bytes
    return blobs,audit

def decode_hybrid(blob,dictionary=None):
    if len(blob)<6 or blob[:4]!=MAGIC:raise ValueError("bad hybrid stream")
    tag=blob[4];data=blob[5:]
    if tag==MODE_CL8:return cl8_decode(data)
    if tag!=MODE_DUAL or dictionary is None:
        raise ValueError("unknown mode or missing required static dictionary")
    words,offsets,width=decode_grouped(data)
    return dictionary.decode(words,offsets,width)

