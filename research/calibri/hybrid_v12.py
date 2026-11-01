"""CALIBRI-CH12: lossless per-matrix adaptive CL8 vs mask-Huffman format.

Two distinct dictionary concepts have been compared in this series:
QMoE 65536-word sequences (stage1), and the support-mask 94-symbol
Huffman alphabet. Neither is a newly invented general compression method.

The compact wrapper chooses the shorter PHYSICAL byte stream, including
the extra 5-byte hybrid header. FP16 row scales are external in BOTH
formats and counted separately in benchmark totals.
"""
from __future__ import annotations
import numpy as np
from index import encode as cl8_encode,decode as cl8_decode
from huffman_masks_v12 import encode as mask_encode,decode as mask_decode

MAGIC=b'CH12'
CL8=0;MASK=1

def encode(q,group=64):
    q=np.asarray(q,dtype=np.int8)
    clb=cl8_encode(q)
    mask=mask_encode(q,group=group)
    if len(mask)<len(clb):
        return MAGIC+bytes((MASK,))+mask,{"chosen":"MASK_HUFF","cl8_bytes":len(clb),"mask_bytes":len(mask)}
    return MAGIC+bytes((CL8,))+clb,{"chosen":"CL8","cl8_bytes":len(clb),"mask_bytes":len(mask)}

def decode(buf):
    if len(buf)<6 or buf[:4]!=MAGIC:raise ValueError('bad hybrid header')
    if buf[4]==CL8:return cl8_decode(buf[5:])
    if buf[4]==MASK:return mask_decode(buf[5:])
    raise ValueError('unknown mode')
