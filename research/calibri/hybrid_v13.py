"""CALIBRI v0.13: exact physical size-adaptive matrix codec.

Select the truly shortest byte stream among:
  CL8-I combinadic rows (baseline);
  C12I 8-cell image-style Huffman masks;
  C13R count-Huffman + enumerative support (16,32,64).
All physical index/mode metadata is counted, and common FP16 level bytes
are external (identical across candidates). This selection **cannot**
repair any quality lost during the preceding ternarization.

The choice tag adds exactly 5 bytes per matrix. No runtime kernel exists.
"""
import numpy as np
from index import encode as cl8_encode,decode as cl8_decode
from huffman_masks_v12 import encode as mask_encode,decode as mask_decode
from class_rank_v13 import encode as rank_encode,decode as rank_decode

MAGIC=b'CLH3'
SELECTORS=('CL8','MASK_HUFF','CLASS_RANK16','CLASS_RANK32','CLASS_RANK64')
DECODE=(cl8_decode,mask_decode,rank_decode,rank_decode,rank_decode)

def encode(q,group=64):
    q=np.asarray(q,dtype=np.int8)
    options=[cl8_encode(q),
             mask_encode(q,group=group),
             rank_encode(q,block=16,group=group),
             rank_encode(q,block=32,group=group),
             rank_encode(q,block=64,group=group)]
    which=min(range(len(options)),key=lambda i:(len(options[i]),i))
    return (MAGIC+bytes([which])+options[which],
            {'selection':SELECTORS[which],
             'bytes_by_codec':{SELECTORS[i]:len(z) for i,z in enumerate(options)},
             'physical_bytes_with_wrapper':len(options[which])+5})

def decode(blob):
    if len(blob)<6 or blob[:4]!=MAGIC:raise ValueError("bad matrix wrapper")
    i=blob[4]
    if i>=len(DECODE):raise ValueError("unknown codec tag")
    return DECODE[i](blob[5:])
