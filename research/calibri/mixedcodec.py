"""Concrete little-endian CL4Q/CL8Q row-scaled symmetric weight formats.

Physical format:
  header 12 bytes <4sII : magic, rows, cols
  scales: rows * 2 bytes, little-endian FP16
  payload: int8 per coefficient (CL8Q), packed nibble signed offset +8 (CL4Q).
Output dequantized numpy.float32 for CPU validation. 
"""
import struct
import numpy as np

HDR='<4sII'
SIZE=struct.calcsize(HDR)
def pack_symmetric(q,scales,bits):
    q=np.asarray(q,dtype=np.int8)
    scales=np.asarray(scales,dtype='<f2').reshape(-1)
    if q.ndim!=2 or q.shape[0]!=len(scales) or bits not in (4,8):
        raise ValueError('bad dimensions')
    maxq=(1<<(bits-1))-1
    if np.any(q<-maxq) or np.any(q>maxq):
        raise ValueError('out of bound')
    magic=b'CL4Q' if bits==4 else b'CL8Q'
    if bits==8:
        payload=q.tobytes(order='C')
    else:
        a=(q.ravel().astype(np.int16)+8).astype(np.uint8)
        if a.size%2:a=np.pad(a,(0,1),constant_values=8)
        payload=(a[0::2] | (a[1::2]<<4)).astype(np.uint8).tobytes()
    return struct.pack(HDR,magic,*q.shape)+scales.tobytes()+payload

def unpack_symmetric(blob):
    if len(blob)<SIZE:raise ValueError('truncated')
    magic,rows,cols=struct.unpack_from(HDR,blob)
    if magic not in (b'CL4Q',b'CL8Q') or rows<1 or cols<1:
        raise ValueError('invalid header')
    bits=4 if magic==b'CL4Q' else 8
    total=rows*cols;plen=(total+1)//2 if bits==4 else total
    if len(blob)!=SIZE+2*rows+plen:raise ValueError('invalid size')
    scales=np.frombuffer(blob,dtype='<f2',count=rows,offset=SIZE).astype(np.float32)
    payload=np.frombuffer(blob,dtype=np.uint8,offset=SIZE+2*rows)
    if bits==8: q=payload.view(np.int8).reshape(rows,cols)
    else:
        arr=np.empty(total+(total%2),dtype=np.int16)
        arr[0::2]=payload&15
        arr[1::2]=payload>>4
        q=(arr[:total]-8).astype(np.int8).reshape(rows,cols)
    return q,scales
