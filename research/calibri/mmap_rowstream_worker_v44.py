#!/usr/bin/env python3
"""CALIBRI v44 mmap-backed row/block reference GEMM (NOT native packed kernel).

Compares the SAME original pretrained 12 MLPs encoded as CT33 mixed3/4
and CF15 fixed INT4, already physically serialized inside CB36.
Packed path mmaps CB36 and decodes at most 16 WHOLE ROWS of ONE matrix
at a time. It never materializes any complete W; the dense path restores
and retains all 12 full dense matrices and invokes NumPy/OpenBLAS GEMM.

Functionally tests reference output against a separately computed and
persisted all-12 fully decoded golden set. Throughput is for 8 real
Pythia module input tokens and a warm OS page cache, not production
generation, GPU, cold storage, or a native AVX kernel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import resource
import statistics
import struct
import time
import tracemalloc
import zlib

import numpy as np

BUNDLE_HEADER=struct.Struct("<4sBBH")
BUNDLE_ITEM=struct.Struct("<BBI")
CT33_HEADER=struct.Struct("<4sHHHHII")
CF15_HEADER=struct.Struct("<4sHHBBII")
ROWS_PER_BLOCK=16
MODULES=12
COUNT_WEIGHTS=786432


def _format_rows(mm, pos, size, mode):
    """Validate exact source bytes and index ROW offsets (no W allocation)."""
    matrix_end=pos+size
    if mode==1:
        if size<CT33_HEADER.size:
            raise ValueError("short CT33 module")
        magic,m,n,br,bc,payload_length,crc=CT33_HEADER.unpack_from(mm,pos)
        if (magic!=b"CT33" or br!=1 or bc!=n
            or m not in (128,512) or n not in (128,512)
            or m*n!=65536 or payload_length+CT33_HEADER.size!=size):
            raise ValueError("v44 requires physical rowwise real Pythia CT33")
        payload_start=pos+CT33_HEADER.size
        if zlib.crc32(memoryview(mm)[payload_start:matrix_end])!=crc:
            raise ValueError("CT33 physical CRC mismatch")
        offset=payload_start
        rows=[]
        for row in range(m):
            if offset+3>matrix_end:
                raise ValueError("short CT33 row metadata")
            bits,scale=struct.unpack_from("<Be",mm,offset)
            offset+=3
            if bits not in (3,4) or not np.isfinite(scale) or scale<=0:
                raise ValueError("invalid CT33 row bitwidth/FP16 scale")
            packed=n*bits//8
            if offset+packed>matrix_end:
                raise ValueError("CT33 packed row truncated")
            rows.append((offset,packed,bits,float(scale)))
            offset+=packed
        if offset!=matrix_end:
            raise ValueError("CT33 tail bytes")
        return m,n,rows
    if mode==2:
        if size<CF15_HEADER.size:
            raise ValueError("short CF15 module")
        magic,m,n,bits,rank,k,crc=CF15_HEADER.unpack_from(mm,pos)
        if (magic!=b"CF15" or bits!=4 or rank or k
            or m*n!=65536 or m not in (128,512) or n not in (128,512)):
            raise ValueError("v44 only tests pretrained CF15 signed INT4")
        qbytes=m*n*bits//8
        data_start=pos+CF15_HEADER.size
        if data_start+qbytes+2*m!=matrix_end:
            raise ValueError("CF15 incorrect physical scale/bitstream length")
        if zlib.crc32(memoryview(mm)[data_start:matrix_end])!=crc:
            raise ValueError("CF15 physical CRC mismatch")
        rows=[]
        scale_start=data_start+qbytes
        for row in range(m):
            scale=struct.unpack_from("<e",mm,scale_start+row*2)[0]
            if not np.isfinite(scale) or scale<=0:
                raise ValueError("corrupt CF15 physical row scale")
            rows.append((data_start+row*(n*bits//8),n*bits//8,bits,float(scale)))
        return m,n,rows
    raise ValueError("unsupported CB36 compression codec")


def load_mmap_index(bundle_path):
    """CRC verify every physical matrix and CB36 outer SHA, index only."""
    fp=open(bundle_path,"rb")
    mm=mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ)
    view=memoryview(mm)
    try:
        if len(view)<BUNDLE_HEADER.size+32:
            raise ValueError("truncated physical file")
        magic,ver,mode,count=BUNDLE_HEADER.unpack_from(mm)
        if magic!=b"CB36" or ver!=1 or mode not in (1,2) or count!=MODULES:
            raise ValueError("bad Pythia CB36 format")
        checksum=hashlib.sha256(view[:-32]).digest()
        if checksum!=view[-32:].tobytes():
            raise ValueError("CB36 outer SHA corruption")
        index={}
        off=BUNDLE_HEADER.size
        data_end=len(view)-32
        for i in range(MODULES):
            if off+BUNDLE_ITEM.size>data_end:
                raise ValueError("short CB36 item descriptor")
            layer,kind,size=BUNDLE_ITEM.unpack_from(mm,off)
            off+=BUNDLE_ITEM.size
            if not (0<=layer<6 and kind in (0,1) and size>0 and off+size<=data_end):
                raise ValueError("invalid CB36 source item geometry")
            name=f"{layer}_{kind}"
            if name in index:
                raise ValueError("duplicate source matrix")
            m,n,rows=_format_rows(mm,off,size,mode)
            if ((kind==0 and (m,n)!=(512,128)) or
                (kind==1 and (m,n)!=(128,512))):
                raise ValueError("wrong pretrained source projection shape")
            index[name]=(m,n,rows)
            off+=size
        if off!=data_end or set(index)!={f"{i}_{k}" for i in range(6) for k in range(2)}:
            raise ValueError("all 12 original MLPs not physically present")
        return fp,mm,index,mode
    except Exception:
        mm.close()
        fp.close()
        raise
    finally:
        view.release()


def decode_one_row(mm,desc,n):
    """Signed B-bit unpack from original FILE-backed mmap bytes; no full W."""
    off,packed,bits,scale=desc
    raw=np.frombuffer(mm,dtype=np.uint8,count=packed,offset=off)
    bitplanes=np.unpackbits(raw)[:n*bits].reshape(n,bits)
    powers=(1<<np.arange(bits-1,-1,-1,dtype=np.int16))
    unsigned=bitplanes.astype(np.int16)@powers
    signed=np.where(unsigned>=(1<<(bits-1)),
                    unsigned-(1<<bits),unsigned).astype(np.float32)
    return signed*np.float32(scale)


def map_forward(mm,index,inputs):
    """True rowwise decode+GEMM of each real pretrained MLP layer; no full W."""
    outputs={}
    largest_scratch=0
    for name in sorted(index,key=lambda k:tuple(map(int,k.split("_")))):
        m,n,rows=index[name]
        x=inputs[name]
        if x.ndim!=2 or x.shape!=(8,n):
            raise AssertionError("real input shape changed")
        y=np.empty((x.shape[0],m),np.float32)
        scratch=np.empty((min(ROWS_PER_BLOCK,m),n),np.float32)
        largest_scratch=max(largest_scratch,scratch.nbytes)
        for st in range(0,m,ROWS_PER_BLOCK):
            used=min(ROWS_PER_BLOCK,m-st)
            for j in range(used):
                scratch[j,:]=decode_one_row(mm,rows[st+j],n)
            y[:,st:st+used]=x@scratch[:used].T
        outputs[name]=y
    return outputs,largest_scratch


def dense_decompress_all(mm,index):
    """Dense physical-file reference: all twelve W retained at FP32."""
    dense={}
    for name,(m,n,rows) in index.items():
        w=np.empty((m,n),np.float32)
        for j,desc in enumerate(rows):
            w[j,:]=decode_one_row(mm,desc,n)
        dense[name]=w
    if sum(a.size for a in dense.values())!=COUNT_WEIGHTS:
        raise AssertionError("dense source geometry mismatch")
    return dense


def dense_forward(dense,inputs):
    return {name:inputs[name]@w.T for name,w in dense.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",required=True)
    p.add_argument("--examples",required=True)
    p.add_argument("--mode",choices=("mmap_stream","dense_restored"),required=True)
    p.add_argument("--repetitions",type=int,default=5)
    p.add_argument("--warmups",type=int,default=2)
    args=p.parse_args()
    if args.repetitions!=5 or args.warmups!=2:
        raise AssertionError("frozen benchmark requires exactly five warm page-cache repeats")
    bundle_path=Path(args.bundle)
    if not bundle_path.is_file():
        raise AssertionError("the physical source file must exist")

    # One process per mode; neither includes the source PyTorch model in RSS.
    # All references and inputs were produced from genuine 2048 TRAIN-A
    # pretrained model module captures before this subprocess is launched.
    with np.load(args.examples,allow_pickle=False) as test:
        inputs={f"{layer}_{kind}":np.ascontiguousarray(test[f"x_{layer}_{kind}"])
                for layer in range(6) for kind in range(2)}
        reference={f"{layer}_{kind}":np.ascontiguousarray(test[f"y_{layer}_{kind}"])
                   for layer in range(6) for kind in range(2)}
    tracemalloc.start()
    fp,mm,index,codec=load_mmap_index(str(bundle_path))
    dense=None
    try:
        if args.mode=="dense_restored":
            dense=dense_decompress_all(mm,index)
        fun=lambda: (map_forward(mm,index,inputs) if dense is None
                     else (dense_forward(dense,inputs),0))
        out,scratch=fun()
        deviations={n:float(np.max(np.abs(reference[n]-out[n])))
                    for n in reference}
        max_diff=max(deviations.values())
        if max_diff>2e-4 or not np.isfinite(max_diff):
            raise AssertionError(f"physical byte-backed GEMM differs from dense: {max_diff}")
        for _ in range(args.warmups):fun()
        durations=[]
        for _ in range(args.repetitions):
            start=time.perf_counter_ns()
            y,_scratch=fun()
            durations.append((time.perf_counter_ns()-start)/1e6)
        _current,alloc_peak=tracemalloc.get_traced_memory()
        # Linux ru_maxrss = KiB, NOT exact per-array allocated memory.
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        matrix_bytes=sum(a.nbytes for a in dense.values()) if dense else 0
        input_bytes=sum(x.nbytes for x in inputs.values())
        report={
            "worker_mode":args.mode,"codec":"CT33_mixed3_4" if codec==1 else "CF15_fixed4",
            "physical_file":bundle_path.name,
            "physical_bytes":bundle_path.stat().st_size,
            "physical_sha256":hashlib.sha256(mm[:]).hexdigest(),
            "pretrained_MLP_matrices":len(index),
            "source_parameter_positions":COUNT_WEIGHTS,
            "tokens_per_matrix":8,
            "MLP_forward_calls_per_trial":len(index),
            "MLP_forward_repetitions":args.repetitions,
            "milliseconds":durations,
            "median_12_matmuls_ms":statistics.median(durations),
            "min_12_matmuls_ms":min(durations),
            "max_abs_diff_from_FULL_dense_golden":max_diff,
            "per_matrix_max_abs_diff":deviations,
            "prebuilt_dense_FP32_weight_bytes":matrix_bytes,
            "max_FP32_weight_scratch_bytes":scratch,
            "actual_MMAPPED_source_bytes":len(mm),
            "preloaded_real_activation_input_bytes":input_bytes,
            "peak_process_RSS_KiB_including_numpy_and_mmap":peak_rss_kib,
            "peak_tracemalloc_bytes_including_indices_and_outputs":alloc_peak,
            "control_note":"Page-cache-warm Python NumPy reference, not a fused/native packed kernel or end-to-end full-model speed. Decoded dense matrices allocated ONLY by dense_restored.",
        }
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:
        mm.close()
        fp.close()


if __name__=="__main__":main()
