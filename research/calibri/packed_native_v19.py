"""CALIBRI v19: actual CPU direct packed INT5 residual execution,
with real pretrained Pythia matrix + end-to-end equivalence and honest latency.

No compressed weights expanded into dense matrix inside PackedResidual5.
No use of user data, no model deployment, no CUDA/Android claims.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import pathlib
import struct
import time
import zlib
import numpy as np
from geometric_v16 import make_layer,unpack as tt_unpack
from geometric_v17 import encode as geometry_encode,native_layer as dense_geometry_layer
from fixedbit_v15 import HEADER as FIXED_HDR,encode as fixed_encode,decode as fixed_decode

BUNDLE_HDR=struct.Struct('<4sIII')


def split_geometry(blob):
    if len(blob)<BUNDLE_HDR.size:raise ValueError('truncated geometry')
    magic,n1,n2,crc=BUNDLE_HDR.unpack_from(blob)
    body=blob[BUNDLE_HDR.size:]
    if magic!=b'CG17' or n1<22 or n2<FIXED_HDR.size or len(body)!=n1+n2 or zlib.crc32(body)!=crc:
        raise ValueError('invalid geometry')
    a=body[:n1];b=body[n1:]
    tt_unpack(a) # validate TT crc and factor finiteness only
    return a,b


def parse_packed5(blob):
    if len(blob)<FIXED_HDR.size:raise ValueError('truncated')
    magic,rows,cols,bits,rank,k,crc=FIXED_HDR.unpack_from(blob)
    if magic!=b'CF15' or bits!=5 or rank!=0 or k!=0 or min(rows,cols)<1:
        raise ValueError('not physical packed signed INT5')
    nbytes=(rows*cols*5+7)//8
    payload=blob[FIXED_HDR.size:]
    if len(payload)!=nbytes+2*rows or zlib.crc32(payload)!=crc:
        raise ValueError('CRC / packed length mismatch')
    stream=np.frombuffer(payload[:nbytes],dtype=np.uint8).copy()
    scales=np.frombuffer(payload,'<f2',count=rows,offset=nbytes).astype(np.float32)
    if not np.isfinite(scales).all():
        raise ValueError('invalid packed scales')
    return rows,cols,stream,scales


class Packed5Operator:
    """Streaming CPU operator: only packed bytes + scales persistent, O(batch*rows) output."""
    def __init__(self,blob,library):
        self.rows,self.cols,self.stream,self.scales=parse_packed5(blob)
        self.lib=library
        self.fn=library.calibri_packed5_matmul
        self.fn.argtypes=[ct.POINTER(ct.c_uint8),ct.c_size_t,
                          ct.POINTER(ct.c_float),ct.POINTER(ct.c_float),
                          ct.POINTER(ct.c_float),ct.c_int,ct.c_int,ct.c_int]
        self.fn.restype=ct.c_int
    def __call__(self,x):
        import torch
        if x.device.type!='cpu':
            raise ValueError('CPU-only prototype')
        if x.shape[-1]!=self.cols:raise ValueError('input width mismatch')
        arr=np.ascontiguousarray(x.detach().numpy().reshape(-1,self.cols),dtype=np.float32)
        out=np.empty((len(arr),self.rows),dtype=np.float32)
        status=self.fn(self.stream.ctypes.data_as(ct.POINTER(ct.c_uint8)),
                       self.stream.size,
                       self.scales.ctypes.data_as(ct.POINTER(ct.c_float)),
                       arr.ctypes.data_as(ct.POINTER(ct.c_float)),
                       out.ctypes.data_as(ct.POINTER(ct.c_float)),
                       self.rows,self.cols,len(arr))
        if status!=0:raise RuntimeError(f'packed5 C error code {status}')
        return torch.from_numpy(out).reshape(*x.shape[:-1],self.rows)
    @property
    def resident_weight_bytes(self):
        return self.stream.nbytes+self.scales.nbytes


def geom_native(blob,bias,library):
    import torch
    from torch import nn
    core,packed=split_geometry(blob)
    c=make_layer(core,bias=None,learnable=False)
    p=Packed5Operator(packed,library)
    class CoreAndPacked(nn.Module):
        def __init__(self):
            super().__init__()
            self.core=c
            self.packed=p
            self.register_buffer('bias',None if bias is None else bias.detach().clone())
        def forward(self,x):
            y=self.core(x)+self.packed(x)
            return y if self.bias is None else y+self.bias
    return CoreAndPacked()


def pct(values,k):
    return float(np.percentile(values,k))


def timed(module,x,iters=25):
    import torch
    with torch.no_grad():
        for _ in range(5):module(x)
        observations=[]
        for _ in range(iters):
            start=time.perf_counter()
            module(x)
            observations.append(1000*(time.perf_counter()-start))
    return {'p50_ms':pct(observations,50),'p95_ms':pct(observations,95),
            'min_ms':float(min(observations)),'iterations':iters}


def validate_geometry(w,bias,library,name,m2,n2,rank,bits,shape_cases=(1,16,128)):
    import torch
    assert bits==5
    blob=geometry_encode(w,rank,m2,n2,bits)
    compressed=geom_native(blob,bias,library)
    dense_reference=dense_geometry_layer(blob,bias)
    errors=[]
    performance={}
    for batch in shape_cases:
        x=torch.from_numpy(np.random.default_rng(19+batch).normal(size=(batch,w.shape[1])).astype(np.float32))
        with torch.no_grad():
            a=compressed(x)
            b=dense_reference(x)
        error=float((a-b).abs().max())
        if not torch.allclose(a,b,atol=2e-4,rtol=3e-4):
            raise AssertionError(f'packed kernel deviates from reference {name}/{batch}: {error}')
        errors.append(error)
        performance[str(batch)]={'native_packed':timed(compressed,x),
                                'dense_residual':timed(dense_reference,x)}
    core,packed=split_geometry(blob)
    pack=Packed5Operator(packed,library)
    return {'name':name,'geometry':[m2,n2,rank,bits],
            'physical_bytes':len(blob),'packed_residual_bytes':len(packed),
            'factor_bytes_plus_headers':len(core)+BUNDLE_HDR.size,
            'no_dense_residual_weight_allocation_in_operator':True,
            'persistent_residual_data_bytes':pack.resident_weight_bytes,
            'max_absolute_output_error_vs_dense_reference':max(errors),
            'cpu_timing':performance},blob,compressed,dense_reference


def test_edge_cases(library):
    import torch
    rng=np.random.default_rng(19001)
    tested=0
    for rows,cols in ((3,7),(7,11),(9,13),(16,24),(128,512)):
        w=rng.normal(size=(rows,cols)).astype(np.float32)
        data=fixed_encode(w,5)
        op=Packed5Operator(data,library)
        x=torch.from_numpy(rng.normal(size=(17,cols)).astype(np.float32))
        reference=x@torch.from_numpy(fixed_decode(data)).T
        got=op(x)
        if not torch.allclose(reference,got,atol=2e-4,rtol=3e-4):
            raise AssertionError(f'broken bit mapping: {rows}x{cols}')
        damaged=bytearray(data);damaged[-1]^=1
        try:Packed5Operator(damaged,library);raise AssertionError('corruption accepted')
        except ValueError:pass
        tested+=1
    print(f'Direct C packed5 GEMM: {tested} shape/CRC equivalence controls PASS')
    return tested


def run(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi
    from quality import MODEL,DATASET,load_token_ids,ppl_eval,module_selection
    torch.set_num_threads(2)
    torch.manual_seed(19)
    library=ct.CDLL(str(pathlib.Path(args.library).resolve()))
    tested=test_edge_cases(library)
    model_rev=HfApi().model_info(MODEL).sha
    data_rev=HfApi().dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=model_rev,torch_dtype=torch.float32)
    model.eval()
    name=sorted(module_selection(model))[0]
    original=model.get_submodule(name)
    w=original.weight.detach().cpu().numpy().copy().astype(np.float32)
    d={}
    for conf in [('geo_small',16,32,2,5),('geo_balanced',8,16,4,5),('geo_accuracy',8,32,8,5)]:
        label,m2,n2,rank,bits=conf
        metrics,blob,compressed,dense=validate_geometry(w,original.bias,library,*conf)
        d[label]=metrics
        if label=='geo_small':
            native=compressed
            restored=dense
            saved_blob=blob
    # End-to-end PPL should be the same for packed and restored (same W).
    tokens=load_token_ids(tok,'validation',data_rev,args.eval_tokens)
    parent_name,leaf=name.rsplit('.',1);parent=model.get_submodule(parent_name)
    score={}
    baseline=ppl_eval(model,tokens,args.context)['ppl']
    for key,layer in [('packed_geometric',native),('restored_geometric',restored)]:
        setattr(parent,leaf,layer)
        with torch.no_grad():
            score[key]=ppl_eval(model,tokens,args.context)['ppl']
        setattr(parent,leaf,original)
    if abs(score['packed_geometric']-score['restored_geometric'])>0.001:
        raise AssertionError('packed operator changes whole-model PPL')
    results={'experiment':'CALIBRI_v19_native_packed_int5_geometry',
        'model':MODEL,'revision':model_rev,'dataset_revision':data_rev,
        'module':name,'shape':list(w.shape),'fp16_reference_weight_bytes':2*w.size,
        'signed_packed5_symbol_tests':tested,
        'physical_geometric_variants':d,
        'whole_model_ppl_validation_1024_tokens':{
            'original':baseline,
            **score,
            'warning':'This validation split has been used throughout project; mathematical consistency check only'
        },
        'pretrained_layer_weight_sha256_fp32':hashlib.sha256(w.tobytes()).hexdigest(),
        'limitations':[
          'One pretrained Pythia-14M dense MLP matrix, not routed MoE/QMoE',
          '5-bit packed residual decoded inline by gcc C, not expanded to a full dense W',
          'Tensor cores are dequantized to FP32 arrays; still uses CPU torch einsum',
          'C code lacks SIMD/NEON and is not an optimized product kernel',
          'CPU timing includes Python/ctypes; platform is x86_64 GitHub runner, not ARM',
          'No training or model quality improvement claimed; PPL repeat only equivalence',
          'Not full-model compression, does not measure RAM peak or token/s',
          'Storage byte accounting includes TT headers, FP16 factors, INT5 scales/indices']
    }
    out=pathlib.Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print(json.dumps(results,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--library',default='research/calibri/libpacked5_v19.so')
    p.add_argument('--output',default='results/calibri_v19_packed_native.json')
    p.add_argument('--eval-tokens',type=int,default=1024)
    p.add_argument('--context',type=int,default=128)
    args=p.parse_args()
    run(args)
