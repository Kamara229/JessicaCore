"""CALIBRI v21: vectorized packed INT5 rowstream, only a row-sized scratch buffer.
Benchmark SIMD vs portable rowstream vs dense exact-quantized reference.
Research-only x86-64 CPU: no proof of ARM/GPU or phone speed.
"""
from __future__ import annotations
import argparse,ctypes as ct,hashlib,json,pathlib,platform
import numpy as np
from rowcache_v20 import OneRowPacked5,row_cached_geometry
from packed_native_v19 import split_geometry,timed
from geometric_v16 import make_layer
from geometric_v17 import encode as geometric_encode,native_layer as dense_geometric
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode


class SimdPacked5(OneRowPacked5):
    def __init__(self,blob,lib):
        super().__init__(blob,lib)
        self.fn=lib.calibri_packed5_matmul_rowcache_simd
        self.fn.argtypes=[ct.POINTER(ct.c_uint8),ct.c_size_t,
                          ct.POINTER(ct.c_float),ct.POINTER(ct.c_float),
                          ct.POINTER(ct.c_float),ct.c_int,ct.c_int,ct.c_int]
        self.fn.restype=ct.c_int


def simd_geometry(blob,bias,library):
    import torch
    from torch import nn
    core,packed=split_geometry(blob)
    g=make_layer(core,None,learnable=False)
    streamed=SimdPacked5(packed,library)
    class Operator(nn.Module):
        def __init__(self):
            super().__init__()
            self.core=g
            self.packed=streamed
            self.register_buffer('bias',None if bias is None else bias.detach().clone())
        def forward(self,x):
            a=self.core(x)+self.packed(x)
            return a if self.bias is None else a+self.bias
    return Operator()


def run(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi
    from quality import MODEL,DATASET,load_token_ids,ppl_eval,module_selection
    torch.set_num_threads(2);torch.manual_seed(21)
    lib=ct.CDLL(str(pathlib.Path(args.library).resolve()))
    rng=np.random.default_rng(21)
    base=fixed_encode(rng.normal(size=(7,11)).astype(np.float32),5)
    trial=SimdPacked5(base,lib)
    cpu_support=True
    try:trial(torch.from_numpy(rng.normal(size=(17,11)).astype(np.float32)))
    except RuntimeError as exc:
        if 'code -5' in str(exc):cpu_support=False
        else:raise
    print('v21 AVX2/FMA available:',cpu_support,flush=True)
    if cpu_support:
        for rows,cols in ((3,7),(7,11),(9,13),(16,24),(128,512)):
            w=rng.normal(size=(rows,cols)).astype(np.float32)
            data=fixed_encode(w,5)
            x=torch.from_numpy(rng.normal(size=(17,cols)).astype(np.float32))
            reference=x@torch.from_numpy(fixed_decode(data)).T
            portable=OneRowPacked5(data,lib)(x)
            simd=SimdPacked5(data,lib)(x)
            if not torch.allclose(reference,simd,atol=3e-4,rtol=5e-4):
                raise AssertionError(f'SIMD numerical mismatch {rows}x{cols}')
            if not torch.allclose(portable,simd,atol=3e-4,rtol=5e-4):
                raise AssertionError('SIMD/portable disagreement')
        print('v21 AVX2 numerical equivalence: PASS',flush=True)
    revision=HfApi().model_info(MODEL).sha
    drevision=HfApi().dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,torch_dtype=torch.float32)
    model.eval();name=sorted(module_selection(model))[0]
    orig=model.get_submodule(name);w=orig.weight.detach().numpy().copy().astype(np.float32)
    variants={}
    active_simd=None;active_dense=None
    for label,m2,n2,rank in (('geo_small',16,32,2),
                              ('geo_balanced',8,16,4),
                              ('geo_accuracy',8,32,8)):
        encoded=geometric_encode(w,rank,m2,n2,5)
        portable=row_cached_geometry(encoded,orig.bias,lib)
        dense=dense_geometric(encoded,orig.bias)
        simd=simd_geometry(encoded,orig.bias,lib) if cpu_support else None
        speed={}
        max_abs=0.
        for count in (1,16,128):
            x=torch.from_numpy(rng.normal(size=(count,w.shape[1])).astype(np.float32))
            with torch.no_grad():
                a=portable(x);b=dense(x)
                c=simd(x) if simd is not None else None
            if not torch.allclose(a,b,atol=3e-4,rtol=5e-4):
                raise AssertionError(f'portable does not match dense {label}')
            if simd is not None:
                if not torch.allclose(c,b,atol=3e-4,rtol=5e-4):
                    raise AssertionError(f'SIMD does not match dense {label}')
                max_abs=max(max_abs,float((c-b).abs().max()))
            speed[str(count)]={'portable_rowcache':timed(portable,x,args.reps),
                               'dense_restored':timed(dense,x,args.reps)}
            if simd is not None:speed[str(count)]['simd_rowcache']=timed(simd,x,args.reps)
        variants[label]={'bytes_physical':len(encoded),
          'scratch_bytes':w.shape[1]*4,'simd_error_max_abs':max_abs,
          'cpu_batch_latency_ms':speed}
        if label=='geo_small':active_simd=simd;active_dense=dense
    quality=None
    if cpu_support:
        ids=load_token_ids(tok,'validation',drevision,args.eval_tokens)
        outer_name,leaf=name.rsplit('.',1);parent=model.get_submodule(outer_name)
        quality={}
        for key,layer in (('dense_exact',active_dense),('streaming_avx2',active_simd)):
            setattr(parent,leaf,layer)
            with torch.no_grad():quality[key]=ppl_eval(model,ids,args.context)['ppl']
            setattr(parent,leaf,orig)
        quality['absolute_ppl_gap']=abs(quality['dense_exact']-quality['streaming_avx2'])
        if quality['absolute_ppl_gap']>quality['dense_exact']*.001:
            raise AssertionError('whole model PPL drift above predeclared 0.1%')
    out={'experiment':'CALIBRI_v21_AVX2_bitpacked_direct_weight_matvec',
         'model':MODEL,'revision':revision,'dataset_revision':drevision,
         'module':name,'shape':list(w.shape),'fp16_reference_weight_bytes':2*w.size,
         'machine_arch':platform.machine(),'cpu_avx2_fma_supported':cpu_support,
         'numeric_geometry_controls_passed':cpu_support,
         'variants':variants,'whole_model_repeat_equivalence':quality,
         'precision_caveat':'FMA reassociates float multiplication/addition, minor rounding differences expected',
         'limits':['x86 AVX2/FMA conditional implementation, not ARM NEON',
          'C rowwise decompress uses O(cols) scratch, no full residual W buffer',
          'Tensor core math still uses PyTorch FP32 einsum',
          'One pretrained Pythia MLP projection, not MoE or whole-model inference',
          'CPU benchmark is one GitHub runner and Python overhead; no throughput/RAM claims',
          'Earlier TinyStories vs WikiText cross-domain disagreement remains unresolved',
          'No claimed originality or production readiness'] }
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps(out,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--library',default='research/calibri/libpacked5_v21.so')
    p.add_argument('--output',default='results/calibri_v21_avx2.json')
    p.add_argument('--reps',type=int,default=20)
    p.add_argument('--eval-tokens',type=int,default=1024)
    p.add_argument('--context',type=int,default=128)
    run(p.parse_args())
