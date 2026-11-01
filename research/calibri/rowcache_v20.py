"""CALIBRI v20: optimized packed5 kernel with bounded one-row scratch.
Baseline is v19's no-scratch per-vector bit decoder and dense restored math.
Research on REAL pretrained weight tensor; no production-ready execution.
"""
from __future__ import annotations
import argparse,ctypes as ct,hashlib,json,pathlib
import numpy as np
from packed_native_v19 import Packed5Operator,geom_native,split_geometry,timed
from geometric_v16 import make_layer
from geometric_v17 import encode as geometry_encode,native_layer as restored_geometry
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode


class OneRowPacked5(Packed5Operator):
    def __init__(self,blob,library):
        super().__init__(blob,library)
        self.fn=library.calibri_packed5_matmul_rowcache
        self.fn.argtypes=[ct.POINTER(ct.c_uint8),ct.c_size_t,
                          ct.POINTER(ct.c_float),ct.POINTER(ct.c_float),
                          ct.POINTER(ct.c_float),ct.c_int,ct.c_int,ct.c_int]
        self.fn.restype=ct.c_int


def row_cached_geometry(blob,bias,lib):
    import torch
    from torch import nn
    core,packed=split_geometry(blob)
    core_layer=make_layer(core,None,learnable=False)
    row_operator=OneRowPacked5(packed,lib)
    class StreamingLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.core=core_layer
            self.packed=row_operator
            self.register_buffer('bias',None if bias is None else bias.detach().clone())
        def forward(self,x):
            y=self.core(x)+self.packed(x)
            return y if self.bias is None else y+self.bias
    return StreamingLayer()


def run(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi
    from quality import MODEL,DATASET,load_token_ids,ppl_eval,module_selection
    torch.set_num_threads(2);torch.manual_seed(20)
    lib=ct.CDLL(str(pathlib.Path(args.library).resolve()))
    g=np.random.default_rng(2001)
    tests=[]
    for rows,cols in ((3,7),(7,11),(16,24),(128,512)):
        w=g.normal(size=(rows,cols)).astype(np.float32)
        blob=fixed_encode(w,5)
        x=torch.from_numpy(g.normal(size=(17,cols)).astype(np.float32))
        ref=x@torch.from_numpy(fixed_decode(blob)).T
        old=Packed5Operator(blob,lib)(x)
        opt=OneRowPacked5(blob,lib)(x)
        if not torch.allclose(opt,ref,rtol=5e-4,atol=3e-4) or not torch.allclose(opt,old,rtol=5e-4,atol=3e-4):
            raise AssertionError(f'rowcache mismatch {rows},{cols}')
        tests.append([rows,cols])
    print('v20 packed5 one-row scratch equivalence: PASS',flush=True)
    revision=HfApi().model_info(MODEL).sha
    data_rev=HfApi().dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,torch_dtype=torch.float32)
    model.eval()
    name=sorted(module_selection(model))[0]
    original=model.get_submodule(name)
    w=original.weight.detach().cpu().numpy().copy().astype(np.float32)
    variants={}
    for key,m2,n2,rank in (('geo_small',16,32,2),
                           ('geo_balanced',8,16,4),
                           ('geo_accuracy',8,32,8)):
        blob=geometry_encode(w,rank,m2,n2,5)
        base=geom_native(blob,original.bias,lib)
        opt=row_cached_geometry(blob,original.bias,lib)
        dense=restored_geometry(blob,original.bias)
        timing={}
        output_max_abs=0.
        for batch in (1,16,128):
            x=torch.from_numpy(g.normal(size=(batch,w.shape[1])).astype(np.float32))
            with torch.no_grad():
                a=base(x);b=opt(x);c=dense(x)
            if not torch.allclose(a,b,rtol=5e-4,atol=3e-4) or not torch.allclose(b,c,rtol=5e-4,atol=3e-4):
                raise AssertionError(f'tensor bitstream output drift {key}:{batch}')
            output_max_abs=max(output_max_abs,float((b-c).abs().max()))
            timing[str(batch)]={'repeat_decode_v19':timed(base,x,args.reps),
                                'one_row_cache_v20':timed(opt,x,args.reps),
                                'unpacked_dense_ref':timed(dense,x,args.reps)}
        variants[key]={'physical_bytes':len(blob),'max_abs_kernel_error':output_max_abs,
                       'scratch_row_float32_bytes':w.shape[1]*4,
                       'batch_latency_ms':timing}
        if key=='geo_small':
            best=opt;reference=dense
    # Actual full LLM equivalence; its dev split was already used before.
    ids=load_token_ids(tok,'validation',data_rev,args.eval_tokens)
    parent,leaf=name.rsplit('.',1);outer=model.get_submodule(parent)
    quality={}
    quality['original']=ppl_eval(model,ids,args.context)['ppl']
    for label,layer in (('dense_reference',reference),('rowcache_packed',best)):
        setattr(outer,leaf,layer)
        with torch.no_grad():quality[label]=ppl_eval(model,ids,args.context)['ppl']
        setattr(outer,leaf,original)
    quality['abs_ppl_gap']=abs(quality['dense_reference']-quality['rowcache_packed'])
    if quality['abs_ppl_gap']>0.001*quality['dense_reference']:
        raise AssertionError('v20 full model outputs differ >0.1% PPL from exact-quantized reference')
    result={'experiment':'CALIBRI_v20_CPU_one_row_bitpacked_residual',
      'model':MODEL,'model_revision':revision,'data_revision':data_rev,
      'module':name,'trained_weight_fp32_sha256':hashlib.sha256(w.tobytes()).hexdigest(),
      'shape':list(w.shape),'physical_vs_FP16_reference_bytes':2*w.size,
      'one_row_scratch_array_only':True,
      'one_row_scratch_bytes':w.shape[1]*4,
      'numeric_shape_controls_passed':tests,
      'variants':variants,'full_lm_equivalence_only':quality,
      'constraints':['One real dense Pythia 14M weight projection, not MoE or QMoE',
        'No other model modules are compressed',
        'No long-run Android/ARM/neon benchmarks',
        'One-row decoded scratch, no full dense residual matrix is stored',
        'Bit-packing, streaming and TT tensor networks are not novel by themselves',
        'CPU timing is from single GitHub runner and Python ctypes wrappers',
        'Full-model perplexity evaluation is repeated, not fresh independent quality']}
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--library',default='research/calibri/libpacked5_v20.so')
    p.add_argument('--output',default='results/calibri_v20_rowcache.json')
    p.add_argument('--eval-tokens',type=int,default=1024)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--reps',type=int,default=20)
    run(p.parse_args())
