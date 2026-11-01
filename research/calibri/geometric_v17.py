"""CALIBRI v17: topology/rank/bitwidth search for actually multidimensional
FP16 tensor cores + physically packed INT4/INT5 residual. Real model only.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,struct,zlib
import numpy as np
from geometric_v16 import tt_svd,reconstruct,pack as pack_tt,unpack as unpack_tt,make_layer
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode

HEADER=struct.Struct('<4sIII')


def encode(w,rank,m2,n2,residual_bits):
    u,v=tt_svd(w,rank,m2,n2)
    core_blob=pack_tt(u,v)
    a,b,res=unpack_tt(core_blob)
    assert res is None
    residual=w-reconstruct(a,b)  # physical FP16-rounded cores
    residual_blob=fixed_encode(residual,residual_bits)
    payload=core_blob+residual_blob
    return HEADER.pack(b'CG17',len(core_blob),len(residual_blob),zlib.crc32(payload))+payload


def unpack(blob):
    if len(blob)<HEADER.size:raise ValueError('short geometry bundle')
    magic,n1,n2,crc=HEADER.unpack_from(blob)
    payload=blob[HEADER.size:]
    if magic!=b'CG17' or n1<22 or n2<18 or len(payload)!=n1+n2 or zlib.crc32(payload)!=crc:
        raise ValueError('bad bundle framing or CRC')
    core=payload[:n1]
    rblob=payload[n1:]
    a,b,_=unpack_tt(core)
    residual=fixed_decode(rblob)
    if residual.shape!=(a.shape[0]*b.shape[1],a.shape[1]*b.shape[2]):
        raise ValueError('TT/quant residual shape mismatch')
    return core,residual


def native_layer(blob,bias=None):
    import torch
    from torch import nn
    core,coeff=unpack(blob)
    class NativeHybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.core=make_layer(core)
            self.register_buffer('residual',torch.from_numpy(coeff.copy()))
            self.register_buffer('bias',None if bias is None else bias.detach().clone())
        def forward(self,x):
            out=self.core(x)+torch.nn.functional.linear(x,self.residual)
            return out if self.bias is None else out+self.bias
    return NativeHybrid()


def smoke():
    import torch
    g=np.random.default_rng(17)
    w=g.normal(size=(16,24)).astype(np.float32)
    x=g.normal(size=(7,24)).astype(np.float32)
    for bits in (4,5):
        for rank in (2,4):
            for m2,n2 in ((4,6),(8,3)):
                blob=encode(w,rank,m2,n2,bits)
                core,r=unpack(blob)
                u,v,_=unpack_tt(core)
                y=native_layer(blob)(torch.from_numpy(x)).detach().numpy()
                reference=x@(reconstruct(u,v)+r).T
                assert np.allclose(y,reference,rtol=4e-5,atol=4e-5)
                broken=bytearray(blob);broken[-1]^=1
                try:unpack(broken);raise AssertionError('corruption undetected')
                except ValueError:pass
    print('v17 topology, exact physical roundtrip, hybrid native TT + residual, CRC: PASS')


def run(args):
    import torch
    from torch import nn
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi
    from quality import load_token_ids,ppl_eval,module_selection,MODEL,DATASET
    torch.set_num_threads(2);torch.manual_seed(17)
    revision=HfApi().model_info(MODEL).sha;data_rev=HfApi().dataset_info(DATASET).sha
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,torch_dtype=torch.float32)
    model.eval()
    name=sorted(module_selection(model))[0]
    original=model.get_submodule(name)
    w=original.weight.detach().cpu().numpy().copy().astype(np.float32)
    parent_name,leaf=name.rsplit('.',1);parent=model.get_submodule(parent_name)
    dev=load_token_ids(tokenizer,'validation',data_rev,args.dev_tokens)
    test=load_token_ids(tokenizer,'test',data_rev,args.test_tokens)
    original_dev=ppl_eval(model,dev,args.context)['ppl']
    # Controls are the exact same physical codec used in CALIBRI v15/v16.
    blobs={'int4':fixed_encode(w,4),'int5':fixed_encode(w,5),
           'int6':fixed_encode(w,6),'int8':fixed_encode(w,8)}
    byte_cap=int(len(blobs['int4'])*1.5)
    topologies=((4,32),(8,16),(8,32),(16,16),(16,32),(2,64))
    searched=0;excluded_by_size=0
    for m2,n2 in topologies:
        for rank in (2,4,6,8):
            for bits in (4,5):
                key=f'geo_m{m2}_n{n2}_r{rank}_q{bits}'
                blob=encode(w,rank,m2,n2,bits)
                searched+=1
                if len(blob)>byte_cap:
                    excluded_by_size+=1
                    continue
                blobs[key]=blob
    def candidate(name):
        blob=blobs[name]
        if name.startswith('geo_'):return native_layer(blob,original.bias)
        lin=nn.Linear(w.shape[1],w.shape[0],bias=original.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(fixed_decode(blob)))
            if original.bias is not None:lin.bias.copy_(original.bias)
        return lin
    rows=[]
    for key in blobs:
        setattr(parent,leaf,candidate(key))
        with torch.no_grad():ppl=ppl_eval(model,dev,args.context)['ppl']
        rows.append({'name':key,'bytes':len(blobs[key]),'development_ppl':ppl,
                     'dev_ratio':ppl/original_dev,'native_tt':key.startswith('geo_')})
        setattr(parent,leaf,original)
    limit=original_dev*1.01
    approved=[r for r in rows if r['bytes']<=byte_cap and r['development_ppl']<=limit]
    chosen=min(approved,key=lambda r:(r['bytes'],r['development_ppl'])) if approved else None
    hybrids_sorted=sorted((r for r in rows if r['native_tt']),
                          key=lambda r:(r['development_ppl'],r['bytes']))
    hybrid_passes=[r for r in approved if r['native_tt']]
    # Test is exploratory because project has observed this split in v13-v16.
    # Prespecified controls, selected model and best development hybrid only.
    targets=['int4','int5','int6']
    if chosen:targets.append(chosen['name'])
    if hybrids_sorted:targets.append(hybrids_sorted[0]['name'])
    original_test=ppl_eval(model,test,args.context)['ppl']
    test_scores={}
    for name in dict.fromkeys(targets):
        setattr(parent,leaf,candidate(name))
        with torch.no_grad():test_scores[name]=ppl_eval(model,test,args.context)['ppl']
        setattr(parent,leaf,original)
    record={'experiment':'CALIBRI_v17_geometry_topology_rank_residual_precision',
        'model':MODEL,'model_revision':revision,'data_revision':data_rev,
        'matrix_name':sorted(module_selection(model))[0],'matrix_shape':list(w.shape),
        'fp16_reference_weight_bytes':2*w.size,'development_tokens':len(dev),
        'test_tokens':len(test),'development_sha256':hashlib.sha256(dev.numpy().tobytes()).hexdigest(),
        'test_sha256':hashlib.sha256(test.numpy().tobytes()).hexdigest(),
        'original_development_ppl':original_dev,'original_test_ppl':original_test,
        'max_bytes':byte_cap,'max_development_ppl':limit,
        'total_geometry_candidates':searched,'excluded_by_byte_budget':excluded_by_size,
        'evaluated_variant_count':len(rows),
        'selected_min_bytes_approved':chosen['name'] if chosen else 'NO_APPROVED_VARIANT',
        'approved_geometries':[r['name'] for r in hybrid_passes],
        'best_development_geometry':hybrids_sorted[0] if hybrids_sorted else None,
        'variants':rows,'exploratory_predeclared_test':test_scores,
        'limitations':['One trained Pythia-14m dense MLP matrix, no MoE or QMoE',
          'TT and bit quantization are existing methods, no claim of novelty',
          'TT core executes without dense W, residual runs as dequantized dense tensor',
          'No fused streaming residual code or measured ARM/Android latency',
          'Data/test split reused from prior research; exploratory, not confirmatory',
          'No retraining, full model storage, or memory-residency measurements'] }
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    print(json.dumps(record,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',default='results/calibri_v17_topology.json')
    p.add_argument('--dev-tokens',type=int,default=1024)
    p.add_argument('--test-tokens',type=int,default=1024)
    p.add_argument('--context',type=int,default=128)
    arg=p.parse_args()
    smoke() if arg.smoke else run(arg)
