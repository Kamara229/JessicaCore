"""CALIBRI v15 control: real 5- and 6-bit row-quant baselines against fusion.

Signed fixed-bit physically packed, CRC, rowwise FP16 scales. Genuine model and
held-out PPL; choose exclusively from independent development split.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,struct,zlib
import numpy as np
from fusion_v13 import candidates
HEADER=struct.Struct('<4sHHBBII')


def encode(w,bits):
    w=np.asarray(w,dtype=np.float32)
    if w.ndim!=2 or bits not in (4,5,6,8) or min(w.shape)==0 or max(w.shape)>65535 or not np.isfinite(w).all():
        raise ValueError('bad tensor/bit-width')
    m,n=w.shape;bound=(1<<(bits-1))-1
    scales=(np.max(np.abs(w),axis=1)/bound).astype('<f2')
    scales[scales==0]=np.float16(1.)
    q=np.clip(np.rint(w/scales.astype(np.float32)[:,None]),-bound,bound).astype(np.int16)
    symbols=(q.ravel() & ((1<<bits)-1)).astype(np.uint8)
    if bits==8:body=symbols.tobytes()
    else:
        ordered=((symbols[:,None]>>np.arange(bits-1,-1,-1,dtype=np.uint8))&1).ravel()
        body=np.packbits(ordered).tobytes()
    payload=body+scales.tobytes()
    return HEADER.pack(b'CF15',m,n,bits,0,0,zlib.crc32(payload))+payload


def decode(blob):
    if len(blob)<HEADER.size:raise ValueError('too short')
    magic,m,n,bits,rank,k,crc=HEADER.unpack_from(blob)
    if magic!=b'CF15' or bits not in (4,5,6,8) or min(m,n)<1 or rank or k:raise ValueError('invalid header')
    qbytes=(m*n*bits+7)//8
    payload=blob[HEADER.size:]
    if len(payload)!=qbytes+2*m or zlib.crc32(payload)!=crc:raise ValueError('invalid length or CRC')
    raw=np.frombuffer(payload[:qbytes],np.uint8)
    if bits==8:q=raw[:m*n].astype(np.int16)
    else:
        nib=np.unpackbits(raw)[:m*n*bits].reshape(-1,bits)
        q=nib.astype(np.int16)@ (2**np.arange(bits-1,-1,-1,dtype=np.int16))
    q=np.where(q>=(1<<(bits-1)),q-(1<<bits),q).astype(np.float32).reshape(m,n)
    scales=np.frombuffer(payload,'<f2',count=m,offset=qbytes).astype(np.float32)
    if not np.isfinite(scales).all():raise ValueError('bad scale')
    return q*scales[:,None]


def smoke():
    r=np.random.default_rng(15);w=r.normal(size=(9,13)).astype(np.float32)
    for bits in (4,5,6,8):
        blob=encode(w,bits)
        assert len(blob)==HEADER.size+(w.size*bits+7)//8+2*len(w)
        assert np.isfinite(decode(blob)).all()
        corrupt=bytearray(blob);corrupt[-1]^=1
        try:decode(corrupt);raise AssertionError('CRC failed')
        except ValueError:pass
    print('v15 physical 4/5/6/8-bit pack, CRC and length PASS')


def run(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi
    from quality import load_token_ids,ppl_eval,module_selection,calibrate,MODEL,DATASET
    torch.set_num_threads(2);torch.manual_seed(15)
    rev=HfApi().model_info(MODEL).sha;drev=HfApi().dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,torch_dtype=torch.float32)
    model.eval();mods=module_selection(model);name=sorted(mods)[0]
    mod=mods[name];w=mod.weight.detach().cpu().numpy().copy().astype(np.float32)
    train=load_token_ids(tok,'train',drev,256)
    dev=load_token_ids(tok,'validation',drev,1024)
    test=load_token_ids(tok,'test',drev,1024)
    x=calibrate(model,{name:mod},train,128)[name].numpy()
    multi=candidates(w,x)
    matched={r['name']:r for r in multi if r['name'] in ('q4','q4_rank8_sparse1','rank8_q4_sparse1','q8')}
    options=[]
    for bits in (4,5,6,8):
        blob=encode(w,bits)
        options.append({'name':f'fixed{bits}','bytes':len(blob),'recon':decode(blob),'codec':'fixed_bits'})
    for variant_name,r in matched.items():
        options.append({'name':variant_name,'bytes':r['bytes'],'recon':r['recon'],'codec':'fusion_v13'})
    q4=matched['q4'];budget=int(q4['bytes']*1.5)
    original=mod.weight.detach().clone()
    original_dev=ppl_eval(model,dev,128)
    for opt in options:
        with torch.no_grad():mod.weight.copy_(torch.from_numpy(opt['recon']))
        opt['dev_ppl']=ppl_eval(model,dev,128)['ppl']
    with torch.no_grad():mod.weight.copy_(original)
    cap=1.01*original_dev['ppl']
    eligible=[o for o in options if o['bytes']<=budget and o['dev_ppl']<=cap]
    chosen=min(eligible,key=lambda o:(o['bytes'],o['dev_ppl'])) if eligible else None
    original_test=ppl_eval(model,test,128)
    test_targets=['fixed4','fixed6','rank8_q4_sparse1']
    if chosen:test_targets.append(chosen['name'])
    result_test={}
    for variant_name in dict.fromkeys(test_targets):
        c=next(o for o in options if o['name']==variant_name)
        with torch.no_grad():mod.weight.copy_(torch.from_numpy(c['recon']))
        result_test[variant_name]=ppl_eval(model,test,128)['ppl']
    with torch.no_grad():mod.weight.copy_(original)
    for opt in options:del opt['recon']
    report={'experiment':'CALIBRI_v15_fixedbit_matched_budget_control',
      'model':MODEL,'revision':rev,'data_revision':drev,
      'token_sha256':{k:hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                      for k,ids in [('train',train),('dev',dev),('test',test)]},
      'module':name,'shape':list(w.shape),'q4_budget_reference':q4['bytes'],
      'max_bytes':budget,'dev_ppl_reference':original_dev['ppl'],
      'dev_ppl_cap':cap,'approved_min_bytes_option':chosen['name'] if chosen else 'NONE',
      'test_ppl_reference':original_test['ppl'],'predeclared_test_results':result_test,
      'variants':options,'limits':['One real dense Pythia projection, no MoE',
          'No direct compressed execution or model-wide physical bundle',
          'Simple symmetric row scales; not AWQ/GPTQ',
          'No test-driven selection, only development PPL']}
    out=pathlib.Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',default='results/calibri_v15_fixedbit.json')
    a=p.parse_args();smoke() if a.smoke else run(a)
