"""Pretrained Pythia-14M tensor benchmark; never production Jessica inference.

Only public safetensors. Approximate 88.5%-zero row-wise ternarization is
NOT QMoE quantization; evaluate its quantization error separately.
"""
import argparse
import hashlib
import json
import pathlib
import time
import zlib
import numpy as np
from safetensors import safe_open
from huggingface_hub import HfApi,hf_hub_download
from codec import encode,decode
from index import encode as compact_encode,decode as compact_decode
from lift import measure,fused_matvec

MODEL='EleutherAI/pythia-14m'

def ternarize(w,target_zero):
    v=w.astype(np.float32)
    threshold=np.quantile(np.abs(v),target_zero,axis=1,keepdims=True)
    q=np.where(np.abs(v)>threshold,np.sign(v),0).astype(np.int8)
    lo=np.zeros(q.shape[0],dtype=np.float64);hi=lo.copy()
    for i in range(len(lo)):
        p=v[i][q[i]>0];n=v[i][q[i]<0]
        hi[i]=float(np.float16(p.mean())) if len(p) else 0.
        lo[i]=float(np.float16(n.mean())) if len(n) else 0.
    return q,lo,hi

def two_bit_zlib(q):
    a=(q.ravel()+1).astype(np.uint8)
    if len(a)%4:a=np.pad(a,(0,4-len(a)%4),constant_values=1)
    packed=(a[0::4]|a[1::4]<<2|a[2::4]<<4|a[3::4]<<6).tobytes()
    return len(zlib.compress(packed,9))+12

def run(output,target_zero=0.885,max_matrices=12):
    sha=HfApi().model_info(MODEL).sha
    path=hf_hub_download(repo_id=MODEL,filename='model.safetensors',revision=sha)
    digest=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    rng=np.random.default_rng(260701)
    rows=[];total={k:0 for k in ('parameters','clb_bytes','lift_bytes','packed_zlib_bytes','scale_bytes','compact_bytes')}
    with safe_open(path,framework='np',device='cpu') as f:
        keys=[k for k in f.keys() if '.mlp.dense_' in k and k.endswith('.weight')]
        for key in sorted(keys)[:max_matrices]:
            w=f.get_tensor(key).astype(np.float32)
            if w.ndim!=2 or w.shape[1]%32:continue
            t0=time.perf_counter()
            q,lo,hi=ternarize(w,target_zero)
            cb=encode(q)
            if not np.array_equal(q,decode(cb)):raise AssertionError('CLB roundtrip')
            ic=compact_encode(q)
            if not np.array_equal(q,compact_decode(ic)):raise AssertionError('CLB8 roundtrip')
            liftb,delta_zero,parts=measure(q,True)
            x=rng.normal(size=q.shape[1]);a,s,d,ps=parts
            direct=fused_matvec(a,s,d,ps,x,lo,hi)
            qreal=np.where(q<0,lo[:,None],np.where(q>0,hi[:,None],0.))
            gold=qreal@x
            absolute=float(np.max(np.abs(direct-gold)))
            relative=float(np.linalg.norm(direct-gold)/max(np.linalg.norm(gold),1e-20))
            qdist=float(np.linalg.norm(w@x-gold)/max(np.linalg.norm(w@x),1e-20))
            row={'key':key,'shape':list(q.shape),'parameters':int(q.size),
                 'zero_fraction':float(np.mean(q==0)),'clb_bytes':len(cb),
                 'lift_bytes':liftb,'packed_zlib_bytes':two_bit_zlib(q),'compact_bytes':len(ic),
                 'scale_bytes':4*q.shape[0],
                 'delta_zero_fraction':delta_zero,
                 'direct_matvec_max_abs_diff':absolute,
                 'direct_matvec_rel_diff':relative,
                 'float_to_ternary_rel_diff':qdist,
                 'seconds':round(time.perf_counter()-t0,3)}
            if relative>1e-10 or absolute>1e-10:raise AssertionError('lift operator mismatch')
            rows.append(row)
            for field in total:total[field]+=row[field]
    if not rows:raise RuntimeError('No compatible FFN matrices in safetensors')
    report={'model':MODEL,'revision':sha,'safetensors_sha256':digest,
            'subset':'MLP dense matrices only, not embedding/attention/full model',
            'target_zero':target_zero,
            'status':'REAL WEIGHTS, CPU only; not QMoE parity or full-model quality',
            'total':total,'measured':rows}
    for prefix in ('clb','lift','packed_zlib','compact'):
        total[prefix+'_bits_per_ternary_weight']=8*(total[prefix+'_bytes']+total['scale_bytes'])/total['parameters']
    total['lift_saving_vs_clb_pct']=100*(1-total['lift_bytes']/total['clb_bytes'])
    total['compact_saving_vs_clb_pct']=100*(1-total['compact_bytes']/total['clb_bytes'])
    pathlib.Path(output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'model':MODEL,'revision':sha,'sha256':digest,
                      'matrix_count':len(rows),'total':total},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='calibri-v07.json')
    p.add_argument('--target-zero',type=float,default=.885)
    p.add_argument('--max-matrices',type=int,default=12)
    a=p.parse_args()
    run(a.output,a.target_zero,a.max_matrices)
