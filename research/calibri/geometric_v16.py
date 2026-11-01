"""CALIBRI v16: physically serialized 2-core TT matrices, native tensor contraction,
and train-split-only teacher-logit recovery. Research/CI only; no app deployment.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,struct,time,zlib
import numpy as np
from scipy.linalg import svd
from fusion_v13 import _pack_q,_unpack_q,_quant
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode

HEADER=struct.Struct('<4sHHHHBBII')  # magic, m1,m2,n1,n2,rank,mode,payload_bytes,crc


def reconstruct(g1,g2):
    return np.einsum('acr,rkd->akcd',g1,g2,optimize=True).reshape(
        g1.shape[0]*g2.shape[1],g1.shape[1]*g2.shape[2]).astype(np.float32)


def tt_svd(w,rank,m2=8,n2=32):
    w=np.asarray(w,np.float32);m,n=w.shape
    if m%m2 or n%n2 or rank<1 or rank>min(m//m2*(n//n2),m2*n2) or rank>255:
        raise ValueError('incompatible TT dimensions')
    m1,n1=m//m2,n//n2
    a=w.reshape(m1,m2,n1,n2).transpose(0,2,1,3).reshape(m1*n1,m2*n2)
    u,s,vt=svd(a,full_matrices=False,lapack_driver='gesdd')
    root=np.sqrt(s[:rank])
    return (u[:,:rank]*root).reshape(m1,n1,rank).astype(np.float32),(
        root[:,None]*vt[:rank]).reshape(rank,m2,n2).astype(np.float32)


def quant_residual(w,g1,g2):
    g1=g1.astype('<f2').astype(np.float32)
    g2=g2.astype('<f2').astype(np.float32)
    return _quant(np.asarray(w,np.float32)-reconstruct(g1,g2),4)


def pack(g1,g2,q=None,scales=None):
    g1=np.asarray(g1,np.float32);g2=np.asarray(g2,np.float32)
    if g1.ndim!=3 or g2.ndim!=3 or g1.shape[-1]!=g2.shape[0]:
        raise ValueError('TT shapes mismatch')
    m1,n1,r=g1.shape;_,m2,n2=g2.shape
    if not all(0<d<=65535 for d in (m1,n1,m2,n2)) or not 1<=r<=255:
        raise ValueError('TT dims/rank invalid')
    a=g1.astype('<f2');b=g2.astype('<f2')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite TT cores')
    mode=int(q is not None)
    payload=a.tobytes()+b.tobytes()
    if mode:
        q=np.asarray(q,np.int8);scales=np.asarray(scales,'<f2')
        if (q.shape!=(m1*m2,n1*n2) or scales.shape!=(m1*m2,)
                or np.any(np.abs(q)>7) or not np.isfinite(scales).all()):
            raise ValueError('quantized residual invalid')
        payload+=_pack_q(q,4)+scales.tobytes()
    elif scales is not None:raise ValueError('orphan scales')
    return HEADER.pack(b'CG16',m1,m2,n1,n2,r,mode,len(payload),zlib.crc32(payload))+payload


def unpack(blob):
    if len(blob)<HEADER.size:raise ValueError('short TT block')
    magic,m1,m2,n1,n2,r,mode,nbytes,crc=HEADER.unpack_from(blob)
    if magic!=b'CG16' or not all((m1,m2,n1,n2)) or r<1 or mode not in (0,1):
        raise ValueError('bad TT header')
    corebytes=2*r*(m1*n1+m2*n2)
    n=m1*m2*n1*n2
    expected=corebytes+((n+1)//2+2*m1*m2 if mode else 0)
    data=blob[HEADER.size:]
    if nbytes!=expected or len(data)!=expected or zlib.crc32(data)!=crc:
        raise ValueError('TT length/CRC mismatch')
    p=2*m1*n1*r
    a=np.frombuffer(data,'<f2',count=m1*n1*r).astype(np.float32).reshape(m1,n1,r)
    b=np.frombuffer(data,'<f2',count=r*m2*n2,offset=p).astype(np.float32).reshape(r,m2,n2)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('bad TT values')
    residual=None
    if mode:
        count=(n+1)//2
        q=_unpack_q(data[corebytes:corebytes+count],n,4).reshape(m1*m2,n1*n2)
        s=np.frombuffer(data,'<f2',count=m1*m2,offset=corebytes+count).astype(np.float32)
        if not np.isfinite(s).all():raise ValueError('bad residual scale')
        residual=q*s[:,None]
    return a,b,residual


def from_weight(w,rank,residual):
    a,b=tt_svd(w,rank)
    if not residual:return pack(a,b)
    q,s=quant_residual(w,a,b)
    return pack(a,b,q,s)


def make_layer(blob,bias=None,learnable=False):
    import torch
    from torch import nn
    a,b,residual=unpack(blob)
    class TensorCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=nn.Parameter(torch.from_numpy(a.copy()),requires_grad=learnable)
            self.b=nn.Parameter(torch.from_numpy(b.copy()),requires_grad=learnable)
            self.register_buffer('residual',None if residual is None else torch.from_numpy(residual.copy()))
            self.register_buffer('bias',None if bias is None else bias.detach().clone())
        def forward(self,x):
            # Straight-through FP16 rounding: train FP32 factors, execute stored FP16.
            a=self.a+(self.a.half().float()-self.a).detach()
            b=self.b+(self.b.half().float()-self.b).detach()
            xx=x.reshape(-1,a.shape[1],b.shape[2])
            z=torch.einsum('bcd,rkd->brkc',xx,b)
            y=torch.einsum('acr,brkc->bak',a,z).reshape(*x.shape[:-1],-1)
            if self.residual is not None:
                y=y+torch.nn.functional.linear(x,self.residual)
            if self.bias is not None:y=y+self.bias
            return y
    return TensorCore()


def smoke():
    import torch
    rng=np.random.default_rng(16)
    w=rng.normal(size=(16,24)).astype(np.float32)
    x=rng.normal(size=(5,24)).astype(np.float32)
    for rank in (2,4):
        for residual in (False,True):
            a,b=tt_svd(w,rank,m2=4,n2=6)
            if residual:
                q,s=quant_residual(w,a,b);blob=pack(a,b,q,s)
            else:blob=pack(a,b)
            u,v,r=unpack(blob)
            expected=HEADER.size+2*rank*(4*4+4*6)+(16*24+1)//2*residual+2*16*residual
            assert len(blob)==expected,(len(blob),expected)
            y=make_layer(blob)(torch.from_numpy(x)).detach().numpy()
            target=x@(reconstruct(u,v)+(r if r is not None else 0)).T
            assert np.allclose(y,target,atol=3e-5,rtol=3e-5),np.max(np.abs(y-target))
            tampered=bytearray(blob);tampered[-1]^=1
            try:unpack(tampered);raise AssertionError('accepted corruption')
            except ValueError:pass
    print('TT physical and DIRECT tensor contraction equivalence + CRC PASS')


def run(args):
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi
    from quality import load_token_ids,ppl_eval,module_selection,MODEL,DATASET
    torch.set_num_threads(2);torch.manual_seed(16)
    rev=HfApi().model_info(MODEL).sha
    d_rev=HfApi().dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,torch_dtype=torch.float32)
    model.eval()
    modules=module_selection(model)
    name=sorted(modules)[0]
    orig=modules[name]
    w=orig.weight.detach().numpy().copy().astype(np.float32)
    parent_name,leaf=name.rsplit('.',1)
    parent=model.get_submodule(parent_name)
    train=load_token_ids(tok,'train',d_rev,args.train_tokens)
    dev=load_token_ids(tok,'validation',d_rev,args.dev_tokens)
    test=load_token_ids(tok,'test',d_rev,args.test_tokens)
    teacher=[];windows=[]
    with torch.no_grad():
        for start in range(0,len(train),args.train_context):
            tokens=train[start:start+args.train_context][None]
            windows.append(tokens)
            teacher.append(model(input_ids=tokens,use_cache=False).logits.detach().clone())
    original_dev=ppl_eval(model,dev,args.eval_context)['ppl']
    blobs={}
    for r in (8,16,32):blobs[f'tt{r}']=from_weight(w,r,False)
    for r in (8,16):blobs[f'tt{r}_q4']=from_weight(w,r,True)
    for b in (4,5,6,8):blobs[f'int{b}']=fixed_encode(w,b)
    # TRAIN-only distillation; fixed quantized residual; no other parameters updated.
    student=make_layer(blobs['tt8_q4'],orig.bias,learnable=True)
    model.requires_grad_(False)
    setattr(parent,leaf,student)
    opt=torch.optim.AdamW([student.a,student.b],lr=args.lr,weight_decay=0.)
    history=[]
    for step in range(args.steps):
        idx=step%len(windows)
        opt.zero_grad(set_to_none=True)
        y=model(input_ids=windows[idx],use_cache=False).logits
        t=teacher[idx]
        T=2.
        loss=F.kl_div(F.log_softmax(y/T,dim=-1),
                      F.softmax(t/T,dim=-1),reduction='sum')*T*T/y.shape[1]
        if not torch.isfinite(loss):raise ValueError('nonfinite teacher KL')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([student.a,student.b],1.)
        opt.step()
        history.append(float(loss.detach()))
    q,s=quant_residual(w,*tt_svd(w,8))
    reference_residual=unpack(blobs['tt8_q4'])[2]
    assert np.allclose(reference_residual,q.astype(np.float32)*s.astype(np.float32)[:,None])
    blobs['tt8_q4_distilled']=pack(student.a.detach().numpy(),student.b.detach().numpy(),q,s)
    setattr(parent,leaf,orig)
    model.eval()
    # Microbenchmark only pure TT against dense FP32 CPU (batch=64).
    inp=torch.from_numpy(np.random.default_rng(16).normal(size=(64,w.shape[1])).astype(np.float32))
    weight=torch.from_numpy(w)
    direct=make_layer(blobs['tt8'])
    with torch.no_grad():
        for _ in range(5):direct(inp);F.linear(inp,weight)
        tt_times=[];dense_times=[]
        for _ in range(20):
            t=time.perf_counter();direct(inp);tt_times.append(1000*(time.perf_counter()-t))
            t=time.perf_counter();F.linear(inp,weight);dense_times.append(1000*(time.perf_counter()-t))
    def layer_for(key):
        blob=blobs[key]
        if key.startswith('tt'):return make_layer(blob,orig.bias)
        obj=nn.Linear(w.shape[1],w.shape[0],bias=orig.bias is not None)
        with torch.no_grad():
            obj.weight.copy_(torch.from_numpy(fixed_decode(blob)))
            if orig.bias is not None:obj.bias.copy_(orig.bias)
        return obj
    variants=[]
    for key,blob in blobs.items():
        setattr(parent,leaf,layer_for(key))
        with torch.no_grad():v=ppl_eval(model,dev,args.eval_context)['ppl']
        variants.append({'name':key,'bytes':len(blob),'dev_ppl':v,'dev_ppl_ratio':v/original_dev,
          'execution':'native TT cores with dense residual' if 'q4' in key and key.startswith('tt')
                      else ('native TT only' if key.startswith('tt') else 'restored dense INT')})
        setattr(parent,leaf,orig)
    budget=int(len(blobs['int4'])*1.5);cap=original_dev*1.01
    eligible=[a for a in variants if a['bytes']<=budget and a['dev_ppl']<=cap]
    selected=min(eligible,key=lambda r:(r['bytes'],r['dev_ppl'])) if eligible else None
    reference_test=ppl_eval(model,test,args.eval_context)['ppl']
    test_keys=['int4','int6','tt8_q4','tt8_q4_distilled']
    if selected:test_keys.append(selected['name'])
    scores={}
    for key in dict.fromkeys(test_keys):
        setattr(parent,leaf,layer_for(key))
        with torch.no_grad():scores[key]=ppl_eval(model,test,args.eval_context)['ppl']
        setattr(parent,leaf,orig)
    result={'experiment':'CALIBRI_v16_multidimensional_tensor_weights',
      'model':MODEL,'revision':rev,'dataset_revision':d_rev,'module':name,
      'matrix_shape':list(w.shape),'tt_indexing':'(m1,m2,n1,n2)=(16,8,16,32)',
      'fp16_reference_weight_bytes':int(2*w.size),
      'train_tokens':len(train),'dev_tokens':len(dev),'test_tokens':len(test),
      'token_sha256':{kind:hashlib.sha256(ids.numpy().tobytes()).hexdigest()
         for kind,ids in [('train',train),('dev',dev),('test',test)]},
      'baseline_dev_ppl':original_dev,'baseline_test_ppl':reference_test,
      'byte_budget_1_5x_int4':budget,'development_quality_cap_1pct':cap,
      'approved_smallest':selected['name'] if selected else 'NO_APPROVED_VARIANT',
      'variants':variants,'predeclared_test_ppl':scores,
      'distillation':{'steps':args.steps,'lr':args.lr,'initial_teacher_KL':history[0],
                      'final_teacher_KL':history[-1],'train_split_only':True},
      'cpu_microbenchmark_ms':{'direct_tt8_p50':float(np.median(tt_times)),
          'direct_tt8_p95':float(np.percentile(tt_times,95)),
          'dense_fp32_p50':float(np.median(dense_times)),
          'dense_fp32_p95':float(np.percentile(dense_times,95))},
      'limitations':['One 65,536-parameter trained dense matrix, not MoE/QMoE',
         'Tensor train and knowledge distillation are prior art, not new discoveries',
         'Full LM reference unchanged except one substituted projection',
         'TT-only has direct tensor contraction, TT+INT4 uses dense residual buffer',
         'CPU Python einsum timing is not ARM/GPU and not packed low-bit execution',
         'No end-to-end package, inference throughput, thermal, RSS, or whole-model bytes',
         'No tuning on held-out test; test split seen previously and is exploratory']}
    out=pathlib.Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',default='results/calibri_v16_geometric.json')
    p.add_argument('--train-tokens',type=int,default=128)
    p.add_argument('--train-context',type=int,default=64)
    p.add_argument('--dev-tokens',type=int,default=1024)
    p.add_argument('--test-tokens',type=int,default=1024)
    p.add_argument('--eval-context',type=int,default=128)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--lr',type=float,default=.001)
    args=p.parse_args()
    smoke() if args.smoke else run(args)
