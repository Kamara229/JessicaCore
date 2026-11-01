"""CALIBRI v22: TRAIN the native multidimensional weight parameters themselves.

Frozen pretrained teacher Pythia-14m. Train only TT cores (1536 params)
of a physical rank2 TT + immutable packed INT5 residual on real teacher
logits; optional local-output loss. Never train on eval splits. No app changes.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import pathlib
import struct
import zlib
import numpy as np
from geometric_v16 import pack as pack_tt
from geometric_v17 import encode as geo_encode
from packed_native_v19 import split_geometry
from rowcache_v20 import row_cached_geometry
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode

BUNDLE=struct.Struct('<4sIII')


def rebuild_physical(student,previous_blob):
    old_tt,packed5=split_geometry(previous_blob)
    a=student.core.a.detach().cpu().numpy()
    b=student.core.b.detach().cpu().numpy()
    tt=pack_tt(a,b)
    assert len(tt)==len(old_tt)
    payload=tt+packed5
    blob=BUNDLE.pack(b'CG17',len(tt),len(packed5),zlib.crc32(payload))+payload
    assert len(blob)==len(previous_blob)
    return blob


def make_teacher_windows(model,original,tokens,context):
    import torch
    windows=[]
    captures=[]
    def hook(mod,inp,out):
        captures.append((inp[0].detach().cpu().clone().half(),
                         out.detach().cpu().clone().half()))
    handle=original.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0,len(tokens),context):
                ids=tokens[start:start+context][None]
                output=model(input_ids=ids,use_cache=False).logits.detach().cpu().half()
                if len(captures)!=len(windows)+1:raise ValueError('teacher hook missing')
                x,y=captures[-1]
                windows.append({'ids':ids,'logits':output,'input':x,'output':y})
    finally:
        handle.remove()
    return windows


def train_method(model,outer,leaf,orig,base_blob,library,teacher_windows,steps,lr,mode):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(22)
    student=row_cached_geometry(base_blob,orig.bias,library)
    student.core.a.requires_grad_(True)
    student.core.b.requires_grad_(True)
    params=[student.core.a,student.core.b]
    n=sum(p.numel() for p in params)
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.)
    model.requires_grad_(False)
    for p in params:p.requires_grad_(True)
    setattr(outer,leaf,student)
    records=[]
    try:
        for step in range(steps):
            w=teacher_windows[step%len(teacher_windows)]
            optimizer.zero_grad(set_to_none=True)
            output=model(input_ids=w['ids'],use_cache=False).logits
            target=w['logits'].float()
            T=2.0
            kl=F.kl_div(F.log_softmax(output/T,dim=-1),
                        F.softmax(target/T,dim=-1),
                        reduction='sum')*T*T/output.shape[1]
            candidate=student(w['input'].float())
            local=((candidate-w['output'].float())**2).mean() / (
                w['output'].float().square().mean()+1e-9)
            loss=kl+(0.1*local if mode=='kl_plus_local' else 0.)
            if not torch.isfinite(loss):raise ValueError('teacher loss not finite')
            loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            optimizer.step()
            records.append({'step':step,'KL':float(kl.detach()),
                'local_relative_mse':float(local.detach()),
                'objective':float(loss.detach()),'clip_norm':norm})
        trained=rebuild_physical(student,base_blob)
        rebuilt=row_cached_geometry(trained,orig.bias,library)
        x=teacher_windows[0]['input'].float()
        with torch.no_grad():
            max_diff=float((student(x)-rebuilt(x)).abs().max())
        if max_diff>0.0002:raise AssertionError('learned TT core lost on physical serialization')
    finally:
        setattr(outer,leaf,orig)
        model.requires_grad_(False)
    return trained,{'mode':mode,'steps':steps,'trainable_TT_parameters':n,
                    'initial_teacher_KL':records[0]['KL'],
                    'last_train_teacher_KL':records[-1]['KL'],
                    'initial_local_rel_MSE':records[0]['local_relative_mse'],
                    'last_local_rel_MSE':records[-1]['local_relative_mse'],
                    'max_serialized_core_output_diff':max_diff,
                    'training_history':records}


def run(args):
    import torch
    from torch import nn
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from quality import load_token_ids,ppl_eval,module_selection,DATASET,MODEL
    from geometric_v18 import load_new_corpus
    torch.set_num_threads(2);torch.manual_seed(22)
    lib=ct.CDLL(str(pathlib.Path(args.library).resolve()))
    revision=HfApi().model_info(MODEL).sha
    wikitext_rev=HfApi().dataset_info(DATASET).sha
    tiny_rev=HfApi().dataset_info('roneneldan/TinyStories').sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,torch_dtype=torch.float32)
    model.eval()
    name=sorted(module_selection(model))[0]
    orig=model.get_submodule(name)
    parent,leaf=name.rsplit('.',1)
    outer=model.get_submodule(parent)
    w=orig.weight.detach().cpu().numpy().copy().astype(np.float32)
    # Hyperparameters/predeclared geometry are fixed BEFORE evaluation:
    # Geo-small: m2=16,n2=32,r=2,5-bit residual from v17/v18.
    base=geo_encode(w,2,16,32,5)
    train=load_token_ids(tok,'train',wikitext_rev,args.train_tokens)
    dev=load_token_ids(tok,'validation',wikitext_rev,args.dev_tokens)
    tiny,_=load_new_corpus(tok,args.tiny_tokens,tiny_rev)
    teacher=make_teacher_windows(model,orig,train,args.context)
    reference_wikitext=ppl_eval(model,dev,args.context)['ppl']
    reference_tiny=ppl_eval(model,tiny,args.context)['ppl']
    learned={}
    traces={}
    for method in ('kl_only','kl_plus_local'):
        learned[method],traces[method]=train_method(model,outer,leaf,orig,base,
             lib,teacher,args.steps,args.lr,method)
    fixed6=fixed_encode(w,6)
    def make_int6():
        lin=nn.Linear(w.shape[1],w.shape[0],bias=orig.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(fixed_decode(fixed6)))
            if orig.bias is not None:lin.bias.copy_(orig.bias)
        return lin
    entries=[('original',None,orig),('INT6',fixed6,make_int6()),
             ('geo_untrained',base,row_cached_geometry(base,orig.bias,lib))]
    for method,blob in learned.items():
        entries.append(('geo_trained_'+method,blob,row_cached_geometry(blob,orig.bias,lib)))
    results=[]
    for key,blob,layer in entries:
        setattr(outer,leaf,layer)
        with torch.no_grad():
            d=ppl_eval(model,dev,args.context)['ppl']
            t=ppl_eval(model,tiny,args.context)['ppl']
        setattr(outer,leaf,orig)
        results.append({'name':key,
          'physical_weight_bytes':len(blob) if blob is not None else 2*w.size,
          'wikitext_dev_ppl':d,'wikitext_ratio_to_original':d/reference_wikitext,
          'tinystories_eval_ppl':t,'tinystories_ratio_to_original':t/reference_tiny})
    report={'experiment':'CALIBRI_v22_direct_training_of_multidimensional_TT_cores',
     'model':MODEL,'model_revision':revision,'matrix':name,'shape':list(w.shape),
     'teacher_train_tokens':len(train),'teacher_train_sha256':hashlib.sha256(train.numpy().tobytes()).hexdigest(),
     'Wikitext_dev_tokens':len(dev),'Wikitext_dev_sha256':hashlib.sha256(dev.numpy().tobytes()).hexdigest(),
     'TinyStories_eval_tokens':len(tiny),'TinyStories_eval_sha256':hashlib.sha256(tiny.numpy().tobytes()).hexdigest(),
     'data_revisions':{'WikiText2':wikitext_rev,'TinyStories':tiny_rev},
     'teacher_train_windows':len(teacher),'optimizer':'AdamW with fixed physical FP16 STE for TT cores',
     'initial_geometry':'TT rank 2 (m2=16,n2=32) + FIXED packed INT5 residual',
     'physical_initial_bytes':len(base),'trained_physical_bytes':{k:len(v) for k,v in learned.items()},
     'lr':args.lr,'steps_per_variant':args.steps,'methods':traces,
     'scores':results,'constraints':[
      'Post-hoc TT core distillation on REAL pretrained model; NOT training a new language model from scratch',
      'One 128x512 dense Pythia projection; 1536 trainable parameters, residual stays frozen',
      'No evaluation split was used for training, yet these splits appeared in earlier research and are not new holdouts',
      'No real Switch MoE routes, QMoE checkpoint, or fully compressed model',
      'The packed INT5 C operator uses row-wise float scratch; no GPU/ARM execution claims',
      'Teacher KL and local MSE are established losses and TT is prior art; no novelty claim',
      'No automatic deployment to Jessica or main branch'] }
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--library',default='research/calibri/libpacked5_v22.so')
    p.add_argument('--output',default='results/calibri_v22_native_training.json')
    p.add_argument('--train-tokens',type=int,default=2048)
    p.add_argument('--dev-tokens',type=int,default=1024)
    p.add_argument('--tiny-tokens',type=int,default=2048)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--lr',type=float,default=0.0005)
    run(p.parse_args())
