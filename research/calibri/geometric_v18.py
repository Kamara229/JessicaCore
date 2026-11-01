"""CALIBRI v18: PREDECLARED and unchanged v17 candidate transfer to new corpus.
Do NOT select methods on this dataset; this is a cross-domain confirmatory pilot.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib
import numpy as np
from geometric_v17 import encode as geo_encode,native_layer
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode


def load_new_corpus(tokenizer,max_tokens,revision):
    import torch
    from datasets import load_dataset
    ds=load_dataset('roneneldan/TinyStories',split='validation',revision=revision)
    text=[];total_chars=0
    for row in ds:
        item=str(row['text']).strip()
        if not item:continue
        text.append(item)
        total_chars+=len(item)
        if total_chars>max_tokens*12:break
    ids=tokenizer('\n\n'.join(text),return_tensors='pt',add_special_tokens=False).input_ids[0,:max_tokens]
    if len(ids)<max_tokens:raise ValueError('insufficient TinyStories tokens')
    return ids,len(text)


def run(args):
    import torch
    from torch import nn
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from quality import MODEL,ppl_eval,module_selection
    torch.set_num_threads(2);torch.manual_seed(18)
    model_rev=HfApi().model_info(MODEL).sha
    corpus_rev=HfApi().dataset_info('roneneldan/TinyStories').sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=model_rev,torch_dtype=torch.float32)
    model.eval()
    name=sorted(module_selection(model))[0]
    source=model.get_submodule(name)
    parent,leaf=name.rsplit('.',1)
    outer=model.get_submodule(parent)
    w=source.weight.detach().cpu().numpy().copy().astype(np.float32)
    # All candidate names and TT topology/rank/precision were fixed in v17
    # using only WikiText-2 development perplexity, BEFORE this new-corpus run.
    manifest=(
      ('INT5',None,None,None,5),
      ('INT6',None,None,None,6),
      ('Geo-small',16,32,2,5),
      ('Geo-balanced',8,16,4,5),
      ('Geo-accuracy',8,32,8,5),
    )
    ids,num_examples=load_new_corpus(tok,args.tokens,corpus_rev)
    baseline=ppl_eval(model,ids,args.context)['ppl']
    results=[]
    for label,m2,n2,rank,bits in manifest:
        if m2 is None:
            blob=fixed_encode(w,bits)
            layer=nn.Linear(w.shape[1],w.shape[0],bias=source.bias is not None)
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(fixed_decode(blob)))
                if source.bias is not None:layer.bias.copy_(source.bias)
        else:
            blob=geo_encode(w,rank,m2,n2,bits)
            layer=native_layer(blob,source.bias)
        setattr(outer,leaf,layer)
        with torch.no_grad():ppl=ppl_eval(model,ids,args.context)['ppl']
        setattr(outer,leaf,source)
        results.append({'label':label,'bytes':len(blob),'test_ppl':ppl,
             'ppl_ratio_to_original':ppl/baseline,'delta_vs_original_pct':100*(ppl/baseline-1),
             'physical_storage_ratio_to_fp16':len(blob)/(2*w.size),
             'configuration':{'m2':m2,'n2':n2,'rank':rank,'residual_bits':bits},
             'execution':'native TT + dequantized dense residual' if m2 else 'restored dense low-bit'})
    result={'experiment':'CALIBRI_v18_frozen_v17_geometries_new_corpus',
       'model':MODEL,'model_revision':model_rev,'new_evaluation_dataset':'roneneldan/TinyStories',
       'dataset_revision':corpus_rev,'split':'validation','requested_tokens':args.tokens,
       'evaluated_tokens':len(ids)-len(ids)//args.context,'examples_concatenated':num_examples,
       'token_sha256':hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
       'model_projection':name,'shape':list(w.shape),'fp16_baseline_weight_bytes':2*w.size,
       'original_test_ppl_new_corpus':baseline,
       'v17_predeclared_candidates_not_selected_using_new_corpus':True,'results':results,
       'limitations':['Only one trained dense Pythia MLP matrix, not MoE or QMoE',
         'A fresh corpus, not a novel-compression proof; domain differs from WikiText',
         'No tuning/selection based on TinyStories labels or perplexity',
         'Physical bundle includes headers, TT FP16, residual low-bit stream and scales',
         'Native TT operator plus DENSE dequantized residual; no fused packed inference',
         'No ARM/GPU speed, peak memory or whole-model compressed runtime',
         'TinyStories was not used in prior CALIBRI experiment, but prior runs saw WikiText-2 test']}
    out=pathlib.Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/calibri_v18_new_corpus.json')
    p.add_argument('--tokens',type=int,default=8192)
    p.add_argument('--context',type=int,default=128)
    run(p.parse_args())
