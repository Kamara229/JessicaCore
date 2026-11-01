"""CALIBRI v25 — reproducibility audit for pretrained routed Switch MoE.

Prior v24 runs with SAME checkpoint and token SHA gave nonidentical KL.
This experiment tests whether original BF16 model, router and weight restore
are bitwise repeatable within one CI process and records checkpoint-loading
mismatches. Not an innovation/quality benchmark.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,platform
import numpy as np
import torch
from transformers import AutoTokenizer,SwitchTransformersForConditionalGeneration
from huggingface_hub import HfApi
from moe_allocation_v24 import (
 MODEL,DATASET,EXPERTS,PINNED_SWITCH_SHA,NUM_EXPERTS,
 token_data,load_new_corpus,make_windows,identify_next_router,evaluation
)
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode

def hash_bytes(obj):
    return hashlib.sha256(obj).hexdigest()

def fingerprint(t):
    return hash_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())

def audit_repeat(model,windows,router,repeat):
    observed=[]
    for k in range(repeat):
        logits,masks=evaluation(model,windows,router)
        observed.append({
          'repeat':k,
          'logits_sha256':hash_bytes(b''.join(o.numpy().tobytes() for o in logits)),
          'router_mask_sha256':hash_bytes(b''.join(m.numpy().tobytes() for m in masks)),
          'first_sequence_logits_checksum':float(logits[0].double().sum()),
          'logit_samples':int(sum(o.numel() for o in logits)),
        })
    return {'runs':observed,
      'all_logits_bitwise_identical':len({x['logits_sha256'] for x in observed})==1,
      'all_router_masks_bitwise_identical':
          len({x['router_mask_sha256'] for x in observed})==1}

def run(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(25)
    api=HfApi()
    revision=api.model_info(MODEL).sha
    if revision!=PINNED_SWITCH_SHA:
        raise ValueError('source checkpoint revision changed')
    wiki_revision=api.dataset_info(DATASET).sha
    tiny_revision=api.dataset_info('roneneldan/TinyStories').sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=PINNED_SWITCH_SHA)
    model,loading=SwitchTransformersForConditionalGeneration.from_pretrained(
       MODEL,revision=PINNED_SWITCH_SHA,torch_dtype=torch.bfloat16,
       low_cpu_mem_usage=True,output_loading_info=True)
    model.eval()
    names={n:m for n,m in model.named_modules() if n.endswith('.mlp.router')}
    routers=[m for n,m in names.items()]
    if not routers:raise ValueError('no pretrained sparse routers')
    router=identify_next_router(model)
    if not router:raise ValueError('missing downstream decoder router')
    for r in routers:
        if r.training:raise AssertionError('router training unexpectedly true')
    wiki=token_data(tok,wiki_revision,'validation',args.tokens)
    tiny,_=load_new_corpus(tok,args.tokens,tiny_revision)
    windows={'WikiText2_validation':list(make_windows(wiki,args.encoder_len,args.decoder_len,tok.pad_token_id)),
             'TinyStories_validation':list(make_windows(tiny,args.encoder_len,args.decoder_len,tok.pad_token_id))}
    fingerprints={}
    for name in ('decoder.block.1.layer.2.mlp.router.classifier',
                 f'{EXPERTS}.expert_0.wi',f'{EXPERTS}.expert_0.wo'):
        module=model.get_submodule(name)
        fingerprints[name]=fingerprint(module.weight)
    originals={}
    for i in range(NUM_EXPERTS):
        m=model.get_submodule(f'{EXPERTS}.expert_{i}')
        originals[i]=(m.wi.weight.detach().clone(),m.wo.weight.detach().clone())
    report={'original_eval':{},'zero_router_jitter_eval':{},'int5_eval':{},'restored_eval':{}}
    for corpus,win in windows.items():
        report['original_eval'][corpus]=audit_repeat(model,win,router,args.repeats)
    jitters=[float(getattr(m,'jitter_noise',float('nan'))) for m in routers]
    for m in routers:m.jitter_noise=0.
    for corpus,win in windows.items():
        report['zero_router_jitter_eval'][corpus]=audit_repeat(model,win,router,args.repeats)
    # Keep the exact same pinned-physical quantization mechanics as v23/v24.
    try:
        with torch.no_grad():
            for i in range(NUM_EXPERTS):
                m=model.get_submodule(f'{EXPERTS}.expert_{i}')
                for layer,w in ((m.wi,originals[i][0]),(m.wo,originals[i][1])):
                    restored=fixed_decode(fixed_encode(w.float().numpy(),5))
                    layer.weight.copy_(torch.from_numpy(restored))
        for corpus,win in windows.items():
            report['int5_eval'][corpus]=audit_repeat(model,win,router,args.repeats)
    finally:
        with torch.no_grad():
            for i in range(NUM_EXPERTS):
                m=model.get_submodule(f'{EXPERTS}.expert_{i}')
                m.wi.weight.copy_(originals[i][0])
                m.wo.weight.copy_(originals[i][1])
    same=True
    for i in range(NUM_EXPERTS):
        m=model.get_submodule(f'{EXPERTS}.expert_{i}')
        same &= torch.equal(m.wi.weight,originals[i][0])
        same &= torch.equal(m.wo.weight,originals[i][1])
    if not same:raise AssertionError('incomplete expert restoration')
    for corpus,win in windows.items():
        report['restored_eval'][corpus]=audit_repeat(model,win,router,args.repeats)
    # The same source model with explicit jitter disabled MUST not change
    # after copying original bits back, except potential unsupported CPU kernels.
    before={k:v['runs'][0]['logits_sha256'] for k,v in report['zero_router_jitter_eval'].items()}
    after={k:v['runs'][0]['logits_sha256'] for k,v in report['restored_eval'].items()}
    result={
      'experiment':'CALIBRI_v25_true_MoE_reproducibility_and_router_stability',
      'source_model':MODEL,'model_revision':revision,'dataset_revisions':
         {'WikiText2':wiki_revision,'TinyStories':tiny_revision},
      'evaluation_token_sha256':
         {'WikiText2_validation':hash_bytes(wiki.numpy().tobytes()),
          'TinyStories_validation':hash_bytes(tiny.numpy().tobytes())},
      'model_library':'transformers==4.46.3','torch_version':torch.__version__,
      'cpu_arch':platform.machine(),'torch_threads':torch.get_num_threads(),
      'torch_mkldnn_enabled':torch.backends.mkldnn.enabled,
      'router_layer_count':len(routers),'router_jitter_before':jitters,
      'router_was_eval_mode':True,
      'loading_info':{
        'missing_key_count':len(loading.get('missing_keys',[])),
        'unexpected_key_count':len(loading.get('unexpected_keys',[])),
        'mismatched_key_count':len(loading.get('mismatched_keys',[])),
        'missing_keys':loading.get('missing_keys',[]),
        'unexpected_keys':loading.get('unexpected_keys',[]),
        'error_msgs':loading.get('error_msgs',[])},
      'source_key_weight_SHA256':fingerprints,
      'all_8_expert_pair_weights_restored_exactly':bool(same),
      'original_vs_restored_logits_bitwise_identical_with_zero_jitter':before==after,
      'traces':report,
      'limits':['Only within-run replay; repeat runs across CPUs for between-run reproducibility',
       'Router disable-jitter is an audit control only; no training or production',
       'The underlying samples are tiny and previously used; not task quality',
       'No compressed speed or size gain asserted; data comes from public pretrained Switch',
       'No main/Render/Supabase/APK or user data changed']
    }
    out=pathlib.Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    summary={k:result[k] for k in ('experiment','model_revision','router_layer_count',
              'router_jitter_before','loading_info',
              'all_8_expert_pair_weights_restored_exactly',
              'original_vs_restored_logits_bitwise_identical_with_zero_jitter')}
    summary['repeat_identical']={
       category:{corpus:{
           'logits':v['all_logits_bitwise_identical'],
           'router_masks':v['all_router_masks_bitwise_identical']}
          for corpus,v in contexts.items()}
       for category,contexts in report.items()}
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/calibri_v25_moe_replay.json')
    p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--encoder-len',type=int,default=64)
    p.add_argument('--decoder-len',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    run(p.parse_args())
