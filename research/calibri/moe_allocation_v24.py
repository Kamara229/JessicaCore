"""CALIBRI v24 — exact allocation of 5/6 bit expert precision on REAL MoE routing.

Input v23 routing statistics are a frozen, independently measured manifest
from actual pretrained google/switch-base-8, run 41339418910.
No data from held-out evaluation is used for allocation.
All eight paired wi/wo quantized from original pretrained BF16.
Return NO_FEASIBLE rather than silently violating rare-expert cap.
"""
from __future__ import annotations
import argparse
from itertools import product
import hashlib
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode
from routed_moe_v23 import MODEL,DATASET,EXPERTS,token_data,slices,model_teacher_kl
from geometric_v18 import load_new_corpus

PINNED_SWITCH_SHA="92fe2d22b024d9937146fe097ba3d3a7ba146e1b"
V23_EXPERIMENT_ID="https://github.com/Kamara229/JessicaCore/actions/runs/41339418910"
V23_JSON_SHA256="ecbc5daeceeb71d1fe3ab1d78dbce6807acce61f16253d28e3db4d04786e6f27"
TRAIN_N=(19,19,12,29,14,12,12,11)
VAL_N=(8,13,9,19,21,17,20,21)
TRAIN5=(.005900459978296473,.00011852018891137593,.0033864663661671134,
        .00563585826922069,.007554948417898813,.004828111615978336,
        .006225233105560058,.0036746129716653197)
TRAIN6=(.001389530375923571,.000010545375779251608,.0007965425430139849,
        .0013876852314271466,.0018515740578048758,.0011297915641769198,
        .001549808644610631,.0007979494215760508)
VAL5=(.004418386338566552,.00011682446999442328,.003036882529753518,
      .0044279399184171345,.005483835455463475,.004220387572231853,
      .006072526560079468,.003630814601408967)
VAL6=(.0010963592555665183,.000010523537666920264,.0007632759318760839,
      .0010385855271599412,.0013985711184823489,.0009265884638619592,
      .0015899328303105112,.0008516778411950517)
PAIR_BYTES={5:2956836,6:3546660}
NUM_EXPERTS=8

def weighted_proxy(bits,phase="train"):
    if phase=="train":count=TRAIN_N;errs={5:TRAIN5,6:TRAIN6}
    elif phase=="validation":count=VAL_N;errs={5:VAL5,6:VAL6}
    else:raise ValueError("phase")
    return sum(c*errs[b][i] for i,(c,b) in enumerate(zip(count,bits)))/sum(count)

def storage_bytes(bits):
    return sum(PAIR_BYTES[b] for b in bits)

def exact_allocation(budget_bytes,cap=None):
    best=None
    evaluated=0
    feasible=0
    for bits in product((5,6),repeat=NUM_EXPERTS):
        evaluated+=1
        physical=storage_bytes(bits)
        if physical>budget_bytes:continue
        if cap is not None and any((TRAIN5 if b==5 else TRAIN6)[i]>cap
                                    for i,b in enumerate(bits)):
            continue
        feasible+=1
        objective=weighted_proxy(bits,"train")
        ranking=(objective,physical,bits)
        if best is None or ranking<best[0]:
            best=(ranking,bits)
    return {'budget':budget_bytes,'rare_training_MSE_cap':cap,
            'combinations_searched':evaluated,'feasible':feasible,
            'bits':list(best[1]) if best is not None else None,
            'train_routed_proxy':best[0][0] if best is not None else None}

def design():
    all6=(6,)*NUM_EXPERTS
    budget90=int(storage_bytes(all6)*.90)
    budget95=int(storage_bytes(all6)*.95)
    allocation90=exact_allocation(budget90,None)
    allocation90_cap=exact_allocation(budget90,.005)
    allocation95=exact_allocation(budget95,.005)
    if allocation90['bits'] is None:raise AssertionError("90% cannot allocate")
    if allocation95['bits'] is None:raise AssertionError("95% guarded cannot allocate")
    if allocation90_cap['bits'] is not None:
        raise AssertionError("expected 90% with strict cap to fail")
    # Same number of INT6 experts as optimized 90% allocation, but picks
    # experts based ONLY on training frequency (no sensitivity).
    num6=allocation90['bits'].count(6)
    freq=sorted(range(NUM_EXPERTS),key=lambda i:(-TRAIN_N[i],i))[:num6]
    freq_bits=[6 if i in freq else 5 for i in range(NUM_EXPERTS)]
    first=[6 if i<num6 else 5 for i in range(NUM_EXPERTS)]
    return {
      'int5_uniform':[5]*NUM_EXPERTS,
      'int6_uniform':[6]*NUM_EXPERTS,
      '90pct_training_routed_optimal':allocation90['bits'],
      '90pct_training_frequency_only':freq_bits,
      '90pct_fixed_expert_order':first,
      '95pct_rare_guarded_optimal':allocation95['bits'],
    },{'budget90':budget90,'budget95':budget95,
       'allocation90':allocation90,'guarded90':allocation90_cap,
       'guarded95':allocation95,
       'selection_uses_only_real_routed_TRAIN_error':True}

def smoke():
    designs,meta=design()
    assert meta['allocation90']['feasible']>0
    assert meta['guarded90']['feasible']==0
    assert meta['guarded95']['feasible']>0
    assert sum(designs['90pct_training_routed_optimal'])==sum(designs['90pct_training_frequency_only'])
    for values in designs.values():assert len(values)==NUM_EXPERTS
    assert len(set(storage_bytes(v) for v in (
        designs['90pct_training_routed_optimal'],
        designs['90pct_training_frequency_only'])))==1
    print(json.dumps({'result':'PASS','portfolios':designs,
                      'guarded90':'NO_FEASIBLE','budget':meta},indent=2),flush=True)

def make_windows(ids,enc_len,dec_len,pad_id):
    for input_ids,_,prev in slices(ids,enc_len,dec_len):
        decoder=torch.cat([torch.full((1,1),pad_id,dtype=torch.long),prev],dim=1)
        yield input_ids,decoder

def identify_next_router(model):
    names=sorted(n for n,_ in model.named_modules()
       if n.startswith("decoder.block.") and n.endswith(".mlp.router"))
    candidate=[n for n in names if int(n.split(".")[2])>1]
    return candidate[0] if candidate else None

def evaluation(model,windows,record_next_router):
    masks=[]
    hook=None
    if record_next_router:
        def trace(mod,args,out):
            if not isinstance(out,(tuple,list)) or not out:
                raise ValueError("router output contract changed")
            m=out[0].detach().float().cpu()
            masks.append(m)
        hook=model.get_submodule(record_next_router).register_forward_hook(trace)
    logits=[]
    try:
        with torch.inference_mode():
            for enc,dec in windows:
                logits.append(model(input_ids=enc,decoder_input_ids=dec,
                     use_cache=False,return_dict=True).logits.detach().float().cpu())
    finally:
        if hook is not None:hook.remove()
    if record_next_router and len(masks)!=len(windows):
        raise ValueError('missing router trace')
    return logits,masks

def compare_to_teacher(teacher,student,mask_teacher,mask_student):
    if len(teacher)!=len(student):raise ValueError("different contexts")
    kls=[];agreements=[];router_match=[];router_bit_match=[]
    for a,b in zip(teacher,student):
        kls.append(model_teacher_kl(a,b))
        agreements.append(float((a.argmax(dim=-1)==b.argmax(dim=-1)).float().mean()))
    if len(mask_teacher)!=len(mask_student):
        raise ValueError('mismatched count of downstream router traces')
    for a,b in zip(mask_teacher,mask_student):
        if a.shape!=b.shape:raise ValueError('next-router shape mismatch')
        # Comparing individual one-hot elements overstates token-level
        # agreement: a changed top-1 expert flips only 2 of 8 bits.
        router_match.append(float((a==b).all(dim=-1).float().mean()))
        router_bit_match.append(float((a==b).float().mean()))
    return {'teacher_KL_per_sequence_mean':float(np.mean(kls)),
       'teacher_top1_agreement_mean':float(np.mean(agreements)),
       'next_router_dispatch_exact_per_token_agreement_mean':
         float(np.mean(router_match)) if router_match else None,
       'next_router_mask_bitwise_agreement_mean_DIAGNOSTIC_ONLY':
         float(np.mean(router_bit_match)) if router_bit_match else None,
       'contexts':len(kls)}

def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(24)
    configs,allocator=design()
    api=HfApi()
    model_rev=api.model_info(MODEL).sha
    if model_rev!=PINNED_SWITCH_SHA:
        raise ValueError('checkpoint HEAD moved, refusing unverified manifest')
    wiki_rev=api.dataset_info(DATASET).sha
    tiny_rev=api.dataset_info('roneneldan/TinyStories').sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=PINNED_SWITCH_SHA)
    model=SwitchTransformersForConditionalGeneration.from_pretrained(
        MODEL,revision=PINNED_SWITCH_SHA,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    model.eval()
    if model.config.num_experts!=NUM_EXPERTS:raise ValueError('expert count changed')
    original={}
    bybits={5:{},6:{}}
    for i in range(NUM_EXPERTS):
        exp=model.get_submodule(f'{EXPERTS}.expert_{i}')
        original[i]={'wi':exp.wi.weight.detach().clone(),'wo':exp.wo.weight.detach().clone()}
        for bits in (5,6):
            bybits[bits][i]={
                'wi':fixed_encode(original[i]['wi'].float().numpy(),bits),
                'wo':fixed_encode(original[i]['wo'].float().numpy(),bits)}
            if sum(map(len,bybits[bits][i].values()))!=PAIR_BYTES[bits]:
                raise ValueError('real pretrained pair physical size mismatch')
    wiki=token_data(tok,wiki_rev,'validation',args.tokens)
    tiny,_=load_new_corpus(tok,args.tokens,tiny_rev)
    corpora={'WikiText2_validation':wiki,'TinyStories_validation':tiny}
    next_router=identify_next_router(model)
    if next_router is None:raise ValueError("no later real decoder router")
    teacher={}
    for corpus,ids in corpora.items():
        teacher[corpus]=evaluation(model,list(make_windows(ids,args.encoder_len,
                    args.decoder_len,tok.pad_token_id)),next_router)
    report=[]
    for label,bits in configs.items():
        with torch.inference_mode():
            for i in range(NUM_EXPERTS):
                ex=model.get_submodule(f'{EXPERTS}.expert_{i}')
                ex.wi.weight.copy_(torch.from_numpy(fixed_decode(bybits[bits[i]][i]['wi'])))
                ex.wo.weight.copy_(torch.from_numpy(fixed_decode(bybits[bits[i]][i]['wo'])))
        corpus_checks={}
        try:
            for corpus,ids in corpora.items():
                candidate=evaluation(model,list(make_windows(ids,args.encoder_len,
                           args.decoder_len,tok.pad_token_id)),next_router)
                t=teacher[corpus]
                corpus_checks[corpus]=compare_to_teacher(t[0],candidate[0],t[1],candidate[1])
        finally:
            with torch.inference_mode():
                for i in range(NUM_EXPERTS):
                    ex=model.get_submodule(f'{EXPERTS}.expert_{i}')
                    ex.wi.weight.copy_(original[i]['wi'])
                    ex.wo.weight.copy_(original[i]['wo'])
        record={'label':label,'expert_bits':bits,
         'physical_all_eight_wi_wo_bytes':storage_bytes(bits),
         'fraction_of_uniform_int6_storage':storage_bytes(bits)/storage_bytes([6]*8),
         'train_routed_weighted_MSE_proxy':weighted_proxy(bits,'train'),
         'previously_measured_val_routed_weighted_MSE_proxy':weighted_proxy(bits,'validation'),
         'worst_expert_previous_val_relative_output_MSE':max(
                  (VAL5 if b==5 else VAL6)[i] for i,b in enumerate(bits)),
         'corpus_teacher_comparisons':corpus_checks}
        report.append(record)
        print(json.dumps({'completed':label,
             'bits':bits,'bytes':record['physical_all_eight_wi_wo_bytes'],
             'teacher':corpus_checks}),flush=True)
    out={'experiment':'CALIBRI_v24_real_routed_8_expert_joint_precision_allocation',
         'model':MODEL,'model_revision':PINNED_SWITCH_SHA,
         'v23_source_run':V23_EXPERIMENT_ID,'v23_metrics_sha256':V23_JSON_SHA256,
         'fresh_manifest_from_real_pretrained_MoE_routing':True,
         'train_routed_counts':TRAIN_N,'validation_routed_counts':VAL_N,
         'exact_search':allocator,
         'target_decoder_layer':'decoder.block.1.layer.2.mlp',
         'subsequent_router':next_router,
         'corpus_revisions':{'WikiText2':wiki_rev,'TinyStories':tiny_rev},
         'corpus_token_sha256':{name:hashlib.sha256(ids.numpy().tobytes()).hexdigest()
             for name,ids in corpora.items()},
         'expert_pair_fp16_reference_bytes':75497472,
         'portfolios':report,
         'limitations':[
            'Mixed precision/routed sensitivity allocation are established prior art, NOT a new invention',
            'Exact exhaustive global budget for 8 binary choices, train-only surrogate',
            'Local weighted relative MSE is NOT exact global logit loss',
            'Only one decoder sparse layer and 256 Wikitext2 train routed sources used in v23',
            'Evaluation uses reloaded BF16 matrix weights, NOT packed live MoE kernel',
            'Pseudo teacher-forced decoder inputs: teacher-logit fidelity, not task accuracy',
            'TinyStories is new for this Switch model allocation, but previously used for dense Pythia',
            'Very small sample and shifted rare-expert frequencies; no reliable generalization',
            'No QMoE official checkpoint or 0.8bpw comparison, no mobile CPU/flash runtime',
            'No changes to Jessica production, Render, Supabase, APK or live chat']}
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps({'experiment':out['experiment'],
       'exact_search':allocator,
       'portfolios':[{k:v for k,v in p.items() if k in (
           'label','expert_bits','physical_all_eight_wi_wo_bytes',
           'train_routed_weighted_MSE_proxy',
           'previously_measured_val_routed_weighted_MSE_proxy',
           'worst_expert_previous_val_relative_output_MSE',
           'corpus_teacher_comparisons')} for p in report]},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',default='results/calibri_v24_real_moe_allocation.json')
    parser.add_argument('--tokens',type=int,default=256)
    parser.add_argument('--encoder-len',type=int,default=64)
    parser.add_argument('--decoder-len',type=int,default=32)
    a=parser.parse_args()
    if a.smoke:smoke()
    else:run(a)
