"""CALIBRI v23 — REAL routed Switch Transformer experts (research only).

Loads Google's actual pretrained switch-base-8; captures only activations
dispatched by the REAL decoder MoE router. Compare *paired* wi/wo matrices,
native TT+packed INT5 execution versus INT5/INT6 reference, and real teacher
logits after replacing all 8 wi+wo pairs. Never calls itself QMoE/GPTQ.
No user data, no model checkpoints committed, no production modifications.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from transformers.activations import ACT2FN

from fixedbit_v15 import encode as fixed_encode, decode as fixed_decode
from geometric_v17 import encode as geometric_encode, unpack as geometric_unpack
from geometric_v16 import reconstruct
from rowcache_v20 import row_cached_geometry

MODEL="google/switch-base-8"
DATASET="Salesforce/wikitext"
CONFIG="wikitext-2-raw-v1"
EXPERTS="decoder.block.1.layer.2.mlp.experts"
ROUTER="decoder.block.1.layer.2.mlp.router"

def token_data(tokenizer,revision,split,count):
    ds=load_dataset(DATASET,CONFIG,split=split,revision=revision)
    text="\n\n".join(str(t) for t in ds["text"] if t and t.strip())
    ids=tokenizer(text,add_special_tokens=False,return_tensors="pt").input_ids[0,:count]
    if len(ids)!=count:raise ValueError("missing source text")
    return ids

def slices(ids,source_len,decoder_len):
    if len(ids)%source_len:
        raise ValueError("non-divisible capture sample")
    for a in range(0,len(ids),source_len):
        batch=ids[a:a+source_len][None]
        start=torch.full((1,1),0,dtype=torch.long) # overwritten with actual tokenizer pad below
        yield batch,start,batch[:,:decoder_len-1]

def collect(model,ids,source_len,decoder_len,pad_id,target,with_logits=False):
    # wi prehook sees ONLY tokens that were really dispatched to each expert.
    xs=defaultdict(list)
    routed_counts=np.zeros(8,dtype=np.int64)
    actual_calls=np.zeros(8,dtype=np.int64)
    router_outputs=[]
    logits=[]
    names={i:model.get_submodule(f"{EXPERTS}.expert_{i}.wi") for i in range(8)}
    hooks=[]
    for i,layer in names.items():
        def recorder(mod,args,idx=i):
            if args and args[0].numel():
                inp=args[0].detach().reshape(-1,args[0].shape[-1]).float().cpu()
                xs[idx].append(inp)
                actual_calls[idx]+=len(inp)
        hooks.append(layer.register_forward_pre_hook(recorder))
    def routing(mod,args,output):
        if not isinstance(output,(tuple,list)) or len(output)<1:
            raise ValueError("unexpected Switch router contract")
        dispatch=output[0].detach().float().cpu()
        if dispatch.shape[-1]!=8:raise ValueError("router must return eight experts")
        routed_counts[:]+=dispatch.reshape(-1,8).sum(dim=0).numpy().astype(np.int64)
        router_outputs.append(dispatch.reshape(-1,8).numpy().copy())
    hooks.append(model.get_submodule(ROUTER).register_forward_hook(routing))
    try:
        with torch.inference_mode():
            for enc,_,previous in slices(ids,source_len,decoder_len):
                shifted=torch.cat((torch.full((1,1),pad_id,dtype=torch.long),previous),dim=1)
                out=model(input_ids=enc,decoder_input_ids=shifted,
                          use_cache=False,return_dict=True)
                if with_logits:logits.append(out.logits.detach().float().cpu())
    finally:
        for h in hooks:h.remove()
    if not np.array_equal(routed_counts,actual_calls):
        raise AssertionError(f"actual expert invocations differ from router mask: {routed_counts} vs {actual_calls}")
    return {'xs':{i:torch.cat(xs[i],dim=0) if xs[i] else torch.empty(0,target)
                   for i in range(8)},
            'route_counts':routed_counts.tolist(),
            'routed_total':int(routed_counts.sum()),
            'routed_mask':router_outputs,
            'teacher_logits':logits}

def pair_output(x,wi,wo,act):
    return F.linear(act(F.linear(x,wi)),wo)

def rel_mse(a,b):
    num=float(((a-b).double()**2).sum())
    den=float((a.double()**2).sum())
    return num/max(den,1e-30)

def geo_dense(blob):
    base,res=geometric_unpack(blob)
    from geometric_v16 import unpack as tt_unpack
    a,b,none=tt_unpack(base)
    if none is not None:raise AssertionError("unexpected separate TT residual")
    return reconstruct(a,b)+res

def local_compare(x_train,x_eval,wi,wo,act,candidates,token_budget):
    # Train and eval routing are independent (no candidate choice on eval).
    x_train=x_train[:token_budget]
    x_eval=x_eval[:token_budget]
    out={}
    for label,(wi_blob,wo_blob,kind) in candidates.items():
        a=(fixed_decode(wi_blob) if kind=="fixed" else geo_dense(wi_blob))
        b=(fixed_decode(wo_blob) if kind=="fixed" else geo_dense(wo_blob))
        a=torch.from_numpy(np.asarray(a,dtype=np.float32))
        b=torch.from_numpy(np.asarray(b,dtype=np.float32))
        for split,x in (("train",x_train),("validation",x_eval)):
            if not len(x):
                out.setdefault(label,{})[split]={'tokens':0,'relative_output_mse':None}
                continue
            with torch.inference_mode():
                ref=pair_output(x,wi,wo,act)
                other=pair_output(x,a,b,act)
                value=rel_mse(ref,other)
            out.setdefault(label,{})[split]={'tokens':len(x),'relative_output_mse':value}
    return out

def model_teacher_kl(original,modified):
    # Compare same token positions, with no task labels or external answer key.
    prob=F.softmax(original.float(),dim=-1)
    return float(F.kl_div(F.log_softmax(modified.float(),dim=-1),
                          prob,reduction="sum")/original.shape[1])

def main(args):
    import ctypes as ct
    import gc
    torch.set_num_threads(2)
    torch.manual_seed(23)
    api=HfApi()
    revision=api.model_info(MODEL).sha
    data_revision=api.dataset_info(DATASET).sha
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision)
    if tokenizer.pad_token_id is None:raise ValueError("T5 requires pad token")
    # CPU BF16 checkpoint is significantly smaller than a FP32 duplicate.
    # accelerate low_cpu_mem_usage loads a single model without a second FP32 state dict.
    model=SwitchTransformersForConditionalGeneration.from_pretrained(
        MODEL,revision=revision,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    model.eval()
    if model.config.dense_act_fn not in ACT2FN:
        raise ValueError("unknown nonlinearity")
    act=ACT2FN[model.config.dense_act_fn]
    if model.config.num_experts!=8:raise ValueError("unexpected Switch base-8 config")
    ids_train=token_data(tokenizer,data_revision,"train",args.train_tokens)
    ids_val=token_data(tokenizer,data_revision,"validation",args.val_tokens)
    train=collect(model,ids_train,args.source_len,args.decoder_len,tokenizer.pad_token_id,
                  model.config.d_model,False)
    val=collect(model,ids_val,args.source_len,args.decoder_len,tokenizer.pad_token_id,
                model.config.d_model,True)
    print("actual router counts",json.dumps({'train':train['route_counts'],
                                              'validation':val['route_counts']}),flush=True)
    weights={}
    for i in range(8):
        exp=model.get_submodule(f"{EXPERTS}.expert_{i}")
        weights[i]={'wi':exp.wi.weight.detach().clone(),
                    'wo':exp.wo.weight.detach().clone()}
    # Choose geometric experts using TRAIN routing counts ONLY.
    rank_by_train=sorted(range(8),key=lambda i:(-train['route_counts'][i],i))
    geometry_experts=set(rank_by_train[:args.geo_experts])
    library=ct.CDLL(str(pathlib.Path(args.library).resolve()))
    details={}
    packed_fixed={b:{} for b in (5,6)}
    geo_packed={}
    for i in range(8):
        original=weights[i]
        wi=original['wi'].float().numpy()
        wo=original['wo'].float().numpy()
        candidate={}
        for bits in (5,6):
            enc_wi=fixed_encode(wi,bits)
            enc_wo=fixed_encode(wo,bits)
            candidate[f"int{bits}"]=(enc_wi,enc_wo,"fixed")
            packed_fixed[bits][i]={'wi':enc_wi,'wo':enc_wo}
        if i in geometry_experts:
            wigeo=geometric_encode(wi,2,16,32,5)
            wogeo=geometric_encode(wo,2,16,32,5)
            candidate["tt2_int5"]=(wigeo,wogeo,"geometry")
            geo_packed[i]={'wi':wigeo,'wo':wogeo}
            # Direct execution on ACTUALLY ROUTED expert activations:
            x=val['xs'][i][:args.max_expert_tokens]
            if len(x):
                with torch.inference_mode():
                    native= row_cached_geometry(wigeo,None,library)
                    native_wo= row_cached_geometry(wogeo,None,library)
                    direct=native_wo(act(native(x)))
                    dense=pair_output(x,torch.from_numpy(geo_dense(wigeo)),
                                      torch.from_numpy(geo_dense(wogeo)),act)
                if not torch.allclose(direct,dense,atol=1e-3,rtol=1e-3):
                    raise AssertionError("true routed native TT+INT5 differs from restored same weights")
                direct_gap=float((direct-dense).abs().max())
            else:direct_gap=None
        else:direct_gap=None
        assessed=local_compare(train['xs'][i],val['xs'][i],
                               torch.from_numpy(wi),torch.from_numpy(wo),
                               act,candidate,args.max_expert_tokens)
        details[i]={'train_routed_tokens':train['route_counts'][i],
                    'validation_routed_tokens':val['route_counts'][i],
                    'real_router_activations_used':True,
                    'samples_per_split_capped_at':args.max_expert_tokens,
                    'full_pair_fp16_bytes':int(2*(wi.size+wo.size)),
                    'candidate_physical_bytes':{label:len(b1)+len(b2)
                                     for label,(b1,b2,kind) in candidate.items()},
                    'native_packed_TT_routed_max_abs_gap':direct_gap,
                    'paired_expert_output_metrics':assessed}
        print("real expert finished",i,flush=True)
        gc.collect()
    teacher_logits=val['teacher_logits']
    total_fp16=sum(d['full_pair_fp16_bytes'] for d in details.values())
    options=[]
    baseline={'model_label':'original_pretrained_BF16',
              'decoder_layer_pair_bytes_FP16':total_fp16,
              'teacher_logits_KL':0.,'teacher_argmax_agreement':1.}
    for bits in (5,6):
        # Runtime restored BF16 original architecture for FULL-model check.
        # Physical disk byte streams measured separately, no fast packed claim.
        try:
            with torch.no_grad():
                for i in range(8):
                    expert=model.get_submodule(f"{EXPERTS}.expert_{i}")
                    blob=packed_fixed[bits][i]
                    expert.wi.weight.copy_(torch.from_numpy(fixed_decode(blob['wi'])))
                    expert.wo.weight.copy_(torch.from_numpy(fixed_decode(blob['wo'])))
            snippets=[];kl=[];exact=[]
            with torch.inference_mode():
                for j,(enc,_,prev) in enumerate(slices(ids_val,args.source_len,args.decoder_len)):
                    decoder=torch.cat((torch.full((1,1),tokenizer.pad_token_id,dtype=torch.long),
                                       prev),dim=1)
                    logits=model(input_ids=enc,decoder_input_ids=decoder,
                                 use_cache=False,return_dict=True).logits.detach().float().cpu()
                    target=teacher_logits[j]
                    kl.append(model_teacher_kl(target,logits))
                    exact.append(float((target.argmax(-1)==logits.argmax(-1)).float().mean()))
            options.append({'variant':f"int{bits}_all8_expert_wi_wo",
                'pair_storage_bytes':int(sum(len(e['wi'])+len(e['wo'])
                                             for e in packed_fixed[bits].values())),
                'teacher_KL_sum_over_sequences':float(np.sum(kl)),
                'teacher_KL_mean_per_sequence':float(np.mean(kl)),
                'teacher_argmax_agreement':float(np.mean(exact)),
                'runtime':'dense BF16 restored weights, NOT packed execution'})
        finally:
            with torch.no_grad():
                for i in range(8):
                    expert=model.get_submodule(f"{EXPERTS}.expert_{i}")
                    expert.wi.weight.copy_(weights[i]['wi'])
                    expert.wo.weight.copy_(weights[i]['wo'])
        print("whole model eval finished INT",bits,flush=True)
    total_packed={f'int{b}':sum(
        len(p['wi'])+len(p['wo']) for p in packed_fixed[b].values())
        for b in (5,6)}
    observed={str(i):details[i] for i in range(8)}
    report={
      'experiment':'CALIBRI_v23_real_routed_pretrained_Switch_MoE_pairs',
      'source_model':MODEL,'revision':revision,
      'dataset':DATASET,'data_revision':data_revision,
      'model_checkpoint_precision':'BF16',
      'actual_moe_path':EXPERTS,'actual_router_path':ROUTER,
      'router_dispatch_count_matches_expert_forward_hooks':True,
      'nonlinearity':model.config.dense_act_fn,
      'token_sha256':{'train':hashlib.sha256(ids_train.numpy().tobytes()).hexdigest(),
                      'validation':hashlib.sha256(ids_val.numpy().tobytes()).hexdigest()},
      'encoder_tokens_per_window':args.source_len,
      'decoder_tokens_per_window':args.decoder_len,
      'train_router_dispatch':train['route_counts'],
      'validation_router_dispatch':val['route_counts'],
      'train_total_dispatched':train['routed_total'],
      'validation_total_dispatched':val['routed_total'],
      'geometry_expert_ids_SELECTED_BY_TRAIN_FREQUENCY_ONLY':sorted(geometry_experts),
      'total_paired_weight_fp16_bytes':total_fp16,
      'total_8_expert_pair_bitstreams':total_packed,
      'reference':baseline,'all_expert_model_teacher_checks':options,
      'experts':observed,
      'limitations':[
        'Real Google Switch-base-8 MoE checkpoint and TRUE router-dispatched activations',
        'Only one decoder sparse MoE layer; other layers remain BF16',
        'INT5/INT6 compare two linked matrices wi and wo; TT+INT5 tested on two experts only',
        'True native TT + streamed 5bit execution checked locally, not for full-model forward',
        'Whole-model teacher KL checks use restored BF16 matrices, not packed kernels',
        'Teacher logits on arbitrary teacher-forced WikiText tokens, NOT a downstream quality benchmark',
        'Router capacity / dropped tokens faithfully reflected by actual wi prehooks',
        'Too few routed tokens per expert gives unreliable rare-expert conclusions',
        'Neither QMoE checkpoint, GPTQ quality equivalence nor 0.8 bpw reproduced',
        'No on-device ARM/flash bandwidth, sustained throughput, or end-to-end size improvement claim',
        'No production branch, Render, Supabase, app or user data modified']}
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    overview={k:report[k] for k in ('experiment','revision',
         'train_router_dispatch','validation_router_dispatch',
         'geometry_expert_ids_SELECTED_BY_TRAIN_FREQUENCY_ONLY',
         'total_paired_weight_fp16_bytes',
         'total_8_expert_pair_bitstreams',
         'all_expert_model_teacher_checks')}
    print(json.dumps(overview,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/calibri_v23_real_routed_moe.json")
    p.add_argument("--library",default="research/calibri/libpacked5_v23.so")
    p.add_argument("--train-tokens",type=int,default=256)
    p.add_argument("--val-tokens",type=int,default=256)
    p.add_argument("--source-len",type=int,default=64)
    p.add_argument("--decoder-len",type=int,default=32)
    p.add_argument("--geo-experts",type=int,default=2)
    p.add_argument("--max-expert-tokens",type=int,default=64)
    args=p.parse_args()
    if args.train_tokens%args.source_len or args.val_tokens%args.source_len:
        raise ValueError('token windows must divide sample lengths')
    main(args)
