"""CALIBRI v14: independent dev whole-model quality gate after v13's negative test.

No selection on final test. No merged code, model export, inference kernel or device.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib
import numpy as np
from fusion_v13 import candidates,pareto


def choose(rows, byte_budget, original_dev_ppl, allowed_increase=0.01):
    feasible=[r for r in rows if r['bytes']<=byte_budget and
              r['dev_ppl']<=original_dev_ppl*(1+allowed_increase)]
    return min(feasible,key=lambda c:(c['bytes'],c['dev_ppl'])) if feasible else None


def run(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi
    from quality import load_token_ids,ppl_eval,calibrate,module_selection,MODEL,DATASET
    torch.set_num_threads(2);torch.manual_seed(14)
    revision=HfApi().model_info(MODEL).sha
    d_rev=HfApi().dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,torch_dtype=torch.float32)
    model.eval();mods=module_selection(model)
    name=sorted(mods)[0];mod=mods[name]
    w=mod.weight.detach().cpu().numpy().copy().astype(np.float32)
    ids_train=load_token_ids(tok,'train',d_rev,args.calibration_tokens)
    ids_dev=load_token_ids(tok,'validation',d_rev,args.dev_tokens)
    ids_test=load_token_ids(tok,'test',d_rev,args.test_tokens)
    x_train=calibrate(model,{name:mod},ids_train,args.sequence_length)[name].numpy()
    x_dev=calibrate(model,{name:mod},ids_dev,args.sequence_length)[name].numpy()
    rows=candidates(w,x_train)
    q4=next(r for r in rows if r['name']=='q4')
    budget=int(q4['bytes']*args.relative_budget)
    original=mod.weight.detach().clone()
    reference_dev=ppl_eval(model,ids_dev,args.sequence_length)
    den=np.sum((x_dev@w.T).astype(np.float64)**2)
    for row in rows:
        row['dev_rel_mse']=float(np.sum((x_dev@(w-row['recon']).T).astype(np.float64)**2)/max(den,1e-30))
        with torch.no_grad():mod.weight.copy_(torch.from_numpy(row['recon']))
        row['dev_ppl']=ppl_eval(model,ids_dev,args.sequence_length)['ppl']
    with torch.no_grad():mod.weight.copy_(original)
    selected=choose(rows,budget,reference_dev['ppl'],args.allowed_dev_ppl_increase)
    train_selected=min((r for r in rows if r['bytes']<=budget),
                       key=lambda r:(r['cal_rel_mse'],r['bytes']))
    selected_names=['q4','q8',train_selected['name']]
    if selected is not None:selected_names.append(selected['name'])
    selected_names=list(dict.fromkeys(selected_names))
    test_ppl={}
    original_test=ppl_eval(model,ids_test,args.sequence_length)
    for chosen_name in selected_names:
        row=next(r for r in rows if r['name']==chosen_name)
        with torch.no_grad():mod.weight.copy_(torch.from_numpy(row['recon']))
        test_ppl[chosen_name]=ppl_eval(model,ids_test,args.sequence_length)
    with torch.no_grad():mod.weight.copy_(original)
    hashes={k:hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            for k,ids in [('train',ids_train),('development',ids_dev),('test',ids_test)]}
    for r in rows:del r['recon'],r['blob']
    report={'experiment':'CALIBRI_v14_no_test_selection_quality_gate',
      'model':MODEL,'model_revision':revision,'dataset':DATASET,'dataset_revision':d_rev,
      'token_sha256':hashes,'target_module':name,'shape':list(w.shape),
      'train_activation_count':len(x_train),'development_activation_count':len(x_dev),
      'fp16_reference_weight_bytes':2*w.size,'int4_bytes':q4['bytes'],
      'byte_budget':budget,'relative_byte_budget_to_int4':args.relative_budget,
      'allowed_dev_ppl_increase':args.allowed_dev_ppl_increase,
      'original_development_ppl':reference_dev,'original_test_ppl':original_test,
      'train_error_selected':train_selected['name'],
      'quality_gate_selected':selected['name'] if selected else 'NO_APPROVED_VARIANT',
      'no_approved_variant_means_fallback':'FP16 original, outside compression budget',
      'test_ppl_prespecified_controls_and_selected':test_ppl,
      'variants':rows,'calibration_pareto':pareto(rows),
      'limits':['Single real Pythia dense FFN matrix; not MoE or QMoE',
                'Dev gate prevents tuning to test split, not generalization proof',
                'Only one fixed module and a tiny token sample',
                'No training, no fused runtime, no whole-model storage comparison',
                'FP16 factors are not shared across blocks',
                'Outcomes relative to naive per-row INT4, not AWQ/GPTQ']}
    output=pathlib.Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


def smoke():
    rows=[{'name':'a','bytes':100,'dev_ppl':100.5},
          {'name':'b','bytes':80,'dev_ppl':120},
          {'name':'c','bytes':95,'dev_ppl':100.7}]
    assert choose(rows,100,100,0.01)['name']=='c'
    assert choose(rows,90,100,0.01) is None
    assert choose(rows,100,100,0.001) is None
    print('v14 selection smoke PASS: byte-constrained quality gate, fail closed')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',default='results/calibri_v14_quality_gate.json')
    p.add_argument('--calibration-tokens',type=int,default=256)
    p.add_argument('--dev-tokens',type=int,default=1024)
    p.add_argument('--test-tokens',type=int,default=1024)
    p.add_argument('--sequence-length',type=int,default=128)
    p.add_argument('--relative-budget',type=float,default=1.5)
    p.add_argument('--allowed-dev-ppl-increase',type=float,default=.01)
    a=p.parse_args()
    if a.smoke:smoke()
    else:run(a)
