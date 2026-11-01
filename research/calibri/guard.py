"""CALIBRI v0.8 quality-gated mixed-codec selection on real pretrained LM.

Uses WikiText-2 TRAIN for statistics, VALIDATION for choosing modules, TEST
for truly held-out comparison. Never feeds selected codec to production.
"""
from __future__ import annotations
import argparse, json, pathlib, math, time, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import HfApi
from quality import MODEL,DATASET,CONFIG,load_token_ids,module_selection,calibrate,ppl_eval,quantize
from mixedcodec import pack_symmetric,unpack_symmetric

torch.set_num_threads(2)

@torch.no_grad()
def uniform_quant(w,bits):
    if bits not in (4,8):raise ValueError("bits")
    levels=(1<<(bits-1))-1
    vmax=w.abs().amax(dim=1).clamp_min(1e-12)
    # For INT4 use values -7..7 with 4-bit signed nibble; INT8 uses -127..127.
    scale=(vmax/levels).to(torch.float16).float().clamp_min(1e-12)
    q=torch.round(w/scale[:,None]).clamp(-levels,levels).to(torch.int8)
    reconstructed=q.float()*scale[:,None]
    rows,cols=w.shape
    blob=pack_symmetric(q.numpy(),scale.numpy(),bits)
    q2,s2=unpack_symmetric(blob)
    verified=torch.from_numpy(q2.copy()).float()*torch.from_numpy(s2.copy())[:,None]
    if not torch.equal(verified,reconstructed):
        raise AssertionError('mixed codec binary roundtrip changed reconstructed weights')
    return reconstructed,len(blob),{"quantization":f"int{bits}","row_scales":"fp16",
                                  "levels":2*levels+1}

@torch.no_grad()
def get_candidates(w,x):
    v=w.detach().float().cpu()
    base=2*v.numel()
    result={}
    for bits in (8,4):
        rec,bsize,extra=uniform_quant(v,bits)
        result[f"int{bits}"]={"recon":rec,"bytes":bsize,"savings":base-bsize,"meta":extra}
    for z in (.25,.5):
        # MAE-sensitive ternary calibrator is a candidate, not assumed safe.
        recon,met=quantize(v,x,z,"activation_ls")
        name=f"ternary{int(100*z)}"
        result[name]={"recon":recon,"bytes":met["size_cl8_bytes_with_scales"],
                      "savings":base-met["size_cl8_bytes_with_scales"],
                      "meta":{"quantization":"ternary+LS","target_zero":z,"codec":"CL8-I"}}
    return result

@torch.no_grad()
def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--output",default="results/calibri_v08_guard.json")
    pa.add_argument("--development-tokens",type=int,default=384)
    pa.add_argument("--test-tokens",type=int,default=1024)
    pa.add_argument("--calibration-tokens",type=int,default=512)
    pa.add_argument("--seq-len",type=int,default=128)
    pa.add_argument("--max-dev-ppl-ratio",type=float,default=1.05)
    a=pa.parse_args()
    for count in (a.development_tokens,a.test_tokens,a.calibration_tokens):
        if count%a.seq_len:raise ValueError("samples must align")
    torch.manual_seed(7)
    started=time.time()
    rev_model=HfApi().model_info(MODEL).sha
    rev_data=HfApi().dataset_info(DATASET).sha
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev_model,torch_dtype=torch.float32)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=rev_model)
    model.eval()
    train_ids=load_token_ids(tokenizer,"train",rev_data,a.calibration_tokens)
    val_ids=load_token_ids(tokenizer,"validation",rev_data,a.development_tokens)
    test_ids=load_token_ids(tokenizer,"test",rev_data,a.test_tokens)
    modules=module_selection(model)
    if len(modules)!=12:raise AssertionError("Unexpected module list")
    original={k:m.weight.detach().clone() for k,m in modules.items()}
    baseline_dev=ppl_eval(model,val_ids,a.seq_len)
    baseline_test=ppl_eval(model,test_ids,a.seq_len)
    acts=calibrate(model,modules,train_ids,a.seq_len)
    candidate={}
    alone=[]
    for name,module in modules.items():
        candidate[name]=get_candidates(original[name],acts[name])
        for typ,entry in candidate[name].items():
            module.weight.copy_(entry["recon"])
            quality=ppl_eval(model,val_ids,a.seq_len)
            module.weight.copy_(original[name])
            delta=quality["nll"]-baseline_dev["nll"]
            record={"name":name,"codec":typ,"single_dev_delta_nll":delta,
                    "single_dev_ppl_ratio":quality["ppl"]/baseline_dev["ppl"],
                    "saved_bytes":entry["savings"],"encoded_bytes":entry["bytes"]}
            alone.append(record)
    # Order candidates by estimated quality cost per byte; choose cumulatively.
    options=sorted(alone,key=lambda r:(max(r["single_dev_delta_nll"],1e-4)
                                        /max(r["saved_bytes"],1),-r["saved_bytes"]))
    budget=math.log(a.max_dev_ppl_ratio)
    selected={}
    decisions=[]
    for option in options:
        name=option["name"]
        if name in selected:continue
        mod=modules[name]
        mod.weight.copy_(candidate[name][option["codec"]]["recon"])
        joint=ppl_eval(model,val_ids,a.seq_len)
        delta=joint["nll"]-baseline_dev["nll"]
        accepted=delta<=budget
        if accepted:
            selected[name]=option["codec"]
        else:
            mod.weight.copy_(original[name])
        decisions.append({"name":name,"codec":option["codec"],
                          "accepted":accepted,"joint_dev_delta_nll":delta})
    final_dev=ppl_eval(model,val_ids,a.seq_len)
    final_test=ppl_eval(model,test_ids,a.seq_len)
    all_params=sum(p.numel() for p in model.parameters())
    saved=sum(candidate[name][typ]["savings"] for name,typ in selected.items())
    result={"experiment":"CALIBRI_v08_heldout_guarded_mixed_coding",
            "model":MODEL,"model_revision":rev_model,"dataset":DATASET,
            "dataset_revision":rev_data,"development_tokens":a.development_tokens,
            "test_tokens":a.test_tokens,"calibration_tokens":a.calibration_tokens,
            "train_token_sha256":hashlib.sha256(train_ids.numpy().tobytes()).hexdigest(),
            "validation_token_sha256":hashlib.sha256(val_ids.numpy().tobytes()).hexdigest(),
            "test_token_sha256":hashlib.sha256(test_ids.numpy().tobytes()).hexdigest(),
            "methodology":"train for X, validation for codec selection, TEST for unbiased heldout assessment",
            "warning":"Mixed compressed model is DECODED to dense matrices before evaluating perplexity",
            "baseline":{"dev":baseline_dev,"test":baseline_test},
            "final":{"dev":final_dev,"test":final_test},
            "dev_ppl_ratio":final_dev["ppl"]/baseline_dev["ppl"],
            "heldout_test_ppl_ratio":final_test["ppl"]/baseline_test["ppl"],
            "dev_threshold":a.max_dev_ppl_ratio,
            "total_parameters":all_params,
            "full_model_assumed_fp16_baseline_bytes":2*all_params,
            "hypothetical_mixed_checkpoint_bytes":2*all_params-saved,
            "total_estimated_checkpoint_reduction_percent":100*saved/(2*all_params),
            "selection":selected,"selected_count":len(selected),
            "single_module_screen":alone,"greedy_decisions":decisions,
            "runtime_seconds":time.time()-started}
    p=pathlib.Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({"baseline_dev_ppl":baseline_dev["ppl"],
                      "final_dev_ppl":final_dev["ppl"],
                      "baseline_test_ppl":baseline_test["ppl"],
                      "final_test_ppl":final_test["ppl"],
                      "test_ppl_ratio":result["heldout_test_ppl_ratio"],
                      "selected_count":len(selected),
                      "checkpoint_reduction_pct":result["total_estimated_checkpoint_reduction_percent"],
                      "selection":selected,
                      "runtime_seconds":round(result["runtime_seconds"],1)}),flush=True)

if __name__=="__main__":main()
