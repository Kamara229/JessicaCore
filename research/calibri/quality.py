"""CALIBRI v0.8: activation-calibrated ternarization + held-out LM loss.

A real, pinned Pythia-14M is evaluated on disjoint WikiText-2 train/validation
token windows. Compression covers ONLY 12 MLP projections; model executes
restored DENSE weights (no fused compressed kernel, no GPU claims).
"""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, time, os
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from index import encode as cl8_encode, decode as cl8_decode
from codec import encode as cl7_encode

MODEL = "EleutherAI/pythia-14m"
DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
torch.set_num_threads(2)

def load_token_ids(tokenizer, split, revision, limit):
    ds=load_dataset(DATASET,CONFIG,split=split,revision=revision)
    # Each split is concatenated independently, with genuine text paragraph separators.
    raw="\n\n".join(str(s) for s in ds["text"] if s and s.strip())
    ids=tokenizer(raw,return_tensors="pt",add_special_tokens=False).input_ids[0,:limit]
    if len(ids)<limit:raise ValueError(f"too little {split} data")
    return ids

def module_selection(model):
    return {name:mod for name,mod in model.named_modules()
            if isinstance(mod,torch.nn.Linear)
            and (name.endswith(".mlp.dense_h_to_4h") or name.endswith(".mlp.dense_4h_to_h"))}

@torch.no_grad()
def calibrate(model,modules,tokens,seq_len):
    captures=defaultdict(list)
    def hook(name):
        def capture(mod,inp,out):
            x=inp[0].detach().reshape(-1,inp[0].shape[-1]).float().cpu()
            captures[name].append(x)
        return capture
    handles=[mod.register_forward_hook(hook(name)) for name,mod in modules.items()]
    try:
        for i in range(0,len(tokens),seq_len):
            model(input_ids=tokens[i:i+seq_len][None],use_cache=False)
    finally:
        for h in handles:h.remove()
    return {k:torch.cat(v,dim=0) for k,v in captures.items()}

@torch.no_grad()
def ppl_eval(model,ids,seq_len):
    nll=0.; count=0
    for st in range(0,len(ids)-seq_len+1,seq_len):
        sub=ids[st:st+seq_len][None]
        loss=model(input_ids=sub,labels=sub,use_cache=False).loss
        nll+=float(loss)*(seq_len-1)
        count+=seq_len-1
    if count==0:raise ValueError("empty validation")
    loss=nll/count
    return {"nll":loss,"ppl":float(math.exp(loss)),"tokens":int(count),
            "context":seq_len,"evaluation":"non-overlapping blocks"}

def quantize(w,x,target_zero,method):
    """Produce q in {-1,0,+1}; per-row asym fp16 levels for fair storage."""
    v=w.detach().float().cpu()
    X=x.detach().float().cpu()
    nout,nin=v.shape
    k=max(1,int(round(nin*(1-target_zero))))
    var=X.square().mean(dim=0).clamp_min(1e-12)
    score=v.abs() if method=="magnitude" else v.abs()*var.sqrt()[None,:]
    top=torch.topk(score,k=k,dim=1,largest=True,sorted=False).indices
    mask=torch.zeros_like(v,dtype=torch.bool);mask.scatter_(1,top,True)
    q=torch.where(mask,v.sign(),torch.zeros_like(v)).to(torch.int8)
    neg=(q==-1);pos=(q==1)
    if method=="activation_ls":
        # Two-parameter least squares from genuine routed module activations:
        # min_{lo,hi} || X(W_j-q_j) ||_2 on calibration data.
        A=(X @ neg.float().T)
        B=(X @ pos.float().T)
        Y=X @ v.T
        aa=(A*A).sum(0); bb=(B*B).sum(0);ab=(A*B).sum(0)
        ay=(A*Y).sum(0);by=(B*Y).sum(0)
        # Ridge only prevents degenerate 2x2 normal equations.
        ridge=1e-5*(aa+bb).clamp_min(1.)
        aa=aa+ridge; bb=bb+ridge
        det=(aa*bb-ab.square()).clamp_min(1e-20)
        lo=(ay*bb-by*ab)/det
        hi=(by*aa-ay*ab)/det
        lo=lo.clamp(max=0.);hi=hi.clamp(min=0.)
    else:
        weights=var[None,:] if method=="activation_diag" else torch.ones_like(v)
        lo=(v*weights*neg).sum(1)/(weights*neg).sum(1).clamp_min(1e-12)
        hi=(v*weights*pos).sum(1)/(weights*pos).sum(1).clamp_min(1e-12)
    # Cost levels encoded as FP16, not arbitrarily precise FP32.
    lo=lo.to(torch.float16).float();hi=hi.to(torch.float16).float()
    recon=torch.where(neg,lo[:,None],torch.where(pos,hi[:,None],0.))
    with torch.no_grad():
        y0=X @ v.T
        y1=X @ recon.T
        err=float(torch.linalg.vector_norm(y0-y1)/torch.linalg.vector_norm(y0).clamp_min(1e-20))
    arr=q.numpy()
    binary=cl8_encode(arr)
    if not np.array_equal(arr,cl8_decode(binary)):
        raise AssertionError("lossless codec reconstruction failure")
    binary7=cl7_encode(arr)
    metrics={"shape":list(v.shape),"weights":int(arr.size),
             "zero_fraction":float(np.mean(arr==0)),
             "size_cl8_bytes_with_scales":len(binary)+4*nout,
             "size_cl7_bytes_with_scales":len(binary7)+4*nout,
             "calibration_output_rel_rmse":err,
             "negative_levels":int((lo<0).sum()),"positive_levels":int((hi>0).sum())}
    return recon,metrics

@torch.no_grad()
def trial(model,modules,originals,captures,target_zero,method,validation,seq_len):
    infos=[]
    for name,module in modules.items():
        rec,info=quantize(originals[name],captures[name],target_zero,method)
        module.weight.copy_(rec)
        info["name"]=name
        infos.append(info)
    quality=ppl_eval(model,validation,seq_len)
    for name,module in modules.items():
        module.weight.copy_(originals[name])
    count=sum(x["weights"] for x in infos)
    bytes8=sum(x["size_cl8_bytes_with_scales"] for x in infos)
    bytes7=sum(x["size_cl7_bytes_with_scales"] for x in infos)
    total_params=sum(p.numel() for p in model.parameters())
    # Estimate: all other parameters in FP16, compressed MLP projections
    # replaced by CL8; excludes runtime buffers, framework, tokenizer.
    total_size=2*(total_params-count)+bytes8
    return {"strategy":method,"target_zero":target_zero,
            "quality":quality,"mlp_weights":count,
            "mlp_bpw_cl8":8*bytes8/count,
            "mlp_bpw_cl7":8*bytes7/count,
            "mlp_cl8_bytes":bytes8,"mlp_cl7_bytes":bytes7,
            "full_model_hypothetical_mixed_checkpoint_bytes":total_size,
            "full_model_fp16_baseline_bytes":2*total_params,
            "full_model_estimated_reduction_percent":100*(1-total_size/(2*total_params)),
            "mlp_calibration_rel_rmse_avg":float(np.mean([m["calibration_output_rel_rmse"] for m in infos])),
            "modules":infos}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="results/calibri_v08.json")
    ap.add_argument("--calibration-tokens",type=int,default=512)
    ap.add_argument("--validation-tokens",type=int,default=1024)
    ap.add_argument("--sequence-length",type=int,default=128)
    args=ap.parse_args()
    if args.calibration_tokens%args.sequence_length or args.validation_tokens%args.sequence_length:
        raise ValueError("token counts must be multiples of sequence length")
    torch.manual_seed(7)
    started=time.time()
    revisions={"model":HfApi().model_info(MODEL).sha,
               "dataset":HfApi().dataset_info(DATASET).sha}
    # A pinned source snapshot avoids silent model substitution on reruns.
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revisions["model"])
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=revisions["model"],torch_dtype=torch.float32)
    model.eval()
    calibration=load_token_ids(tokenizer,"train",revisions["dataset"],args.calibration_tokens)
    validation=load_token_ids(tokenizer,"validation",revisions["dataset"],args.validation_tokens)
    source_hash=hashlib.sha256(calibration.numpy().tobytes()).hexdigest()
    valid_hash=hashlib.sha256(validation.numpy().tobytes()).hexdigest()
    modules=module_selection(model)
    if len(modules)!=12:raise AssertionError(f"Expected 12 MLP modules; got {len(modules)}")
    originals={name:m.weight.detach().clone() for name,m in modules.items()}
    quality_fp=ppl_eval(model,validation,args.sequence_length)
    captures=calibrate(model,modules,calibration,args.sequence_length)
    tested=[]
    for zero in (0.50,0.75,0.885):
        for method in ("magnitude","activation_diag","activation_ls"):
            result=trial(model,modules,originals,captures,zero,method,validation,args.sequence_length)
            tested.append(result)
            print(json.dumps({"zero":zero,"method":method,
                              "ppl":result["quality"]["ppl"],
                              "rel_ppl":result["quality"]["ppl"]/quality_fp["ppl"],
                              "cl8_bpw":result["mlp_bpw_cl8"],
                              "total_reduction_pct":result["full_model_estimated_reduction_percent"]}),flush=True)
    output={"experiment":"CALIBRI_v0.8_activation_aware_real_ppl",
            "model":MODEL,"dataset":DATASET,"dataset_configuration":CONFIG,
            "revisions":revisions,"train_calibration_token_sha256":source_hash,
            "validation_token_sha256":valid_hash,
            "validation_independent_of_calibration":True,
            "architecture_scope":"12 MLP modules only; attention and embeddings remain FP32 in PPL execution",
            "quality_metric":"WikiText-2 validation nonoverlapping windows; sample subset, not full benchmark",
            "runtime_warning":"Dense restored weights used for PPL; compression does NOT accelerate inference",
            "quantization_warning":"Not QMoE GPTQ; imposed sparsity; diagonal sensitivity and LS are ablations",
            "baseline":{"quality":quality_fp,"model_parameters":sum(p.numel() for p in model.parameters())},
            "trial_count":len(tested),"trials":tested,
            "runtime_seconds":time.time()-started}
    p=pathlib.Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print(json.dumps({"baseline_ppl":quality_fp["ppl"],"trials":len(tested),"elapsed_s":round(output["runtime_seconds"],1)}),flush=True)

if __name__=="__main__":
    main()
