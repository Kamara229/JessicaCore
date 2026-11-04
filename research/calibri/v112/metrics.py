"""Evidence-only metrics: actual reconstructed operator signals, teacher KL, NLL."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .archive import Archive

# Predeclared distinct sentences; none are used for compression fitting.
VALID = (
    "As the rain cleared, the children discovered small tracks beside the riverbank.",
    "Every astronomer in the observatory watched the pale comet cross the sky.",
    "In programming, a function receives arguments and returns a computed value.",
    "The museum guide explained how the wooden ships navigated the open sea.",
)
GENERATE = (
    "The small robot looked at the wooden door and",
    "Scientists studied the bright planet because",
)
MAX_NEW_TOKENS = 24

@torch.inference_mode()
def original_reference(model,tok,source):
    """Save original logits, labels, source activation matrices and their hashes."""
    captures={name:[] for name in source}
    hooks=[]
    try:
        for name in source:
            module=model.get_submodule(name.removesuffix(".weight"))
            def hook(_mod,arguments,*,key=name):
                h=arguments[0].detach().reshape(-1,arguments[0].shape[-1])
                captures[key].append(h.cpu().float().numpy().copy())
            hooks.append(module.register_forward_pre_hook(hook))
        sequences=[]; originals=[]
        for text in VALID:
            ids=tok(text,return_tensors="pt")["input_ids"]
            if ids.shape[1] < 8:
                raise ValueError("validation text too short")
            scores=model(input_ids=ids,use_cache=False).logits[0,:-1].detach().float()
            sequences.append(ids)
            originals.append(scores)
        x={name:np.concatenate(arr,axis=0) for name,arr in captures.items()}
        return sequences,originals,x
    finally:
        for h in hooks:
            h.remove()

def activation_error(source,archive,inputs):
    numer=0.;denom=0.; per_role={}
    from .roles import classify
    for name,w in source.items():
        x=inputs[name]
        rec=archive.decode(name)
        a=x @ w.T
        b=x @ rec.T
        # Accumulate absolute sums of squared output errors, not mean-of-means.
        dif=float(np.square(a-b,dtype=np.float64).sum())
        base=float(np.square(a,dtype=np.float64).sum())
        numer+=dif;denom+=base
        _,role=classify(name)
        role_item=per_role.setdefault(str(role),{"sse":0.,"signal":0.})
        role_item["sse"]+=dif
        role_item["signal"]+=base
    return {"real_teacher_activation_rel_rmse":math.sqrt(numer/max(denom,1e-20)),
            "by_role_rel_rmse":{k:math.sqrt(item["sse"]/max(item["signal"],1e-20))
                                for k,item in per_role.items()}}

@torch.inference_mode()
def evaluate(model,tok,sequences,teacher_logits):
    total_kl=0.; total_nll=0.; total=0
    for ids,reference in zip(sequences,teacher_logits):
        logits=model(input_ids=ids,use_cache=False).logits[0,:-1].float()
        log_t=torch.log_softmax(reference,dim=-1)
        log_q=torch.log_softmax(logits,dim=-1)
        kl=(log_t.exp()*(log_t-log_q)).sum(-1)
        labels=ids[0,1:]
        nll=-torch.gather(log_q,1,labels[:,None]).flatten()
        total_kl+=float(kl.sum());total_nll+=float(nll.sum());total+=len(labels)
    return {"tokens":total, "teacher_relative_kl":total_kl/total,
            "token_nll":total_nll/total,"perplexity":math.exp(min(50,total_nll/total))}

@torch.inference_mode()
def generate(model,tok):
    answer=[]
    for text in GENERATE:
        tokens=tok(text,return_tensors="pt")["input_ids"]
        with torch.no_grad():
            ids=model.generate(tokens,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
                               pad_token_id=tok.eos_token_id)
        answer.append({"prompt":text,"text":tok.decode(ids[0],skip_special_tokens=True),
                       "new_ids":ids[0,len(tokens[0]):].tolist()})
    return answer

@torch.inference_mode()
def patch_model(model,source,archive):
    for name in source:
        weight=model.get_submodule(name.removesuffix(".weight")).weight
        restored=archive.decode(name)
        if tuple(weight.shape)!=restored.shape:
            raise AssertionError("shape drift while patching genuine GPT-NeoX")
        weight.copy_(torch.from_numpy(restored))

@torch.inference_mode()
def restore_model(model,source):
    for name,w in source.items():
        model.get_submodule(name.removesuffix(".weight")).weight.copy_(torch.from_numpy(w))
    for name,w in source.items():
        actual=model.get_submodule(name.removesuffix(".weight")).weight.detach().cpu().numpy()
        if not np.array_equal(actual,w):
            raise AssertionError("original pretrained weights not restored: "+name)

def source_digest(source):
    sha=hashlib.sha256()
    for name,w in source.items():
        sha.update(name.encode()+b"\0")
        sha.update(np.asarray(w,dtype="<f4").tobytes())
    return sha.hexdigest()

def write_report(out,report):
    out.mkdir(parents=True,exist_ok=True)
    (out/"v112_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
