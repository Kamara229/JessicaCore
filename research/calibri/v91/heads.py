"""Decompose frozen CB88 head impact into original logits, 4D tensor and bias.
Only TRAIN-only BoolQ; exact 2-vocabulary-row contraction, no extra inference.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from v88.tensor import OUT2, IN1, IN2, delta, normalized

MODES=("CB59","bias_only","tensor_only","CB88")

class HeadComponents:
    def __init__(self,A,B,bias):
        self.A=torch.as_tensor(np.asarray(A,dtype=np.float32).copy())
        self.B=torch.as_tensor(np.asarray(B,dtype=np.float32).copy())
        self.bias=torch.as_tensor(np.asarray(bias,dtype=np.float32).copy())

    def token_tensor(self,x,token_id):
        z=normalized(x.float()).reshape(IN1,IN2)
        o1,o2=divmod(int(token_id),OUT2)
        # Exactly W[o1,o2,i1,i2] = sum_r A[o1,i1,r] B[r,o2,i2].
        return float(torch.einsum("ij,ir,rj->",z,self.A[o1],self.B[:,o2,:]).item())

    def margins(self,x,base_logits,answer_tokens):
        yes,no=map(int,answer_tokens)
        base=float((base_logits[yes]-base_logits[no]).item())
        tensor=self.token_tensor(x,yes)-self.token_tensor(x,no)
        bias=float((self.bias[yes]-self.bias[no]).item())
        return {"CB59":base,"bias_only":base+bias,
                "tensor_only":base+tensor,"CB88":base+bias+tensor,
                "bias_yes_minus_no":bias,"tensor_yes_minus_no":tensor}

def category_margin(margin,gold):
    pred=0 if margin>=0 else 1
    return {"prediction":pred,"correct":int(pred==gold),
            "yes_no_logit_margin":float(margin),
            "gold_signed_margin":float(margin if gold==0 else -margin),
            "prediction_confidence":float(1/(1+math.exp(-min(80,abs(margin)))))}

def summarize(rows):
    if not rows:raise AssertionError("empty TRAIN-only head ablation")
    info={}
    for mode in MODES:
        data=[r["modes"][mode] for r in rows]
        n_yes=sum(r["gold"]==0 for r in rows)
        n_no=len(rows)-n_yes
        info[mode]={"accuracy":sum(x["correct"] for x in data)/len(rows),
                    "correct":sum(x["correct"] for x in data),
                    "mean_yes_no_margin":sum(x["yes_no_logit_margin"] for x in data)/len(data),
                    "predicted_yes":sum(x["prediction"]==0 for x in data),
                    "correct_yes":sum(x["correct"] for x,r in zip(data,rows) if r["gold"]==0),
                    "total_yes":n_yes,
                    "correct_no":sum(x["correct"] for x,r in zip(data,rows) if r["gold"]==1),
                    "total_no":n_no}
    info["component_effects"]={key:sum(r[key] for r in rows)/len(rows)
        for key in ("bias_yes_minus_no","tensor_yes_minus_no")}
    return info

@torch.inference_mode()
def run_head_ablations(student,cases,answer_tokens,components):
    captured=[]
    def hook(_module,inputs):
        captured.append(inputs[0].detach()[0,-1,:].clone())
    handle=student.lm_head.register_forward_pre_hook(hook)
    findings={}
    try:
        for split,items in cases.items():
            rows=[]
            for entry in items:
                captured.clear()
                x=torch.tensor(entry["tokens"],dtype=torch.long)[None,:]
                logits=student(input_ids=x,use_cache=False).logits[0,-1,:].float()
                if len(captured)!=1:raise AssertionError("missing packed head activation capture")
                margins=components.margins(captured[0],logits,answer_tokens)
                rows.append({"record_index":entry["record_index"],"gold":entry["gold"],
                    "prompt_SHA":entry["prompt_SHA"],
                    "modes":{mode:category_margin(margins[mode],entry["gold"]) for mode in MODES},
                    "bias_yes_minus_no":margins["bias_yes_minus_no"],
                    "tensor_yes_minus_no":margins["tensor_yes_minus_no"]})
            findings[split]={"summary":summarize(rows),"examples":rows}
    finally:handle.remove()
    return findings
