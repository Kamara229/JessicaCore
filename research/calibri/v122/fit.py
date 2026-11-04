"""Differentiable real-CB59 Q/A training of the SHARED 8MB Green Tunnel.

Original source and paid v119 U/V NEVER modified. No teacher KL. No
VALID or formal TEST inspection during this training phase.
"""
from __future__ import annotations
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from v108.generation import build_chat
from v119.native import build as build_vjp,install as install_vjp,restore as restore_vjp
from .steer import Installed
from .protocol import SEED,STEPS,LR,TRAIN

def fit(model,tok,source,policy,out):
    random.seed(SEED);torch.manual_seed(SEED)
    for p in model.parameters():
        p.requires_grad_(False)
    policy.train()
    fw,bw=build_vjp(out)
    frozen_modules=install_vjp(model,fw,bw)
    bridge=None
    params=[p for p in policy.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(params,lr=LR,weight_decay=.0001)
    history=[]
    begin=time.perf_counter()
    try:
        bridge=Installed(model,source)
        for step,(question,answer) in enumerate(TRAIN):
            prompt=tok.encode(build_chat(tok,question),add_special_tokens=False)
            target=tok.encode(" "+answer,add_special_tokens=False)+[tok.eos_token_id]
            sample=torch.tensor([prompt+target[:-1]],dtype=torch.long)
            gains=bridge.set_question(policy,question)
            logits=model(input_ids=sample,use_cache=False).logits
            selected=logits[0,len(prompt)-1:len(prompt)-1+len(target)]
            if selected.shape[0]!=len(target):
                raise ValueError("gold target shift differs from actual instruction")
            # Gold-token supervised objective; not old teacher logits.
            gold_loss=F.cross_entropy(selected,torch.tensor(target))
            regularizer=.02*((gains-1)**2).mean()
            loss=gold_loss+regularizer
            optim.zero_grad(set_to_none=True)
            loss.backward()
            last=policy.network.score.weight.grad
            if last is None or not bool(torch.isfinite(last).all()):
                raise AssertionError("native end-to-end controller gradient missing")
            torch.nn.utils.clip_grad_norm_(params,1.)
            optim.step()
            gains_cpu=[float(z) for z in gains.detach()]
            history.append({
                "step":step,"correct_gold_training_loss":float(gold_loss.detach()),
                "mean_gain":sum(gains_cpu)/len(gains_cpu),
                "minimum_gain":min(gains_cpu),"maximum_gain":max(gains_cpu),
                "controller_grad_L2":float(last.norm().detach())})
            bridge.clear_question()
            if step%4==0:
                print({"stage":"v122_GREEN_training","step":step,
                       "real_gold_loss":round(history[-1]["correct_gold_training_loss"],3),
                       "gate_range":[round(min(gains_cpu),3),round(max(gains_cpu),3)]},
                      flush=True)
    finally:
        if bridge is not None:bridge.restore()
        restore_vjp(model,frozen_modules)
    policy.eval()
    return {"real_pretrained_CB59_full_gradient":True,
            "original_pretrained_weights_changed":False,
            "under_10MB_neural_policy_trained":True,
            "train_questions":len(TRAIN),"training_steps":STEPS,
            "objective":"true text targets, not teacher KL/NLL",
            "duration_seconds":time.perf_counter()-begin,
            "history":history}
