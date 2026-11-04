"""Train actual 135M CB59 output operator on GROUND TRUTH, not original logits.

Student hidden states/base logits are frozen observations. New rank2 output
U/V and FP16 bias are optimized to emit task-correct token sequences.
This is a restricted head-only pilot; no claim of repairing corrupted
internal transformer blocks.
"""
import torch
import numpy as np
from torch.nn import functional as F
from v108.generation import build_chat
from .protocol import FIT,SEED,TRAIN_STEPS,LR,VOCAB,HEAD_DIM,HEAD_RANK

def capture(model,tok):
    x_all=[];base_all=[];gold_all=[];counts=[]
    for q,a in FIT:
        prompt=tok.encode(build_chat(tok,q),add_special_tokens=False)
        target=tok.encode(" "+a,add_special_tokens=False)+[tok.eos_token_id]
        sequence=torch.tensor([prompt+target[:-1]],dtype=torch.long)
        snapshot=[]
        def hook(_module,args):
            snapshot.append(args[0].detach().float().clone())
        handle=model.lm_head.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                out=model(input_ids=sequence,use_cache=False).logits
                s=len(prompt)-1
                h=snapshot[0][0,s:s+len(target)]
                y=out[0,s:s+len(target)].float()
                if tuple(y.shape)!=(len(target),VOCAB) or h.shape[-1]!=HEAD_DIM:
                    raise AssertionError("actual packed 135M head source mismatch")
                x_all.append(h.cpu().numpy().copy())
                base_all.append(y.cpu().numpy().copy())
                gold_all.extend(target)
                counts.append(len(target))
        finally:
            handle.remove()
    x=torch.from_numpy(np.concatenate(x_all))
    base=torch.from_numpy(np.concatenate(base_all))
    gold=torch.tensor(gold_all,dtype=torch.long)
    if x.shape[0]!=len(gold) or x.shape[0]<65:
        raise AssertionError("too few genuine TRAIN tokens")
    return x,base,gold,counts

def fit(model,tok):
    torch.manual_seed(SEED)
    hidden,base,gold,counts=capture(model,tok)
    u=torch.nn.Parameter(torch.randn(VOCAB,HEAD_RANK)*.008)
    v=torch.nn.Parameter(torch.randn(HEAD_RANK,HEAD_DIM)*.01)
    bias=torch.nn.Parameter(torch.zeros(VOCAB))
    opt=torch.optim.Adam([u,v,bias],lr=LR)
    loss_curve=[]
    for k in range(TRAIN_STEPS):
        output=F.linear(F.linear(hidden,v),u)+bias
        # True, independently labeled completions. No teacher KL.
        objective=F.cross_entropy(base+output,gold)
        objective=objective+.003*output.square().mean()
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([u,v,bias],1.0)
        opt.step()
        with torch.no_grad():
            u.clamp_(-8,8);v.clamp_(-8,8);bias.clamp_(-8,8)
        if k%20==0 or k==TRAIN_STEPS-1:
            loss_curve.append([k,round(float(objective.detach()),6)])
    result=tuple(t.detach().cpu().numpy().astype("<f2")
                 for t in (u,v,bias))
    if all(not np.any(t) for t in result):raise AssertionError("no trained head")
    return result,{
        "training_questions":len(FIT),"packed_source_target_tokens":len(gold),
        "teacher_forcing_ONLY_TRAIN":True,"correct_target_labels_not_teacher_logits":True,
        "optimizer_steps":TRAIN_STEPS,"objective_loss_training_only":loss_curve,
        "nonzero_fp16_head":True,
    }
