"""End-to-end ground-truth trainable LoRA over FROZEN CB59 native C VJP."""
from __future__ import annotations
import random
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from v117.protocol import roles
from v108.generation import build_chat
from v118.protocol import FIT
from .protocol import TRAIN_SELECT,SEED,LR
class RankOne(nn.Module):
    def __init__(self,base,u,v):
        super().__init__()
        self.base=base
        self.u=nn.Parameter(torch.tensor(np.array(u,dtype=np.float32,copy=True)))
        # Zero U and tiny random V ensures a first-step signal to U.
        src=np.asarray(v,dtype=np.float32)
        if not np.any(src):src=torch.randn_like(self.u.T.new_zeros(1,base.original.in_features)).numpy()*.01
        self.v=nn.Parameter(torch.tensor(np.array(src,copy=True)))
    def forward(self,x):
        return self.base(x)+F.linear(F.linear(x,self.v),self.u)
class RankHead(nn.Module):
    def __init__(self,base,u,v,b):
        super().__init__()
        self.base=base
        self.u=nn.Parameter(torch.tensor(np.array(u,dtype=np.float32,copy=True)))
        self.v=nn.Parameter(torch.tensor(np.array(v,dtype=np.float32,copy=True)))
        self.b=nn.Parameter(torch.tensor(np.array(b,dtype=np.float32,copy=True)))
    def forward(self,x):
        return self.base(x)+F.linear(F.linear(x,self.v),self.u)+self.b
def apply(model,paid):
    changed={}
    blocks={}
    for name,rows,cols in roles():
        base=model.get_submodule(name)
        u,v=paid.role(name)
        adapter=RankOne(base,u,v)
        parent,_,leaf=name.rpartition(".")
        changed[name]=base;blocks[name]=adapter
        setattr(model.get_submodule(parent),leaf,adapter)
    changed["lm_head"]=model.lm_head
    head=RankHead(model.lm_head,*paid.head())
    model.lm_head=head
    return changed,blocks,head
def restore(model,changed):
    for name,base in changed.items():
        parent,sep,leaf=name.rpartition(".")
        owner=model.get_submodule(parent) if sep else model
        setattr(owner,leaf,base)

def fit(model,tok,paid):
    torch.manual_seed(SEED);random.seed(SEED)
    # Model trained pretrained tensors remain non-trainable throughout.
    for p in model.parameters():p.requires_grad_(False)
    old,blocks,head=apply(model,paid)
    trained=list(head.parameters())+[p for b in blocks.values() for p in (b.u,b.v)]
    if not all(p.requires_grad for p in trained):
        raise AssertionError("sidecar gradient disabled")
    optimizer=torch.optim.Adam(trained,lr=LR)
    trace=[];observed_grad=0
    try:
        model.eval()
        for step,idx in enumerate(TRAIN_SELECT):
            question,answer=FIT[idx]
            prompt=tok.encode(build_chat(tok,question),add_special_tokens=False)
            target=tok.encode(" "+answer,add_special_tokens=False)+[tok.eos_token_id]
            sequence=torch.tensor([prompt+target[:-1]],dtype=torch.long)
            logits=model(input_ids=sequence,use_cache=False).logits
            span=logits[0,len(prompt)-1:len(prompt)-1+len(target)]
            loss=F.cross_entropy(span,torch.tensor(target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_count=sum(int(b.u.grad is not None and float(b.u.grad.abs().sum())>0)
                           for b in blocks.values())
            observed_grad=max(observed_grad,grad_count)
            if not torch.isfinite(loss).all():raise FloatingPointError("non-finite gold task fit")
            torch.nn.utils.clip_grad_norm_(trained,1.)
            optimizer.step()
            with torch.no_grad():
                for param in trained:param.clamp_(-8,8)
            trace.append({"step":step,"source_FIT_index":idx,
                          "gold_answer_training_loss":round(float(loss.detach()),5),
                          "structural_blocks_received_nonzero_grad":grad_count})
            if step%4==0:
                print({"stage":"v119_native_joint_TRAIN","step":step,
                       "loss":round(float(loss.detach()),3),
                       "gradient_blocks":grad_count},flush=True)
        if observed_grad<1:
            raise AssertionError("CB59 true native reverse path did not train any body adapter")
        newblocks={name:tuple(x.detach().cpu().numpy().astype("<f2")
                              for x in (b.u,b.v)) for name,b in blocks.items()}
        newhead=tuple(x.detach().cpu().numpy().astype("<f2")
                      for x in (head.u,head.v,head.b))
        return newblocks,newhead,{"steps":len(trace),"max_nonzero_body_block_grads":observed_grad,
                                  "trace":trace,"no_teacher_KL_fit":True}
    finally:
        restore(model,old)
