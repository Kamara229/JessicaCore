"""Supervised factual critic: compare REAL hard negatives vs synthetic-only.

The exact same <=10MB encoder/architecture is used for both hosts; one
checkpoint is frozen after fitting and evaluated on disjoint questions.
A critic alone cannot rewrite hopeless model responses.
"""
import hashlib
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from v120.critic import GreenTunnel
from v120.bytes_adapter import pairs
from .protocol import SEED,STEPS,LR,BATCH,MAX_CHECKPOINT_BYTES,V120_CRITIC_SHA
from .hard_examples import canonical

def restore_previous(file):
    blob=file.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=V120_CRITIC_SHA:
        raise AssertionError("previous 8MB pretrained shared neural critic not locked")
    model=GreenTunnel()
    model.load_state_dict(load_file(str(file)),strict=True)
    return model

def simple_synthetic(rows):
    result=[]
    for q,correct,label,kind in rows:
        if kind in ("independently_correct_reference","plausible_false_reference"):
            result.append((q,correct,label,kind))
        if kind=="independently_correct_reference":
            result.extend([(q,"Yes Yes Yes Yes",0,"fake_repetition"),
                           (q,"",0,"fake_empty")])
    return result

def train(model,rows,seed=SEED):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(2)
    for p in model.parameters():p.requires_grad_(False)
    # Keep whole external AI unchanged in architecture and size, adapt only
    # last transformer and discriminator head.
    for p in model.blocks.layers[-1].parameters():p.requires_grad_(True)
    for p in model.score.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=LR)
    tokens=pairs([(q,a) for q,a,label,kind in rows])
    gold=torch.tensor([float(label) for q,a,label,kind in rows])
    rng=random.Random(seed)
    curve=[]
    for step in range(STEPS):
        indices=[rng.randrange(len(rows)) for _ in range(BATCH)]
        predictions=model(tokens[indices])
        targets=gold[indices]
        objective=F.binary_cross_entropy_with_logits(
            predictions,targets,pos_weight=torch.tensor(2.0))
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),1.0)
        opt.step()
        if step%35==0 or step==STEPS-1:
            curve.append([step,round(float(objective.detach()),5)])
    model.eval()
    return curve

@torch.inference_mode()
def infer(model,rows):
    model.eval()
    collected=[]
    for j in range(0,len(rows),16):
        b=rows[j:j+16]
        p=torch.sigmoid(model(pairs([(q,a) for q,a,l,kind in b])))
        collected.extend(float(x) for x in p)
    return collected

def grade(rows,prob,subset_kind=None):
    selected=[(r,p) for r,p in zip(rows,prob)
              if subset_kind is None or r[3] in subset_kind]
    if not selected:
        return {"count":0}
    correct=sum(1 for (q,a,lab,kind),p in selected if (p>=.5)==bool(lab))
    positives=[p for (q,a,lab,kind),p in selected if lab]
    negatives=[p for (q,a,lab,kind),p in selected if not lab]
    return {"count":len(selected),
            "balanced_accuracy":(
                (sum(p>=.5 for p in positives)/len(positives)+
                 sum(p<.5 for p in negatives)/len(negatives))/2
                 if positives and negatives else None),
            "accuracy":correct/len(selected),
            "real_negative_rejection":sum(p<.5 for p in negatives)/len(negatives) if negatives else None,
            "positive_acceptance":sum(p>=.5 for p in positives)/len(positives) if positives else None,
            "mean_probability":sum(p for _,p in selected)/len(selected)}

def file_checkpoint(model,path):
    save_file({k:v.detach().cpu().contiguous()
               for k,v in model.state_dict().items()},str(path))
    if path.stat().st_size>MAX_CHECKPOINT_BYTES:
        raise AssertionError("shared neural weights too large")
    return {"file_bytes":path.stat().st_size,
            "full_SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "trained_neural_parameter_count":sum(p.numel() for p in model.parameters()),
            "under_10MB":True}
