"""TRAIN-only learnt tensor-network residual, VALID-only selection.

Never uses any official TEST text, labels from AGNews or v86 TEST outcomes.
The original native compressed student and original FP32 teacher are frozen.
"""
from __future__ import annotations
import numpy as np
import torch
from .tensor import OUT1,OUT2,IN1,IN2,RANK,VOCAB,delta,normalized

STEPS=(8,16,24)
MIN_KL_GAIN_EACH_DOMAIN=0.05
BATCH=64

def snapshot(A,B,b):
    return tuple(np.asarray(z.detach().cpu().numpy(),dtype="<f2").copy()
                 for z in (A,B,b))

def empty():
    return (np.zeros((OUT1,IN1,RANK),dtype="<f2"),
            np.zeros((RANK,OUT2,IN2),dtype="<f2"),
            np.zeros(VOCAB,dtype="<f2"))

def logits_with_paid_tensors(x,base,params):
    A,B,b=(torch.as_tensor(z.astype(np.float32)) for z in params)
    return base+delta(normalized(x),A,B)+b

@torch.inference_mode()
def measure(part,params):
    rows=[]
    for row in part["rows"]:
        z=logits_with_paid_tensors(row["x"],row["baseline"],params)
        y=row["teacher"]
        p=y.double().log_softmax(-1)
        q=z.double().log_softmax(-1)
        kl=float((p.exp()*(p-q)).sum(-1).mean())
        nll=float(-q.gather(1,row["labels"].long()[:,None]).mean())
        rows.append((kl,nll,len(row["labels"])))
    n=sum(r[2] for r in rows)
    return {"teacher_full_vocab_KL":sum(r[0]*r[2] for r in rows)/n,
            "original_next_token_NLL":sum(r[1]*r[2] for r in rows)/n,
            "scored_positions":n}

def fit(train,valid):
    torch.manual_seed(8800)
    A=torch.nn.Parameter(torch.zeros(OUT1,IN1,RANK))
    B=torch.nn.Parameter(torch.randn(RANK,OUT2,IN2)*.009)
    b=torch.nn.Parameter(torch.zeros(VOCAB))
    optimizer=torch.optim.Adam([A,B,b],lr=.025)
    x=torch.cat([row["x"] for domain in ("wiki_fit","news_fit")
                for row in train[domain]["rows"]])
    source=torch.cat([row["baseline"] for domain in ("wiki_fit","news_fit")
                for row in train[domain]["rows"]])
    teacher=torch.cat([row["teacher"] for domain in ("wiki_fit","news_fit")
                     for row in train[domain]["rows"]])
    labels=torch.cat([row["labels"] for domain in ("wiki_fit","news_fit")
                    for row in train[domain]["rows"]])
    if len(labels)<128 or source.shape!=(len(labels),VOCAB) or x.shape[1]!=IN1*IN2:
        raise AssertionError("original TRAIN failed to capture complete model")
    baseline=empty()
    candidates=[{"step":0,"mode":"original_CB59","params":baseline}]
    for step in range(1,max(STEPS)+1):
        ids=(torch.arange(BATCH)+(step-1)*BATCH)%len(labels)
        target=teacher[ids].detach().log_softmax(-1).exp()
        z=source[ids]+delta(normalized(x[ids]),A,B)+b
        lp=z.log_softmax(-1)
        kl_cross_entropy=-(target*lp).sum(-1).mean()
        nll=-lp.gather(1,labels[ids,None]).mean()
        loss=(kl_cross_entropy+.025*nll+.002*A.square().mean()+
              .002*B.square().mean()+.0001*b.square().mean())
        if not bool(torch.isfinite(loss)):
            raise AssertionError("multidimensional learned source diverged")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            A.clamp_(-.4,.4)
            B.clamp_(-.4,.4)
            b.clamp_(-2.,2.)
        if step in STEPS:
            packed=snapshot(A,B,b)
            candidates.extend([
                {"step":step,"mode":"4D_tensor_plus_bias","params":packed},
                {"step":step,"mode":"equal_paid_size_bias_only",
                 "params":(np.zeros_like(packed[0]),np.zeros_like(packed[1]),packed[2])},
            ])
            print({"stage":"v88_multidim_real_TRAIN_checkpoint",
                   "step":step,"loss":round(float(loss.detach()),5)},flush=True)
    baseline_metrics={domain:measure(valid[domain],baseline)
                      for domain in ("wiki_valid","news_valid")}
    summary=[]
    for item in candidates:
        item["VALID"]={domain:measure(valid[domain],item["params"])
                       for domain in baseline_metrics}
        item["min_domain_KL_improvement"]=min(
            baseline_metrics[d]["teacher_full_vocab_KL"]-
            item["VALID"][d]["teacher_full_vocab_KL"] for d in baseline_metrics)
        item["dual_NLL_nonregression"]=all(
            item["VALID"][d]["original_next_token_NLL"]<=
            baseline_metrics[d]["original_next_token_NLL"] for d in baseline_metrics)
        summary.append({k:v for k,v in item.items() if k!="params"})
    ordered=sorted(range(len(candidates)),
                   key=lambda i:(candidates[i]["min_domain_KL_improvement"],-i),
                   reverse=True)
    raw_best=candidates[ordered[0]]
    guarded=raw_best if (
        raw_best["min_domain_KL_improvement"]>=MIN_KL_GAIN_EACH_DOMAIN and
        raw_best["dual_NLL_nonregression"]) else candidates[0]
    return {"selected_physical_params":guarded["params"],
            "raw_best_exploratory_params":raw_best["params"],
            "source_train_splits_only":True,
            "predeclared_checkpoint_steps":STEPS,
            "baseline_VALID":baseline_metrics,
            "all_predeclared_candidates":summary,
            "raw_best_mode":raw_best["mode"],"raw_best_step":raw_best["step"],
            "guarded_selected_step":guarded["step"],
            "guarded_selected_mode":guarded["mode"],
            "research_VALID_go":guarded["step"]!=0,
            "separate_tensor_wins_against_bias_only":(
                guarded["mode"]=="4D_tensor_plus_bias")}
