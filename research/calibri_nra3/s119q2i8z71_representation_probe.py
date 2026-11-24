"""Z71 diagnostic-only linear probes on actual frozen Z55 states.
Fit on six TRAIN domains, test on the seventh, with all positions of
a given example confined to one fold. No probe enters model inference.
"""
import torch
from torch.nn import functional as F
from collections import Counter
from research.calibri_nra3.s119q2i8z63_utf8_transition_objective import byte_pieces,advance
from research.calibri_nra3.s119q2i8z55_gpt_curriculum import KINDS
LAMBDA=.1
def check(ok,msg):
    if not ok:raise ValueError(msg)
def positions(tok,rows):
    pieces=byte_pieces(tok);out=[]
    for r in rows:
        pending=b"";gold=r["answer_native_with_eos"]
        for t,target in enumerate(gold):
            out.append({"id":r["id"],"kind":r["kind"],"position":t+1,
                        "prefix":r["prefix"]+gold[:t],"features":r["prompt_features"],
                        "pending":bool(pending)})
            if target==tok.eos_token_id:
                check(not pending and t==len(gold)-1,"complete reference EOS")
            else:
                pending=advance(pending,pieces[target])
                check(pending is not None,"valid reference byte state")
    check(len(out)==842 and sum(r["pending"] for r in out)==57,"exact registered historical TRAIN42")
    return out
def fit_predict(x,y,test):
    # No test statistics in the training transform, no hyperparameter search.
    x=x.double();test=test.double();y=y.double()
    check(x.ndim==test.ndim==2 and x.shape[1]==test.shape[1] and y.shape==(len(x),)
          and bool(torch.isfinite(x).all()) and bool(torch.isfinite(test).all())
          and set(y.tolist())=={0.,1.},"finite train/test matrices, both train labels")
    mean=x.mean(0);sd=x.std(0,unbiased=False).clamp_min(1e-6)
    z=(x-mean)/sd;zt=(test-mean)/sd
    # Equal aggregate class weights; intercept not penalized.
    w=torch.where(y==1,.5/y.sum(),.5/(1-y).sum())
    z=torch.cat((z,torch.ones(len(z),1,dtype=z.dtype)),1)
    zt=torch.cat((zt,torch.ones(len(zt),1,dtype=z.dtype)),1)
    reg=torch.eye(z.shape[1],dtype=z.dtype)*LAMBDA;reg[-1,-1]=0
    coef=torch.linalg.solve(z.T@(w[:,None]*z)+reg,z.T@(w*(2*y-1)))
    return (zt@coef>=0).long()
def stats(y,p):
    check(y.shape==p.shape and set(y.tolist())=={0,1},"both held-out classes")
    tp=int(((y==1)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum())
    pos=int(y.sum());neg=len(y)-pos
    return {"TP":tp,"TN":tn,"positive":pos,"negative":neg,"sensitivity":tp/pos,
            "specificity":tn/neg,"balanced_accuracy":.5*(tp/pos+tn/neg)}
def crossfit(x,records):
    check(len(x)==len(records) and len({r["id"] for r in records})==42
          and set(r["kind"] for r in records)==set(KINDS),"exact TRAIN42 grouping")
    y=torch.tensor([int(r["pending"]) for r in records]);pred=torch.full_like(y,-1)
    folds={}
    for k in KINDS:
        test=torch.tensor([r["kind"]==k for r in records])
        check(not {r["id"] for r,t in zip(records,test) if t}.intersection(
                  r["id"] for r,t in zip(records,test) if not t),"row group separation")
        p=fit_predict(x[~test],y[~test],x[test]);pred[test]=p
        folds[k]=stats(y[test],p)
    check(bool((pred>=0).all()),"one held-out prediction per position")
    return {"aggregate":stats(y,pred),"folds":folds,"ridge_lambda":LAMBDA}
def rank_stats(x):
    x=x.double();s=torch.linalg.svdvals(x-x.mean(0))
    q=s.square();total=float(q.sum())
    check(total>0,"nonconstant real features")
    p=q/q.sum();p=p[p>0]
    return {"dimension":x.shape[1],"effective_rank":float(torch.exp(-(p*p.log()).sum())),
            "first4_energy_fraction":float(q[:4].sum()/q.sum()),
            "singular_values_first8":[float(v) for v in s[:8]]}
