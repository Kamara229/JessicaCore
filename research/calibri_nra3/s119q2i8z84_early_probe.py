"""Z84 diagnostic-only native first/second target separability.
Actual frozen representations, fixed ridge regression, no model updates.
In-sample fit is capacity evidence only. Cross-domain folds use only
training-fold labels and normalization. Unseen target labels count as misses.
"""
import torch
from collections import Counter
from research.calibri_nra3.s119q2i8z55_gpt_curriculum import KINDS
LAMBDA=.1
def require(ok,msg):
    if not ok:raise ValueError(msg)
def predict(x,y,test):
    x=x.double();test=test.double();y=y.long()
    require(x.ndim==test.ndim==2 and x.shape[1]==test.shape[1] and len(x)==len(y)
            and len(y)>0 and bool(torch.isfinite(x).all()) and bool(torch.isfinite(test).all()),"finite aligned features")
    classes,idx=torch.unique(y,sorted=True,return_inverse=True)
    mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-6)
    a=(x-mean)/scale;b=(test-mean)/scale
    # Normalize by feature count to make regularization comparable.
    a=a/(a.shape[1]**.5);b=b/(b.shape[1]**.5)
    a=torch.cat((a,torch.ones(len(a),1,dtype=a.dtype)),1)
    b=torch.cat((b,torch.ones(len(b),1,dtype=b.dtype)),1)
    targets=torch.nn.functional.one_hot(idx,num_classes=len(classes)).double()
    reg=torch.eye(a.shape[1],dtype=a.dtype)*LAMBDA;reg[-1,-1]=0
    w=torch.linalg.solve(a.T@a+reg,a.T@targets)
    return classes[(b@w).argmax(1)]
def evaluate(x,y,kinds):
    require(len(x)==len(y)==len(kinds) and set(kinds)==set(KINDS),"all seven domains")
    pred=predict(x,y,x)
    folds={};correct=covered=baseline=0
    for k in KINDS:
        test=torch.tensor([v==k for v in kinds],dtype=torch.bool)
        known=set(y[~test].tolist());gold=y[test]
        p=predict(x[~test],y[~test],x[test])
        frequency=Counter(y[~test].tolist())
        majority=min(frequency,key=lambda i:(-frequency[i],i))
        n=int((p==gold).sum());c=sum(int(t in known) for t in gold.tolist());m=int((gold==majority).sum())
        folds[k]={"n":len(gold),"covered":c,"correct":n,"frequency_correct":m}
        correct+=n;covered+=c;baseline+=m
    return {"n":len(y),"distinct_gold":len(set(y.tolist())),"in_sample_correct":int((pred==y).sum()),
            "cross_domain_correct":correct,"cross_domain_covered":covered,"cross_domain_frequency_correct":baseline,
            "folds":folds,"lambda":LAMBDA}
def native_metrics(rows):
    return {"n":len(rows),"correct":sum(r["target"]==r["top1"] for r in rows),
            "mean_gold_rank":sum(r["rank"] for r in rows)/len(rows),
            "distinct_targets":len({r["target"] for r in rows}),
            "distinct_top1":len({r["top1"] for r in rows})}
