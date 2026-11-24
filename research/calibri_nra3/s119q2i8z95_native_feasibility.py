"""Z95 frozen-state full-native-logit linear feasibility diagnostic.
Compare exact same observed target columns, full competitor vocabulary,
and fixed ridge fit for actual H320 versus actual rank4 GELU.
The fitted diagnostic is NOT a model, not connected to any generation.
In-sample interpolation is capacity evidence, not generalization.
"""
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119a_hard_four_goals import check
LAMBDA=.001
MARGIN=2.
def feasibility(features,base,gold):
    check(features.ndim==2 and base.ndim==2 and len(features)==len(base)==len(gold)
          and gold.ndim==1 and bool(torch.isfinite(features).all()) and bool(torch.isfinite(base).all()),
          "finite aligned real native states")
    check(bool(((gold>=0)&(gold<base.shape[1])).all()),"legal gold IDs")
    x=features.double()
    x=(x-x.mean(0))/x.std(0,unbiased=False).clamp_min(1e-6)
    x=torch.cat((x/(features.shape[1]**.5),torch.ones(len(x),1,dtype=x.dtype)),1)
    ids,indices=torch.unique(gold,sorted=True,return_inverse=True)
    # Desired additive residual only on observed target columns. All
    # other columns retain exact original logits and remain competitors.
    desired=torch.zeros(len(gold),len(ids),dtype=torch.float64)
    desired[torch.arange(len(gold)),indices]=(base.max(1).values-base.gather(1,gold[:,None]).squeeze(1)+MARGIN).double()
    gram=x@x.T
    coef=x.T@torch.linalg.solve(gram+LAMBDA*torch.eye(len(x),dtype=x.dtype),desired)
    delta=x@coef
    fitted=base.double().clone()
    fitted[:,ids]+=delta
    other=torch.ones(base.shape[1],dtype=torch.bool);other[ids]=False
    check(torch.equal(fitted[:,other],base[:,other].double()),"all non-target competitors preserved")
    g=fitted.gather(1,gold[:,None]).squeeze(1)
    wrong=fitted.clone();wrong[torch.arange(len(gold)),gold]=-torch.inf
    margins=g-wrong.max(1).values
    s=torch.linalg.svdvals(x)
    return {"feature_dimension":features.shape[1],"n":len(gold),"target_columns":len(ids),
        "auxiliary_probe_coefficients":coef.numel(),"feature_matrix_rank":int((s>s[0]*1e-8).sum()),
        "top1":int((fitted.argmax(1)==gold).sum()),"strict_positive_margin":int((margins>0).sum()),
        "mean_gold_margin":float(margins.mean()),"min_gold_margin":float(margins.min()),
        "native_NLL":float(F.cross_entropy(fitted,gold)),
        "residual_RMSE":float((delta-desired).square().mean().sqrt()),
        "max_abs_logit_correction":float(delta.abs().max()),"all_non_target_logits_unchanged":True}
def diagnose(child,cache):
    from research.calibri_nra3.s119q2i8z94_micro_overfit import batch,logits_for,metrics
    h,b,gold=batch(child,cache)
    with torch.no_grad():
        base=logits_for(child.language_head,h,b)
        h320=F.layer_norm(h,(320,))[:,0,:]
        r4=F.gelu(child.language_head.down(F.layer_norm(h,(320,))))[:,0,:]
        check(h320.shape==(51,320) and r4.shape==(51,4),"actual existing features")
        return {"original_native":metrics(base,gold),"H320":feasibility(h320,base,gold),
                "rank4":feasibility(r4,base,gold)}
