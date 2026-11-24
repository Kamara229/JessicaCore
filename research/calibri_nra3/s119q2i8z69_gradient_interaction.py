"""Z69 gradient compatibility diagnosis on 14 legal TRAIN98 positions.
No optimizer step, no teacher, no DEV, no model selection.
Measure representative regional tensors, not whole-region gradient norms.
"""
import math
import torch
from research.calibri_nra3.s119q2i8z65_stable_gate_train import objective_loss,ARMS
from research.calibri_nra3.s119q2i8z59_gpt_compositional_curriculum import KINDS
def check(ok,msg):
    if not ok:raise ValueError(msg)
def select(rows,contrasts):
    out=[]
    for k in KINDS:
        domain=[r for r in rows if r["kind"]==k]
        pending=next((r,t) for r in domain for t,m in enumerate(r["utf8_masks"]) if not bool(m[2]))
        paired=next((r,contrasts[r["id"]][0]) for r in domain if r["id"] in contrasts
                    and (r["id"],contrasts[r["id"]][0])!=(pending[0]["id"],pending[1]))
        out.extend([("pending_UTF8",*pending),("paired_divergence",*paired)])
    check(len(out)==14 and len({(r["id"],t) for _,r,t in out})==14,"fixed first eligible two distinct cases per kind")
    return out
def comparison(a,b):
    a=a.detach().double().reshape(-1);b=b.detach().double().reshape(-1)
    na=float(torch.linalg.vector_norm(a));nb=float(torch.linalg.vector_norm(b))
    dot=float(torch.dot(a,b))
    return {"CE_norm":na,"aux_norm":nb,"dot":dot,
            "cosine":max(-1.,min(1.,dot/(na*nb))) if na>1e-15 and nb>1e-15 else None}
def diagnose(logits,gold,t,contrast,mask,views):
    total,ce,pair,eos,utf8=objective_loss(logits,gold,t,arm=ARMS[1],contrast=contrast,mask=mask)
    losses={"CE":ce,"pair":pair,"early_EOS":eos,"UTF8":utf8,"total":total}
    params=tuple(views.values());grads={};unused={}
    for i,(key,loss) in enumerate(losses.items()):
        g=torch.autograd.grad(loss,params,retain_graph=i<len(losses)-1,allow_unused=True) if loss.requires_grad else (None,)*len(params)
        unused[key]=[name for name,grad in zip(views,g) if grad is None]
        grads[key]={name:grad.detach() if grad is not None else torch.zeros_like(p)
                    for (name,p),grad in zip(views.items(),g)}
        check(all(bool(torch.isfinite(x).all()) for x in grads[key].values()),"all measured gradients finite")
    regions={}
    for name in views:
        regions[name]={k:comparison(grads["CE"][name],grads[k][name]) for k in ("pair","early_EOS","UTF8","total")}
        expected=sum(grads[k][name] for k in ("CE","pair","early_EOS","UTF8"))
        actual=grads["total"][name]
        check(torch.allclose(expected,actual,rtol=2e-3,atol=2e-6),"combined gradient additive within numeric tolerance")
    gates=views["GT12"].detach()
    idx=int(gates.abs().argmax())
    # Positive means a descent step would increase absolute value of current largest gate.
    outward={k:float(-g["GT12"][idx]*gates[idx].sign()) for k,g in grads.items()}
    return {"losses":{k:float(v.detach()) for k,v in losses.items()},"regions":regions,
            "unused_gradient_tensors":unused,"max_abs_gate":float(gates.abs().max()),
            "max_gate_index":idx,"max_gate_outward_descent_gradient":outward}
def summarize(records):
    out={}
    for group in ("pending_UTF8","paired_divergence"):
        rows=[r for r in records if r["group"]==group]
        check(len(rows)==7,"one fixed position per domain and group")
        def avg(x):return sum(x)/len(x) if x else None
        regions={}
        for name in rows[0]["regions"]:
            regions[name]={}
            for aux in ("pair","early_EOS","UTF8","total"):
                v=[r["regions"][name][aux] for r in rows]
                cos=[x["cosine"] for x in v if x["cosine"] is not None]
                regions[name][aux]={"nonzero_comparisons":len(cos),"negative_cosines":sum(x<0 for x in cos),
                    "mean_cosine":avg(cos),"mean_CE_norm":avg([x["CE_norm"] for x in v]),
                    "mean_aux_norm":avg([x["aux_norm"] for x in v])}
        out[group]={"n":7,"representative_tensors":regions,
                    "mean_max_gate_outward_descent_gradient":{k:avg([r["max_gate_outward_descent_gradient"][k] for r in rows])
                         for k in ("CE","pair","early_EOS","UTF8","total")}}
    return out
