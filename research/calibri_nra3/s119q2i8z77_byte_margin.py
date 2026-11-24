"""Z77 diagnostic margins on already-known incomplete TRAIN42 byte prefixes.
Masks are used only to measure unmodified native logits, never to select
the output or train the model. No new parameters.
"""
import torch
from research.calibri_nra3.s119q2i8z68_pending_byte_audit import measure,collect,summarize as basic
def measure_margin(logits,target,mask):
    result=measure(logits,target,mask)
    assert bool(mask[target]) and bool(mask.any()) and bool((~mask).any())
    x=logits[0].float()
    best_valid=x[mask].max()
    best_invalid=x[~mask].max()
    result.update(best_valid_rank=1+int((x>best_valid).sum()),
        best_valid_minus_invalid=float(best_valid-best_invalid),
        gold_minus_best_valid=float(x[target]-best_valid),
        gold_minus_best_invalid=float(x[target]-best_invalid),
        conditional_gold_given_valid=float(torch.exp(x[target]-torch.logsumexp(x[mask],0))))
    return result
def summarize(records):
    result=basic(records)
    for k in ("best_valid_rank","best_valid_minus_invalid","gold_minus_best_valid","gold_minus_best_invalid","conditional_gold_given_valid"):
        result["mean_"+k]=sum(r[k] for r in records)/len(records)
    result["valid_in_top5"]=sum(r["best_valid_rank"]<=5 for r in records)
    result["valid_in_top20"]=sum(r["best_valid_rank"]<=20 for r in records)
    return result
def paired(a,b):
    a={(r["id"],r["position"]):r for r in a}
    b={(r["id"],r["position"]):r for r in b}
    assert set(a)==set(b) and len(a)==57
    return {k:{"improved":sum(b[i][k]>a[i][k] for i in a),"worse":sum(b[i][k]<a[i][k] for i in a),
        "equal":sum(b[i][k]==a[i][k] for i in a)} for k in ("best_valid_minus_invalid","valid_mass","conditional_gold_given_valid")}
