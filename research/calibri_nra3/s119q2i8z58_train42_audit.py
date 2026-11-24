"""TRAIN42 diagnostic only; no DEV answers, FP32 or optimizer."""
import torch
from torch.nn import functional as F
from math import isfinite
ARMS=("Z10_A_PARENT","Z55_GPT_ONLY")
def check(cond,msg):
    if not cond:raise ValueError(msg)
def audit_row(prefix,gold,step,*,vocab=49152,eos=2):
    check(len(gold)>=2 and gold[-1]==eos and len(prefix)+len(gold)<160,"complete reference")
    rows=[]
    for i,target in enumerate(gold):
        x=step(list(prefix)+gold[:i])
        check(x.shape==(1,vocab) and not x.requires_grad and bool(torch.isfinite(x).all()),"frozen native logits")
        lp=F.log_softmax(x[0].float(),dim=0)
        top=int(lp.argmax())
        rank=1+int((lp>lp[target]).sum())
        nll=-float(lp[target])
        check(isfinite(nll),"finite loss")
        rows.append({"position":i+1,"target":target,"top1":top,"rank":rank,"nll":nll,
                     "correct":top==target,"target_EOS":target==eos,
                     "premature_EOS_top1":top==eos and target!=eos,
                     "EOS_probability":float(lp[eos].exp())})
    return rows
def summarize(records):
    def agg(group):
        rows=[v for r in group for v in r["positions"]]
        if not rows:return {"n":0,"tokens":0}
        return {"n":len(group),"tokens":len(rows),
          "mean_NLL":sum(x["nll"] for x in rows)/len(rows),
          "top1_correct_tokens":sum(x["correct"] for x in rows),
          "first_token_correct":sum(r["positions"][0]["correct"] for r in group),
          "teacher_forced_all_tokens_correct":sum(all(x["correct"] for x in r["positions"]) for r in group),
          "terminal_EOS_correct":sum(x["correct"] for x in rows if x["target_EOS"]),
          "premature_EOS_top1":sum(x["premature_EOS_top1"] for x in rows),
          "mean_nonEOS_EOS_probability":sum(x["EOS_probability"] for x in rows if not x["target_EOS"])/sum(not x["target_EOS"] for x in rows)}
    check(len(records)==42 and len({r["id"] for r in records})==42,"exact TRAIN42")
    return {"all":agg(records),"new":agg([r for r in records if r["new"]]),
            "replay":agg([r for r in records if not r["new"]]),
            "per_kind":{k:agg([r for r in records if r["kind"]==k]) for k in sorted({r["kind"] for r in records})}}
