"""Z54 frozen teacher on actual student prefixes; no optimizer."""
from __future__ import annotations
import torch
from torch.nn import functional as F
VOCAB=49152
MAX_STEPS=6
EOS=2
def check(cond,msg):
    if not cond: raise ValueError(msg)
def trace(prompt,gold,student_step,teacher_step,*,vocab=VOCAB):
    check(len(prompt)>0 and len(gold)>=1 and gold[-1]==EOS,"valid reference")
    generated=[]
    rows=[]
    for pos in range(MAX_STEPS):
        prefix=list(prompt)+generated
        s=student_step(prefix)
        t=teacher_step(prefix)
        check(s.shape==t.shape==(1,vocab) and not s.requires_grad and
              not t.requires_grad and bool(torch.isfinite(s).all()) and
              bool(torch.isfinite(t).all()),"frozen finite native distributions")
        ls=F.log_softmax(s[0].float(),dim=0)
        lt=F.log_softmax(t[0].float(),dim=0)
        ps,pt=ls.exp(),lt.exp()
        ids=torch.topk(s[0],min(64,vocab)).indices
        si,ti=int(s.argmax()),int(t.argmax())
        aligned=(generated==gold[:pos] and pos<len(gold))
        target=gold[pos] if aligned else None
        rows.append({
          "position":pos+1,"aligned_reference_prefix":aligned,
          "student_id":si,"teacher_id":ti,"reference_id_if_aligned":target,
          "student_correct_if_aligned":si==target if aligned else None,
          "teacher_correct_if_aligned":ti==target if aligned else None,
          "teacher_rescues_if_aligned":ti==target and si!=target if aligned else None,
          "teacher_entropy_nats":float(-(pt*lt).sum()),
          "teacher_mass_in_student_top64":float(pt[ids].sum()),
          "teacher_eos_probability":float(pt[EOS]),
          "student_eos_probability":float(ps[EOS]),
          "student_eos":si==EOS,"teacher_eos":ti==EOS,
          "student_premature_eos_if_aligned":si==EOS and target!=EOS if aligned else None,
          "teacher_premature_eos_if_aligned":ti==EOS and target!=EOS if aligned else None,
          "reverse_KL_student_teacher":float((ps*(ls-lt)).sum()),
        })
        generated.append(si)
        if si==EOS: break
    return rows
def aggregate(traces):
    rows=[r for x in traces for r in x["steps"]]
    def summary(sub):
        n=len(sub)
        if not n:return {"positions":0}
        return {"positions":n,
          "teacher_student_top1_agree":sum(r["teacher_id"]==r["student_id"] for r in sub),
          "teacher_eos":sum(r["teacher_eos"] for r in sub),
          "student_eos":sum(r["student_eos"] for r in sub),
          **{"mean_"+k:sum(r[k] for r in sub)/n for k in (
              "teacher_entropy_nats","teacher_mass_in_student_top64",
              "teacher_eos_probability","student_eos_probability","reverse_KL_student_teacher")}}
    aligned=[r for r in rows if r["aligned_reference_prefix"]]
    diverged=[r for r in rows if not r["aligned_reference_prefix"]]
    a=summary(aligned)
    a.update({k:sum(bool(r[k]) for r in aligned) for k in (
        "student_correct_if_aligned","teacher_correct_if_aligned",
        "teacher_rescues_if_aligned","student_premature_eos_if_aligned",
        "teacher_premature_eos_if_aligned")})
    return {"positions_total":len(rows),"aligned":a,
            "diverged_no_reference_correctness_claim":summary(diverged),
            "by_position":{str(i):summary([r for r in rows if r["position"]==i]) for i in range(1,MAX_STEPS+1)},
            "student_terminated_eos":sum(x["steps"][-1]["student_eos"] for x in traces),
            "student_first1_correct":sum(x["steps"][0]["student_correct_if_aligned"] is True for x in traces)}
