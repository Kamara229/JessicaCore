"""Teacher-normalized KL and actual next-token NLL, not response matching."""
from statistics import mean
import torch
from torch.nn import functional as F


@torch.inference_mode()
def compare_logits(teacher,student,sequence):
    if teacher.shape!=student.shape or teacher.ndim!=3:
        raise ValueError("teacher and student logits must be same [1,seq,vocab]")
    if sequence.ndim!=2 or sequence.shape!=teacher.shape[:2] or len(sequence[0])<2:
        raise ValueError("unmatched validation source sequence")
    if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
        raise ValueError("nonfinite LLM logits")
    next_token=sequence[0,1:].long()
    reference=teacher[0,:-1].to(torch.float64)
    predicted=student[0,:-1].to(torch.float64)
    reference_logp=F.log_softmax(reference,dim=-1)
    candidate_logp=F.log_softmax(predicted,dim=-1)
    distribution=reference_logp.exp()
    target=(next_token[:,None])
    NLL=-candidate_logp.gather(-1,target).squeeze(-1)
    teacher_nll=-reference_logp.gather(-1,target).squeeze(-1)
    kl=(distribution*(reference_logp-candidate_logp)).sum(-1)
    if torch.any(kl< -1e-6):raise ValueError("unexpectedly negative teacher KL")
    greedy_match=(reference.argmax(-1)==predicted.argmax(-1))
    return {
        "tokens_scored":int(next_token.numel()),
        "sum_NLL":float(NLL.sum()),
        "sum_teacher_NLL":float(teacher_nll.sum()),
        "sum_teacher_to_student_KL":float(kl.clamp_min(0).sum()),
        "exact_FP32_next_token_argmax_count":int(greedy_match.sum()),
        "first_EOS_count":None,
    }


def aggregate(records):
    if not records:raise ValueError("no validation windows")
    count=sum(r["tokens_scored"] for r in records)
    if count<2:raise ValueError("too few validation tokens")
    nll=sum(r["sum_NLL"] for r in records)/count
    teacher_nll=sum(r["sum_teacher_NLL"] for r in records)/count
    kl=sum(r["sum_teacher_to_student_KL"] for r in records)/count
    return {
        "tokens_scored":count,
        "mean_TRUE_next_token_NLL":nll,
        "perplexity":float(torch.exp(torch.tensor(nll,dtype=torch.float64))),
        "teacher_mean_TRUE_NLL":teacher_nll,
        "teacher_to_candidate_KL":kl,
        "teacher_next_token_argmax_agreement_fraction":
            sum(r["exact_FP32_next_token_argmax_count"] for r in records)/count,
    }


def contrast(baseline,alternative):
    out={}
    for domain in sorted(baseline):
        original=aggregate(baseline[domain])
        new=aggregate(alternative[domain])
        if original["tokens_scored"]!=new["tokens_scored"]:
            raise ValueError("different source validation windows")
        out[domain]={
            "CB59":original,"paid_CB140_LUT":new,
            "candidate_minus_baseline_NLL":
                new["mean_TRUE_next_token_NLL"]-original["mean_TRUE_next_token_NLL"],
            "candidate_minus_baseline_teacher_KL":
                new["teacher_to_candidate_KL"]-original["teacher_to_candidate_KL"],
            "candidate_vs_CB59_NLL_better":
                new["mean_TRUE_next_token_NLL"]<original["mean_TRUE_next_token_NLL"],
        }
    return out
