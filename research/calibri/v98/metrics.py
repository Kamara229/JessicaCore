"""Stable metrics for residual error accumulation, not selection objectives.

Token sequence hidden states are matched at each layer under exactly same
prefix input, not same samples from diverged autoregressive decoding.
"""
from __future__ import annotations
import math
import torch

@torch.inference_mode()
def drift(exact,compressed):
    if exact.shape!=compressed.shape or not exact.numel():
        raise AssertionError("teacher/native states must share nonempty geometry")
    a=exact.detach().float().reshape(-1)
    b=compressed.detach().float().reshape(-1)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError("nonfinite hidden residual")
    ar=float(a.square().mean().sqrt().item())
    br=float(b.square().mean().sqrt().item())
    err=float((a-b).square().mean().sqrt().item())
    denom=max(ar,1e-10)
    cosine=float(torch.nn.functional.cosine_similarity(a[None,:],b[None,:],dim=-1).item())
    return {"teacher_rms":ar,"native_rms":br,
            "absolute_rms_error":err,
            "relative_rms_error":err/denom,
            "cosine_similarity":cosine,
            "native_teacher_norm_ratio":br/denom}

@torch.inference_mode()
def logits_topk(logits,tokenizer):
    if logits.ndim!=1 or not bool(torch.isfinite(logits).all()):
        raise AssertionError("expected finite one-token full-vocabulary logits")
    x=logits.detach().float()
    p=torch.softmax(x,dim=-1)
    top=torch.topk(p,min(10,p.numel()))
    entropy=float(-(p*(p.clamp_min(1e-30)).log()).sum().item())
    scores={"entropy_nats":entropy,
        "effective_support_size":float(math.exp(min(entropy,25))),
        "top1_probability":float(top.values[0].item()),
        "top10_probability_mass":float(top.values.sum().item()),
        "top10_token_ids":[int(x) for x in top.indices.tolist()],
        "top10_text":[tokenizer.decode([int(x)]) for x in top.indices.tolist()]}
    return scores

def layer_delta(previous,now):
    if previous is None:return None
    return float(now["relative_rms_error"]-previous["relative_rms_error"])

def rank_first_divergences(rows):
    if not rows:return {"first_relative_rms_above_0p25":None,
        "first_relative_rms_above_0p50":None,"top_positive_drift_jumps":[]}
    first={}
    for threshold in (0.25,0.50):
        first[str(threshold)]=next((r["stage"] for r in rows
           if r["metrics"]["relative_rms_error"]>=threshold),None)
    return {"first_relative_rms_above_0p25":first["0.25"],
            "first_relative_rms_above_0p50":first["0.5"],
            "top_positive_drift_jumps":sorted(
               ({"stage":r["stage"],"positive_jump":r["delta_from_previous_stage"]}
                for r in rows if r["delta_from_previous_stage"] is not None),
                key=lambda r:r["positive_jump"],reverse=True)[:5],
            "ranking_is_observational_not_causal_or_fit":True}
