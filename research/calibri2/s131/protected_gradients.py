"""S131: project TRAIN gradient against independently protected old losses.

This is only a first-order guard on TRAIN gradients. It is NOT a proof of
no forgetting: actual persisted 456 full/EOS CPU J7 guard is mandatory.
No external teacher, QA/TEST, inspected S118, or original core mutation.
"""
from __future__ import annotations
import torch

def project_against_old_loss_increase(
    proposal:torch.Tensor, protected:list[torch.Tensor], *,
    passes:int=8, min_norm:float=1e-12,
)->tuple[torch.Tensor,dict]:
    """Enforce near-nonnegative g·g_old for each old TRAIN gradient.

    The optimizer updates -g. Thus positive g·g_old locally reduces old loss.
    Repeated cyclic halfspace corrections are conservative; true quality is
    independently guarded by real FP16/Q2 CPU answer identity checks.
    """
    if not proposal.is_floating_point() or not bool(torch.isfinite(proposal).all()):
        raise ValueError("S131 invalid proposed TRAIN gradient")
    if not protected or not 1<=passes<=32:
        raise ValueError("S131 requires protected TRAIN gradients and finite passes")
    shape=proposal.shape
    unit=[]
    for q in protected:
        if q.shape!=shape or q.device!=proposal.device or not bool(torch.isfinite(q).all()):
            raise ValueError("S131 protected gradient shape/device/finite mismatch")
        denom=torch.linalg.vector_norm(q.float())
        if float(denom)>min_norm:
            unit.append(q.float()/denom)
    if not unit:
        raise AssertionError("S131 no nonzero historical TRAIN constraints")
    current=proposal.float().clone()
    before=[float(torch.sum(current*u)) for u in unit]
    corrections=0
    for _ in range(passes):
        adjusted=0
        for u in unit:
            dot=torch.sum(current*u)
            if float(dot)<0:
                current=current-dot*u
                adjusted+=1
        corrections+=adjusted
        if adjusted==0:
            break
    after=[float(torch.sum(current*u)) for u in unit]
    if not bool(torch.isfinite(current).all()):
        raise AssertionError("S131 gradient projection nonfinite")
    return current.to(dtype=proposal.dtype),{
        "protected_gradient_constraints":len(unit),
        "negative_dot_before":sum(v< -1e-8 for v in before),
        "negative_dot_after":sum(v< -1e-8 for v in after),
        "min_normalized_alignment_after":min(after),
        "cyclic_projection_corrections":corrections,
    }
