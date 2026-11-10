"""S117: stable first-order protected gradient projection for 340 S111 gates.

Research tool only: a protected-loss non-increasing first-order direction
is NOT actual token retention. MUST still run physical all-456 greedy CPU
test, all-inclusive bpp and CPU speed before any learned candidate.
No optimizer steps or accepted models in S117.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProjectedDirection:
    direction: torch.Tensor
    original: torch.Tensor
    constraints: int
    originally_violated: int
    remaining_violations: int
    projected_norm_ratio: float
    new_loss_first_order_decrease: float
    iterations: int
    converged: bool


def project_for_old_skills(
    new_gradient:torch.Tensor,old_gradients:torch.Tensor, *,
    max_iter:int=800, eps:float=1e-7,
) -> ProjectedDirection:
    """Project -g_new onto intersections of 0-loss-increase halfspaces.

    Dykstra cyclic Euclidean convex projection; each protected normal is
    independently normalized. Computation in CPU float64, zero norm
    protected gradients skipped. Numerical failures return a zero vector,
    not an unsafe pseudo-projection.
    """
    g=new_gradient.detach().flatten().double().cpu()
    olds=old_gradients.detach().double().cpu()
    if (olds.ndim!=2 or olds.shape[1]!=g.numel()
            or olds.shape[0]<1 or g.numel()<1):
        raise ValueError("Gradient and protected matrix dimension mismatch")
    if (not bool(torch.isfinite(g).all())
            or not bool(torch.isfinite(olds).all())
            or not torch.isfinite(g.norm())
            or not eps>0 or max_iter<1 or g.norm().item()==0):
        raise ValueError("Nonfinite or empty gradients")
    normals=olds / olds.norm(dim=1,keepdim=True).clamp_min(1e-20)
    normals=normals[olds.norm(dim=1)>1e-14]
    original=-g
    direction=original.clone()
    corrections=torch.zeros_like(normals)
    converged=False
    count=0
    for step in range(max_iter):
        count=step+1
        prev=direction.clone()
        for i in range(normals.shape[0]):
            shifted=direction+corrections[i]
            violation=float(torch.dot(shifted,normals[i]))
            new=shifted-max(0.0,violation)*normals[i]
            corrections[i]=shifted-new
            direction=new
        violations=normals@direction if normals.numel() else torch.zeros(0)
        if ((not violations.numel() or float(violations.max())<=eps)
                and float((direction-prev).norm())<=eps):
            converged=True
            break
    if normals.numel() and float((normals@direction).max())>eps:
        direction.zero_()
    if float(torch.dot(direction,g))>eps:
        direction.zero_()
    raw_violate=normals@original if normals.numel() else torch.zeros(0)
    final_violate=normals@direction if normals.numel() else torch.zeros(0)
    return ProjectedDirection(
        direction=direction,
        original=original,
        constraints=int(normals.shape[0]),
        originally_violated=int((raw_violate>eps).sum()),
        remaining_violations=int((final_violate>eps).sum()),
        projected_norm_ratio=float(direction.norm()/original.norm()),
        new_loss_first_order_decrease=float(-torch.dot(g,direction)),
        iterations=count,
        converged=converged,
    )
