"""CALIBRI S121 — project 340D new-task gradients to protected null space.

An orthogonal null-space direction has zero first-order dot product with
ALL captured old-task surrogate-loss gradients (stronger than GEM's
non-increase halfspaces), but this does NOT guarantee old complete/EOS
greedy outputs stay correct. Physical S101 full456 check is mandatory.

Inspired by parameter-efficient gradient projection (PEGP, 2025), and
null-space continual learning (NESS, 2026), NOT a copy of either method.
This implementation uses a small exact SVD on the 23+ original TRAIN
gradient constraints and reuses the 340 paid S95 dual-GT coefficients.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class OrthogonalUpdate:
    direction: torch.Tensor
    original: torch.Tensor
    constraints: int
    original_violations: int
    remaining_violations: int
    projected_norm_ratio: float
    new_loss_first_order_decrease: float
    iterations: int
    converged: bool
    protected_gradient_rank: int
    protected_max_normalized_dot: float
    singular_values: tuple[float, ...]


def protected_nullspace_direction(
    new_gradient:torch.Tensor, old_gradients:torch.Tensor,
    *, rank_tol_relative:float=1e-10,
    admissible_cosine:float=1e-8,
) -> OrthogonalUpdate:
    """Euclidean orthogonal residual of -new_grad wrt all old gradients.

    Inputs do not modify weights, tokenizer, train samples, QA or model.
    Computation is CPU float64. Zero or strongly degenerate protected
    gradients contribute no nonzero constraint. A direction is returned
    only if residual is finite, orthogonal and minimizes new loss locally.
    """
    g=new_gradient.detach().cpu().double().reshape(-1)
    old=old_gradients.detach().cpu().double()
    if (g.ndim!=1 or old.ndim!=2 or old.shape[1]!=g.numel()
            or old.shape[0]<1 or g.numel()<1
            or not bool(torch.isfinite(g).all())
            or not bool(torch.isfinite(old).all())
            or not math.isfinite(rank_tol_relative)
            or not 0<rank_tol_relative<1
            or not math.isfinite(admissible_cosine)
            or not 0<admissible_cosine<1e-3
            or float(g.norm())<=0):
        raise ValueError("Invalid 340D new/protected FP64 gradient matrix")
    norms=old.norm(dim=1)
    enabled=norms>1e-14
    protected=(old[enabled]/norms[enabled,None])
    original=-g
    if protected.numel():
        _u,values,vt=torch.linalg.svd(protected,full_matrices=False)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("SVD singular values nonfinite")
        tolerance=max(
            torch.finfo(torch.float64).eps*max(protected.shape)*float(values[0]),
            rank_tol_relative*float(values[0]),
        )
        rank=int((values>tolerance).sum())
        if rank:
            basis=vt[:rank]
            projection=(original@basis.T)@basis
            residual=original-projection
        else:
            residual=original.clone()
    else:
        values=torch.empty(0,dtype=torch.float64)
        rank=0
        residual=original.clone()
    if not bool(torch.isfinite(residual).all()):
        raise AssertionError("Non-finite protected null-space projected direction")
    cosine=(
        float((protected@residual).abs().max())/max(float(residual.norm()),1e-30)
        if protected.numel() else 0.0
    )
    violation=cosine>admissible_cosine
    if violation:
        # Never return a gradient with nonzero first-order OLD loss
        # under insufficient SVD precision/conditioning.
        residual=torch.zeros_like(original)
        cosine=0.0
    if float(g@residual)>1e-8:
        residual.zero_()
    return OrthogonalUpdate(
        direction=residual,
        original=original,
        constraints=int(protected.shape[0]),
        original_violations=int(((protected@original)>admissible_cosine).sum())
        if protected.numel() else 0,
        remaining_violations=int(violation),
        projected_norm_ratio=float(residual.norm()/original.norm()),
        new_loss_first_order_decrease=float(-(g@residual)),
        iterations=1,
        converged=not violation,
        protected_gradient_rank=rank,
        protected_max_normalized_dot=cosine,
        singular_values=tuple(float(x) for x in values),
    )
