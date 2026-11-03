"""TRAIN-only original-model-targeted conditional gate learning.

Teacher target y_teacher at each token, student native x at that token.
This may compensate upstream compression drift, unlike v84's local
teacher-input-only codebook training. No native weights or source bits
are modified during gate fitting.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import torch
from direct_runtime_v59 import PackedLinear

STEPS=84
BATCH_TOKENS=64
LR=0.016
REG_W=0.06


def native_dual_outputs(module:PackedLinear,alt_centroids:np.ndarray,
                        x:torch.Tensor):
    if not isinstance(module,PackedLinear) or module.bias is not None:
        raise AssertionError("must use direct native packed original operator")
    extra=PackedLinear(replace(module.handle,centroids=np.asarray(
        alt_centroids,dtype=np.float32).copy()),module.kernel)
    with torch.inference_mode():
        base=module(x)
        modified=extra(x)
    # Avoid inference tensors in later autograd's SavedVariable.
    return base.detach().clone(),modified.detach().clone()


def fit_activation_gate(x:torch.Tensor,teacher_y:torch.Tensor,
                        base_y:torch.Tensor,alt_y:torch.Tensor):
    if (x.ndim!=2 or teacher_y.shape!=base_y.shape or
        teacher_y.shape!=alt_y.shape or x.shape[0]!=teacher_y.shape[0]):
        raise AssertionError("misaligned actual compressed trajectories")
    n,d=x.shape
    if n<128:
        raise AssertionError("too few original genuine TRAIN activations")
    normalized=x/torch.sqrt(x.square().mean(dim=-1,keepdim=True)+1e-6)
    y0=base_y.detach().clone()
    delta=(alt_y-base_y).detach().clone()
    target=teacher_y.detach().clone()
    initial_bias=torch.tensor([-0.7],dtype=torch.float32)
    w=torch.nn.Parameter(torch.zeros(d,dtype=torch.float32))
    bias=torch.nn.Parameter(initial_bias)
    optimizer=torch.optim.Adam((w,bias),lr=LR)
    teacher_energy=float(target.square().mean())
    if not teacher_energy>0:
        raise AssertionError("invalid actual original teacher projection energy")
    losses=[]
    for step in range(1,STEPS+1):
        first=((step-1)*BATCH_TOKENS)%n
        ids=torch.arange(first,first+BATCH_TOKENS)%n
        gate=torch.sigmoid(normalized[ids]@w+bias)
        output=y0[ids]+delta[ids]*gate[:,None]
        loss=(output-target[ids]).square().mean()/teacher_energy
        loss=loss+REG_W*w.square().sum() + .002*bias.square().mean()
        if not bool(torch.isfinite(loss)):
            raise AssertionError("adaptive operator fitting diverged")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            w.clamp_(-.20,.20)
            bias.clamp_(-6.0,6.0)
        if step in (12,36,84):
            with torch.no_grad():
                g=torch.sigmoid(normalized@w+bias)
                all_out=y0+delta*g[:,None]
                train_err=float((all_out-target).square().mean()/teacher_energy)
                losses.append({"checkpoint":step,"TRAIN_relative_squared_error":train_err,
                               "mean_dynamic_gate":float(g.mean()),
                               "std_dynamic_gate":float(g.std())})
    values=w.detach().numpy().astype("<f2")
    b=bias.detach().numpy().astype("<f2")
    if not np.isfinite(values).all() or not np.isfinite(b).all():
        raise AssertionError("physical adaptive operator gate FP16 diverged")
    with torch.no_grad():
        q_w=torch.from_numpy(values.astype(np.float32))
        q_b=torch.tensor(float(b[0]))
        gates=torch.sigmoid(normalized@q_w+q_b)
        candidate=y0+delta*gates[:,None]
        original_error=float((y0-target).square().mean()/teacher_energy)
        calibrated_error=float((candidate-target).square().mean()/teacher_energy)
    return (values.copy(),b.copy()),{
        "TRAIN_student_activation_examples":n,
        "gate_input_dimension":d,
        "original_student_TRAIN_relative_squared_error":original_error,
        "trained_gated_TRAIN_relative_squared_error":calibrated_error,
        "gate_mean":float(gates.mean()),
        "gate_std":float(gates.std()),
        "gate_range":[float(gates.min()),float(gates.max())],
        "checkpoint_trace":losses,
        "trained_on_real_student_activations":True,
        "targets_original_teacher_real_model_output":True}
