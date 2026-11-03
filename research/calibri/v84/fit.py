"""Train new physical CB59 operator codebooks on true pretrained teacher INPUTS.

A differentiable reconstruction of the 5bit-native mathematical operator is
used only during source construction; it is NEVER the inference backend.
The candidate remains exactly 32x8 FP16 centroids per rewritten operator.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from direct_codebook_v59 import K, VECTOR, unpack5

CHECKPOINTS = (12, 36, 72)
BATCH_TOKENS = 32
REGULARIZATION = 0.015
LR = 0.012


@dataclass
class Projection:
    path: str
    code_index: torch.Tensor
    row_scales: torch.Tensor
    input_width: int
    base_codebook: torch.Tensor
    bias: torch.Tensor | None = None

    @classmethod
    def from_native(cls, path, layer):
        h = layer.handle
        m, n = h.shape
        if n % VECTOR != 0:
            raise AssertionError("CB59 operator input not divisible by 8")
        groups = n // VECTOR
        codes = unpack5(h.packed.tobytes(), m * groups).reshape(m, groups)
        if codes.min() < 0 or codes.max() >= K:
            raise AssertionError("corrupt packed 5-bit code")
        if layer.bias is not None:
            raise AssertionError("initial rewrite prototype targets bias-free down_proj only")
        return cls(
            path=path,
            code_index=torch.from_numpy(codes.astype(np.int64)),
            row_scales=torch.from_numpy(h.row_scales.copy()),
            input_width=n,
            base_codebook=torch.from_numpy(h.centroids.copy()))

    def output(self, x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.input_width:
            raise AssertionError("wrong teacher activation shape for operator")
        m = self.code_index.shape[0]
        reconstructed = (centroids[self.code_index].reshape(m, self.input_width)
                         * self.row_scales[:, None])
        return x @ reconstructed.T


def train_codebook(projection: Projection,
                   train_inputs: torch.Tensor,
                   train_targets: torch.Tensor) -> dict[int, np.ndarray]:
    if train_inputs.shape[0] != train_targets.shape[0]:
        raise AssertionError("paired teacher targets absent")
    initial = projection.base_codebook
    centroids = torch.nn.Parameter(initial.clone())
    optim = torch.optim.Adam([centroids], lr=LR)
    n = train_inputs.shape[0]
    if n < 64:
        raise AssertionError("true TRAIN sample too small")
    baseline_energy = train_targets.square().mean().item()
    centroid_energy = initial.square().mean().item()
    if baseline_energy <= 0 or centroid_energy <= 0:
        raise AssertionError("degenerate pretrained teacher source")
    history = {}
    for step in range(1, max(CHECKPOINTS) + 1):
        begin = ((step - 1) * BATCH_TOKENS) % n
        indices = torch.arange(begin, begin + BATCH_TOKENS) % n
        x, y = train_inputs[indices], train_targets[indices]
        predicted = projection.output(x, centroids)
        reconstruction = (predicted - y).square().mean() / baseline_energy
        regularization = (centroids - initial).square().mean() / centroid_energy
        loss = reconstruction + REGULARIZATION * regularization
        if not bool(torch.isfinite(loss)):
            raise AssertionError("codebook rewrite diverged")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        with torch.no_grad():
            centroids.clamp_(-120, 120)
        if step in CHECKPOINTS:
            values = centroids.detach().numpy().astype("<f2").astype(np.float32)
            if not np.isfinite(values).all() or np.array_equal(
                    values, initial.numpy()):
                raise AssertionError("candidate is not a genuine changed FP16 operator")
            history[step] = values.copy()
    return history


def local_relative_error(projection: Projection, x: torch.Tensor,
                         y: torch.Tensor, codebook: np.ndarray) -> float:
    with torch.inference_mode():
        c = torch.as_tensor(np.asarray(codebook, dtype=np.float32))
        output = projection.output(x, c)
        numerator = (output - y).double().square().sum()
        denominator = y.double().square().sum()
        if denominator <= 0:
            raise AssertionError("teacher has zero original output")
        return float((numerator / denominator).sqrt())
