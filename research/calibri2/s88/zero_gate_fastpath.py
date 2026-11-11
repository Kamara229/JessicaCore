"""S88 research-only exact zero-gate inference fast path for S87 internal tunnel.

The S87 pilot trains one optimizer step. Gate parameters are initially zero
and have zero first-step gradients while the upstream projection starts at
zero. Never assume this invariant: inspect the *serialized* gates before
replacing a forward path. This does not create new weights or change bpp.
"""
from __future__ import annotations

from types import MethodType
import torch
from torch.nn import functional as F


def _fast_zero_gate_forward(self, hidden: torch.Tensor) -> torch.Tensor:
    normal = F.layer_norm(hidden, (hidden.shape[-1],))
    return self.up(torch.tanh(self.down(normal))) * 0.5


def install_verified_fastpath(tunnel) -> tuple[str, ...]:
    """Replace only proven-zero trained gates; reject all other candidates."""
    if tunnel.training:
        raise AssertionError("Never patch training: inference only")
    blocks = tunnel.blocks
    for name, block in blocks.items():
        if block.training:
            raise AssertionError("Internal tunnel block not in eval mode")
        with torch.no_grad():
            if (torch.count_nonzero(block.gate.weight).item() or
                    torch.count_nonzero(block.gate.bias).item()):
                raise AssertionError(
                    f"S88 fast path is invalid: trained gate {name} is nonzero")
            # Test the exact float32 residual rather than only asserting
            # an algebraic identity. The gate is identically sigmoid(0)=0.5.
            probe = torch.linspace(-1.0, 1.0, steps=11 * block.down.in_features,
                                   dtype=block.down.weight.dtype,
                                   device=block.down.weight.device).view(
                                       1, 11, block.down.in_features)
            original = block(probe)
            fast = _fast_zero_gate_forward(block, probe)
            if not torch.equal(original, fast):
                raise AssertionError(
                    f"S88 fast path changed the exact residual on layer {name}")
    for block in blocks.values():
        block.forward = MethodType(_fast_zero_gate_forward, block)
    return tuple(blocks.keys())
