"""S87 integrated Green Tunnel: small internal neural residuals for a CB59 Calibri2 core.

RESEARCH ONLY. This module defines the forward interface and a bounded
joint TRAIN step. It does not claim a validated physical candidate, QA gain,
or automatic acceptance. Accepted S81/S82 source files remain immutable.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from v176.operators import clear


class InternalTunnelBlock(nn.Module):
    """Zero-initialized, 320->rank->320 contextual transformer residual."""
    def __init__(self, width: int = 320, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.gate = nn.Linear(width, 1, bias=True)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(h, (h.shape[-1],))
        return self.up(torch.tanh(self.down(x))) * torch.sigmoid(self.gate(x))


class IntegratedGreenTunnel(nn.Module):
    """Attach internal trainable residuals to transformer layers; no answer lookup.

    Hooks are removable and preserve original attention/cache outputs.
    Each block is initially an exact zero residual, so frozen S82 inference
    is not changed before training. Registered on the model to ensure device
    placement and state serialization are explicit.
    """
    def __init__(self, width: int = 320, rank: int = 8,
                 layers: tuple[int, ...] = (3, 9)):
        super().__init__()
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("Need distinct integration layers")
        self.layers = layers
        self.blocks = nn.ModuleDict({
            str(index): InternalTunnelBlock(width, rank) for index in layers
        })
        self._handles = []

    def install(self, model: nn.Module):
        if self._handles:
            raise RuntimeError("Green Tunnel already installed")
        blocks = model.model.layers
        if max(self.layers) >= len(blocks):
            raise ValueError("Target model has fewer transformer layers")
        for index in self.layers:
            block = self.blocks[str(index)]
            def hook(_layer, _args, output, *, adapter=block):
                if isinstance(output, tuple):
                    return (output[0] + adapter(output[0]), *output[1:])
                if isinstance(output, torch.Tensor):
                    return output + adapter(output)
                raise TypeError("Unsupported transformer decoder output")
            self._handles.append(blocks[index].register_forward_hook(hook))
        return self

    def uninstall(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def unique_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


@dataclass(frozen=True)
class JointStep:
    loss: float
    core_gradient_norm: float
    tunnel_gradient_norm: float
    selected_codebook_count: int
    nonzero_core_gradient_tensors: int
    nonzero_tunnel_gradient_tensors: int
    zero_tunnel_gradient_tensors: tuple[str, ...]


def select_paid_core_codebooks(sources: dict, target_layers=(3, 9)):
    """Choose existing paid FP16 dictionaries only; never allocate dense weights.

    Deliberately touch a small subset of old CB59 operators in the first
    prototype. Other CB59 centroids, row scales and discrete codes are frozen.
    """
    for source in sources.values():
        source.centroids.requires_grad_(False)
        source.log_scales.requires_grad_(False)
    selected = {
        name: source for name, source in sources.items()
        if any(name.startswith(f"model.layers.{i}.") and
               name.endswith("mlp.down_proj.weight") for i in target_layers)
    }
    if len(selected) != len(target_layers):
        raise AssertionError("Expected one existing paid MLP projection per layer")
    for source in selected.values():
        source.centroids.requires_grad_(True)
    return selected


def teacher_forced_train_loss(model, tok, question: str, answer: str,
                              *, device="mps"):
    """Train whole response including EOS; no fixed first/second-token gate."""
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    gold = tok.encode(answer, add_special_tokens=False) + [tok.eos_token_id]
    if not prompt_ids or len(gold) < 2:
        raise ValueError("Invalid supervised full answer")
    all_ids = prompt_ids + gold[:-1]
    inp = torch.tensor([all_ids], dtype=torch.long, device=device)
    prediction = model(input_ids=inp, use_cache=False).logits
    gold_logits = prediction[0, len(prompt_ids)-1:len(prompt_ids)-1+len(gold)]
    if gold_logits.shape[0] != len(gold):
        raise AssertionError("Full answer and EOS positions misaligned")
    return F.cross_entropy(gold_logits, torch.tensor(gold, device=device))


def joint_train_step(model, sources, tunnel: IntegratedGreenTunnel,
                     selected_codebooks, tok, examples, optimizer) -> JointStep:
    """One MPS joint gradient step through CB59 dictionaries AND internal tunnel.

    No physical acceptance is performed here: a separate guarded, replayed
    CPU inference + exact source/controller/manifest byte audit is mandatory.
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError("Real MPS GPU is required")
    if not examples:
        raise ValueError("TRAIN examples are required")
    if any(p.device.type != "mps" for p in tunnel.parameters()):
        raise AssertionError("Internal Green Tunnel is not on MPS GPU")
    if any(s.centroids.device.type != "mps" for s in selected_codebooks.values()):
        raise AssertionError("Selected core dictionaries are not on MPS GPU")
    clear(sources)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    try:
        for question, gold_answer in examples:
            clear(sources)
            losses.append(teacher_forced_train_loss(
                model, tok, question, gold_answer))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite joint objective")
        loss.backward()
        core = [s.centroids for s in selected_codebooks.values()]
        side = list(tunnel.parameters())
        if any(p.grad is None for p in core + side):
            raise AssertionError("Missing gradients in compressed core or tunnel")
        # A finite scalar loss does not guarantee finite per-parameter gradients.
        # Reject NaN/Inf before clipping or mutating the physically paid weights.
        if any(not bool(torch.isfinite(p.grad).all().item()) for p in core + side):
            raise AssertionError("Nonfinite joint gradient in paid core or internal tunnel")
        # A zero-initialized up projection blocks first-step down/gate gradients.
        # Count actual nonzero gradients rather than only checking grad != None.
        nonzero_core = sum(bool(torch.count_nonzero(p.grad).item()) for p in core)
        nonzero_tunnel = sum(bool(torch.count_nonzero(p.grad).item()) for p in side)
        zero_tunnel = tuple(name for name, p in tunnel.named_parameters()
                            if not bool(torch.count_nonzero(p.grad).item()))
        up_names = tuple(name for name, _ in tunnel.named_parameters()
                         if name.endswith(".up.weight"))
        if nonzero_core != len(core) or any(name in zero_tunnel for name in up_names):
            raise AssertionError("No real joint gradient in paid core or tunnel up")
        core_norm = float(torch.nn.utils.clip_grad_norm_(core, 1.0))
        tunnel_norm = float(torch.nn.utils.clip_grad_norm_(side, 1.0))
        optimizer.step()
        with torch.no_grad():
            for p in core: p.clamp_(-16.0, 16.0)
        return JointStep(float(loss.detach()), core_norm, tunnel_norm,
                         len(selected_codebooks), nonzero_core, nonzero_tunnel,
                         zero_tunnel)
    finally:
        clear(sources)
