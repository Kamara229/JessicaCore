"""S87 integrated Green Tunnel: small internal neural residuals for a CB59 Calibri2 core.

RESEARCH ONLY. This module defines the forward interface and a bounded
joint TRAIN step. It does not claim a validated physical candidate, QA gain,
or automatic acceptance. Accepted S81/S82 source files remain immutable.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
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
    supervised_example_count: int
    protected_replay_count: int
    protected_replay_weight: float


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
                     selected_codebooks, tok, examples, optimizer, *,
                     replay_examples=(), replay_weight=0.5, post_step_guard=None,
                     require_deep_tunnel_gradients=False) -> JointStep:
    """One MPS joint gradient step through CB59 dictionaries AND internal tunnel.

    No physical acceptance is performed here: a separate guarded, replayed
    CPU inference + exact source/controller/manifest byte audit is mandatory.
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError("Real MPS GPU is required")
    examples = list(examples)
    replay_examples = list(replay_examples)
    if not examples:
        raise ValueError("New ChatGPT-authored TRAIN examples are required")
    if replay_examples and not 0.0 < replay_weight < 1.0:
        raise ValueError("Protected replay weight must be strictly between zero and one")
    if not replay_examples:
        replay_weight = 0.0
    if not callable(post_step_guard):
        raise ValueError("Transactional post-step TRAIN guard is mandatory")
    if any(p.device.type != "mps" for p in tunnel.parameters()):
        raise AssertionError("Internal Green Tunnel is not on MPS GPU")
    if any(s.centroids.device.type != "mps" for s in selected_codebooks.values()):
        raise AssertionError("Selected core dictionaries are not on MPS GPU")
    clear(sources)
    optimizer.zero_grad(set_to_none=True)
    # Stream each cohort separately, releasing each sequence graph.
    # Project conflicting new gradients against protected replay before Adam.
    loss_terms = []
    groups = (
        [(examples, 1.0)] if not replay_examples else
        [(examples, 1.0 - replay_weight), (replay_examples, replay_weight)]
    )
    core = [s.centroids for s in selected_codebooks.values()]
    side = list(tunnel.parameters())
    params = core + side
    new_grads = None
    try:
        for group_index, (group, group_weight) in enumerate(groups):
            if not group:
                continue
            per_example_weight = group_weight / len(group)
            for question, gold_answer in group:
                clear(sources)
                example_loss = teacher_forced_train_loss(
                    model, tok, question, gold_answer)
                if not bool(torch.isfinite(example_loss.detach()).all().item()):
                    raise AssertionError("Nonfinite per-example joint objective")
                loss_terms.append(example_loss.detach() * per_example_weight)
                (example_loss * per_example_weight).backward()
                del example_loss
            if replay_examples and group_index == 0:
                if any(p.grad is None for p in params):
                    raise AssertionError("Missing new TRAIN gradients")
                new_grads = [p.grad.detach().float().clone() for p in params]
                optimizer.zero_grad(set_to_none=True)
        if new_grads is not None:
            # One-sided projection: protect replay, never project replay away.
            # Compute separately for core and tunnel because they are clipped
            # and optimized at different scales. The physical guard remains
            # authoritative: Adam and FP16 rounding can still cause regression.
            if any(p.grad is None for p in params):
                raise AssertionError("Missing replay gradients")
            replay_grads = [p.grad.detach().float().clone() for p in params]
            for start, stop in ((0, len(core)), (len(core), len(params))):
                dot = sum((new_grads[i] * replay_grads[i]).sum()
                          for i in range(start, stop))
                norm_sq = sum(replay_grads[i].square().sum()
                              for i in range(start, stop))
                if not bool(torch.isfinite(dot).item()) or not bool(torch.isfinite(norm_sq).item()):
                    raise AssertionError("Nonfinite gradient projection")
                coefficient = float(dot.item()) / float(norm_sq.item()) if float(dot.item()) < 0 and float(norm_sq.item()) > 0 else 0.0
                for i in range(start, stop):
                    combined = new_grads[i] - coefficient * replay_grads[i] + replay_grads[i]
                    params[i].grad.copy_(combined.to(params[i].grad.dtype))
        loss = torch.stack(loss_terms).sum()
        if not bool(torch.isfinite(loss).item()):
            raise AssertionError("Nonfinite accumulated joint objective")
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
        if require_deep_tunnel_gradients:
            missing = [name for name in zero_tunnel if name.endswith(
                (".down.weight", ".gate.weight", ".gate.bias"))]
            if missing:
                raise AssertionError(
                    f"Deep internal tunnel gradients absent before Adam step: {missing}")
        core_norm = float(torch.nn.utils.clip_grad_norm_(core, 1.0))
        tunnel_norm = float(torch.nn.utils.clip_grad_norm_(side, 1.0))
        # The optimizer mutates both parameters and Adam moments. A failed
        # post-step check must restore BOTH, or a later retry starts from a
        # corrupted state despite reporting a rejected candidate.
        snapshot = [p.detach().clone() for p in core + side]
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        try:
            optimizer.step()
            # Finite gradients do not guarantee finite optimizer-updated FP16
            # centroids or adapter parameters. Reject before physical packing.
            with torch.no_grad():
                for p in core:
                    p.clamp_(-16.0, 16.0)
                if any(not bool(torch.isfinite(p).all().item()) for p in core + side):
                    raise AssertionError("Nonfinite joint parameter after optimizer step")
            # A guard may serialize the compact candidate and replay protected
            # TRAIN identities. Its rejection must roll back Adam moments too.
            post_step_guard()
        except Exception:
            with torch.no_grad():
                for p, old in zip(core + side, snapshot):
                    p.copy_(old)
            optimizer.load_state_dict(optimizer_snapshot)
            raise
        return JointStep(float(loss.detach().cpu()), core_norm, tunnel_norm,
                         len(selected_codebooks), nonzero_core, nonzero_tunnel,
                         zero_tunnel, len(examples), len(replay_examples),
                         float(replay_weight))
    finally:
        clear(sources)
