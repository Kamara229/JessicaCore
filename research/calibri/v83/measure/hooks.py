"""Direct paired local operator probes and propagated hidden-state probes.

Local operator probes apply each original CB59 packed projection to EXACTLY
the teacher's own inputs; upstream quantization cannot contaminate this metric.
Layer output probes separately measure cumulative whole-model drift.
"""
from __future__ import annotations
from contextlib import ExitStack
import torch
from torch import nn
from direct_runtime_v59 import PackedLinear
from .metrics import MetricBook

PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def _hidden(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def probe_one_window(teacher: nn.Module, packed: nn.Module,
                     tokens: torch.Tensor, book: MetricBook) -> dict:
    """Compute 7 projections x all layers and full-block outputs on TRAIN."""
    if tokens.ndim != 1 or tokens.numel() < 4:
        raise AssertionError("one nonempty true token window is required")
    n = len(teacher.model.layers)
    if n != len(packed.model.layers) or not n:
        raise AssertionError("same complete original transformer depth required")
    hits: dict[str, int] = {}
    teacher_layers: dict[str, torch.Tensor] = {}
    handles = []
    try:
        for i in range(n):
            layer = f"model.layers.{i}"
            for role in PROJECTIONS:
                path = f"{layer}.{role}"
                real = teacher.get_submodule(path)
                compressed = packed.get_submodule(path)
                if not isinstance(real, nn.Linear) or not isinstance(compressed, PackedLinear):
                    raise AssertionError("native C packed trained projection absent: " + path)
                hits[path] = 0

                def local_hook(_module, inputs, output, key=path, fn=compressed):
                    if len(inputs) != 1:
                        raise AssertionError("projection has unexpected input count")
                    observed = fn(inputs[0].detach())
                    book.observe("teacher_forced_local_projection", key,
                                 output, observed)
                    hits[key] += 1

                handles.append(real.register_forward_hook(local_hook))
            teacher_key = layer

            def save_teacher(_module, inputs, output, key=teacher_key):
                if key in teacher_layers:
                    raise AssertionError("teacher layer invoked multiple times")
                teacher_layers[key] = _hidden(output).detach().clone()

            def score_packed(_module, inputs, output, key=teacher_key):
                if key not in teacher_layers:
                    raise AssertionError("packed ran without corresponding teacher")
                book.observe("propagated_layer_output", key,
                             teacher_layers.pop(key), _hidden(output))

            handles.append(teacher.model.layers[i].register_forward_hook(save_teacher))
            handles.append(packed.model.layers[i].register_forward_hook(score_packed))
        with torch.inference_mode():
            _ = teacher(input_ids=tokens.unsqueeze(0), use_cache=False)
            _ = packed(input_ids=tokens.unsqueeze(0), use_cache=False)
        if teacher_layers or any(count != 1 for count in hits.values()):
            raise AssertionError("missing/duplicated teacher-forced projection or layer")
    finally:
        for h in handles:
            h.remove()
    return {"layers": n, "local_projection_count": len(hits),
            "source_tokens": int(tokens.numel()),
            "teacher_input_conditioned": True,
            "propagated_error_separately_recorded": True}
