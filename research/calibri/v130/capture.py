"""Read-only matched hidden states on the same tokenized original prompt."""
from contextlib import contextmanager, ExitStack

import torch

from .metrics import similarity, swap_alignment

POINTS = (
    "model.embed_tokens",
    "model.layers.0", "model.layers.7", "model.layers.15",
    "model.layers.23", "model.layers.29", "model.norm",
)


@contextmanager
def observations(model):
    latest = {}
    with ExitStack() as stack:
        for name in POINTS:
            module = model.get_submodule(name)

            def hook(_module, _inputs, output, key=name):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(value, torch.Tensor) or value.ndim != 3:
                    raise ValueError("incompatible hidden output: " + key)
                last = value[0, -1].detach().float().cpu().clone()
                if not torch.isfinite(last).all():
                    raise ValueError("nonfinite hidden state: " + key)
                latest[key] = last

            handle = module.register_forward_hook(hook)
            stack.callback(handle.remove)
        yield latest


def read_snapshot(latest):
    if set(latest) != set(POINTS):
        raise ValueError("incomplete hidden state capture")
    return {key: value.clone() for key, value in latest.items()}


def compare(reference, candidate, cases):
    by_id = {case["id"]: case for case in cases}
    if set(reference) != set(by_id) or set(candidate) != set(by_id):
        raise ValueError("missing or additional activation cases")
    layer_metrics = {}
    for point in POINTS:
        matched = [similarity(reference[ident][point], candidate[ident][point])
                   for ident in by_id]
        swaps = []
        for case in cases:
            if case["orientation"] != "original":
                continue
            pair, focus = case["pair"], case["focus"]
            forward = pair + ":" + focus + ":original"
            reverse = pair + ":" + focus + ":swapped"
            score = swap_alignment(reference[forward][point],
                                   reference[reverse][point],
                                   candidate[forward][point],
                                   candidate[reverse][point])
            if score is not None:
                swaps.append(score)
        avg = lambda key, values: (sum(x[key] for x in values) / len(values)
                                   if values else None)
        layer_metrics[point] = {
            "mean_relative_hidden_error": avg("relative_error", matched),
            "mean_hidden_cosine": avg("cosine", matched),
            "swap_count_with_nonzero_teacher_signal": len(swaps),
            "mean_swap_direction_cosine": avg("cosine", swaps),
            "mean_swap_delta_relative_error": avg("relative_error", swaps),
        }
    return layer_metrics
