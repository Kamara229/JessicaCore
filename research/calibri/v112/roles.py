"""Structural roles are the known GPT-NeoX architecture, not discovered synapses."""
import re
import numpy as np

ROLE_NAMES = ("attention_qkv", "attention_out", "mlp_up", "mlp_down")
MAPPING = {
    "attention.query_key_value.weight": 0,
    "attention.dense.weight": 1,
    "mlp.dense_h_to_4h.weight": 2,
    "mlp.dense_4h_to_h.weight": 3,
}
PATTERN = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.*)$")

def classify(name):
    m = PATTERN.fullmatch(name)
    if not m or m.group(2) not in MAPPING:
        raise ValueError("not a predeclared GPT-NeoX matrix: " + name)
    return int(m.group(1)), MAPPING[m.group(2)]

def weights(model):
    result = {}
    counts = {}
    for name, value in model.named_parameters():
        try:
            layer, role = classify(name)
        except ValueError:
            continue
        matrix = value.detach().cpu().float().numpy().copy()
        if matrix.ndim != 2 or matrix.shape[1] % 8:
            raise AssertionError("expected source trained two-dimensional width multiple of eight")
        if not np.isfinite(matrix).all():
            raise AssertionError("non-finite source weights")
        result[name] = matrix
        counts[layer] = counts.get(layer, 0) + 1
    if not result or any(count != 4 for count in counts.values()):
        raise AssertionError("incomplete GPT-NeoX attention/MLP role quartet")
    return dict(sorted(result.items()))

def bank_for(mode, name):
    layer, role = classify(name)
    if mode == "global":
        return 0
    if mode == "role":
        return role
    if mode == "layer_control":
        return layer % 4
    raise ValueError(mode)
