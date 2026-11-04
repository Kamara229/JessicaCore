"""Temporary source-paid module replacements, with mandatory rollback.

No hybrid is a <=0.7 bpp checkpoint: INT6 components have not been encoded
inside CB59's physical budget. Never benchmark hybrid inference as compact.
"""
from contextlib import contextmanager
import torch
from .protocol import FACTORS, validate_geometry

@contextmanager
def swap(packed, int6, *, head=False, block_ids=(), factor=None):
    n = len(packed.model.layers)
    validate_geometry(n)
    if len(int6.model.layers) != n:
        raise AssertionError("source mismatch")
    if factor is not None and factor not in FACTORS:
        raise ValueError("invalid transformer submodule")
    indices = tuple(block_ids)
    if len(indices) != len(set(indices)) or any(not isinstance(i, int) or i < 0 or i >= n for i in indices):
        raise ValueError("duplicate or invalid block")
    updates = []
    def replace(obj, name, item):
        original = getattr(obj, name)
        updates.append((obj, name, original))
        setattr(obj, name, item)
    try:
        if head:
            original_input = int6.model.embed_tokens
            original_output = int6.lm_head
            if original_input.weight.data_ptr() != original_output.weight.data_ptr():
                raise AssertionError("INT6 input/output must be tied")
            replace(packed.model, "embed_tokens", original_input)
            replace(packed, "lm_head", original_output)
        for i in indices:
            if factor is None:
                # ModuleList uses __setitem__, not attribute replacement.
                old = packed.model.layers[i]
                updates.append((packed.model.layers, i, old))
                packed.model.layers[i] = int6.model.layers[i]
            else:
                replace(packed.model.layers[i], factor, getattr(int6.model.layers[i], factor))
        if head and packed.model.embed_tokens.weight.data_ptr() != packed.lm_head.weight.data_ptr():
            raise AssertionError("hybrid tied matrix alias was broken")
        yield packed
    finally:
        for obj, key, old in reversed(updates):
            if isinstance(key, int):
                obj[key] = old
            else:
                setattr(obj, key, old)

def permutations():
    return (
        ("CB59", False, ()),
        ("INT6_tied_head_CB_blocks", True, ()),
        ("CB_head_INT6_blocks", False, tuple(range(30))),
        ("INT6", True, tuple(range(30))),
    )

@torch.inference_mode()
def distributions(model, inputs):
    import torch.nn.functional as F
    logits = model(input_ids=inputs, use_cache=False).logits[0, -1].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite hybrid logits")
    return logits

def compare(reference, target):
    import torch.nn.functional as F
    import torch
    p = F.softmax(reference, dim=-1)
    lp = F.log_softmax(reference, dim=-1)
    lq = F.log_softmax(target, dim=-1)
    k = 10
    a = torch.topk(reference, k).indices
    b = torch.topk(target, k).indices
    return {"kl_from_INT6_nats": float((p * (lp - lq)).sum()),
            "top10_overlap": len(set(a.tolist()) & set(b.tolist())),
            "top1_equal": bool(a[0] == b[0]),
            "first_token_id": int(b[0])}
