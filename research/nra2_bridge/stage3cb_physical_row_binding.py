"""S3CB research-only physical row binding primitive. No training or evaluation.

A manifest must be generated in an independent, SHA-locked capture stage.
This module never generates an expected manifest from optimizer inputs.
"""
from __future__ import annotations
import hashlib
import json
import torch

FIELDS = ("prompt_sha256", "token_sha256", "mask_sha256",
          "length_sha256", "query_mask_sha256", "source_sha256", "target_sha256")

def digest_tensor(t):
    if not isinstance(t, torch.Tensor):
        raise TypeError("tensor required")
    x = t.detach().cpu().contiguous()
    meta = json.dumps({"dtype": str(x.dtype), "shape": list(x.shape)},
                      sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(meta + b"\0" + x.view(torch.uint8).numpy().tobytes()).hexdigest()

def row_record(prompt_sha, token, mask, length, query_mask, source, target):
    if not isinstance(prompt_sha, str) or len(prompt_sha) != 64:
        raise ValueError("invalid prompt SHA")
    int(prompt_sha, 16)
    return dict(zip(FIELDS, (
        prompt_sha, digest_tensor(token), digest_tensor(mask),
        digest_tensor(length), digest_tensor(query_mask),
        digest_tensor(source), digest_tensor(target)
    )))

def verify_rows(expected, actual):
    """Fail closed on any physical row, order, mask, role or token mismatch."""
    if not isinstance(expected, list) or not isinstance(actual, list):
        raise ValueError("manifest must be ordered lists")
    if len(expected) != len(actual) or not expected:
        raise ValueError("manifest length mismatch")
    for i, (e, a) in enumerate(zip(expected, actual)):
        if set(e) != set(FIELDS) or set(a) != set(FIELDS) or e != a:
            raise ValueError(f"physical row binding mismatch at {i}")
    return True

def self_test():
    def row(i):
        return row_record(f"{i+1:064x}", torch.tensor([[i, i+1]], dtype=torch.float16),
                          torch.tensor([True, True]), torch.tensor(2),
                          torch.tensor([True, False]),
                          torch.tensor(i), torch.tensor(i+1))
    sealed = [row(0), row(1)]
    assert verify_rows(sealed, [row(0), row(1)])
    mutations = [
        [row(1), row(0)],                    # equal-shaped row swap
        [row(0), {**row(1), "token_sha256": row(0)["token_sha256"]}],
        [row(0), {**row(1), "query_mask_sha256": row(0)["query_mask_sha256"]}],
        [row(0)],                            # missing row
    ]
    for candidate in mutations:
        try:
            verify_rows(sealed, candidate)
        except ValueError:
            continue
        raise AssertionError("S3CB negative control unexpectedly accepted")
    changed = row_record(f"{1:064x}", torch.tensor([[0., 1.01]], dtype=torch.float16),
                         torch.tensor([True, True]), torch.tensor(2),
                         torch.tensor([True, False]), torch.tensor(0), torch.tensor(1))
    try:
        verify_rows(sealed, [changed, row(1)])
    except ValueError:
        pass
    else:
        raise AssertionError("S3CB FP16 corruption unexpectedly accepted")
    print("S3CB standalone physical-binding negative controls PASS; no optimizer integration")

if __name__ == "__main__":
    self_test()
