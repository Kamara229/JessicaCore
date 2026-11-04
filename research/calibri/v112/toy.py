"""Frozen synthetic controls: role templates and unstructured vectors."""
from __future__ import annotations
import numpy as np
from .roles import MAPPING

def make_toy(structured=True, layers=4, rows=64, cols=64, seed=112):
    """Positive control has 4 distinct *source-defined* vector families.

    Negative control shares one family across all four architectural roles.
    Neither is evidence that real language-model tensors behave this way.
    """
    rng=np.random.default_rng(seed)
    books=rng.normal(0, 1, (4, 25, 8)).astype(np.float32)
    # Separate families by shapes/directions, not just amplitude.
    for k in range(4):
        books[k] += 0.75*np.sin(np.arange(8)[None,:]*(k+1)+k).astype(np.float32)
    result={}
    for layer in range(layers):
        for suffix,role in MAPPING.items():
            group_id = role if structured else 0
            picks = rng.integers(0, 25, (rows,cols//8))
            vectors=books[group_id,picks]
            noise=rng.normal(0,0.065,vectors.shape)
            scales=rng.uniform(0.4,1.7,(rows,1))
            w=((vectors+noise).reshape(rows,cols)*scales).astype(np.float32)
            result[f"gpt_neox.layers.{layer}.{suffix}"]=w
    return dict(sorted(result.items()))
