"""Preregistered *new synthetic* stimuli; do not reuse v97/v98 for fitting.

Each intervention temporarily substitutes EXACT original FP32 hidden residual
at one boundary while leaving EVERY CB59 trained packed weight unchanged.
"""
from __future__ import annotations
PROBES=(
 ("arithmetic","Question: What is six plus eight? Answer:"),
 ("factual","Question: What is the chemical symbol for gold? Answer:"),
 ("instruction","Write one clear sentence explaining how to wash a cup safely:"),
 ("creative","Write a short sentence about a small boat on a quiet lake:"),
)
STAGES=(
 "model.embed_tokens",
 "model.layers.0",
 "model.layers.8",
 "model.layers.9",
 "model.layers.10",
 "model.layers.29",
 "model.norm",
)
GENERATE_ON_CATEGORIES=("arithmetic","instruction")
GENERATE_TOKENS=12
assert len(PROBES)==len({p for _,p in PROBES})
assert len(STAGES)==len(set(STAGES))
assert GENERATE_TOKENS>=8
