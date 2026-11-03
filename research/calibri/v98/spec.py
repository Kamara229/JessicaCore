"""Preregistered independent synthetic diagnostic contexts, not official TEST.

Use the *same prefix tokens* in FP32 and CB59 to separate quantization drift
from generated-token feedback; generated free text is assessed separately.
"""
from __future__ import annotations
PROBES=(
 ("arithmetic","Question: What is twelve plus six? Answer:"),
 ("factual","Question: Which planet is the closest to the Sun? Answer:"),
 ("instruction","Explain in one sentence how to water a small houseplant:"),
 ("creative","Finish this sentence: On a foggy morning, the bicycle"),
)
MAX_CONTEXT_TOKENS=48
GENERATE_TOKENS=32
for kind,prompt in PROBES:
    assert kind and len(prompt)>15
assert len({prompt for _,prompt in PROBES})==len(PROBES)
