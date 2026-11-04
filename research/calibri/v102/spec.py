"""Predeclared NEW synthetic prompts; not official independent TEST nor fitting."""
from __future__ import annotations
TRAIN_SOURCE_SAMPLE_GROUPS=32768
ROW_CHUNK=256
TEST_PROMPTS=(
 ("arithmetic","Question: What is sixteen minus three? Answer:"),
 ("factual","Question: What is the capital of Switzerland? Answer:"),
 ("instruction","Write one clear sentence explaining how to clean a window:"),
 ("creative","Write a short sentence about a red kite in the summer sky:"),
)
GENERATE_PROMPTS=TEST_PROMPTS[0:2]
assert len(TEST_PROMPTS)==len(set(p for _,p in TEST_PROMPTS))
