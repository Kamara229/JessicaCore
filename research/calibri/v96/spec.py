"""New public synthetic canaries, NOT official dataset TEST or model selection.
All outputs are interpreted as answers to plainly stated questions; compare
FP32, CB59 and same-byte CB59 rewrite to ACTUAL answer keys, never logits.
"""
from __future__ import annotations

CANARIES=(
 ("arithmetic","Question: What is thirteen plus nine? Give just the number. Answer:",("22",)),
 ("arithmetic","Question: What is 30 minus 7? Give just the number. Answer:",("23",)),
 ("arithmetic","Question: What is five times six? Give just the number. Answer:",("30",)),
 ("arithmetic","Question: Half of 18 is what number? Answer:",("9",)),
 ("factual","Question: What is the capital city of Norway? Answer:",("Oslo",)),
 ("factual","Question: Which ocean is the world's largest? Answer:",("Pacific","Pacific Ocean")),
 ("factual","Question: What substance do humans breathe in to stay alive? Answer:",("Oxygen",)),
 ("factual","Question: What is solid frozen water called? Answer:",("Ice",)),
)
OPEN_ENDED=(
 ("instruction","Give two useful steps for cleaning a kitchen table safely:"),
 ("instruction","Explain in plain words how to plant a sunflower seed:"),
 ("reasoning","A child missed the bus to school. Suggest a safe next step:"),
 ("creative","Write one coherent sentence about a lighthouse in a storm:"),
)
SOURCE="fresh synthetic quality canaries, no official TEST; never TRAIN on them"
TOKENS_TO_GENERATE=32
assert len({prompt for _,prompt,_ in CANARIES})==len(CANARIES)
assert len({prompt for _,prompt in OPEN_ENDED})==len(OPEN_ENDED)
assert not {p for _,p,_ in CANARIES}&{p for _,p in OPEN_ENDED}
