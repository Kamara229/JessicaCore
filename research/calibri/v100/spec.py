"""New diagnostic text stimuli, NOT independent validation or TEST."""
from __future__ import annotations
PROBES=(
 ("arithmetic","Question: What is nine multiplied by three? Answer:"),
 ("factual","Question: Which gas do plants absorb from the air? Answer:"),
 ("instruction","Explain in one short sentence why we wash hands before eating:"),
 ("creative","Finish one sentence: The small lantern flickered when"),
)
FACTORS=("native","norm_only","head_only","norm_and_head")
GENERATE_DOMAINS=("arithmetic","instruction")
GENERATE_TOKENS=12
assert len(PROBES)==len({p for _,p in PROBES})
assert len(FACTORS)==len(set(FACTORS)) and len(FACTORS)==4
