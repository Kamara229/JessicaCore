"""Preregistered *new public TRAIN-only* quality probes for native 0.6678 bpp.

Ablations temporarily use uncompressed FP32 modules and are NOT physical
candidates. This can only localize failures and prioritize the next search.
"""
from __future__ import annotations

BOOLQ_RANGE=(8720,8736)
ANSWER_PROMPTS=(
    ("arithmetic","Question: How many days are in one week? Answer:",("7","seven")),
    ("factual","Question: What is the capital of Italy? Answer:",("Rome",)),
)
FREE_PROMPTS=(
    ("instruction","Describe two simple ways to keep a houseplant healthy:"),
    ("creative","Write one short coherent sentence about a lost umbrella:"),
)
# All selected modules exist in original SmolLM2-135M-Instruct.
# Replacement modules MUST come from same exact pretrained model SHA.
TRIALS=(
    ("input_embedding_oracle",("model.embed_tokens",)),
    ("output_head_oracle",("lm_head",)),
    ("tied_input_and_head_oracle",("model.embed_tokens","lm_head")),
    ("early_mlp_oracle",("model.layers.0.mlp.down_proj",)),
    ("middle_attention_value_oracle",("model.layers.13.self_attn.v_proj",)),
    ("late_mlp_oracle",("model.layers.29.mlp.down_proj",)),
)
for kind,mods in TRIALS:
    assert kind and len(mods)==len(set(mods))
    assert len(mods)<=2
assert len({name for name,_ in TRIALS})==len(TRIALS)
assert 0<=BOOLQ_RANGE[0]<BOOLQ_RANGE[1]<=9427
assert BOOLQ_RANGE[0]>8414  # Never reread earlier v95 VALID nor v94.
assert not set(p for _,p,_ in ANSWER_PROMPTS)&set(p for _,p in FREE_PROMPTS)
