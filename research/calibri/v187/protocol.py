"""Frozen 4D cross-row-group geometry, physical budget and source pilot."""
from v175.protocol import SOURCE_SHA,SOURCE_BYTES,TOTAL_UNIQUE
from v184.protocol import MAX_BYTES
MODEL_PILOT=(
 "model.embed_tokens.weight",
 "model.layers.0.self_attn.o_proj.weight",
 "model.layers.0.mlp.gate_proj.weight",
 "model.layers.0.mlp.down_proj.weight",
 "model.layers.15.self_attn.o_proj.weight",
 "model.layers.15.mlp.down_proj.weight",
 "model.layers.29.self_attn.o_proj.weight",
 "model.layers.29.mlp.down_proj.weight",
)
ROW_AXIS=2
COL_AXIS=4
K=32
BITS_PER_TILE=5
VALUES_PER_TILE=8
SOURCE_BATCH=128
SEED=187
REPEATS=5
assert ROW_AXIS*COL_AXIS==VALUES_PER_TILE
assert SOURCE_BYTES==11229303 and MAX_BYTES==11770063
assert len(MODEL_PILOT)==8
